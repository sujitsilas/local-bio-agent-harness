"""The one thing the agent does: call cell types. Standalone, ablatable, instrumented.

Given a dataset, the agent clusters INDEPENDENTLY (leiden), then calls each cluster's
identity de novo from its DE markers + (optionally) NCBI literature + (optionally) its own
previously-discovered signatures. Every knob the benchmark ablates is a field on `Ablation`.
Returns per-cell agent/reference labels + per-cluster confidence + structured-failure stats
— everything metrics.py needs. No downstream analysis, figures, or reports.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from bioagent.eval.annotation import SCORE_CODE, SYSTEM, Calls, ground_markers
from bioagent.eval.datasets import DatasetSpec
from bioagent.llm.structured import generate_structured
from bioagent.models import Sampling


@dataclass
class Ablation:
    grounding: bool = True         # NCBI literature grounding on the DE markers
    thinking: bool = True          # stream reasoning on the first attempt
    reuse_signatures: bool = True  # score against the agent's own discovered signatures
    critic: bool = False           # a critic pass that can revise low-confidence calls
    model_profile: str = "primary"
    resolution: float = 1.0        # leiden resolution for the agent's clustering
    batch_size: int = 2
    critic_threshold: float = 0.6  # clusters below this get a critic re-look

    def label(self) -> str:
        on = lambda b: "on" if b else "off"  # noqa: E731
        return (f"ground={on(self.grounding)},think={on(self.thinking)},"
                f"reuse={on(self.reuse_signatures)},critic={on(self.critic)}")


@dataclass
class CallingResult:
    agent_labels: list[str]
    ref_labels: list[str]
    per_cell_confidence: list[float]
    cluster_calls: list[dict]      # {cluster, label, confidence, n}
    stats: dict                    # {calls, repaired, failed}
    seconds: float
    n_clusters: int
    species: str = "unknown"


_CRITIC_SYS = """You are a validation critic re-examining a low-confidence cell-type call.
Given a cluster's DE markers, literature, and the initial call, decide the most defensible
label and a calibrated confidence in [0,1]. If the initial call is well supported, keep it;
if the markers point elsewhere, revise it. Be honest — do not inflate confidence."""


def run_calling(ctx, run_id: str, spec: DatasetSpec, ablation: Ablation) -> CallingResult:
    t0 = time.time()
    ctx.engine.provision(run_id)
    stats: dict = {}

    # 1. load + independent clustering (the agent never sees the reference column)
    ctx.engine.run_code(run_id, spec.load_code, timeout_s=1200)
    _prepare_clustering(ctx, run_id, spec, ablation.resolution)
    species = _species(ctx, run_id)

    # 2. per-cluster evidence: DE markers (+ optional self-discovered signature scores)
    sigs = ({s.name: s.genes for s in ctx.kb.signatures() if s.genes}
            if ablation.reuse_signatures else {})
    out = ctx.engine.run_code(
        run_id, SCORE_CODE.format(h="adata", key="agent_leiden",
                                  sigs_json=json.dumps(sigs).replace("'", "\\'")), timeout_s=900)
    evidence = _parse(out.stdout, {})

    # 3. optional NCBI literature grounding on the DE markers
    pubmed = _pubmed(ctx) if ablation.grounding else None
    literature = ground_markers(ctx.kb, evidence, pubmed, species, ctx.tracer)

    # 4. de novo calls, batched (thinking on first attempt per ablation)
    calls = _call_clusters(ctx, evidence, literature, sigs, species, ablation, stats)

    # 5. optional critic pass on low-confidence calls (a controlled multi-agent variable)
    if ablation.critic:
        calls = _critic_pass(ctx, calls, evidence, literature, ablation, stats)

    # 6. write labels, extract per-cell vectors (labels only — never expression)
    mapping = {c["cluster"]: c["label"] for c in calls}
    conf = {c["cluster"]: c["confidence"] for c in calls}
    vectors = _extract_vectors(ctx, run_id, spec.ref_col, mapping, conf)

    counts = {c["cluster"]: 0 for c in calls}
    for cl in vectors["cluster"]:
        counts[cl] = counts.get(cl, 0) + 1
    cluster_calls = [{"cluster": c["cluster"], "label": c["label"],
                      "confidence": c["confidence"], "n": counts.get(c["cluster"], 0)}
                     for c in calls]
    return CallingResult(
        agent_labels=vectors["agent"], ref_labels=vectors["ref"],
        per_cell_confidence=[conf.get(cl, 0.0) for cl in vectors["cluster"]],
        cluster_calls=cluster_calls, stats=stats, seconds=round(time.time() - t0, 1),
        n_clusters=len(calls), species=species)


# --------------------------------------------------------------------------- #
def _prepare_clustering(ctx, run_id: str, spec: DatasetSpec, resolution: float) -> None:
    code = "import scanpy as sc\n"
    if not spec.normalized:
        code += "sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)\n"
    emb = spec.embedding
    if not emb:
        code += ("sc.pp.highly_variable_genes(adata, n_top_genes=2000)\n"
                 "sc.pp.pca(adata, n_comps=50)\n")
        emb = "X_pca"
    code += (f"sc.pp.neighbors(adata, use_rep={emb!r})\n"
             f"sc.tl.leiden(adata, resolution={resolution}, key_added='agent_leiden')\n"
             "print(adata.obs['agent_leiden'].nunique())")
    ctx.engine.run_code(run_id, code, timeout_s=900)


def _species(ctx, run_id: str) -> str:
    out = ctx.engine.run_code(
        run_id, "print('mouse' if any(str(g)[:1].isupper() and str(g)[1:].islower() "
                "for g in list(adata.var_names[:20])) else 'human')")
    for line in reversed(out.stdout.strip().splitlines()):
        if line.strip() in ("mouse", "human"):
            return line.strip()
    return "unknown"


def _call_clusters(ctx, evidence, literature, sigs, species, ablation: Ablation, stats) -> list[dict]:
    sig_names = list(sigs)
    clusters = list(evidence)
    calls: list[dict] = []
    for i in range(0, len(clusters), ablation.batch_size):
        batch = clusters[i:i + ablation.batch_size]
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (
                f"Species: {species}. Self-discovered signatures available: {sig_names}\n\n"
                f"Call THESE clusters: {batch}\n\n"
                f"Evidence:\n{json.dumps({c: evidence[c] for c in batch}, indent=2)}\n\n"
                f"Literature:\n{json.dumps({c: literature.get(c, []) for c in batch})}\n\n"
                "Return grounded, cited calls as JSON (one per cluster).")},
        ]
        batch_calls = generate_structured(
            ctx.provider, messages, Calls, model_profile=ablation.model_profile,
            sampling=Sampling(temperature=0.1),
            first_thinking=ablation.thinking, stats=stats).calls
        for c in batch_calls:
            calls.append({"cluster": c.cluster, "label": c.label,
                          "confidence": float(c.confidence),
                          "markers": c.supporting_markers})
    # any cluster the model skipped -> explicit low-confidence unassigned
    called = {c["cluster"] for c in calls}
    for cl in clusters:
        if cl not in called:
            calls.append({"cluster": cl, "label": "unassigned", "confidence": 0.0, "markers": []})
    return calls


def _critic_pass(ctx, calls, evidence, literature, ablation: Ablation, stats) -> list[dict]:
    from pydantic import BaseModel

    class Revision(BaseModel):
        label: str
        confidence: float
        rationale: str = ""

    revised = []
    for c in calls:
        if c["confidence"] >= ablation.critic_threshold or c["cluster"] not in evidence:
            revised.append(c)
            continue
        ev = evidence[c["cluster"]]
        messages = [
            {"role": "system", "content": _CRITIC_SYS},
            {"role": "user", "content": (
                f"Initial call: {c['label']} (confidence {c['confidence']}).\n"
                f"Top DE markers: {ev.get('top_markers', [])}\n"
                f"Literature: {literature.get(c['cluster'], [])}\n\nReturn the revised call.")},
        ]
        r = generate_structured(ctx.provider, messages, Revision,
                                model_profile=ablation.model_profile,
                                sampling=Sampling(temperature=0.0),
                                first_thinking=ablation.thinking, stats=stats)
        revised.append({**c, "label": r.label, "confidence": float(r.confidence)})
    return revised


def _extract_vectors(ctx, run_id: str, ref_col: str, mapping: dict, conf: dict) -> dict:
    code = (
        "import json as _j\n"
        f"_m = _j.loads('''{json.dumps(mapping)}''')\n"
        "adata.obs['agent_cell_type'] = adata.obs['agent_leiden'].astype(str).map(_m).fillna('unassigned')\n"
        f"_ref = adata.obs[{ref_col!r}].astype(str).tolist()\n"
        "print(_j.dumps({'agent': adata.obs['agent_cell_type'].astype(str).tolist(), "
        "'ref': _ref, 'cluster': adata.obs['agent_leiden'].astype(str).tolist()}))"
    )
    out = ctx.engine.run_code(run_id, code, timeout_s=300)
    return _parse(out.stdout, {"agent": [], "ref": [], "cluster": []})


def _parse(stdout: str, default):
    for line in reversed(stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return default


def _pubmed(ctx):
    try:
        from bioagent.tools.apis.base import KnowledgeClient
        from bioagent.tools.apis.pubmed import PubMedPlugin

        return PubMedPlugin(KnowledgeClient(ctx.config))
    except Exception:
        return None
