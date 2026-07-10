"""De novo cell-type annotation primitives (self-contained; no pipeline dependency).

The agent calls each cluster's identity from its DE marker genes + NCBI literature. There
is NO pre-loaded gene→cell-type dictionary; any signature scores shown are ones the agent
discovered on prior data (weak hints only). These pieces are shared by the calling core.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from bioagent.bio.priors_kb import PriorsKB


class Call(BaseModel):
    cluster: str
    label: str
    confidence: float = 0.5
    supporting_markers: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    rationale: str = ""


class Calls(BaseModel):
    calls: list[Call]


# Runs in the Run kernel: DE markers per cluster (+ optional signature scores). Prints JSON.
SCORE_CODE = r"""
import json as _j, scanpy as _sc, numpy as _np, pandas as _pd
_a = {h}; _key = {key!r}
_lower = {{str(g).lower(): g for g in _a.var_names}}
_sigs = _j.loads('''{sigs_json}''')
_scored = []
for _name, _genes in _sigs.items():
    _present = [_lower[g.lower()] for g in _genes if g.lower() in _lower]
    if len(_present) >= 2:
        _sc.tl.score_genes(_a, _present, score_name='sig_'+_name)
        _scored.append(_name)
_sc.tl.rank_genes_groups(_a, _key, method='wilcoxon', n_genes=15)
_names = _pd.DataFrame(_a.uns['rank_genes_groups']['names'])
_cats = list(_a.obs[_key].cat.categories) if hasattr(_a.obs[_key],'cat') else sorted(map(str,_a.obs[_key].unique()))
_out = {{}}
for _cl in _cats:
    _mask = _a.obs[_key].astype(str) == str(_cl)
    _scores = {{_n: round(float(_a.obs.loc[_mask, 'sig_'+_n].mean()), 3) for _n in _scored}}
    _top = [str(x) for x in _names[str(_cl)].head(12).tolist()] if str(_cl) in _names else []
    _out[str(_cl)] = {{'n': int(_mask.sum()), 'top_markers': _top, 'sig_scores': _scores}}
print(_j.dumps(_out))
"""

SYSTEM = """You call single-cell cluster identities DE NOVO. There is NO predefined marker
dictionary — you must recognize the cell type from the cluster's differentially expressed
genes and the scientific literature retrieved for those genes. (Any signature scores you
are shown were discovered by the agent on prior data, not curated — treat them as weak
hints only, never as ground truth.) For each cluster you get its top DE marker genes,
optional self-discovered signature scores, and literature snippets for its markers. Return
per cluster:
  - label: the single most likely cell type, whatever the DE genes + literature support.
  - supporting_markers: the 3-6 genes (from the DE list) that define the call.
  - citations: reference ids (PMID:…) that support the call — ONLY ids present in the
    provided evidence; never invent one.
  - confidence in [0,1]; rationale in one line naming the DE genes / literature you used.
Do not claim literature support you were not given. If nothing grounds a marker, lower
confidence and say so.
"""


def ground_markers(kb: PriorsKB, evidence: dict, pubmed, species: str, tracer=None) -> dict:
    """Objectively search NCBI/PubMed for each cluster's DE markers. Best-effort; returns
    {cluster -> [literature snippets]}. Emits `search` events for any live viewer."""
    literature: dict[str, list[str]] = {}
    if pubmed is None:
        return literature
    failures = 0
    for cl, ev in evidence.items():
        markers = ev.get("top_markers", [])[:4]
        if not markers:
            continue
        sp = "" if species.lower() in ("", "unknown") else f" AND {species}"
        term = f"({' OR '.join(markers)}){sp} AND (marker OR \"cell type\" OR \"single-cell\")"
        try:
            hits = pubmed.search(term, retmax=3)
            literature[cl] = [f"[{h['citation']}] {h['title']} ({h.get('year','')})" for h in hits]
            if tracer:
                tracer.emit("search", source="pubmed", query=", ".join(markers), hits=len(hits))
        except Exception:
            failures += 1
            if failures >= 3:
                break
    return literature
