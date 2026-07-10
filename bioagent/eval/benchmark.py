"""Benchmark runner: cell-type calling → metrics → ablation table.

Reports numbers, not vibes: accuracy + ARI + macro-F1 against held-out reference labels,
structured-output failure rate, calibration (ECE), and latency — one row per ablation.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from bioagent.eval.calling import Ablation, CallingResult, run_calling
from bioagent.eval.datasets import DatasetSpec
from bioagent.eval.metrics import calibration, majority_map, score


def evaluate(ctx, spec: DatasetSpec, ablation: Ablation) -> dict:
    run_id = f"bench_{spec.name}_{uuid.uuid4().hex[:6]}"
    try:
        res: CallingResult = run_calling(ctx, run_id, spec, ablation)
    finally:
        ctx.engine.teardown(run_id, keep_venv=True)

    s = score(res.agent_labels, res.ref_labels)
    mapping = s["label_map"]
    predicted = [mapping.get(a, a) for a in res.agent_labels]
    correct = [p == r for p, r in zip(predicted, res.ref_labels)]
    cal = calibration(res.per_cell_confidence, correct)
    calls = res.stats.get("calls", 0)
    return {
        "ablation": ablation.label(),
        "accuracy": s["accuracy"], "ari": s["ari"], "macro_f1": s["macro_f1"],
        "ece": cal["ece"], "n_clusters": res.n_clusters,
        "n_agent_labels": s["n_agent_labels"], "n_ref_labels": s["n_ref_labels"],
        "structured_failure_rate": round(res.stats.get("failed", 0) / calls, 4) if calls else 0.0,
        "repairs": res.stats.get("repaired", 0), "llm_calls": calls,
        "seconds": res.seconds,
        "per_class": s["per_class"], "calibration": cal, "cluster_calls": res.cluster_calls,
    }


def default_sweep() -> list[Ablation]:
    """Baseline (everything on) then one knob toggled at a time + the critic variable."""
    return [
        Ablation(),                              # full system
        Ablation(grounding=False),               # no NCBI literature
        Ablation(thinking=False),                # no reasoning
        Ablation(reuse_signatures=False),        # no self-discovered-signature reuse
        Ablation(critic=True),                   # + critic pass
    ]


def run_sweep(ctx, spec: DatasetSpec, ablations: list[Ablation] | None = None,
              out_dir: Path | None = None) -> list[dict]:
    ablations = ablations or default_sweep()
    rows = []
    for ab in ablations:
        row = evaluate(ctx, spec, ab)
        rows.append(row)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{spec.name}_{ab.label()}.json").write_text(json.dumps(row, indent=2))
    if out_dir:
        (out_dir / f"{spec.name}_summary.json").write_text(json.dumps(rows, indent=2))
    return rows


def render_table(dataset: str, rows: list[dict]) -> str:
    cols = ["ablation", "accuracy", "ari", "macro_f1", "ece",
            "structured_failure_rate", "n_clusters", "seconds"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [f"### {dataset} — cell-type calling benchmark ({time.strftime('%Y-%m-%d')})",
             "", head, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    # calibration reliability for the baseline row
    base = rows[0] if rows else {}
    bins = (base.get("calibration") or {}).get("bins", [])
    if bins:
        lines += ["", f"**Calibration (baseline, ECE={base['calibration']['ece']}):**",
                  "", "| confidence bin | mean conf | accuracy | n |", "| --- | --- | --- | --- |"]
        for b in bins:
            lines.append(f"| {b['range']} | {b['mean_confidence']} | {b['accuracy']} | {b['count']} |")
    return "\n".join(lines)
