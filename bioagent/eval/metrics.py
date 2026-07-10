"""Cell-type-calling metrics — the "report numbers, not vibes" core.

The agent invents free-text labels (e.g. "Inflammatory Monocyte") that don't string-match
the reference labels (e.g. "Mφ1-Inf"). So we compare partitions two ways:

  * Adjusted Rand Index — label-name-agnostic agreement of the two partitions.
  * Majority-vote mapping — map each agent label to the reference label it most overlaps,
    then report overall accuracy and per-reference-class precision/recall/F1. This mirrors
    standard scRNA annotation benchmarking and needs no name matching.

Plus calibration: does the agent's stated confidence track whether the call is correct
(ECE + a reliability table)? All pure functions — unit-tested without a model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import comb

import numpy as np


def contingency(agent: list[str], ref: list[str]) -> tuple[list[str], list[str], np.ndarray]:
    a_labels = sorted(set(agent))
    r_labels = sorted(set(ref))
    ai = {l: i for i, l in enumerate(a_labels)}
    ri = {l: i for i, l in enumerate(r_labels)}
    m = np.zeros((len(a_labels), len(r_labels)), dtype=int)
    for a, r in zip(agent, ref):
        m[ai[a], ri[r]] += 1
    return a_labels, r_labels, m


def adjusted_rand_index(agent: list[str], ref: list[str]) -> float:
    """ARI from the contingency table (Hubert & Arabie). Chance-corrected in [-~0, 1]."""
    _, _, m = contingency(agent, ref)
    n = m.sum()
    if n < 2:
        return 1.0
    sum_comb_c = sum(comb(int(x), 2) for x in m.sum(axis=0))
    sum_comb_k = sum(comb(int(x), 2) for x in m.sum(axis=1))
    sum_comb = sum(comb(int(x), 2) for x in m.flatten())
    total = comb(int(n), 2)
    expected = (sum_comb_k * sum_comb_c) / total if total else 0.0
    max_index = 0.5 * (sum_comb_k + sum_comb_c)
    denom = max_index - expected
    return 1.0 if denom == 0 else (sum_comb - expected) / denom


def majority_map(agent: list[str], ref: list[str]) -> dict[str, str]:
    """Map each agent label to the reference label it overlaps most (the standard
    interpretable mapping; multiple agent labels may map to one reference label)."""
    overlap: dict[str, Counter] = defaultdict(Counter)
    for a, r in zip(agent, ref):
        overlap[a][r] += 1
    return {a: counts.most_common(1)[0][0] for a, counts in overlap.items()}


def score(agent: list[str], ref: list[str]) -> dict:
    """Overall accuracy + per-reference-class precision/recall/F1 under majority mapping."""
    mapping = majority_map(agent, ref)
    predicted = [mapping[a] for a in agent]
    ref_labels = sorted(set(ref))

    correct = sum(p == r for p, r in zip(predicted, ref))
    accuracy = correct / len(ref) if ref else 0.0

    per_class = {}
    for cls in ref_labels:
        tp = sum(p == cls and r == cls for p, r in zip(predicted, ref))
        fp = sum(p == cls and r != cls for p, r in zip(predicted, ref))
        fn = sum(p != cls and r == cls for p, r in zip(predicted, ref))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[cls] = {"precision": round(prec, 4), "recall": round(rec, 4),
                          "f1": round(f1, 4), "support": tp + fn}
    macro_f1 = round(float(np.mean([c["f1"] for c in per_class.values()])), 4) if per_class else 0.0
    return {"accuracy": round(accuracy, 4), "ari": round(adjusted_rand_index(agent, ref), 4),
            "macro_f1": macro_f1, "per_class": per_class, "label_map": mapping,
            "n_agent_labels": len(set(agent)), "n_ref_labels": len(ref_labels)}


def calibration(confidences: list[float], correct: list[bool], n_bins: int = 10) -> dict:
    """Expected Calibration Error + a reliability table. `correct[i]` is whether the call
    for item i (with stated `confidences[i]`) matched the reference."""
    if not confidences:
        return {"ece": None, "bins": [], "n": 0}
    conf = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece, n = [], 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        acc = float(corr[sel].mean())
        mean_conf = float(conf[sel].mean())
        ece += (cnt / n) * abs(acc - mean_conf)
        bins.append({"range": [float(round(lo, 2)), float(round(hi, 2))], "count": cnt,
                     "mean_confidence": round(mean_conf, 4), "accuracy": round(acc, 4)})
    return {"ece": round(ece, 4), "bins": bins, "n": n}
