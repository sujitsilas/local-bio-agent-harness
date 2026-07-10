"""Cell-type-calling metrics: ARI, majority-mapped accuracy/PRF, calibration/ECE."""
from __future__ import annotations

from bioagent.eval.metrics import adjusted_rand_index, calibration, majority_map, score


def test_ari_perfect_and_permutation_invariant():
    agent = ["a", "a", "b", "b", "c", "c"]
    ref = ["x", "x", "y", "y", "z", "z"]
    assert adjusted_rand_index(agent, ref) == 1.0
    # relabeling the agent partition must not change ARI (name-agnostic)
    relabel = ["p", "p", "q", "q", "r", "r"]
    assert adjusted_rand_index(relabel, ref) == 1.0
    # an independent partition scores far below perfect
    indep = ["a", "b", "a", "b", "a", "b"]
    assert adjusted_rand_index(indep, ref) < 0.5


def test_majority_map_and_scoring():
    agent = ["Mono", "Mono", "Mac", "Mac", "Mac"]
    ref = ["M1", "M1", "M2", "M2", "M1"]
    assert majority_map(agent, ref) == {"Mono": "M1", "Mac": "M2"}
    s = score(agent, ref)
    assert s["accuracy"] == 0.8  # 4/5 correct under the mapping
    assert s["per_class"]["M1"]["precision"] == 1.0
    assert round(s["per_class"]["M1"]["recall"], 3) == 0.667
    assert round(s["per_class"]["M2"]["precision"], 3) == 0.667
    assert s["per_class"]["M2"]["recall"] == 1.0


def test_over_splitting_still_maps():
    # agent over-splits M1 into two labels -> both map to M1, accuracy stays perfect
    agent = ["a1", "a2", "b", "b"]
    ref = ["M1", "M1", "M2", "M2"]
    s = score(agent, ref)
    assert s["accuracy"] == 1.0
    assert s["n_agent_labels"] == 3 and s["n_ref_labels"] == 2


def test_calibration_ece():
    conf = [0.9, 0.9, 0.5, 0.5]
    correct = [True, True, False, True]
    cal = calibration(conf, correct, n_bins=10)
    # bin (0.8,0.9]: acc 1.0 vs conf 0.9 -> gap .1, weight .5 -> .05
    # bin (0.4,0.5]: acc 0.5 vs conf 0.5 -> gap 0
    assert cal["ece"] == 0.05
    assert cal["n"] == 4


def test_calibration_empty():
    assert calibration([], [])["ece"] is None
