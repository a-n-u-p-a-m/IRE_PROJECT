"""
test_outsiders.py - Promotion, false negatives and seed variation, on frames
built so the right answer is known.

    python -m pytest experiments_shubham/tests/test_outsiders.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments_shubham import outsiders
from shared.schema import E_GATE, NON_SURVIVOR, SURVIVOR

FAILURES = pd.DataFrame({"query_id": ["q1", "q1", "q2"], "doc_id": ["a", "b", "c"]})


def _promoted(rows):
    return pd.DataFrame(rows, columns=["policy", "t", "seed", "query_id", "doc_id",
                                       "judged", "utility_treatment"])


def test_gating_failure_set_keeps_useful_non_survivors_only():
    stage = pd.DataFrame({
        "effect_type": [E_GATE] * 4, "stratum": [NON_SURVIVOR, NON_SURVIVOR, SURVIVOR,
                                                NON_SURVIVOR],
        "utility_control": [1.0, 0.0, 2.0, 0.5], "query_id": ["q1"] * 4,
        "doc_id": ["a", "b", "c", "d"]})
    assert set(outsiders.gating_failure_set(stage)["doc_id"]) == {"a", "d"}


def test_false_negatives_count_failures_left_out():
    p = _promoted([("boundary", 0.5, pd.NA, "q1", "a", True, 1.0),
                   ("boundary", 0.5, pd.NA, "q1", "x", False, 0.0),
                   ("boundary", 0.9, pd.NA, "q2", "y", False, 0.0)])
    fn = outsiders.false_negatives(p, FAILURES).set_index("t")
    assert fn.loc[0.5, "found"] == 1 and fn.loc[0.5, "false_negative_rate"] == pytest.approx(2 / 3)
    assert fn.loc[0.5, "queries_all_missed"] == 1                  # q2 had a failure, none found
    assert fn.loc[0.9, "false_negative_rate"] == pytest.approx(1.0)


def test_random_seeds_are_averaged_not_pooled():
    p = _promoted([("random", 0.5, 42, "q1", "a", True, 1.0),
                   ("random", 0.5, 43, "q1", "z", False, 0.0)])
    fn = outsiders.false_negatives(p, FAILURES).iloc[0]
    assert fn["found"] == pytest.approx(0.5) and fn["n_seeds"] == 2
    pr = outsiders.promotion(p, FAILURES, relevant={("q1", "a")}, n_queries=2).iloc[0]
    assert pr["share_relevant"] == pytest.approx(0.5)
    assert pr["gating_failures_promoted"] == pytest.approx(0.5)
    assert pr["promoted_per_query"] == pytest.approx(0.5)


def test_seed_variation_runs_h7_per_seed():
    rng = np.random.default_rng(1)
    qs = [f"q{i}" for i in range(12)]
    diag = pd.DataFrame({"query_id": qs, "g_q": rng.normal(size=12), "reranker_harms": False,
                         "label": ["gate_sensitive"] * 4 + ["rerank_sensitive"] * 8})
    budget = pd.DataFrame([{"query_id": q, "doc_id": "d", "t": t, "seed": s,
                            "effect": float(rng.normal())}
                           for q in qs for t in (0.5, 0.9) for s in (42, 43)])
    sv = outsiders.seed_variation(budget, diag, n_permutations=50, seed=0)
    assert len(sv) == 4 and set(sv["seed"]) == {42, 43}
    one = budget[(budget["seed"] == 42) & (budget["t"] == 0.5)]["effect"].mean()
    got = sv[(sv["seed"] == 42) & (sv["t"] == 0.5)]["mean_delta"].iloc[0]
    assert got == pytest.approx(one)
