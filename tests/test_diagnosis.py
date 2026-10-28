"""
Query-level diagnosis, H7 and the power simulation.

H7 is the only test in the project that points outward: the diagnosis comes
from the reference worlds, the outcome comes from the budget intervention, and
those are genuinely different interventions. It is also the test most likely to
come back null for want of sample size, since it runs only on the reference
subset. `test_power_simulation_*` is what stops that from being discovered
after the GPU budget is spent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.diagnosis import (
    GATE_SENSITIVE,
    MIXED_STABLE,
    RERANK_SENSITIVE,
    diagnose_queries,
    h7_table,
    h7_test,
    query_delta_dcg,
    query_diagnosis,
    required_reference_size,
    smallest_sufficient_size,
)
from shared.worlds import Worlds

TREC_DL = {0: 0.0, 1: 0.0, 2: 3.0, 3: 7.0}


def _worlds(pi_act, pi_no2, pi_no1, candidates, discarded=()):
    return Worlds(
        query_id="q", k_c=len(candidates), candidates=tuple(candidates),
        discarded=tuple(discarded), pi_act=tuple(pi_act), pi_no2=tuple(pi_no2),
        pi_no1=tuple(pi_no1),
    )


def test_toy_query_has_equal_gate_and_rerank_sensitivity(worlds, toy):
    """
    An edge the frozen labelling rule contains, worth pinning so nobody is
    surprised by it in the results.

    In the toy cascade both counterfactual worlds put a gain-7 document at rank
    1 and b at rank 2, so G_q and R_q come out EXACTLY equal. The rule uses
    `G_q > R_q` for gate-sensitive but `R_q >= max(G_q, 0)` for
    reranking-sensitive, so an exact tie breaks towards reranking-sensitive.

    Ties are vanishingly unlikely on real graded data, but the asymmetry is
    arbitrary: it should be a deliberate choice recorded with tau, not an
    artefact of which comparison happened to be written strictly.
    """
    d = query_diagnosis(worlds, toy["relevances"], toy["k"], toy["gain_map"], 0.05)
    assert d["g_q"] == pytest.approx(d["r_q"])
    assert d["label"] == RERANK_SENSITIVE
    assert d["dcg_no_gate"] > d["dcg_actual"]


def test_gate_sensitive_when_the_gate_costs_more_than_the_reranker(worlds, toy):
    """
    Break the tie by making the reranker harmless: give a the same Stage-2
    rank it has in the no-reranker world. The gate still discards e, so the
    query becomes unambiguously gate-sensitive.
    """
    rel = dict(toy["relevances"])
    w = _worlds(
        pi_act=["b", "c", "d", "a"],
        pi_no2=["b", "c", "d", "a"],     # reranking changes nothing
        pi_no1=worlds.pi_no1,
        candidates=worlds.candidates,
        discarded=worlds.discarded,
    )
    d = query_diagnosis(w, rel, toy["k"], toy["gain_map"], 0.05)
    assert d["label"] == GATE_SENSITIVE
    assert d["g_q"] > 0
    assert d["r_q"] == pytest.approx(0.0)


def test_rerank_sensitive_when_the_reranker_does_the_work():
    rel = {"a": 3, "b": 0, "c": 0}
    w = _worlds(pi_act=["a", "b"], pi_no2=["b", "a"], pi_no1=["a", "b", "c"],
                candidates=["a", "b"], discarded=["c"])
    d = query_diagnosis(w, rel, 2, TREC_DL, 0.05)
    assert d["label"] == RERANK_SENSITIVE
    assert d["rerank_signed"] > 0
    assert not d["reranker_harms"]


def test_reranker_harm_is_recorded_separately():
    """
    R_q is an absolute value, so a query the reranker DAMAGES earns the same
    label as one it helps - with the opposite repair. The signed value is kept
    so the two can be split without touching the frozen labelling rule.
    """
    rel = {"a": 3, "b": 0, "c": 0}
    w = _worlds(pi_act=["b", "a"], pi_no2=["a", "b"], pi_no1=["a", "b", "c"],
                candidates=["a", "b"], discarded=["c"])
    d = query_diagnosis(w, rel, 2, TREC_DL, 0.05)
    assert d["label"] == RERANK_SENSITIVE
    assert d["rerank_signed"] < 0
    assert d["reranker_harms"] is True


def test_stable_query_when_nothing_moves():
    rel = {"a": 3, "b": 0, "c": 0}
    w = _worlds(pi_act=["a", "b"], pi_no2=["a", "b"], pi_no1=["a", "b", "c"],
                candidates=["a", "b"], discarded=["c"])
    assert query_diagnosis(w, rel, 2, TREC_DL, 0.05)["label"] == MIXED_STABLE


def test_tau_controls_the_labelling():
    """tau gates group membership for H7, which is why it is frozen in config."""
    rel = {"a": 2, "b": 0, "c": 3}
    w = _worlds(pi_act=["a", "b"], pi_no2=["a", "b"], pi_no1=["c", "a", "b"],
                candidates=["a", "b"], discarded=["c"])
    assert query_diagnosis(w, rel, 2, TREC_DL, 0.05)["label"] == GATE_SENSITIVE
    assert query_diagnosis(w, rel, 2, TREC_DL, 100.0)["label"] == MIXED_STABLE


def test_diagnose_queries_returns_one_row_per_query(worlds, toy):
    out = diagnose_queries({"q1": worlds}, {"q1": toy["relevances"]},
                           toy["k"], toy["gain_map"], 0.05)
    assert len(out) == 1
    assert set(["query_id", "g_q", "r_q", "label"]).issubset(out.columns)


# ── outcomes ──────────────────────────────────────────────────────────

def test_delta_dcg_is_zero_at_the_baseline(t_worlds, toy):
    deltas = query_delta_dcg(t_worlds, toy["relevances"], toy["k"], toy["gain_map"])
    assert deltas[1.0] == pytest.approx(0.0)


def test_lower_t_helps_the_toy_query(t_worlds, toy):
    """
    The toy query is gate-sensitive, and promoting the discarded document e
    improves it. That is the mechanism H7 tests at scale.
    """
    deltas = query_delta_dcg(t_worlds, toy["relevances"], toy["k"], toy["gain_map"])
    assert deltas[0.5] > 0


def test_h7_table_drops_the_baseline_row():
    diag = pd.DataFrame([{"query_id": "q1", "label": GATE_SENSITIVE,
                          "reranker_harms": False}])
    table = h7_table(diag, {"q1": {1.0: 0.0, 0.5: 1.0}})
    assert set(table["t"]) == {0.5}


# ── the test itself ───────────────────────────────────────────────────

def _h7_frame(gate_values, rerank_values, t=0.5):
    rows = [
        {"query_id": f"g{i}", "t": t, "label": GATE_SENSITIVE,
         "reranker_harms": False, "delta_dcg": v}
        for i, v in enumerate(gate_values)
    ] + [
        {"query_id": f"r{i}", "t": t, "label": RERANK_SENSITIVE,
         "reranker_harms": False, "delta_dcg": v}
        for i, v in enumerate(rerank_values)
    ]
    return pd.DataFrame(rows)


def test_h7_detects_a_real_separation():
    rng = np.random.default_rng(0)
    gate = rng.normal(2.0, 0.3, 25)
    rerank = rng.normal(0.0, 0.3, 25)
    res = h7_test(_h7_frame(gate, rerank), n_permutations=2000, seed=1)
    assert len(res) == 1
    assert res.iloc[0]["difference"] > 1.0
    assert res.iloc[0]["p_value"] < 0.01


def test_h7_does_not_cry_wolf_on_noise():
    rng = np.random.default_rng(3)
    gate = rng.normal(0.0, 1.0, 25)
    rerank = rng.normal(0.0, 1.0, 25)
    res = h7_test(_h7_frame(gate, rerank), n_permutations=2000, seed=2)
    assert res.iloc[0]["p_value"] > 0.05


def test_h7_refuses_to_test_a_tiny_group_and_says_so():
    """
    A null on groups of one means nothing, so the test reports the reason
    instead of a p-value. ALWAYS read the group sizes before reading the
    p-value.
    """
    res = h7_test(_h7_frame([1.0], [0.0, 0.1, 0.2]), n_permutations=100)
    assert np.isnan(res.iloc[0]["p_value"])
    assert "too few queries" in res.iloc[0]["note"]
    assert res.iloc[0]["n_gate_sensitive"] == 1


def test_h7_reports_group_sizes_always():
    res = h7_test(_h7_frame([1, 2, 3], [0, 0, 0]), n_permutations=500)
    assert res.iloc[0]["n_gate_sensitive"] == 3
    assert res.iloc[0]["n_rerank_sensitive"] == 3


# ── power ─────────────────────────────────────────────────────────────

def _simulator(effect_size: float):
    def simulate(rng):
        label = rng.choice([GATE_SENSITIVE, RERANK_SENSITIVE, MIXED_STABLE],
                           p=[0.35, 0.35, 0.30])
        mean = effect_size if label == GATE_SENSITIVE else 0.0
        return label, float(rng.normal(mean, 1.0))
    return simulate


def _rare_gate_simulator(effect_size: float, gate_share: float):
    """
    Gate-sensitive queries are rare, as the Week-1 simulation found them to be
    (2-7% across universe sizes and candidate depths). With a small subset,
    most draws then put fewer than two queries in that group and cannot be
    tested at all.
    """
    def simulate(rng):
        label = rng.choice(
            [GATE_SENSITIVE, RERANK_SENSITIVE, MIXED_STABLE],
            p=[gate_share, 1.0 - gate_share - 0.05, 0.05],
        )
        mean = effect_size if label == GATE_SENSITIVE else 0.0
        return label, float(rng.normal(mean, 1.0))
    return simulate


def test_power_simulation_finds_a_sufficient_size():
    out = required_reference_size(
        _simulator(2.0), candidate_sizes=[10, 30, 60],
        n_simulations=40, alpha=0.05, target_power=0.8, seed=5,
    )
    assert list(out["reference_size"]) == [10, 30, 60]
    assert (out["power"].diff().dropna() >= -0.2).all(), "power should rise with n"
    n = smallest_sufficient_size(out)
    assert n is not None and n <= 60


def test_power_simulation_reports_failure_rather_than_guessing():
    """
    With no real separation nothing reaches the target, and the function says
    so instead of returning a number someone would then put in the config.
    """
    out = required_reference_size(
        _simulator(0.0), candidate_sizes=[10, 20],
        n_simulations=30, alpha=0.05, target_power=0.8, seed=6,
    )
    assert smallest_sufficient_size(out) is None


def test_untestable_draws_count_as_misses():
    """
    The denominator is n_simulations, not the number of testable draws.

    A subset that puts fewer than two queries in a group yields no p-value, and
    in a real run at that size you would have no answer - a failure to detect,
    not a sample to throw away. When gate-sensitive queries are rare most draws
    are untestable, and conditioning on the testable ones overstates power
    several-fold. This is the exact trap the H7 Week-1 simulation hit.
    """
    out = required_reference_size(
        _rare_gate_simulator(5.0, gate_share=0.04), candidate_sizes=[10],
        n_simulations=60, alpha=0.05, target_power=0.8, seed=11,
    )
    row = out.iloc[0]
    assert row["n_testable"] < row["n_simulations"], (
        "the fixture no longer produces untestable draws, so this test proves "
        "nothing; lower gate_share or the reference size"
    )
    assert row["power"] == pytest.approx(
        row["power_given_testable"] * row["n_testable"] / row["n_simulations"]
    )
    assert row["power"] < row["power_given_testable"], (
        "conditioning on testable draws must inflate the figure, which is why "
        "meets_target keys off the unconditional one"
    )
    assert row["testable_fraction"] == pytest.approx(
        row["n_testable"] / row["n_simulations"]
    )


def test_meets_target_uses_the_unconditional_power():
    out = required_reference_size(
        _rare_gate_simulator(5.0, gate_share=0.04), candidate_sizes=[10],
        n_simulations=60, alpha=0.05, target_power=0.8, seed=11,
    )
    row = out.iloc[0]
    assert bool(row["meets_target"]) == bool(row["power"] >= 0.8)
    assert smallest_sufficient_size(out) is None


def test_n_usable_is_gone_so_stale_corrections_break_loudly():
    """
    `experiments_shubham/power_h7.py` rescaled `power * n_usable /
    n_simulations` to work around the old conditional denominator. With the
    denominator fixed, that rescaling would wrongly shrink the figure a second
    time. Dropping the column makes the stale line raise a KeyError instead of
    silently returning a smaller number.
    """
    out = required_reference_size(
        _simulator(2.0), candidate_sizes=[10], n_simulations=10, seed=1,
    )
    assert "n_usable" not in out.columns
