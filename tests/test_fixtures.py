"""
The shared toy cascade.

Section 14 asks each member to keep one GPU-free smoke test and one minimal
fixture. This is the common one, so three people disagreeing means a real
disagreement rather than two different toys. Run `shared.fixtures.smoke_test()`
before pushing anything.
"""

from __future__ import annotations

from shared.fixtures import (
    TOY_BUDGET,
    TOY_K,
    TOY_K_C,
    smoke_test,
    toy_cascade,
    toy_effects,
    toy_t_worlds,
    toy_worlds,
)
from shared.schema import DELTA_T, E_GATE, E_RERANK, T_STRATA


def test_smoke_test_passes():
    assert smoke_test() is True


def test_fixture_exercises_every_stratum_with_a_nonzero_value():
    """
    A fixture where the interesting cases are all empty passes every test and
    proves nothing, so check the coverage of the fixture itself.
    """
    eff = toy_effects()
    for effect_type in (E_RERANK, E_GATE, DELTA_T):
        block = eff[eff["effect_type"] == effect_type]
        assert len(block), f"no rows for {effect_type}"
        nonzero = block[block["effect"].abs() > 1e-9]
        assert len(nonzero), f"{effect_type} has no nonzero effect anywhere"

    delta = eff[eff["effect_type"] == DELTA_T]
    assert set(delta["stratum"].unique()) == set(T_STRATA), (
        "the t-allocation does not populate all four membership strata"
    )


def test_fixture_contains_both_a_gate_failure_and_a_reranker_failure():
    eff = toy_effects()
    gate = eff[(eff["effect_type"] == E_GATE) & (eff["doc_id"] == "e")]
    rerank = eff[(eff["effect_type"] == E_RERANK) & (eff["doc_id"] == "a")]
    assert float(gate["effect"].iloc[0]) < -1.0, "no gate failure in the fixture"
    assert float(rerank["effect"].iloc[0]) < -1.0, "no reranker failure in the fixture"


def test_budget_equals_candidate_depth():
    """So t=1 is the conventional cascade rather than a budget baseline."""
    assert TOY_BUDGET == TOY_K_C
    assert toy_t_worlds()[1.0].is_conventional_baseline


def test_cutoff_is_smaller_than_the_candidate_set():
    assert TOY_K < TOY_K_C < len(toy_cascade()["universe"])


def test_deep_outsider_is_invisible_to_the_boundary_policy():
    """
    Document i has a high Stage-2 score but a low Stage-1 rank, so a boundary
    policy never reaches it. That is the false-negative mode the proposal
    commits to measuring, built into the fixture on purpose.
    """
    boundary = toy_t_worlds(t_grid=(1.0, 0.5), policy="boundary")[0.5]
    assert "i" not in boundary.submitted
    assert toy_cascade()["s2"]["i"] > toy_cascade()["s2"]["c"]


def test_worlds_are_stable_across_calls():
    assert toy_worlds().pi_act == toy_worlds().pi_act
    assert toy_t_worlds()[0.5].pi_t == toy_t_worlds()[0.5].pi_t
