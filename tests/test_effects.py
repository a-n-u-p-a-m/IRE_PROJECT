"""
The three estimands and their SEPARATE invariants.

The point this file defends: e_rerank, e_gate and delta_t do not share sign
invariants, so one unconditional validator would either reject correct delta_t
rows or wave through incorrect e_gate rows. `test_validator_dispatches_on_
effect_type` is the direct check that the dispatch works in both directions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from shared.effects import (
    GATING_FAILURE,
    budget_effects_for_query,
    check_additivity,
    check_invariants,
    concat_effects,
    gating_failures,
    label_documents,
    recovered_outsiders,
    stage_effects_for_query,
    summarise,
)
from shared.schema import (
    DELTA_T,
    E_GATE,
    E_RERANK,
    NON_SURVIVOR,
    REMOVED,
    RETAINED,
    SELECTED_OUTSIDER,
    SURVIVOR,
    UNSELECTED_OUTSIDER,
    SchemaError,
)
from shared.strata import check_strata_valid
from shared.worlds import build_t_grid, build_worlds


def _pick(eff, effect_type, doc_id, t=None):
    sel = (eff["effect_type"] == effect_type) & (eff["doc_id"] == doc_id)
    if t is not None:
        sel &= eff["t"] == t
    rows = eff[sel]
    assert len(rows) == 1, f"expected one row, got {len(rows)}"
    return rows.iloc[0]


# ── hand-computed values ──────────────────────────────────────────────

def test_rerank_effect_hand_computation(effects):
    # pi_no2 = (a,b,c,d): a is rank 1 with gain 7. pi_act = (b,c,d,a): a falls
    # to rank 4, outside k=2, so it contributes nothing.
    assert _pick(effects, E_RERANK, "a")["effect"] == pytest.approx(-7.0)
    # b rises from rank 2 to rank 1.
    assert _pick(effects, E_RERANK, "b")["effect"] == pytest.approx(
        3.0 - 3.0 / math.log2(3)
    )


def test_gate_effect_hand_computation(effects):
    # e has the best Stage-2 score of all ten documents and the gate discarded
    # it. In the no-gate world it ranks first and contributes gain 7.
    row = _pick(effects, E_GATE, "e")
    assert row["effect"] == pytest.approx(-7.0)
    assert row["stratum"] == NON_SURVIVOR
    assert row["utility_control"] == pytest.approx(7.0)


def test_delta_t_hand_computation(effects):
    # At t=0.5 the boundary policy promotes e, which takes rank 1.
    assert _pick(effects, DELTA_T, "e", t=0.5)["effect"] == pytest.approx(7.0)
    # b is retained but loses rank 1 to e.
    assert _pick(effects, DELTA_T, "b", t=0.5)["effect"] == pytest.approx(
        3.0 / math.log2(3) - 3.0
    )
    # i is never selected by the boundary policy: structurally zero.
    assert _pick(effects, DELTA_T, "i", t=0.5)["effect"] == pytest.approx(0.0)


# ── the invariants, per effect type ───────────────────────────────────

def test_rerank_is_zero_off_the_candidate_set(effects):
    block = effects[effects["effect_type"] == E_RERANK]
    outside = block[block["stratum"] == NON_SURVIVOR]
    assert len(outside) == 6
    assert (outside["effect"].abs() < 1e-12).all()


def test_gate_signs_follow_survivorship(effects):
    block = effects[effects["effect_type"] == E_GATE]
    assert (block.loc[block["stratum"] == SURVIVOR, "effect"] >= -1e-12).all()
    assert (block.loc[block["stratum"] == NON_SURVIVOR, "effect"] <= 1e-12).all()


def test_delta_t_signs_follow_allocation_membership(effects):
    block = effects[effects["effect_type"] == DELTA_T]
    assert (block.loc[block["stratum"] == REMOVED, "effect"] <= 1e-12).all()
    assert (block.loc[block["stratum"] == SELECTED_OUTSIDER, "effect"] >= -1e-12).all()
    assert (
        block.loc[block["stratum"] == UNSELECTED_OUTSIDER, "effect"].abs() < 1e-12
    ).all()
    # The retained stratum is the only one free to move either way, which is
    # why it is the one carrying magnitude information.
    retained = block.loc[block["stratum"] == RETAINED, "effect"]
    assert (retained < -1e-12).any(), "no retained document lost ground; the "\
        "fixture no longer exercises the interesting stratum"


def test_strata_belong_to_their_effect_type(effects):
    check_strata_valid(effects)


def test_validator_dispatches_on_effect_type(effects):
    """
    The same numeric value is legal for one effect type and illegal for
    another. This is exactly what a single unconditional validator gets wrong.
    """
    # A negative value in the retained stratum is fine.
    ok = effects.copy()
    idx = ok.index[(ok["effect_type"] == DELTA_T) & (ok["stratum"] == RETAINED)][0]
    ok.loc[idx, "utility_treatment"] = ok.loc[idx, "utility_control"] - 1.0
    ok.loc[idx, "effect"] = ok.loc[idx, "utility_treatment"] - ok.loc[idx, "utility_control"]
    check_invariants(ok)            # must not raise

    # The same negative value in the selected-outsider stratum is not.
    bad = effects.copy()
    idx = bad.index[
        (bad["effect_type"] == DELTA_T) & (bad["stratum"] == SELECTED_OUTSIDER)
    ][0]
    bad.loc[idx, "utility_treatment"] = bad.loc[idx, "utility_control"] - 1.0
    bad.loc[idx, "effect"] = bad.loc[idx, "utility_treatment"] - bad.loc[idx, "utility_control"]
    with pytest.raises(SchemaError, match="selected_outsider"):
        check_invariants(bad)


def test_definition_violation_is_caught(effects):
    bad = effects.copy()
    bad.loc[bad.index[0], "effect"] = bad.loc[bad.index[0], "effect"] + 0.5
    with pytest.raises(SchemaError, match="definition violated"):
        check_invariants(bad)


def test_survivor_gaining_from_gate_removal_is_caught(effects):
    bad = effects.copy()
    idx = bad.index[(bad["effect_type"] == E_GATE) & (bad["stratum"] == SURVIVOR)][0]
    bad.loc[idx, "utility_control"] = bad.loc[idx, "utility_treatment"] + 1.0
    bad.loc[idx, "effect"] = bad.loc[idx, "utility_treatment"] - bad.loc[idx, "utility_control"]
    with pytest.raises(SchemaError, match="survivors have effect < 0"):
        check_invariants(bad)


def test_listwise_relaxation_only_touches_gate_signs(effects):
    """
    A listwise Stage 2 re-scores survivors when the candidate set changes, so
    a survivor gaining is admissible there. It is a different estimand, and
    pointwise=False is how that is declared - but delta_t's structural zeros
    stay enforced, because they do not depend on the reranker at all.
    """
    df = effects.copy()
    idx = df.index[(df["effect_type"] == E_GATE) & (df["stratum"] == SURVIVOR)][0]
    df.loc[idx, "utility_control"] = df.loc[idx, "utility_treatment"] + 1.0
    df.loc[idx, "effect"] = df.loc[idx, "utility_treatment"] - df.loc[idx, "utility_control"]
    check_invariants(df, pointwise=False)       # must not raise

    idx2 = df.index[
        (df["effect_type"] == DELTA_T) & (df["stratum"] == UNSELECTED_OUTSIDER)
    ][0]
    df.loc[idx2, "utility_treatment"] = 1.0
    df.loc[idx2, "effect"] = df.loc[idx2, "utility_treatment"] - df.loc[idx2, "utility_control"]
    with pytest.raises(SchemaError, match="unselected_outsider"):
        check_invariants(df, pointwise=False)


def test_non_monotone_utility_is_refused(effects):
    with pytest.raises(ValueError, match="not monotone"):
        check_invariants(effects, utility="made_up", pointwise=True)


def test_budget_rows_must_carry_their_budget(effects):
    bad = effects.copy()
    idx = bad.index[bad["effect_type"] == DELTA_T][0]
    bad.loc[idx, "t"] = pd.NA
    with pytest.raises(SchemaError, match="budget"):
        check_invariants(bad)


# ── non-additivity ────────────────────────────────────────────────────

def test_stage_effects_are_not_additive(effects):
    """
    E_gate + E_rerank != U_d(pi_act) in general. The two interventions compare
    against different reference worlds, so any sum-to-one presentation would be
    a category error. This is evidence for the write-up, not a failure.
    """
    residual = check_additivity(effects)["additivity_residual"].abs()
    assert residual.max() > 1e-6


# ── fuzz ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k_c", [3, 5, 8])
@pytest.mark.parametrize("utility", ["dcg", "topk", "rbp"])
def test_invariants_hold_across_random_cascades(k_c, utility):
    rng = np.random.default_rng(20261021 + k_c)
    gain_map = {0: 0.0, 1: 0.0, 2: 3.0, 3: 7.0}
    frames = []
    for q in range(20):
        universe = [f"d{i:03d}" for i in range(16)]
        s1 = {d: float(rng.normal()) for d in universe}
        s2 = {d: float(rng.normal()) for d in universe}
        rel = {d: int(rng.integers(0, 4)) for d in universe}
        judged = {d: bool(rng.random() > 0.3) for d in universe}
        w = build_worlds(f"q{q}", universe, s1, s2, k_c)
        tw = build_t_grid(w, s1, s2, k_c, (1.0, 0.75, 0.5), "boundary")
        frames.append(stage_effects_for_query(
            w, rel, judged, "fuzz", 3, gain_map, utility
        ))
        frames.append(budget_effects_for_query(
            w, tw, rel, judged, "fuzz", 3, gain_map, utility
        ))
    eff = concat_effects(frames)
    check_invariants(eff, utility=utility, pointwise=True)
    check_strata_valid(eff)


@pytest.mark.parametrize("policy", ["boundary", "random"])
def test_invariants_hold_across_policies(policy):
    rng = np.random.default_rng(99)
    gain_map = {0: 0.0, 1: 0.0, 2: 3.0, 3: 7.0}
    frames = []
    for q in range(15):
        universe = [f"d{i:03d}" for i in range(20)]
        s1 = {d: float(rng.normal()) for d in universe}
        s2 = {d: float(rng.normal()) for d in universe}
        rel = {d: int(rng.integers(0, 4)) for d in universe}
        judged = dict.fromkeys(universe, True)
        w = build_worlds(f"q{q}", universe, s1, s2, 6)
        tw = build_t_grid(w, s1, s2, 6, (1.0, 0.5, 0.0), policy, seed=q)
        frames.append(budget_effects_for_query(
            w, tw, rel, judged, "fuzz", 3, gain_map
        ))
    check_invariants(concat_effects(frames))


# ── derived views ─────────────────────────────────────────────────────

def test_gating_failures_finds_the_discarded_document(effects):
    failures = gating_failures(effects)
    assert set(failures["doc_id"]) == {"e"}


def test_document_labels(effects):
    labelled = label_documents(effects, tau_document=0.05)
    label_of = dict(zip(labelled["doc_id"], labelled["label"]))
    assert label_of["e"] == GATING_FAILURE


def test_recovered_outsiders_flags_entry_into_top_k(effects):
    rec = recovered_outsiders(effects, k=2)
    entered = rec[rec["entered_top_k"]]
    assert "e" in set(entered["doc_id"])
    assert "f" not in set(entered["doc_id"])


def test_summarise_splits_by_effect_type_and_stratum(effects):
    s = summarise(effects)
    assert set(s["effect_type"]) == {E_RERANK, E_GATE, DELTA_T}
    assert int(s["n"].sum()) == len(effects)


def test_concat_of_nothing_is_a_typed_empty_table():
    out = concat_effects([])
    assert out.empty
    assert "effect_type" in out.columns
