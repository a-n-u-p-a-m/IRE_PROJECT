"""
Stratified fidelity, and the shortcuts it exists to remove.

Two demonstrations carry this file. `test_membership_baseline_scores_well_
pooled` shows that a predictor carrying no magnitude information at all earns
a large pooled correlation against E_gate. `test_stratum_mean_baseline_scores_
well_pooled_on_delta_t` shows the same trap one layer along, on the budget
experiment. Both are the reason `evaluate_within_strata` returns no pooled row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.baselines import (
    attach_all,
    baseline_report,
    delta_t_stratum_report,
    fit_stratum_means,
    membership_baseline,
    zero_baseline,
)
from shared.effects import (
    budget_effects_for_query,
    concat_effects,
    stage_effects_for_query,
)
from shared.metrics import (
    bootstrap_ci,
    calibration_curve,
    cost_fidelity_curve,
    evaluate_within_strata,
    macro_fidelity,
    mae,
    pearson,
    pooled_fidelity,
    rmse,
    signed_bias,
    spearman,
    top_m_recovery,
)
from shared.schema import (
    DELTA_T,
    E_GATE,
    E_RERANK,
    NON_SURVIVOR,
    RETAINED,
    SURVIVOR,
    UNSELECTED_OUTSIDER,
)
from shared.strata import stratum_sizes, thin_strata
from shared.utility import ranks_of
from shared.worlds import build_t_grid, build_worlds

TREC_DL = {0: 0.0, 1: 0.0, 2: 3.0, 3: 7.0}


@pytest.fixture(scope="module")
def fuzz_effects():
    """40 random cascades with a plausible cheap predictor attached."""
    rng = np.random.default_rng(11)
    frames = []
    preds = []
    for q in range(40):
        qid = f"q{q:02d}"
        universe = [f"d{i:03d}" for i in range(40)]
        s1 = {d: float(rng.normal()) for d in universe}
        s2 = {d: float(rng.normal()) for d in universe}
        rel = {d: int(rng.integers(0, 4)) for d in universe}
        judged = dict.fromkeys(universe, True)

        w = build_worlds(qid, universe, s1, s2, 12)
        tw = build_t_grid(w, s1, s2, 12, (1.0, 0.5), "boundary")
        frames.append(stage_effects_for_query(w, rel, judged, "fuzz", 5, TREC_DL))
        frames.append(budget_effects_for_query(w, tw, rel, judged, "fuzz", 5, TREC_DL))

        # A cheap signal: Stage-2 margin to the document at the final cutoff.
        pivot = s2[w.pi_act[4]] if len(w.pi_act) > 4 else s2[w.pi_act[-1]]
        r_no2, r_act = ranks_of(w.pi_no2), ranks_of(w.pi_act)
        for d in universe:
            preds.append(
                {
                    "query_id": qid, "doc_id": d,
                    "h_margin": float(s2[d] - pivot),
                    "h_disp": float(r_no2.get(d, 0) - r_act.get(d, 0)),
                }
            )

    eff = concat_effects(frames)
    pred = pd.DataFrame(preds)
    pred[["query_id", "doc_id"]] = pred[["query_id", "doc_id"]].astype("string")
    return eff.merge(pred, on=["query_id", "doc_id"], how="left")


# ── point metrics ─────────────────────────────────────────────────────

def test_mae_and_signed_bias_separate_systematic_error_from_noise():
    assert mae([1, 2, 3], [0, 1, 2]) == pytest.approx(1.0)
    assert signed_bias([1, 2, 3], [0, 1, 2]) == pytest.approx(1.0)
    assert mae([1, 0, 1], [0, 1, 0]) == pytest.approx(1.0)
    assert signed_bias([1, 0, 1], [0, 1, 0]) == pytest.approx(1 / 3)
    assert rmse([1, 1], [0, 0]) == pytest.approx(1.0)


def test_correlations_return_nan_on_constants_rather_than_raising():
    assert np.isnan(spearman(np.ones(10), np.arange(10.0)))
    assert np.isnan(pearson(np.ones(10), np.arange(10.0)))
    assert np.isnan(spearman([1.0, 2.0], [1.0, 2.0]))     # fewer than 3 points


def test_top_m_recovery_measures_localisation():
    ref = np.array([0.0, 0.0, 5.0, -6.0])
    assert top_m_recovery(ref, ref, 2) == pytest.approx(1.0)
    assert top_m_recovery([9.0, 9.0, 0.0, 0.0], ref, 2) == pytest.approx(0.0)


def test_calibration_curve_is_monotone_for_a_scaled_predictor():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=400)
    curve = calibration_curve(2.0 * ref, ref, n_bins=5)
    assert len(curve) >= 4
    assert curve["mean_predicted"].is_monotonic_increasing


# ── bootstrap ─────────────────────────────────────────────────────────

def test_bootstrap_resamples_queries_not_documents(fuzz_effects):
    """
    Documents are nested within queries. Resampling documents would treat
    correlated observations as independent and give intervals far too narrow,
    so the reported n must be the QUERY count.
    """
    ci = bootstrap_ci(fuzz_effects, lambda d: float(d["effect"].mean()),
                      n_iter=200, seed=1)
    assert ci["n"] == fuzz_effects["query_id"].nunique() == 40
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_bootstrap_on_empty_input_is_nan_not_an_exception(fuzz_effects):
    ci = bootstrap_ci(fuzz_effects.iloc[0:0], lambda d: 0.0, n_iter=10)
    assert np.isnan(ci["point"]) and ci["n"] == 0


# ── stratified evaluation ─────────────────────────────────────────────

def test_gate_evaluation_has_no_pooled_row(fuzz_effects):
    out = evaluate_within_strata(fuzz_effects, "h_margin", E_GATE, n_iter=20)
    assert set(out["stratum"]) <= {SURVIVOR, NON_SURVIVOR}
    assert "pooled" not in set(out["stratum"])
    assert len(out) == out["stratum"].nunique()


def test_delta_t_evaluation_covers_four_strata(fuzz_effects):
    out = evaluate_within_strata(fuzz_effects, "h_margin", DELTA_T, n_iter=20)
    assert len(out) >= 3
    zero_row = out[out["stratum"] == UNSELECTED_OUTSIDER]
    if len(zero_row):
        assert bool(zero_row.iloc[0]["structural_zero"])
        assert np.isnan(zero_row.iloc[0]["spearman"]), (
            "a correlation against a structurally zero target is undefined and "
            "must not be reported"
        )


def test_underpowered_strata_are_flagged(fuzz_effects):
    out = evaluate_within_strata(fuzz_effects, "h_margin", E_GATE, n_iter=10,
                                 min_documents=10_000)
    assert out["underpowered"].all()


def test_missing_predictor_column_is_named(fuzz_effects):
    with pytest.raises(KeyError, match="h_nonexistent"):
        evaluate_within_strata(fuzz_effects, "h_nonexistent", E_GATE, n_iter=5)


def test_macro_average_excludes_structural_zeros(fuzz_effects):
    per = evaluate_within_strata(fuzz_effects, "h_margin", DELTA_T, n_iter=20)
    macro = macro_fidelity(per)
    assert len(macro) == 1
    assert UNSELECTED_OUTSIDER not in macro.iloc[0]["strata"]


# ── the shortcuts ─────────────────────────────────────────────────────

def test_membership_baseline_scores_well_pooled(fuzz_effects):
    """
    The trap, demonstrated. A predictor carrying no magnitude information
    whatever achieves a substantial POOLED correlation against E_gate, purely
    because sign(E_gate) is fixed by survivorship.
    """
    frame = membership_baseline(fuzz_effects)
    report = baseline_report(frame, E_GATE)
    rho = float(report.iloc[0]["pooled_spearman"])
    assert rho > 0.3, (
        "the shortcut baseline should be strong; if it is not, the fixture no "
        "longer contains non-survivors with negative gate effects"
    )
    assert "survivorship" in report.iloc[0]["warning"]


def test_stratum_mean_baseline_scores_well_pooled_on_delta_t(fuzz_effects):
    """The same trap on the budget experiment, one layer along."""
    report = delta_t_stratum_report(fuzz_effects)
    assert len(report) == 1
    assert float(report.iloc[0]["pooled_spearman"]) > 0.3


def test_membership_baseline_is_useless_within_stratum(fuzz_effects):
    """
    Inside a stratum the shortcut is constant, so its correlation is undefined
    rather than high. That is the whole point of stratifying: it removes the
    free signal and leaves only magnitude.
    """
    frame = membership_baseline(fuzz_effects)
    out = evaluate_within_strata(frame, "pred_membership", E_GATE, n_iter=10)
    assert out["spearman"].isna().all()


def test_zero_baseline_is_hard_to_beat_on_survivors(fuzz_effects):
    """
    Most survivors barely move, so "nothing happened" has a low MAE. A cheap
    predictor that cannot beat it has not earned its compute.
    """
    frame = zero_baseline(fuzz_effects)
    out = evaluate_within_strata(frame, "pred_zero", E_GATE, n_iter=10)
    survivors = out[out["stratum"] == SURVIVOR]
    assert len(survivors) == 1
    assert np.isfinite(survivors.iloc[0]["mae"])


def test_attach_all_adds_three_baselines(fuzz_effects):
    out = attach_all(fuzz_effects, E_GATE)
    for col in ("pred_zero", "pred_membership", "pred_stratum_mean"):
        assert col in out.columns


def test_stratum_means_pin_structural_zeros_exactly():
    frame = pd.DataFrame(
        {
            "effect_type": [DELTA_T] * 4,
            "stratum": [RETAINED, RETAINED, UNSELECTED_OUTSIDER, UNSELECTED_OUTSIDER],
            "effect": [1.0, 3.0, 0.0, 0.0],
        }
    )
    means = fit_stratum_means(frame, DELTA_T)
    assert means[RETAINED] == pytest.approx(2.0)
    assert means[UNSELECTED_OUTSIDER] == 0.0


def test_pooled_fidelity_labels_itself_secondary(fuzz_effects):
    out = pooled_fidelity(fuzz_effects, "h_margin", E_GATE)
    assert out.iloc[0]["scope"] == "pooled_secondary"
    assert out.iloc[0]["warning"]


# ── reporting helpers ─────────────────────────────────────────────────

def test_stratum_sizes_and_thin_strata(fuzz_effects):
    sizes = stratum_sizes(fuzz_effects)
    assert {"effect_type", "stratum", "n_documents"}.issubset(sizes.columns)
    thin = thin_strata(fuzz_effects, min_documents=10**9)
    assert len(thin) == len(sizes)


def test_cost_fidelity_curve_sorts_by_budget(fuzz_effects):
    frames = [
        (budget, evaluate_within_strata(fuzz_effects, "h_margin", E_GATE, n_iter=5))
        for budget in (100, 20, 0)
    ]
    curve = cost_fidelity_curve(frames, stratum=NON_SURVIVOR)
    assert list(curve["extra_stage2_calls"]) == [0, 20, 100]


def test_rerank_evaluation_marks_non_survivors_structural(fuzz_effects):
    out = evaluate_within_strata(fuzz_effects, "h_disp", E_RERANK, n_iter=10)
    ns = out[out["stratum"] == NON_SURVIVOR]
    assert len(ns) == 1
    assert bool(ns.iloc[0]["structural_zero"])
    assert np.isnan(ns.iloc[0]["spearman"])
