"""
test_evaluation.py — Behaviour of the shared evaluation module.

Run with:  pytest tests/test_evaluation.py -v
"""

import logging

import numpy as np
import pandas as pd
import pytest

from shared.evaluation import (
    average_over_seeds,
    bootstrap_ci,
    comprehensiveness,
    evaluate_all_real,
    evaluate_all_synthetic,
    ndcg,
    rank_correlation,
    recovery_error,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

DOCS = [f"d{i}" for i in range(5)]


@pytest.fixture
def attribution() -> pd.DataFrame:
    s1 = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    return pd.DataFrame({
        "query_id": "q1",
        "doc_id": DOCS,
        "method_name": "rank_delta",
        "stage_1_attribution": s1,
        "stage_2_attribution": 1 - s1,
    })


def _intervention_rows(itype, displacements, seeds=(None,)):
    rows = []
    for seed in seeds:
        for i, (doc, d) in enumerate(zip(DOCS, displacements)):
            rows.append({
                "query_id": "q1", "doc_id": doc,
                "intervention_type": itype, "seed": seed,
                "rank_before": i + 1, "rank_after": i + 1 + d,
                "displacement": d,
            })
    return rows


# ──────────────────────────────────────────────────────────────────────
# Basic metrics
# ──────────────────────────────────────────────────────────────────────

class TestBasicMetrics:

    def test_ndcg_linear_gain(self):
        # DCG = 3 / log2(3); ideal = 3 / log2(2) = 3
        assert ndcg(np.array([0, 3]), k=10) == pytest.approx(1 / np.log2(3))

    def test_ndcg_uses_full_ideal(self):
        # A relevant doc that was never retrieved lowers nDCG
        assert ndcg(np.array([3]), k=10, ideal_relevances=np.array([3, 3])) < 1.0

    def test_rank_correlation_nan_on_constant(self):
        assert np.isnan(rank_correlation([0.1, 0.5, 0.9], [0.5, 0.5, 0.5]))

    def test_bootstrap_ignores_nan(self):
        ci = bootstrap_ci({"q1": 0.2, "q2": float("nan"), "q3": 0.4})
        assert ci["n_queries"] == 2
        assert ci["mean"] == pytest.approx(0.3)


# ──────────────────────────────────────────────────────────────────────
# Synthetic evaluation
# ──────────────────────────────────────────────────────────────────────

class TestSynthetic:

    def _gt(self, attribution):
        return pd.DataFrame({
            "query_id": "q1",
            "doc_id": DOCS,
            "true_stage_1_attribution": attribution["stage_1_attribution"],
            "setting": "linear_mix",
            "alpha": 0.5,
        })

    def test_perfect_prediction_with_per_doc_truth(self, attribution):
        res = evaluate_all_synthetic(attribution, synthetic_gt_df=self._gt(attribution))
        r = res["rank_delta"]
        assert r["attribution_error"]["mean"] == pytest.approx(0.0)
        assert r["rank_correlation"]["mean"] == pytest.approx(1.0)
        assert r["calibration_error"] == pytest.approx(0.0)

    def test_scalar_alpha(self, attribution):
        r = evaluate_all_synthetic(attribution, true_alphas=0.5)["rank_delta"]
        assert r["attribution_error"]["mean"] == pytest.approx(0.24)
        assert r["rank_correlation"]["n_queries"] == 0  # undefined → NaN → dropped

    def test_no_ground_truth_raises(self, attribution):
        with pytest.raises(ValueError, match="requires either"):
            evaluate_all_synthetic(attribution)

    def test_missing_query_alpha_raises(self, attribution):
        with pytest.raises(ValueError, match="missing 1 query"):
            evaluate_all_synthetic(attribution, true_alphas={"other": 0.5})

    def test_missing_document_truth_raises(self, attribution):
        gt = self._gt(attribution).iloc[:3]
        with pytest.raises(ValueError, match="2 attributed documents"):
            evaluate_all_synthetic(attribution, synthetic_gt_df=gt)

    def test_duplicate_truth_raises(self, attribution):
        gt = self._gt(attribution)
        with pytest.raises(ValueError, match="one setting/alpha"):
            evaluate_all_synthetic(attribution, synthetic_gt_df=pd.concat([gt, gt]))


# ──────────────────────────────────────────────────────────────────────
# Real-data evaluation
# ──────────────────────────────────────────────────────────────────────

class TestReal:

    def test_average_over_seeds(self):
        df = pd.DataFrame(
            _intervention_rows("randomize_stage1_scores", [3] * 5, seeds=(42,))
            + _intervention_rows("randomize_stage1_scores", [6] * 5, seeds=(123,))
            + _intervention_rows("randomize_stage1_scores", [9] * 5, seeds=(456,))
        )
        out = average_over_seeds(df)
        assert len(out) == 5
        assert (out["displacement"] == 6).all()

    def test_seeds_are_averaged_before_metrics(self, attribution):
        # Three seeds whose per-document mean equals the single-seed values
        # must give exactly the single-seed result.
        base = np.array([1, 5, 2, 8, 3])
        one = pd.DataFrame(
            _intervention_rows("remove_gating", base)
            + _intervention_rows("remove_stage2", [4, 1, 7, 2, 6])
        )
        three = pd.DataFrame(
            _intervention_rows("remove_gating", base - 1, seeds=(42,))
            + _intervention_rows("remove_gating", base, seeds=(123,))
            + _intervention_rows("remove_gating", base + 1, seeds=(456,))
            + _intervention_rows("remove_stage2", [4, 1, 7, 2, 6])
        )
        kw = dict(stage1_intervention="remove_gating")
        assert evaluate_all_real(attribution, one, **kw) == evaluate_all_real(attribution, three, **kw)

    def test_stage1_intervention_is_configurable(self, attribution):
        # Stage 1 displacement perfectly tracks stage_1_attribution
        df = pd.DataFrame(
            _intervention_rows("remove_gating", [50, 40, 30, 20, 10])
            + _intervention_rows("remove_stage2", [1, 2, 3, 4, 5])
        )
        r = evaluate_all_real(attribution, df, stage1_intervention="remove_gating")
        assert r["rank_delta"]["displacement_correlation_stage1"]["mean"] == pytest.approx(1.0)
        assert r["rank_delta"]["displacement_correlation_stage2"]["mean"] == pytest.approx(1.0)

    def test_degenerate_stage1_intervention_warns(self, attribution, caplog):
        df = pd.DataFrame(
            _intervention_rows("randomize_stage1_scores", [0] * 5)
            + _intervention_rows("remove_stage2", [1, 2, 3, 4, 5])
        )
        with caplog.at_level(logging.WARNING, logger="shared.evaluation"):
            r = evaluate_all_real(attribution, df)
        assert "zero displacement" in caplog.text
        assert r["rank_delta"]["displacement_correlation_stage1"]["n_queries"] == 0

    def test_recovery_error_bounded(self):
        err = recovery_error([0.5, 0.5], [500, 0], [0, 900])
        assert 0.0 <= err <= 1.0
        assert err == pytest.approx(0.5)

    def test_recovery_error_skips_unmoved_docs(self):
        assert np.isnan(recovery_error([0.5], [0], [0]))
        assert recovery_error([1.0, 0.2], [10, 0], [0, 0]) == pytest.approx(0.0)

    def test_comprehensiveness_uses_per_type_median(self, attribution):
        # remove_gating displacements are 100x larger than remove_stage2;
        # a pooled median would mark every Stage-1-dominant doc as a hit.
        df = pd.DataFrame(
            _intervention_rows("remove_gating", [100, 300, 200, 100, 100])
            + _intervention_rows("remove_stage2", [1, 1, 1, 3, 3])
        )
        comp = comprehensiveness(attribution, df, stage1_intervention="remove_gating")
        # Stage 1 dominant: d0 (100, median 100 → miss), d1 (300 → hit)
        # Stage 2 dominant: d2 (1, median 1 → miss), d3 (3 → hit), d4 (3 → hit)
        # (A pooled median of 51.5 would give d0, d1 hits only → 2/5.)
        assert comp["q1"] == pytest.approx(3 / 5)

    def test_comprehensiveness_median_ignores_unattributed_docs(self, attribution):
        # Intervention files cover every candidate; low-ranked documents that
        # were never attributed must not shift the median.
        rows = (_intervention_rows("remove_gating", [100, 300, 200, 100, 100])
                + _intervention_rows("remove_stage2", [1, 1, 1, 3, 3]))
        extra = [{"query_id": "q1", "doc_id": f"x{i}", "intervention_type": "remove_stage2",
                  "seed": None, "rank_before": 100 + i, "rank_after": 900 + i,
                  "displacement": 800} for i in range(50)]
        comp = comprehensiveness(attribution, pd.DataFrame(rows + extra),
                                 stage1_intervention="remove_gating")
        assert comp["q1"] == pytest.approx(3 / 5)
