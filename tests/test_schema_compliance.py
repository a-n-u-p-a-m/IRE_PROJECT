"""
test_schema_compliance.py — Verify data contract is enforced.

Run with:  pytest tests/test_schema_compliance.py -v
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from shared.schema import (
    SchemaError,
    validate_score_matrix,
    validate_attribution,
    validate_intervention,
    validate_synthetic_gt,
    new_score_matrix,
    new_attribution,
    new_intervention,
    new_synthetic_gt,
    save_parquet,
    load_parquet,
    VALID_METHODS,
    VALID_INTERVENTIONS,
    VALID_SETTINGS,
)
from shared.logger import (
    ScoreLogger,
    AttributionLogger,
    InterventionLogger,
    SyntheticGTLogger,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_score_matrix() -> pd.DataFrame:
    """A minimal valid score matrix."""
    return pd.DataFrame({
        "query_id":        ["q1", "q1", "q1", "q2", "q2"],
        "doc_id":          ["d1", "d2", "d3", "d4", "d5"],
        "stage_1_rank":    [1, 2, 3, 1, 2],
        "stage_1_score":   [10.5, 9.2, 8.1, 11.0, 7.3],
        "in_candidate_set": [True, True, True, True, True],
        "stage_2_rank":    [2, 1, 3, 1, 2],
        "stage_2_score":   [0.85, 0.92, 0.41, 0.95, 0.78],
        "final_rank":      [2, 1, 3, 1, 2],
        "relevance_label": [1, 3, 0, 2, 0],
    })


@pytest.fixture
def sample_attribution() -> pd.DataFrame:
    """A minimal valid attribution result."""
    return pd.DataFrame({
        "query_id":            ["q1", "q1"],
        "doc_id":              ["d1", "d2"],
        "method_name":         ["rank_delta", "rank_delta"],
        "stage_1_attribution": [0.3, 0.6],
        "stage_2_attribution": [0.7, 0.4],
        "gating_attribution":  [pd.NA, pd.NA],
        "ordering_attribution": [pd.NA, pd.NA],
    })


@pytest.fixture
def sample_intervention() -> pd.DataFrame:
    """A minimal valid intervention result."""
    return pd.DataFrame({
        "query_id":          ["q1", "q1"],
        "doc_id":            ["d1", "d2"],
        "intervention_type": ["remove_stage2", "remove_stage2"],
        "seed":              [42, 42],
        "rank_before":       [2, 1],
        "rank_after":        [5, 3],
        "displacement":      [3, 2],
    })


@pytest.fixture
def sample_synthetic_gt() -> pd.DataFrame:
    """A minimal valid synthetic ground-truth result."""
    return pd.DataFrame({
        "query_id":                 ["q1", "q1"],
        "doc_id":                   ["d1", "d2"],
        "true_stage_1_attribution": [0.7, 0.3],
        "setting":                  ["linear_mix", "linear_mix"],
        "alpha":                    [0.7, 0.7],
    })


# ──────────────────────────────────────────────────────────────────────
# Score matrix tests
# ──────────────────────────────────────────────────────────────────────

class TestScoreMatrix:

    def test_valid(self, sample_score_matrix):
        df = validate_score_matrix(sample_score_matrix)
        assert len(df) == 5

    def test_does_not_mutate_caller(self, sample_score_matrix):
        """A2 fix: validation must not modify the caller's DataFrame."""
        original_dtypes = sample_score_matrix.dtypes.copy()
        validate_score_matrix(sample_score_matrix)
        pd.testing.assert_series_equal(sample_score_matrix.dtypes, original_dtypes)

    def test_missing_column(self, sample_score_matrix):
        df = sample_score_matrix.drop(columns=["stage_1_rank"])
        with pytest.raises(SchemaError, match="missing required columns"):
            validate_score_matrix(df)

    def test_null_key_rejected(self, sample_score_matrix):
        df = sample_score_matrix.copy()
        df.loc[0, "query_id"] = None
        with pytest.raises(SchemaError, match="null value"):
            validate_score_matrix(df)

    def test_duplicate_rejected(self, sample_score_matrix):
        df = pd.concat([sample_score_matrix, sample_score_matrix.iloc[:1]])
        with pytest.raises(SchemaError, match="duplicate"):
            validate_score_matrix(df)

    def test_zero_rank_rejected(self, sample_score_matrix):
        df = sample_score_matrix.copy()
        df.loc[0, "stage_1_rank"] = 0
        with pytest.raises(SchemaError, match="values < 1"):
            validate_score_matrix(df)

    def test_non_strict_allows_duplicates(self, sample_score_matrix):
        df = pd.concat([sample_score_matrix, sample_score_matrix.iloc[:1]])
        result = validate_score_matrix(df, strict=False)
        assert len(result) == 6

    def test_empty_df(self):
        df = new_score_matrix()
        result = validate_score_matrix(df)
        assert len(result) == 0


# ──────────────────────────────────────────────────────────────────────
# Attribution tests
# ──────────────────────────────────────────────────────────────────────

class TestAttribution:

    def test_valid(self, sample_attribution):
        df = validate_attribution(sample_attribution)
        assert len(df) == 2

    def test_does_not_mutate_caller(self, sample_attribution):
        """A2 fix: validation must not modify the caller's DataFrame."""
        original_cols = list(sample_attribution.columns)
        validate_attribution(sample_attribution)
        assert list(sample_attribution.columns) == original_cols

    def test_bad_method_rejected(self, sample_attribution):
        df = sample_attribution.copy()
        df.loc[0, "method_name"] = "invalid_method"
        with pytest.raises(SchemaError, match="unknown method"):
            validate_attribution(df)

    def test_sum_not_one_rejected(self, sample_attribution):
        df = sample_attribution.copy()
        df.loc[0, "stage_1_attribution"] = 0.3
        df.loc[0, "stage_2_attribution"] = 0.3  # sum = 0.6, not 1.0
        with pytest.raises(SchemaError, match="does not sum to 1.0"):
            validate_attribution(df)

    def test_all_valid_methods(self):
        """Every method in VALID_METHODS should pass validation."""
        for method in VALID_METHODS:
            df = pd.DataFrame({
                "query_id": ["q1"],
                "doc_id": ["d1"],
                "method_name": [method],
                "stage_1_attribution": [0.5],
                "stage_2_attribution": [0.5],
                "gating_attribution": [pd.NA],
                "ordering_attribution": [pd.NA],
            })
            validate_attribution(df)


# ──────────────────────────────────────────────────────────────────────
# Intervention tests
# ──────────────────────────────────────────────────────────────────────

class TestIntervention:

    def test_valid(self, sample_intervention):
        df = validate_intervention(sample_intervention)
        assert len(df) == 2

    def test_bad_intervention_type(self, sample_intervention):
        df = sample_intervention.copy()
        df.loc[0, "intervention_type"] = "bad_intervention"
        with pytest.raises(SchemaError, match="unknown intervention type"):
            validate_intervention(df)

    def test_displacement_mismatch(self, sample_intervention):
        df = sample_intervention.copy()
        df.loc[0, "displacement"] = 999  # wrong
        with pytest.raises(SchemaError, match="displacement != rank_after - rank_before"):
            validate_intervention(df)

    def test_seed_column_present(self, sample_intervention):
        """Q3: seed column must be accepted."""
        df = validate_intervention(sample_intervention)
        assert "seed" in df.columns

    def test_seed_nullable(self):
        """Q3: seed can be null for deterministic interventions."""
        df = pd.DataFrame({
            "query_id":          ["q1"],
            "doc_id":            ["d1"],
            "intervention_type": ["remove_stage2"],
            "seed":              [pd.NA],
            "rank_before":       [2],
            "rank_after":        [5],
            "displacement":      [3],
        })
        result = validate_intervention(df)
        assert len(result) == 1


# ──────────────────────────────────────────────────────────────────────
# Synthetic GT tests
# ──────────────────────────────────────────────────────────────────────

class TestSyntheticGT:

    def test_valid(self, sample_synthetic_gt):
        df = validate_synthetic_gt(sample_synthetic_gt)
        assert len(df) == 2

    def test_bad_setting_rejected(self, sample_synthetic_gt):
        df = sample_synthetic_gt.copy()
        df.loc[0, "setting"] = "unknown_setting"
        with pytest.raises(SchemaError, match="unknown setting"):
            validate_synthetic_gt(df)

    def test_alpha_out_of_range(self, sample_synthetic_gt):
        df = sample_synthetic_gt.copy()
        df.loc[0, "alpha"] = 1.5
        with pytest.raises(SchemaError, match="alpha must be in"):
            validate_synthetic_gt(df)

    def test_attribution_out_of_range(self, sample_synthetic_gt):
        df = sample_synthetic_gt.copy()
        df.loc[0, "true_stage_1_attribution"] = -0.1
        with pytest.raises(SchemaError, match="true_stage_1_attribution must be in"):
            validate_synthetic_gt(df)

    def test_empty_df(self):
        df = new_synthetic_gt()
        result = validate_synthetic_gt(df)
        assert len(result) == 0


# ──────────────────────────────────────────────────────────────────────
# Logger tests
# ──────────────────────────────────────────────────────────────────────

class TestLoggers:

    def test_score_logger_roundtrip(self):
        log = ScoreLogger()
        log.add(
            query_id="q1", doc_id="d1",
            stage_1_rank=1, stage_1_score=10.0,
            in_candidate_set=True,
            stage_2_rank=2, stage_2_score=0.9,
            final_rank=2, relevance_label=3,
        )
        df = log.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["query_id"] == "q1"

    def test_attribution_logger(self):
        log = AttributionLogger()
        log.add(
            query_id="q1", doc_id="d1",
            method_name="rank_delta",
            stage_1_attribution=0.4,
            stage_2_attribution=0.6,
        )
        df = log.to_dataframe()
        assert len(df) == 1

    def test_intervention_logger_auto_displacement(self):
        log = InterventionLogger()
        log.add(
            query_id="q1", doc_id="d1",
            intervention_type="remove_stage2",
            rank_before=3, rank_after=10,
            seed=42,
        )
        df = log.to_dataframe()
        assert df.iloc[0]["displacement"] == 7

    def test_intervention_logger_with_seed(self):
        """Q3: InterventionLogger must accept seed parameter."""
        log = InterventionLogger()
        log.add(
            query_id="q1", doc_id="d1",
            intervention_type="randomize_stage1_scores",
            rank_before=3, rank_after=10,
            seed=123,
        )
        df = log.to_dataframe()
        assert df.iloc[0]["seed"] == 123

    def test_synthetic_gt_logger(self):
        """Q4: SyntheticGTLogger should produce valid synthetic GT."""
        log = SyntheticGTLogger()
        log.add(
            query_id="q1", doc_id="d1",
            true_stage_1_attribution=0.7,
            setting="linear_mix",
            alpha=0.7,
        )
        df = log.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["setting"] == "linear_mix"

    def test_score_logger_parquet_roundtrip(self, tmp_path):
        log = ScoreLogger()
        log.add(
            query_id="q1", doc_id="d1",
            stage_1_rank=1, stage_1_score=5.0,
            in_candidate_set=True,
            stage_2_rank=1, stage_2_score=0.99,
            final_rank=1, relevance_label=2,
        )
        path = log.save(tmp_path / "test.parquet")
        loaded = load_parquet(path, "score_matrix")
        assert len(loaded) == 1
        assert loaded.iloc[0]["query_id"] == "q1"


# ──────────────────────────────────────────────────────────────────────
# Parquet I/O tests
# ──────────────────────────────────────────────────────────────────────

class TestParquetIO:

    def test_save_and_load(self, sample_score_matrix, tmp_path):
        path = save_parquet(sample_score_matrix, tmp_path / "sm.parquet", "score_matrix")
        loaded = load_parquet(path, "score_matrix")
        assert len(loaded) == len(sample_score_matrix)
        assert list(loaded.columns) == list(sample_score_matrix.columns)

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_parquet(tmp_path / "nope.parquet", "score_matrix")

    def test_synthetic_gt_roundtrip(self, sample_synthetic_gt, tmp_path):
        path = save_parquet(sample_synthetic_gt, tmp_path / "gt.parquet", "synthetic_gt")
        loaded = load_parquet(path, "synthetic_gt")
        assert len(loaded) == len(sample_synthetic_gt)

    def test_intervention_with_seed_roundtrip(self, sample_intervention, tmp_path):
        path = save_parquet(sample_intervention, tmp_path / "interv.parquet", "intervention")
        loaded = load_parquet(path, "intervention")
        assert "seed" in loaded.columns
