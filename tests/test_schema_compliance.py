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
    new_score_matrix,
    new_attribution,
    new_intervention,
    save_parquet,
    load_parquet,
    VALID_METHODS,
    VALID_INTERVENTIONS,
)
from shared.logger import ScoreLogger, AttributionLogger, InterventionLogger


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
        "rank_before":       [2, 1],
        "rank_after":        [5, 3],
        "displacement":      [3, 2],
    })


# ──────────────────────────────────────────────────────────────────────
# Score matrix tests
# ──────────────────────────────────────────────────────────────────────

class TestScoreMatrix:

    def test_valid(self, sample_score_matrix):
        df = validate_score_matrix(sample_score_matrix)
        assert len(df) == 5

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
        )
        df = log.to_dataframe()
        assert df.iloc[0]["displacement"] == 7

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
