"""
schema.py — Data contract definitions for the stage-level attribution benchmark.

Every pipeline output, attribution result, and intervention result must conform
to these schemas.  This module provides:

  1. Column-name constants and dtypes for the three core DataFrames.
  2. Validation functions that raise on schema violations.
  3. Factory helpers to create empty, correctly-typed DataFrames.
  4. I/O wrappers (read/write Parquet) that validate on every round-trip.

Usage:
    from shared.schema import (
        validate_score_matrix,
        validate_attribution,
        validate_intervention,
        new_score_matrix,
        save_parquet,
        load_parquet,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ──────────────────────────────────────────────────────────────────────
# 1. Column specs  (name → pandas dtype string)
# ──────────────────────────────────────────────────────────────────────

SCORE_MATRIX_COLUMNS: dict[str, str] = {
    "query_id":        "string",
    "doc_id":          "string",
    "stage_1_rank":    "Int64",    # nullable int for padding rows
    "stage_1_score":   "float64",
    "in_candidate_set": "boolean",
    "stage_2_rank":    "Int64",
    "stage_2_score":   "float64",
    "final_rank":      "Int64",
    "relevance_label":  "Int64",   # TREC qrel 0/1/2/3, nullable for unjudged
}

ATTRIBUTION_COLUMNS: dict[str, str] = {
    "query_id":             "string",
    "doc_id":               "string",
    "method_name":          "string",
    "stage_1_attribution":  "float64",
    "stage_2_attribution":  "float64",
    "gating_attribution":   "float64",   # only for inclusion-ordering
    "ordering_attribution": "float64",   # only for inclusion-ordering
}

INTERVENTION_COLUMNS: dict[str, str] = {
    "query_id":           "string",
    "doc_id":             "string",
    "intervention_type":  "string",
    "rank_before":        "Int64",
    "rank_after":         "Int64",
    "displacement":       "Int64",
}

# Allowed categorical values
VALID_METHODS = frozenset({
    "rank_delta",
    "inclusion_ordering",
    "stage_shapley",
    "reranker_lime",
})

VALID_INTERVENTIONS = frozenset({
    "remove_stage2",
    "randomize_stage1_scores",
    "remove_gating",
})

# ──────────────────────────────────────────────────────────────────────
# 2. Validation
# ──────────────────────────────────────────────────────────────────────

class SchemaError(Exception):
    """Raised when a DataFrame violates the agreed data contract."""


def _check_columns(df: pd.DataFrame, expected: dict[str, str], name: str) -> None:
    """Verify all expected columns exist (extra columns are allowed)."""
    missing = set(expected) - set(df.columns)
    if missing:
        raise SchemaError(
            f"{name}: missing required columns {sorted(missing)}. "
            f"Got columns: {sorted(df.columns)}"
        )


def _check_no_empty_keys(df: pd.DataFrame, key_cols: list[str], name: str) -> None:
    """Key columns must not contain nulls."""
    for col in key_cols:
        if col in df.columns and df[col].isna().any():
            n_null = df[col].isna().sum()
            raise SchemaError(
                f"{name}: key column '{col}' has {n_null} null value(s)"
            )


def _check_ranks_positive(df: pd.DataFrame, rank_cols: list[str], name: str) -> None:
    """Rank columns must be >= 1 where not null."""
    for col in rank_cols:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if len(valid) > 0 and (valid < 1).any():
            raise SchemaError(
                f"{name}: rank column '{col}' has values < 1 (ranks are 1-indexed)"
            )


def validate_score_matrix(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """
    Validate and coerce a score-matrix DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    strict : bool
        If True (default), also checks:
          - No duplicate (query_id, doc_id) pairs
          - Ranks are 1-indexed positive integers

    Returns
    -------
    pd.DataFrame  — same data, columns cast to canonical dtypes.
    """
    _check_columns(df, SCORE_MATRIX_COLUMNS, "ScoreMatrix")
    _check_no_empty_keys(df, ["query_id", "doc_id"], "ScoreMatrix")

    # Cast to canonical dtypes (lenient — allows int→Int64, etc.)
    for col, dtype in SCORE_MATRIX_COLUMNS.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as exc:
                raise SchemaError(
                    f"ScoreMatrix: cannot cast column '{col}' to {dtype}: {exc}"
                ) from exc

    if strict:
        dupes = df.duplicated(subset=["query_id", "doc_id"], keep=False)
        if dupes.any():
            n = dupes.sum()
            raise SchemaError(
                f"ScoreMatrix: {n} duplicate (query_id, doc_id) rows"
            )
        _check_ranks_positive(
            df, ["stage_1_rank", "stage_2_rank", "final_rank"], "ScoreMatrix"
        )

    return df


def validate_attribution(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Validate an attribution-result DataFrame."""
    _check_columns(df, ATTRIBUTION_COLUMNS, "Attribution")
    _check_no_empty_keys(df, ["query_id", "doc_id", "method_name"], "Attribution")

    for col, dtype in ATTRIBUTION_COLUMNS.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as exc:
                raise SchemaError(
                    f"Attribution: cannot cast '{col}' to {dtype}: {exc}"
                ) from exc

    if strict:
        bad_methods = set(df["method_name"].unique()) - VALID_METHODS
        if bad_methods:
            raise SchemaError(
                f"Attribution: unknown method(s) {bad_methods}. "
                f"Allowed: {sorted(VALID_METHODS)}"
            )

        # stage_1 + stage_2 should sum to ~1.0 (tolerance for float rounding)
        sums = df["stage_1_attribution"] + df["stage_2_attribution"]
        bad = ((sums - 1.0).abs() > 0.01) & sums.notna()
        if bad.any():
            n = bad.sum()
            raise SchemaError(
                f"Attribution: {n} rows where stage_1 + stage_2 attribution "
                f"does not sum to 1.0 (tolerance 0.01)"
            )

    return df


def validate_intervention(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Validate an intervention-result DataFrame."""
    _check_columns(df, INTERVENTION_COLUMNS, "Intervention")
    _check_no_empty_keys(
        df, ["query_id", "doc_id", "intervention_type"], "Intervention"
    )

    for col, dtype in INTERVENTION_COLUMNS.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as exc:
                raise SchemaError(
                    f"Intervention: cannot cast '{col}' to {dtype}: {exc}"
                ) from exc

    if strict:
        bad_types = set(df["intervention_type"].unique()) - VALID_INTERVENTIONS
        if bad_types:
            raise SchemaError(
                f"Intervention: unknown intervention type(s) {bad_types}. "
                f"Allowed: {sorted(VALID_INTERVENTIONS)}"
            )
        _check_ranks_positive(
            df, ["rank_before", "rank_after"], "Intervention"
        )

        # displacement must equal rank_after - rank_before
        computed = df["rank_after"] - df["rank_before"]
        mismatch = (computed != df["displacement"]) & computed.notna()
        if mismatch.any():
            n = mismatch.sum()
            raise SchemaError(
                f"Intervention: {n} rows where displacement != "
                f"rank_after - rank_before"
            )

    return df


# ──────────────────────────────────────────────────────────────────────
# 3. Factory helpers
# ──────────────────────────────────────────────────────────────────────

TableKind = Literal["score_matrix", "attribution", "intervention"]

_SCHEMA_MAP: dict[TableKind, dict[str, str]] = {
    "score_matrix":  SCORE_MATRIX_COLUMNS,
    "attribution":   ATTRIBUTION_COLUMNS,
    "intervention":  INTERVENTION_COLUMNS,
}

_VALIDATOR_MAP = {
    "score_matrix":  validate_score_matrix,
    "attribution":   validate_attribution,
    "intervention":  validate_intervention,
}


def new_dataframe(kind: TableKind) -> pd.DataFrame:
    """Return an empty DataFrame with the correct columns and dtypes."""
    cols = _SCHEMA_MAP[kind]
    return pd.DataFrame({c: pd.array([], dtype=d) for c, d in cols.items()})


def new_score_matrix() -> pd.DataFrame:
    return new_dataframe("score_matrix")


def new_attribution() -> pd.DataFrame:
    return new_dataframe("attribution")


def new_intervention() -> pd.DataFrame:
    return new_dataframe("intervention")


# ──────────────────────────────────────────────────────────────────────
# 4. I/O  (Parquet, with schema validation on every read/write)
# ──────────────────────────────────────────────────────────────────────

def save_parquet(
    df: pd.DataFrame,
    path: str | Path,
    kind: TableKind,
    *,
    strict: bool = True,
) -> Path:
    """
    Validate *df* against the given schema, then write to Parquet.

    Returns the resolved Path for convenience.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    validator = _VALIDATOR_MAP[kind]
    df = validator(df, strict=strict)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    return path


def load_parquet(
    path: str | Path,
    kind: TableKind,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Read a Parquet file and validate it against the given schema.

    Returns the validated DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    table = pq.read_table(path)
    df = table.to_pandas()

    validator = _VALIDATOR_MAP[kind]
    return validator(df, strict=strict)
