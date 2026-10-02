"""
logger.py — Score/rank logger that builds schema-compliant DataFrames.

Instead of manually constructing DataFrames row-by-row, pipeline code
uses this logger to accumulate results incrementally and then flush to
a validated Parquet file.

Usage:

    from shared.logger import ScoreLogger, InterventionLogger

    # Inside a retriever + reranker pipeline:
    log = ScoreLogger()
    for qid, results in pipeline_results.items():
        for doc in results:
            log.add(
                query_id=qid,
                doc_id=doc.pid,
                stage_1_rank=doc.s1_rank,
                stage_1_score=doc.s1_score,
                in_candidate_set=doc.s1_rank <= TOP_K,
                stage_2_rank=doc.s2_rank,
                stage_2_score=doc.s2_score,
                final_rank=doc.final_rank,
            )
    log.save("results/scores/bm25_minilm.parquet")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from shared.schema import (
    ATTRIBUTION_COLUMNS,
    INTERVENTION_COLUMNS,
    SCORE_MATRIX_COLUMNS,
    save_parquet,
    TableKind,
)


class _BaseLogger:
    """Accumulate rows and flush to validated Parquet."""

    _kind: TableKind
    _columns: dict[str, str]

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def _add(self, **kwargs: Any) -> None:
        # Only accept known column names
        unknown = set(kwargs) - set(self._columns)
        if unknown:
            raise ValueError(
                f"Unknown columns for {self._kind}: {sorted(unknown)}"
            )
        self._rows.append(kwargs)

    def to_dataframe(self, *, strict: bool = True) -> pd.DataFrame:
        """Convert accumulated rows to a validated DataFrame."""
        if not self._rows:
            from shared.schema import new_dataframe
            return new_dataframe(self._kind)

        df = pd.DataFrame(self._rows)

        # Add missing optional columns with NaN
        for col in self._columns:
            if col not in df.columns:
                df[col] = pd.NA

        # Reorder columns to canonical order
        ordered = [c for c in self._columns if c in df.columns]
        extra = [c for c in df.columns if c not in self._columns]
        df = df[ordered + extra]

        return df

    def save(
        self,
        path: str | Path,
        *,
        strict: bool = True,
    ) -> Path:
        """Validate and save to Parquet. Returns the output path."""
        df = self.to_dataframe(strict=False)  # validation happens in save_parquet
        return save_parquet(df, path, self._kind, strict=strict)

    def clear(self) -> None:
        """Reset the logger, discarding all accumulated rows."""
        self._rows.clear()


class ScoreLogger(_BaseLogger):
    """
    Accumulate score-matrix rows during pipeline execution.

    Example
    -------
    >>> log = ScoreLogger()
    >>> log.add(query_id="1", doc_id="d1", stage_1_rank=1, ...)
    >>> df = log.to_dataframe()
    >>> log.save("results/scores/bm25_minilm.parquet")
    """

    _kind: TableKind = "score_matrix"
    _columns = SCORE_MATRIX_COLUMNS

    def add(
        self,
        *,
        query_id: str,
        doc_id: str,
        stage_1_rank: int,
        stage_1_score: float,
        in_candidate_set: bool,
        stage_2_rank: int | None = None,
        stage_2_score: float | None = None,
        final_rank: int | None = None,
        relevance_label: int | None = None,
    ) -> None:
        """Add one score-matrix row."""
        self._add(
            query_id=query_id,
            doc_id=doc_id,
            stage_1_rank=stage_1_rank,
            stage_1_score=stage_1_score,
            in_candidate_set=in_candidate_set,
            stage_2_rank=stage_2_rank,
            stage_2_score=stage_2_score,
            final_rank=final_rank,
            relevance_label=relevance_label,
        )

    def fill_final_ranks(self) -> None:
        """
        If stage_2_rank is populated, set final_rank = stage_2_rank for
        candidates and final_rank = stage_1_rank for non-candidates.
        Call this after all rows for a query are added.
        """
        for row in self._rows:
            if row.get("final_rank") is not None:
                continue
            if row.get("in_candidate_set") and row.get("stage_2_rank") is not None:
                row["final_rank"] = row["stage_2_rank"]
            else:
                row["final_rank"] = row.get("stage_1_rank")


class AttributionLogger(_BaseLogger):
    """
    Accumulate attribution results.

    Example
    -------
    >>> log = AttributionLogger()
    >>> log.add(
    ...     query_id="1", doc_id="d1", method_name="rank_delta",
    ...     stage_1_attribution=0.3, stage_2_attribution=0.7,
    ... )
    """

    _kind: TableKind = "attribution"
    _columns = ATTRIBUTION_COLUMNS

    def add(
        self,
        *,
        query_id: str,
        doc_id: str,
        method_name: str,
        stage_1_attribution: float,
        stage_2_attribution: float,
        gating_attribution: float | None = None,
        ordering_attribution: float | None = None,
    ) -> None:
        """Add one attribution row."""
        self._add(
            query_id=query_id,
            doc_id=doc_id,
            method_name=method_name,
            stage_1_attribution=stage_1_attribution,
            stage_2_attribution=stage_2_attribution,
            gating_attribution=gating_attribution,
            ordering_attribution=ordering_attribution,
        )


class InterventionLogger(_BaseLogger):
    """
    Accumulate intervention results.

    Example
    -------
    >>> log = InterventionLogger()
    >>> log.add(
    ...     query_id="1", doc_id="d1",
    ...     intervention_type="remove_stage2",
    ...     rank_before=3, rank_after=15,
    ... )
    """

    _kind: TableKind = "intervention"
    _columns = INTERVENTION_COLUMNS

    def add(
        self,
        *,
        query_id: str,
        doc_id: str,
        intervention_type: str,
        rank_before: int,
        rank_after: int,
        displacement: int | None = None,
    ) -> None:
        """Add one intervention row. Displacement is auto-computed if omitted."""
        if displacement is None:
            displacement = rank_after - rank_before

        self._add(
            query_id=query_id,
            doc_id=doc_id,
            intervention_type=intervention_type,
            rank_before=rank_before,
            rank_after=rank_after,
            displacement=displacement,
        )
