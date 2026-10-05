"""
remove_stage2.py — Intervention: drop the reranker, keep Stage 1's ranking.

Roadmap 1.8: final ranking = Stage 1 ranking.
    rank_before  = final_rank in the full pipeline
    rank_after   = stage_1_rank
    displacement = rank_after − rank_before   (positive = document moved down)

Needs only the full pipeline's score matrix (no retriever, reranker or GPU),
so it works unchanged for every retriever.

Usage:
    from interventions.remove_stage2 import run
    intervention_df = run(score_matrix)
"""

from __future__ import annotations

import pandas as pd

from shared.schema import validate_intervention

INTERVENTION_TYPE = "remove_stage2"


def real_documents(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """Rows for retrieved documents, excluding padding entries."""
    keep = score_matrix["in_candidate_set"].fillna(False).astype(bool) & ~(
        score_matrix["doc_id"].astype(str).str.startswith("__PAD_")
    )
    return score_matrix[keep]


def run(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """Intervention rows (schema: intervention) for every retrieved document."""
    sm = real_documents(score_matrix)
    df = pd.DataFrame({
        "query_id": sm["query_id"].astype(str).values,
        "doc_id": sm["doc_id"].astype(str).values,
        "intervention_type": INTERVENTION_TYPE,
        "seed": pd.NA,                       # deterministic intervention
        "rank_before": sm["final_rank"].values,
        "rank_after": sm["stage_1_rank"].values,
    })
    df["displacement"] = df["rank_after"] - df["rank_before"]
    return validate_intervention(df)
