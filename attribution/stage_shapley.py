"""
stage_shapley.py — Stage-Shapley attribution for a two-stage pipeline.

Each stage is a player in a cooperative game played per document.  The value
of a coalition is the document's utility under the ranking that coalition
produces, u(rank):

    v({})        = 0                      nothing retrieved
    v({S1})      = u(stage_1_rank)        Stage 1 ranking only
    v({S1, S2})  = u(final_rank)          full pipeline
    v({S2})      = 0                      roadmap 1.7: Stage 2 cannot run alone
                 | u(rank of Stage 2 alone)   sensitivity check (stage2_alone)

Two-player Shapley values:

    phi_1 = 1/2 [v(S1) - v({})] + 1/2 [v(S1,S2) - v(S2)]
    phi_2 = 1/2 [v(S2) - v({})] + 1/2 [v(S1,S2) - v(S1)]

Contributions are clipped at zero (a stage that pushed the document down gets
no credit) and normalised to fractions summing to 1.

With v({S2}) = 0, Stage 1 is a *necessary* player, so phi_1 >= phi_2 and
stage_1_attribution is always >= 0.5.  The stage2_alone option relaxes this:
pass Stage 2's ranking over an (ideally ungated) candidate pool, e.g. the
rank_after of the remove_gating intervention.

Usage:
    from attribution.stage_shapley import compute_attribution
    attr = compute_attribution(score_matrix, top_k=10)              # roadmap default
    attr = compute_attribution(score_matrix, value="dcg")            # alternative utility
    attr = compute_attribution(score_matrix,
                               stage2_alone=stage2_alone_from_intervention(gating_df))
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from shared.schema import validate_attribution

METHOD_NAME = "stage_shapley"

# Per-document utility of a rank (1 = best)
VALUE_FUNCTIONS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "rr":   lambda r, k: 1.0 / r,                       # reciprocal rank (default)
    "dcg":  lambda r, k: 1.0 / np.log2(1.0 + r),        # nDCG position discount
    "topk": lambda r, k: (r <= k).astype(float),         # made the top-k cut
}


def shapley_two_stage(
    v1: np.ndarray, v12: np.ndarray, v2: np.ndarray | float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Exact two-player Shapley values (v({}) = 0).  phi_1 + phi_2 = v12."""
    phi1 = 0.5 * v1 + 0.5 * (v12 - v2)
    phi2 = 0.5 * v2 + 0.5 * (v12 - v1)
    return phi1, phi2


def stage2_alone_from_intervention(intervention_df: pd.DataFrame) -> pd.DataFrame:
    """Stage-2-alone ranks from remove_gating results (rank_after)."""
    rows = intervention_df[intervention_df["intervention_type"] == "remove_gating"]
    if rows.empty:
        raise ValueError("intervention_df has no remove_gating rows")
    return rows[["query_id", "doc_id", "rank_after"]].rename(columns={"rank_after": "rank"})


def compute_attribution(
    score_matrix: pd.DataFrame,
    top_k: int = 10,
    value: str = "rr",
    stage2_alone: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Input:  score matrix in the agreed schema (full pipeline)
    Output: attribution rows (agreed schema) for documents in the final top-k,
            plus raw Shapley values in shapley_stage_1 / shapley_stage_2.

    value         "rr" (default), "dcg" or "topk"
    stage2_alone  optional DataFrame (query_id, doc_id, rank): Stage 2's rank
                  without Stage 1's gate, for the sensitivity check
    """
    if value not in VALUE_FUNCTIONS:
        raise ValueError(f"value must be one of {sorted(VALUE_FUNCTIONS)}, got {value!r}")
    u = VALUE_FUNCTIONS[value]

    top = score_matrix[score_matrix["final_rank"] <= top_k].copy()
    top["query_id"] = top["query_id"].astype(str)
    top["doc_id"] = top["doc_id"].astype(str)
    if top["stage_1_rank"].isna().any():
        raise ValueError("score_matrix has top-k documents without a stage_1_rank")

    v1 = u(top["stage_1_rank"].to_numpy(dtype=float), top_k)
    v12 = u(top["final_rank"].to_numpy(dtype=float), top_k)

    v2: np.ndarray | float = 0.0
    if stage2_alone is not None:
        s2 = stage2_alone.copy()
        s2[["query_id", "doc_id"]] = s2[["query_id", "doc_id"]].astype(str)
        top = top.merge(s2[["query_id", "doc_id", "rank"]].rename(columns={"rank": "s2_rank"}),
                        on=["query_id", "doc_id"], how="left")
        missing = int(top["s2_rank"].isna().sum())
        if missing:
            raise ValueError(f"stage2_alone has no rank for {missing} top-{top_k} document(s)")
        v2 = u(top["s2_rank"].to_numpy(dtype=float), top_k)

    phi1, phi2 = shapley_two_stage(v1, v12, v2)

    credit1, credit2 = np.maximum(phi1, 0.0), np.maximum(phi2, 0.0)
    total = credit1 + credit2
    share1 = np.divide(credit1, total, out=np.full_like(total, 0.5), where=total > 0)

    attr = pd.DataFrame({
        "query_id": top["query_id"].values,
        "doc_id": top["doc_id"].values,
        "method_name": METHOD_NAME,
        "stage_1_attribution": share1,
        "stage_2_attribution": 1.0 - share1,
        "gating_attribution": np.nan,       # inclusion-ordering only
        "ordering_attribution": np.nan,     # inclusion-ordering only
        "shapley_stage_1": phi1,
        "shapley_stage_2": phi2,
    })
    return validate_attribution(attr)
