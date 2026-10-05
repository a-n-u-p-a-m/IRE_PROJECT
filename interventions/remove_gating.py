"""
remove_gating.py — Intervention: remove Stage 1's candidate-set gate.

Roadmap 1.8: expand the reranker's input from Stage 1's top-1000 to its
top-5000, re-run Stage 2, and re-rank.  A document that reached the final
ranking only because the gate kept stronger competitors out moves down.

    rank_before  = final_rank in the full (gated) pipeline
    rank_after   = rank among all expanded candidates after reranking
    displacement = rank_after − rank_before   (positive = document moved down)

Rows are produced for the documents of the full pipeline.

The reranker scores each (query, passage) pair independently, so the Stage 2
scores already in the score matrix are reused for the original candidates and
only the newly admitted candidates are scored.

Usage:
    from interventions.remove_gating import run
    intervention_df = run(score_matrix, retriever, reranker, queries,
                          load_passages, expanded_k=5000)
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from interventions.remove_stage2 import real_documents
from shared.schema import validate_intervention

logger = logging.getLogger(__name__)

INTERVENTION_TYPE = "remove_gating"
DEFAULT_EXPANDED_K = 5000


def _expanded_candidates(retriever, queries: dict[str, str], k: int) -> dict[str, list[str]]:
    if hasattr(retriever, "batch_search"):
        results = retriever.batch_search(queries, k=k)
    else:
        results = {qid: retriever.search(text, k=k) for qid, text in queries.items()}
    return {qid: [doc_id for doc_id, _ in hits] for qid, hits in results.items()}


def run(
    score_matrix: pd.DataFrame,
    retriever,
    reranker,
    queries: dict[str, str],
    load_passages: Callable[[Iterable[str]], dict[str, str]],
    expanded_k: int = DEFAULT_EXPANDED_K,
) -> pd.DataFrame:
    """
    Intervention rows (schema: intervention) for every document of the full
    pipeline.

    score_matrix   full pipeline output for these queries (reranked)
    retriever      Stage 1 retriever (.search or .batch_search)
    reranker       Stage 2 reranker with .score(query, texts) -> list[float]
    queries        query_id → query text
    load_passages  doc ids → {doc_id: passage text}
    """
    sm = real_documents(score_matrix).copy()
    sm["query_id"] = sm["query_id"].astype(str)
    sm["doc_id"] = sm["doc_id"].astype(str)
    if sm["stage_2_score"].isna().any():
        raise ValueError("score_matrix has no Stage 2 scores; run the full pipeline first")

    qids = sorted(set(sm["query_id"]))
    missing = [q for q in qids if q not in queries]
    if missing:
        raise ValueError(f"No query text for {len(missing)} query id(s), e.g. {missing[:3]}")

    expanded = _expanded_candidates(retriever, {q: queries[q] for q in qids}, expanded_k)
    original = {q: g for q, g in sm.groupby("query_id")}

    new_docs = {}
    for q in qids:
        seen = set(original[q]["doc_id"])
        new_docs[q] = [d for d in expanded[q] if d not in seen]
    texts = load_passages({d for docs in new_docs.values() for d in docs})
    logger.info("Scoring %d newly admitted candidates across %d queries",
                sum(map(len, new_docs.values())), len(qids))

    frames = []
    for q in qids:
        g = original[q]
        new = new_docs[q]
        new_scores = reranker.score(queries[q], [texts.get(d, "") for d in new]) if new else []

        doc_ids = list(g["doc_id"]) + new
        scores = np.concatenate([g["stage_2_score"].to_numpy(dtype=float),
                                 np.asarray(new_scores, dtype=float)])
        order = np.argsort(-scores, kind="stable")      # ties keep the original order
        rank_after = {doc_ids[i]: r for r, i in enumerate(order, start=1)}

        frames.append(pd.DataFrame({
            "query_id": q,
            "doc_id": g["doc_id"].values,
            "intervention_type": INTERVENTION_TYPE,
            "seed": pd.NA,                   # deterministic intervention
            "rank_before": g["final_rank"].values,
            "rank_after": [rank_after[d] for d in g["doc_id"]],
        }))

    df = pd.concat(frames, ignore_index=True)
    df["displacement"] = df["rank_after"] - df["rank_before"]
    return validate_intervention(df)
