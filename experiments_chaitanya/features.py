"""
features.py - Conventional feature extraction for effect prediction.

Extracts predictor inputs from the FACTUAL run only (world A: the actual
cascade).  No counterfactual outputs are used as features — that would
conflate the predictor with the estimand.  No qrels are used either — they
would make the predictor a relevance classifier masquerading as an effect
predictor.

Features are computed per (query, doc) pair and stored in the feature table
defined in shared/schema.py (FEATURE_COLUMNS).  Each feature has a name, a
version string, and a generation_seconds field recording how long it took.

Feature groups:

  stage1 features  (all documents in U_q — Stage-1 is free)
    stage1_rank       — absolute rank in Stage-1 output (lower = more retrieved)
    stage1_score      — raw Stage-1 retrieval score
    stage1_rank_norm  — rank / candidate_depth (0 = top, 1 = bottom of C_q)
    stage1_percentile — 1 - stage1_rank_norm (0 = bottom, 1 = top)
    is_survivor       — 1 if doc is in C_q (rank <= k_c), 0 otherwise

  stage2 features  (SURVIVORS ONLY: rank <= k_c inside C_q)
    stage2_score      — cross-encoder score from world A
    stage2_score_norm — per-query min-max normalised stage2 score
    rank_drop         — stage1_rank - stage2_rank in world A (reranker displacement)

    NOTE: Stage-2 features are RESTRICTED to C_q documents.  A cached score
    for a document outside C_q exists only because world C was computed — it
    is not a "free" factual observation.  For a non-survivor d,
    e_gate(d) = -U_d(pi_no1), and U_d(pi_no1) is determined exactly by the
    Stage-2 score over U_q; providing that score as a feature does not
    approximate the target — it contains it.  The fix is one line: replace
    the full run doc list with rank-filtered survivors.

  text-length features (from passage text, if available; all documents in U_q)
    passage_words     — word count of the passage (proxy for length penalty)
    passage_words_norm— passage_words / mean_passage_words per query

All features are on training queries only; never apply a fitted normaliser
from training to test queries — use per-query normalisation instead.

Extra Stage-2 calls as cost:
    The feature table includes generation_seconds, which counts the wall time
    for any EXTRA Stage-2 calls needed to compute a feature.  Stage-2 scores
    already in the factual cache cost 0 extra calls; a hypothetical feature
    that required additional scoring would record the inference time here.
    Stage-1 features cost 0 extra calls (no GPU).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from shared.schema import FEATURE_COLUMNS, SchemaError, validate_feature

if TYPE_CHECKING:
    from shared.cache import Stage2Cache

logger = logging.getLogger(__name__)

FEATURE_VERSION = "v2"  # bumped: Stage-2 features now restricted to C_q


# ──────────────────────────────────────────────────────────────────────
# Helper: extract and validate
# ──────────────────────────────────────────────────────────────────────

def _make_feature_row(
    query_id: str,
    doc_id: str,
    name: str,
    value: float,
    generation_seconds: float = 0.0,
    version: str = FEATURE_VERSION,
) -> dict:
    return {
        "query_id": str(query_id),
        "doc_id": str(doc_id),
        "feature_version": version,
        "feature_name": name,
        "value": float(value),
        "generation_seconds": float(generation_seconds),
    }


# ──────────────────────────────────────────────────────────────────────
# Stage-1 features  (all documents in U_q — Stage-1 is GPU-free)
# ──────────────────────────────────────────────────────────────────────

def extract_stage1_features(
    run: pd.DataFrame,
    system: str,
    k_c: int,
) -> pd.DataFrame:
    """
    Extract Stage-1 rank and score features from a validated run table.

    Parameters
    ----------
    run:
        A validated run table (schema.RUN_COLUMNS).
    system:
        'bm25' or 'dpr' — which retriever's run to use.
    k_c:
        Candidate depth (top-K_c documents form C_q).

    Returns
    -------
    pd.DataFrame
        In FEATURE_COLUMNS format.
    """
    t0 = time.perf_counter()
    sub = run[run["system"] == system].copy()
    rows: list[dict] = []

    for qid, grp in sub.groupby("query_id", observed=True):
        qid = str(qid)
        grp = grp.sort_values("rank")
        n = len(grp)

        for _, r in grp.iterrows():
            did = str(r["doc_id"])
            rank = int(r["rank"])
            score = float(r["score"])
            rank_norm = (rank - 1) / max(k_c - 1, 1) if rank <= k_c else 1.0
            is_survivor = 1.0 if rank <= k_c else 0.0

            rows.append(_make_feature_row(qid, did, "stage1_rank", rank))
            rows.append(_make_feature_row(qid, did, "stage1_score", score))
            rows.append(_make_feature_row(qid, did, "stage1_rank_norm", rank_norm))
            rows.append(_make_feature_row(qid, did, "stage1_percentile",
                                          1.0 - (rank - 1) / max(n - 1, 1)))
            rows.append(_make_feature_row(qid, did, "is_survivor", is_survivor))

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    if not df.empty:
        # Stamp generation time on the whole batch (it is shared; not per-pair GPU time)
        df["generation_seconds"] = elapsed / max(len(df) // 5, 1)
    return validate_feature(df)


# ──────────────────────────────────────────────────────────────────────
# Stage-2 features (SURVIVORS ONLY: rank <= k_c — zero extra calls)
# ──────────────────────────────────────────────────────────────────────

def extract_stage2_features(
    cache: "Stage2Cache",
    query_ids: list[str],
    survivor_doc_ids_per_query: dict[str, list[str]],
    model: str,
) -> pd.DataFrame:
    """
    Extract Stage-2 score features from the cache — C_q survivors ONLY.

    IMPORTANT: doc_ids_per_query must contain ONLY documents in C_q
    (rank <= k_c for the relevant system).  Documents outside C_q must
    not be passed here.  A Stage-2 score for a non-survivor exists only
    because world C was computed; providing it as a feature leaks the
    target (e_gate is determined, not approximated).

    Scores come from world A (the factual run) which is already in the cache.
    Extra Stage-2 calls = 0 for all rows (generation_seconds = 0).

    Parameters
    ----------
    cache:
        The Stage2Cache instance.
    query_ids:
        Queries to extract features for.
    survivor_doc_ids_per_query:
        {query_id: [doc_id, ...]} — C_q survivors only (rank <= k_c).
    model:
        The 'model' key in the cache (PointwiseCrossEncoder.checkpoint).
    """
    rows: list[dict] = []

    for qid in query_ids:
        qid = str(qid)
        doc_ids = [str(d) for d in survivor_doc_ids_per_query.get(qid, [])]
        if not doc_ids:
            continue

        scores = cache.get_query(qid, model)
        if not scores:
            logger.warning("query %s: no Stage-2 scores in cache for %s", qid, model)
            continue

        # Per-query normalisation (avoids leaking scale across queries)
        vals = [scores.get(d, float("nan")) for d in doc_ids]
        valid = [v for v in vals if np.isfinite(v)]
        if not valid:
            continue
        vmin, vmax = min(valid), max(valid)
        vrange = max(vmax - vmin, 1e-12)

        for did, raw in zip(doc_ids, vals):
            if not np.isfinite(raw):
                continue
            norm = (raw - vmin) / vrange
            rows.append(_make_feature_row(qid, did, "stage2_score", raw))
            rows.append(_make_feature_row(qid, did, "stage2_score_norm", norm))

    df = pd.DataFrame(rows)
    return validate_feature(df) if not df.empty else _empty_feature_frame()


# ──────────────────────────────────────────────────────────────────────
# Rank-drop feature (reranker displacement from Stage-1 to Stage-2 rank)
# ──────────────────────────────────────────────────────────────────────

def extract_rank_drop_feature(
    run: pd.DataFrame,
    cache: "Stage2Cache",
    system: str,
    model: str,
    k_c: int,
) -> pd.DataFrame:
    """
    Compute rank_drop = stage1_rank - stage2_rank_within_Cq for survivors.

    Non-survivors get rank_drop = NaN (they have no Stage-2 rank in C_q).
    This captures how much the reranker displaced a document within the
    candidate set — documents the reranker promotes heavily should have large
    positive E_rerank values.

    Only survivors (rank <= k_c) are processed here; this is consistent with
    the Stage-2 feature boundary.
    """
    sub = run[run["system"] == system].copy()
    rows: list[dict] = []

    for qid, grp in sub.groupby("query_id", observed=True):
        qid = str(qid)
        grp = grp.sort_values("rank")

        # Survivors only
        survivors = grp[grp["rank"] <= k_c]
        if survivors.empty:
            continue

        s2_scores = cache.get_query(qid, model)
        if not s2_scores:
            continue

        # Stage-2 rank within C_q
        s2_in_cq = {
            did: s2_scores[did]
            for did in survivors["doc_id"].astype(str)
            if did in s2_scores
        }
        if not s2_in_cq:
            continue

        # Sort by s2 descending → rank 1 = highest s2
        s2_ranked = sorted(s2_in_cq.items(), key=lambda x: -x[1])
        s2_rank_map = {did: r + 1 for r, (did, _) in enumerate(s2_ranked)}

        for _, row in survivors.iterrows():
            did = str(row["doc_id"])
            if did not in s2_rank_map:
                continue
            stage1_rank = int(row["rank"])
            stage2_rank = s2_rank_map[did]
            rank_drop = float(stage1_rank - stage2_rank)
            rows.append(_make_feature_row(qid, did, "rank_drop", rank_drop))

    df = pd.DataFrame(rows)
    return validate_feature(df) if not df.empty else _empty_feature_frame()


# ──────────────────────────────────────────────────────────────────────
# Text-length features (from passage texts, if provided; all U_q docs)
# ──────────────────────────────────────────────────────────────────────

def extract_passage_length_features(
    passage_store: dict[str, str],
    query_ids: list[str],
    doc_ids_per_query: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Extract word-count features from passage texts.

    These are a proxy for the length penalty in neural rerankers: very long
    passages are harder for a cross-encoder to rank correctly within 512 tokens.
    Counting words is ~O(n) and needs no GPU.

    Extra Stage-2 calls = 0 (text is already fetched for scoring).

    doc_ids_per_query may contain all U_q documents — passage length is a
    Stage-1 observable (obtained from the passage text alone) and does not
    require world C scores.
    """
    rows: list[dict] = []

    for qid in query_ids:
        qid = str(qid)
        doc_ids = [str(d) for d in doc_ids_per_query.get(qid, [])]
        lengths = []
        for did in doc_ids:
            text = passage_store.get(did, "")
            lengths.append(len(text.split()))

        if not lengths:
            continue

        mean_len = float(np.mean(lengths)) if lengths else 1.0

        for did, wcount in zip(doc_ids, lengths):
            rows.append(_make_feature_row(qid, did, "passage_words", float(wcount)))
            rows.append(_make_feature_row(
                qid, did, "passage_words_norm",
                float(wcount) / max(mean_len, 1.0)
            ))

    df = pd.DataFrame(rows)
    return validate_feature(df) if not df.empty else _empty_feature_frame()


# ──────────────────────────────────────────────────────────────────────
# Combine all features
# ──────────────────────────────────────────────────────────────────────

def extract_all_features(
    run: pd.DataFrame,
    cache: "Stage2Cache",
    system: str,
    model: str,
    k_c: int,
    query_ids: list[str] | None = None,
    passage_store: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Extract all feature groups and return a single concatenated feature table.

    Stage-1 features are extracted for all U_q documents (Stage-1 is free).
    Stage-2 features (stage2_score, stage2_score_norm, rank_drop) are extracted
    ONLY for C_q survivors (rank <= k_c).  This is the critical boundary:
    a Stage-2 score for a non-survivor exists only because world C was computed,
    and providing it as a predictor input leaks the target.

    Parameters
    ----------
    run:
        Validated run table.
    cache:
        Stage2Cache with factual-run scores already loaded.
    system:
        Stage-1 system ('bm25' or 'dpr').
    model:
        Stage-2 model checkpoint string.
    k_c:
        Candidate depth.
    query_ids:
        If given, restrict to these queries (e.g., training split).
    passage_store:
        Optional {doc_id: text} for passage-length features (all U_q docs).

    Returns
    -------
    pd.DataFrame
        Validated feature table.
    """
    sub_run = run[run["system"] == system]
    if query_ids is not None:
        sub_run = sub_run[sub_run["query_id"].isin(query_ids)]

    all_query_ids = sub_run["query_id"].astype(str).unique().tolist()

    # All docs per query (for Stage-1 and passage-length features)
    all_doc_ids_per_query: dict[str, list[str]] = {}
    for qid, grp in sub_run.groupby("query_id", observed=True):
        all_doc_ids_per_query[str(qid)] = grp["doc_id"].astype(str).tolist()

    # Survivor docs per query (for Stage-2 features ONLY — the leakage boundary)
    survivor_doc_ids_per_query: dict[str, list[str]] = {}
    for qid, grp in sub_run.groupby("query_id", observed=True):
        survivors = grp[grp["rank"] <= k_c]["doc_id"].astype(str).tolist()
        survivor_doc_ids_per_query[str(qid)] = survivors

    n_survivors_total = sum(len(v) for v in survivor_doc_ids_per_query.values())
    n_all_total = sum(len(v) for v in all_doc_ids_per_query.values())
    logger.info(
        "Stage-2 features will cover %d survivors (C_q) of %d total U_q docs "
        "(%.1f%% restricted by the C_q boundary)",
        n_survivors_total, n_all_total,
        100.0 * n_survivors_total / max(n_all_total, 1),
    )

    t0 = time.perf_counter()

    parts: list[pd.DataFrame] = []

    # Stage-1 features — all U_q docs, zero GPU cost
    s1_feats = extract_stage1_features(sub_run, system, k_c)
    parts.append(s1_feats)

    # Stage-2 features — C_q survivors ONLY (the leakage-prevention boundary)
    s2_feats = extract_stage2_features(
        cache, all_query_ids, survivor_doc_ids_per_query, model
    )
    parts.append(s2_feats)

    # Rank-drop — C_q survivors ONLY (consistent with Stage-2 boundary)
    rd_feats = extract_rank_drop_feature(sub_run, cache, system, model, k_c)
    parts.append(rd_feats)

    # Passage-length features (optional) — all U_q docs, zero GPU cost
    if passage_store is not None:
        pl_feats = extract_passage_length_features(
            passage_store, all_query_ids, all_doc_ids_per_query
        )
        parts.append(pl_feats)

    parts = [p for p in parts if not p.empty]
    if not parts:
        return _empty_feature_frame()

    combined = pd.concat(parts, ignore_index=True)
    elapsed = time.perf_counter() - t0
    logger.info(
        "extracted %d feature rows for %d queries in %.2fs",
        len(combined), len(all_query_ids), elapsed,
    )
    return validate_feature(combined)


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in FEATURE_COLUMNS.items()})


__all__ = [
    "FEATURE_VERSION",
    "extract_stage1_features",
    "extract_stage2_features",
    "extract_rank_drop_feature",
    "extract_passage_length_features",
    "extract_all_features",
]
