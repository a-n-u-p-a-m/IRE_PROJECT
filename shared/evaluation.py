"""
evaluation.py — All evaluation metrics + bootstrap confidence intervals.

This is the single shared evaluation module. Everyone imports from here.
No one reimplements metrics separately.

Metrics fall into two groups:

  1. **Synthetic evaluation** — comparing attribution predictions against
     planted ground-truth α values.
  2. **Real-data evaluation** — comparing attribution predictions against
     observed intervention displacements.

Both groups include bootstrap 95% CIs (resampling queries, 1000 iterations).

Usage:
    from shared.evaluation import (
        attribution_error,
        rank_correlation,
        calibration_error,
        displacement_correlation,
        comprehensiveness,
        sufficiency,
        recovery_error,
        ndcg,
        bootstrap_ci,
        evaluate_all_synthetic,
        evaluate_all_real,
    )
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Standard IR metrics (for baseline verification)
# ──────────────────────────────────────────────────────────────────────

def dcg(relevances: np.ndarray, k: int | None = None) -> float:
    """
    Discounted Cumulative Gain at rank k.

    Uses linear gain (gain = relevance label), matching trec_eval's
    ndcg_cut — the metric behind published Pyserini / TREC-DL baselines.
    """
    if k is not None:
        relevances = relevances[:k]
    positions = np.arange(1, len(relevances) + 1)
    discounts = np.log2(positions + 1)
    return float(np.sum(relevances / discounts))


def ndcg(
    relevances: np.ndarray,
    k: int = 10,
    ideal_relevances: np.ndarray | None = None,
) -> float:
    """
    Normalized DCG at rank k.

    Parameters
    ----------
    relevances : array-like
        Relevance labels in ranked order (position 0 = rank 1).
    k : int
        Evaluation depth.
    ideal_relevances : array-like, optional
        All known relevance labels for ideal ordering.
        If None, uses sorted(relevances, descending).

    Returns
    -------
    float
        nDCG@k in [0, 1].
    """
    relevances = np.asarray(relevances, dtype=float)
    if ideal_relevances is None:
        ideal_relevances = np.sort(relevances)[::-1]
    else:
        ideal_relevances = np.sort(np.asarray(ideal_relevances, dtype=float))[::-1]

    ideal = dcg(ideal_relevances, k)
    if ideal == 0:
        return 0.0
    return dcg(relevances, k) / ideal


def compute_ndcg_from_score_matrix(
    score_df: pd.DataFrame,
    k: int = 10,
    qrels: dict[str, dict[str, int]] | None = None,
) -> dict[str, float]:
    """
    Compute nDCG@k for each query from a score matrix.

    Returns dict[query_id → nDCG@k].
    Requires 'final_rank' and 'relevance_label' columns.

    Pass *qrels* to match trec_eval: the ideal ranking is then built from
    ALL judged documents (including relevant ones Stage 1 never retrieved),
    and queries without judgments are skipped.  Without qrels, the ideal
    falls back to the labels present in the score matrix, which inflates
    nDCG whenever relevant documents were missed.
    """
    results: dict[str, float] = {}

    for qid, group in score_df.groupby("query_id"):
        qid = str(qid)
        if qrels is not None and qid not in qrels:
            continue

        # Sort by final rank
        sorted_g = group.sort_values("final_rank")

        # Get relevance labels in rank order (unjudged → 0)
        rels = sorted_g["relevance_label"].fillna(0).values.astype(float)

        if qrels is not None:
            all_rels = np.array(list(qrels[qid].values()), dtype=float)
        else:
            all_rels = group["relevance_label"].dropna().values.astype(float)

        results[qid] = ndcg(rels, k=k, ideal_relevances=all_rels)

    return results


# ──────────────────────────────────────────────────────────────────────
# Synthetic evaluation metrics
# ──────────────────────────────────────────────────────────────────────

def attribution_error(
    predicted: np.ndarray,
    true_alpha: float | np.ndarray,
) -> float:
    """
    Mean absolute error between predicted stage_1_attribution and true α.

    Parameters
    ----------
    predicted : array-like
        Predicted stage_1_attribution for documents in top-K.
    true_alpha : float or array-like
        Ground-truth stage-1 contribution.
        - Scalar if the same α was used for all documents.
        - Array of per-document ground truth.

    Returns
    -------
    float
        Mean |predicted - true|.
    """
    predicted = np.asarray(predicted, dtype=float)
    true_alpha = np.asarray(true_alpha, dtype=float)
    return float(np.mean(np.abs(predicted - true_alpha)))


def rank_correlation(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Spearman ρ between predicted per-document stage_1_attribution
    and true per-document contribution.

    Returns 0.0 if all values are identical (no variance).
    """
    predicted = np.asarray(predicted, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)

    if len(predicted) < 2:
        return 0.0
    if np.std(predicted) == 0 or np.std(ground_truth) == 0:
        return 0.0

    rho, _ = scipy_stats.spearmanr(predicted, ground_truth)
    return float(rho)


def calibration_error(
    predicted: np.ndarray,
    true_alpha: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """
    Mean absolute calibration error, binned by true α.

    Returns (overall_mce, bin_details).
    bin_details is a list of dicts with keys:
        bin_start, bin_end, n_samples, mean_predicted, mean_true, abs_error
    """
    predicted = np.asarray(predicted, dtype=float)
    true_alpha = np.asarray(true_alpha, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_details: list[dict] = []
    errors: list[float] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (true_alpha >= lo) & (true_alpha <= hi)
        else:
            mask = (true_alpha >= lo) & (true_alpha < hi)

        n = mask.sum()
        if n == 0:
            bin_details.append({
                "bin_start": float(lo),
                "bin_end": float(hi),
                "n_samples": 0,
                "mean_predicted": None,
                "mean_true": None,
                "abs_error": None,
            })
            continue

        mean_pred = float(predicted[mask].mean())
        mean_true = float(true_alpha[mask].mean())
        err = abs(mean_pred - mean_true)
        errors.append(err)

        bin_details.append({
            "bin_start": float(lo),
            "bin_end": float(hi),
            "n_samples": int(n),
            "mean_predicted": mean_pred,
            "mean_true": mean_true,
            "abs_error": err,
        })

    overall_mce = float(np.mean(errors)) if errors else 0.0
    return overall_mce, bin_details


# ──────────────────────────────────────────────────────────────────────
# Real-data evaluation metrics
# ──────────────────────────────────────────────────────────────────────

def displacement_correlation(
    predicted_attribution: np.ndarray,
    observed_displacement: np.ndarray,
) -> float:
    """
    Spearman ρ between predicted stage importance and measured
    intervention displacement.

    Higher absolute displacement for a stage → that stage was more
    important → predicted attribution should be higher.
    """
    return rank_correlation(predicted_attribution, np.abs(observed_displacement))


def comprehensiveness(
    attribution_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
    top_k: int = 10,
) -> float:
    """
    Fraction of documents where removing the attributed-as-important stage
    causes displacement > median displacement.

    Process:
    1. For each document, determine which stage is attributed as more
       important (stage_1_attribution > 0.5 → stage 1 is dominant).
    2. Look up the intervention that removes that stage.
    3. Check if the resulting displacement exceeds the median.

    Parameters
    ----------
    attribution_df : DataFrame
        Must follow the attribution schema.
    intervention_df : DataFrame
        Must follow the intervention schema.
    top_k : int
        Only consider documents in the top-K of the full pipeline.
    """
    # Build lookup: (query_id, doc_id, intervention_type) → displacement
    interv_lookup: dict[tuple[str, str, str], int] = {}
    for _, row in intervention_df.iterrows():
        key = (row["query_id"], row["doc_id"], row["intervention_type"])
        interv_lookup[key] = row["displacement"]

    # Median displacement across all interventions
    all_displacements = intervention_df["displacement"].abs()
    median_disp = float(all_displacements.median())

    hits = 0
    total = 0

    for _, row in attribution_df.iterrows():
        # Determine which stage is attributed as more important
        if row["stage_1_attribution"] > 0.5:
            # Stage 1 is dominant → check remove_stage2 (should cause LESS
            # displacement) — actually, check what happens when we remove
            # the dominant stage's contribution.
            # remove_stage2 tests Stage 2's importance.
            # To test Stage 1's importance, we use randomize_stage1_scores.
            interv_type = "randomize_stage1_scores"
        else:
            interv_type = "remove_stage2"

        key = (row["query_id"], row["doc_id"], interv_type)
        if key not in interv_lookup:
            continue

        displacement = abs(interv_lookup[key])
        total += 1
        if displacement > median_disp:
            hits += 1

    return hits / total if total > 0 else 0.0


def sufficiency(
    attribution_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
    top_k: int = 10,
) -> float:
    """
    Fraction of documents where keeping ONLY the attributed-as-important
    stage preserves the document in top-K.

    If stage_1_attribution > 0.5 (Stage 1 is dominant), check the
    remove_stage2 intervention: if the document stays in top-K even
    without Stage 2, the attribution is sufficient.
    """
    interv_lookup: dict[tuple[str, str, str], int] = {}
    for _, row in intervention_df.iterrows():
        key = (row["query_id"], row["doc_id"], row["intervention_type"])
        interv_lookup[key] = row["rank_after"]

    hits = 0
    total = 0

    for _, row in attribution_df.iterrows():
        if row["stage_1_attribution"] > 0.5:
            # Stage 1 dominant → keep Stage 1 only → remove_stage2
            interv_type = "remove_stage2"
        else:
            # Stage 2 dominant → keep Stage 2 → the "remove_gating"
            # intervention expands to full corpus, tests if reranker
            # alone suffices.
            interv_type = "remove_gating"

        key = (row["query_id"], row["doc_id"], interv_type)
        if key not in interv_lookup:
            continue

        rank_after = interv_lookup[key]
        total += 1
        if rank_after <= top_k:
            hits += 1

    return hits / total if total > 0 else 0.0


def recovery_error(
    predicted_attribution: np.ndarray,
    observed_displacement: np.ndarray,
) -> float:
    """
    Mean |predicted_attribution − normalized_displacement|.

    Normalizes displacement to [0, 1] before comparison.
    """
    predicted = np.asarray(predicted_attribution, dtype=float)
    displacement = np.abs(np.asarray(observed_displacement, dtype=float))

    # Normalize displacement to [0, 1]
    d_max = displacement.max()
    if d_max > 0:
        norm_disp = displacement / d_max
    else:
        norm_disp = displacement

    return float(np.mean(np.abs(predicted - norm_disp)))


# ──────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ──────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    values_by_query: dict[str, float],
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap 95% CI by resampling queries with replacement.

    Parameters
    ----------
    values_by_query : dict[str, float]
        query_id → metric value for that query.
    n_iterations : int
        Number of bootstrap iterations.
    ci : float
        Confidence level (default 0.95).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: mean, std, ci_low, ci_high, n_queries
    """
    rng = np.random.RandomState(seed)
    query_ids = list(values_by_query.keys())
    values = np.array([values_by_query[q] for q in query_ids])
    n = len(values)

    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_queries": 0}

    bootstrap_means: list[float] = []
    for _ in range(n_iterations):
        sample_idx = rng.choice(n, size=n, replace=True)
        bootstrap_means.append(float(values[sample_idx].mean()))

    bootstrap_means_arr = np.array(bootstrap_means)
    alpha = (1 - ci) / 2

    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "ci_low": float(np.percentile(bootstrap_means_arr, alpha * 100)),
        "ci_high": float(np.percentile(bootstrap_means_arr, (1 - alpha) * 100)),
        "n_queries": n,
    }


# ──────────────────────────────────────────────────────────────────────
# Aggregate evaluation runners
# ──────────────────────────────────────────────────────────────────────

def evaluate_all_synthetic(
    attribution_df: pd.DataFrame,
    true_alphas: dict[str, float] | float,
    method_name: str | None = None,
) -> dict[str, dict]:
    """
    Run all synthetic metrics on an attribution DataFrame.

    Parameters
    ----------
    attribution_df : DataFrame
        Attribution results (must follow schema).
    true_alphas : float or dict[query_id → float]
        The planted ground-truth α.
    method_name : str, optional
        Filter to this method only. If None, evaluate per-method.

    Returns
    -------
    dict[method_name → {metric_name → {mean, ci_low, ci_high, ...}}]
    """
    if method_name:
        attribution_df = attribution_df[
            attribution_df["method_name"] == method_name
        ]

    results: dict[str, dict] = {}

    for mname, mgroup in attribution_df.groupby("method_name"):
        mname = str(mname)
        per_query_error: dict[str, float] = {}
        per_query_corr: dict[str, float] = {}

        for qid, qgroup in mgroup.groupby("query_id"):
            qid = str(qid)
            predicted = qgroup["stage_1_attribution"].values

            if isinstance(true_alphas, dict):
                alpha = true_alphas.get(qid, 0.5)
            else:
                alpha = true_alphas

            per_query_error[qid] = attribution_error(predicted, alpha)

            # For rank correlation, need per-document ground truth
            true_arr = np.full_like(predicted, alpha)
            per_query_corr[qid] = rank_correlation(predicted, true_arr)

        results[mname] = {
            "attribution_error": bootstrap_ci(per_query_error),
            "rank_correlation": bootstrap_ci(per_query_corr),
        }

    return results


def evaluate_all_real(
    attribution_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
    method_name: str | None = None,
    top_k: int = 10,
) -> dict[str, dict]:
    """
    Run all real-data metrics on attribution + intervention DataFrames.

    Returns
    -------
    dict[method_name → {metric_name → {mean, ci_low, ci_high, ...}}]
    """
    if method_name:
        attribution_df = attribution_df[
            attribution_df["method_name"] == method_name
        ]

    results: dict[str, dict] = {}

    for mname, mgroup in attribution_df.groupby("method_name"):
        mname = str(mname)

        # Displacement correlation (per-query)
        per_query_disp_corr: dict[str, float] = {}

        for qid, qg in mgroup.groupby("query_id"):
            qid = str(qid)
            q_interventions = intervention_df[
                intervention_df["query_id"] == qid
            ]

            if q_interventions.empty:
                continue

            # Merge attributions with interventions on doc_id
            merged = qg.merge(
                q_interventions[["doc_id", "displacement"]],
                on="doc_id",
                how="inner",
            )

            if len(merged) < 2:
                continue

            per_query_disp_corr[qid] = displacement_correlation(
                merged["stage_1_attribution"].values,
                merged["displacement"].values,
            )

        results[mname] = {
            "displacement_correlation": bootstrap_ci(per_query_disp_corr),
            "comprehensiveness": comprehensiveness(mgroup, intervention_df, top_k),
            "sufficiency": sufficiency(mgroup, intervention_df, top_k),
        }

    return results


# ──────────────────────────────────────────────────────────────────────
# Pretty-print results
# ──────────────────────────────────────────────────────────────────────

def print_results(results: dict[str, dict], title: str = "Results") -> None:
    """Pretty-print an evaluation results dict."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    for method, metrics in sorted(results.items()):
        print(f"\n  Method: {method}")
        print(f"  {'-'*40}")
        for metric_name, value in sorted(metrics.items()):
            if isinstance(value, dict) and "mean" in value:
                print(
                    f"    {metric_name:30s}  "
                    f"{value['mean']:.4f}  "
                    f"[{value.get('ci_low', 0):.4f}, "
                    f"{value.get('ci_high', 0):.4f}]"
                )
            else:
                print(f"    {metric_name:30s}  {value:.4f}")
    print()
