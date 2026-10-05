"""
evaluation.py — All evaluation metrics + bootstrap confidence intervals.

This is the single shared evaluation module. Everyone imports from here.
No one reimplements metrics separately.

Metrics fall into two groups:

  1. **Synthetic evaluation** — comparing attribution predictions against
     planted ground-truth α values (per-document).
  2. **Real-data evaluation** — comparing attribution predictions against
     observed intervention displacements, with correct stage-intervention
     pairing.

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
        average_over_seeds,
        ndcg,
        bootstrap_ci,
        evaluate_all_synthetic,
        evaluate_all_real,
    )
"""

from __future__ import annotations

import logging
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

    Returns NaN if all values in either array are identical (no variance),
    since rank correlation is undefined for constant inputs.
    """
    predicted = np.asarray(predicted, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)

    if len(predicted) < 2:
        return float("nan")
    # E8: Return NaN for constant input instead of 0.0 which biases averages
    if np.std(predicted) == 0 or np.std(ground_truth) == 0:
        return float("nan")

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

# Stage → intervention pairing.
#
# Stage 1's intervention is an open team decision (review Q1): with a
# pointwise reranker, randomize_stage1_scores inside a fixed candidate set
# cannot move any document, so its displacement is identically zero.  The
# real-data metrics therefore take the Stage 1 intervention as a parameter;
# switch it (e.g. to "remove_gating") once the team settles Q1.
DEFAULT_STAGE1_INTERVENTION = "randomize_stage1_scores"
STAGE2_INTERVENTION = "remove_stage2"

_INTERVENTION_KEYS = ["query_id", "doc_id", "intervention_type"]


def average_over_seeds(intervention_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse seed replicates into one row per (query_id, doc_id,
    intervention_type), averaging rank_before / rank_after / displacement.

    Roadmap 1.8 runs randomised interventions with 3 seeds and reports the
    mean.  Without this, joining attributions to interventions on doc_id
    yields one row per seed and silently triples the sample.
    """
    value_cols = [
        c for c in ("rank_before", "rank_after", "displacement")
        if c in intervention_df.columns
    ]
    df = intervention_df[_INTERVENTION_KEYS + value_cols].copy()
    df[_INTERVENTION_KEYS] = df[_INTERVENTION_KEYS].astype(str)
    df[value_cols] = df[value_cols].astype(float)
    return df.groupby(_INTERVENTION_KEYS, as_index=False)[value_cols].mean()


def _warn_if_degenerate(interv: pd.DataFrame, intervention_type: str) -> None:
    """Log a warning if an intervention never moves any document."""
    disp = interv.loc[
        interv["intervention_type"] == intervention_type, "displacement"
    ]
    if len(disp) > 0 and (disp.abs() == 0).all():
        logger.warning(
            "Intervention '%s' has zero displacement for all %d documents; "
            "metrics that depend on it are uninformative (see review Q1).",
            intervention_type, len(disp),
        )


def _attribution_keys(attribution_df: pd.DataFrame) -> pd.DataFrame:
    df = attribution_df.copy()
    df[["query_id", "doc_id"]] = df[["query_id", "doc_id"]].astype(str)
    return df


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
    stage1_intervention: str = DEFAULT_STAGE1_INTERVENTION,
) -> dict[str, float]:
    """
    Fraction of documents where removing the attributed-as-important stage
    causes displacement > median displacement.

    1. Stage 1 is dominant if stage_1_attribution > 0.5, else Stage 2.
    2. Look up the intervention that removes the dominant stage
       (stage1_intervention or remove_stage2), averaged over seeds.
    3. Hit if |displacement| exceeds the median |displacement| of that
       same intervention type (types are on different scales), taken over
       the attributed documents only: intervention files cover every
       candidate (e.g. 1000 per query), and including unattributed,
       low-ranked documents would distort the median.

    Returns dict[query_id → comprehensiveness] for bootstrap_ci.
    """
    interv = average_over_seeds(intervention_df)
    attr = _attribution_keys(attribution_df)[
        ["query_id", "doc_id", "stage_1_attribution"]
    ]

    evaluated = interv.merge(
        attr[["query_id", "doc_id"]].drop_duplicates(), on=["query_id", "doc_id"]
    )
    medians = (
        evaluated.assign(abs_disp=evaluated["displacement"].abs())
        .groupby("intervention_type")["abs_disp"].median()
    )

    attr["intervention_type"] = np.where(
        attr["stage_1_attribution"] > 0.5,
        stage1_intervention,
        STAGE2_INTERVENTION,
    )
    merged = attr.merge(interv, on=_INTERVENTION_KEYS, how="inner")
    merged["hit"] = (
        merged["displacement"].abs()
        > merged["intervention_type"].map(medians)
    )

    return {
        str(qid): float(g["hit"].mean())
        for qid, g in merged.groupby("query_id")
    }


def sufficiency(
    attribution_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
    top_k: int = 10,
) -> dict[str, float]:
    """
    Fraction of documents where keeping ONLY the attributed-as-important
    stage preserves the document in top-K.

    - Stage 1 dominant → keep Stage 1 only → remove_stage2 intervention.
    - Stage 2 dominant → keep Stage 2 only → remove_gating (the reranker
      sees an expanded candidate set, testing whether it alone suffices).

    Seed replicates are averaged (mean rank_after).
    Returns dict[query_id → sufficiency] for bootstrap_ci.
    """
    interv = average_over_seeds(intervention_df)

    attr = _attribution_keys(attribution_df)[
        ["query_id", "doc_id", "stage_1_attribution"]
    ]
    attr["intervention_type"] = np.where(
        attr["stage_1_attribution"] > 0.5,
        STAGE2_INTERVENTION,
        "remove_gating",
    )
    merged = attr.merge(interv, on=_INTERVENTION_KEYS, how="inner")
    merged["hit"] = merged["rank_after"] <= top_k

    return {
        str(qid): float(g["hit"].mean())
        for qid, g in merged.groupby("query_id")
    }


def recovery_error(
    predicted_stage1_attribution: np.ndarray,
    stage1_displacement: np.ndarray,
    stage2_displacement: np.ndarray,
) -> float:
    """
    Mean |predicted_stage_1_attribution − normalized_displacement|.

    The normalized displacement is Stage 1's share of the total observed
    displacement:  |d1| / (|d1| + |d2|),  where d1 / d2 come from the
    interventions that remove Stage 1 / Stage 2.  This puts the observed
    effect on the same [0, 1] "share" scale as the attribution
    (stage_1 + stage_2 = 1), so the error is bounded in [0, 1].

    Documents where neither intervention moved them carry no signal and are
    skipped.  Returns NaN if no document has signal.
    """
    predicted = np.asarray(predicted_stage1_attribution, dtype=float)
    d1 = np.abs(np.asarray(stage1_displacement, dtype=float))
    d2 = np.abs(np.asarray(stage2_displacement, dtype=float))

    total = d1 + d2
    mask = total > 0
    if not mask.any():
        return float("nan")

    share = d1[mask] / total[mask]
    return float(np.mean(np.abs(predicted[mask] - share)))


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
        NaN values are excluded before bootstrapping.
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

    # E8: Filter out NaN values before bootstrapping
    query_ids = [q for q, v in values_by_query.items() if not np.isnan(v)]
    values = np.array([values_by_query[q] for q in query_ids])
    n = len(values)

    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n_queries": 0}

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

def _per_document_truth(
    attribution_df: pd.DataFrame,
    synthetic_gt_df: pd.DataFrame | None,
    true_alphas: dict[str, float] | float | None,
) -> pd.DataFrame:
    """
    Return attribution_df with a `true_stage_1` column.

    Per-document truth from synthetic_gt_df takes precedence; otherwise the
    planted α (scalar, or per query) is used for every document.  Missing
    ground truth raises instead of being silently skipped or defaulted.
    """
    df = _attribution_keys(attribution_df)

    if synthetic_gt_df is not None:
        keys = ["query_id", "doc_id"]
        gt = synthetic_gt_df[keys + ["true_stage_1_attribution"]].copy()
        gt[keys] = gt[keys].astype(str)
        if gt.duplicated(keys).any():
            raise ValueError(
                "synthetic_gt_df has multiple rows per (query_id, doc_id); "
                "evaluate one setting/alpha at a time."
            )
        df = df.merge(gt, on=keys, how="left").rename(
            columns={"true_stage_1_attribution": "true_stage_1"}
        )
        n_missing = int(df["true_stage_1"].isna().sum())
        if n_missing:
            raise ValueError(
                f"{n_missing} attributed documents have no row in "
                "synthetic_gt_df."
            )
    elif isinstance(true_alphas, dict):
        missing = set(df["query_id"]) - {str(q) for q in true_alphas}
        if missing:
            raise ValueError(
                f"true_alphas is missing {len(missing)} query id(s), "
                f"e.g. {sorted(missing)[:5]}"
            )
        alphas = {str(q): float(a) for q, a in true_alphas.items()}
        df["true_stage_1"] = df["query_id"].map(alphas)
    else:
        df["true_stage_1"] = float(true_alphas)

    return df


def evaluate_all_synthetic(
    attribution_df: pd.DataFrame,
    synthetic_gt_df: pd.DataFrame | None = None,
    true_alphas: dict[str, float] | float | None = None,
    method_name: str | None = None,
) -> dict[str, dict]:
    """
    Run all synthetic metrics on an attribution DataFrame.

    Parameters
    ----------
    attribution_df : DataFrame
        Attribution results (must follow schema).
    synthetic_gt_df : DataFrame, optional
        Per-document ground truth (synthetic_gt schema) for ONE setting/α.
        Preferred: attribution_error, rank_correlation and calibration all
        use the per-document true_stage_1_attribution.
    true_alphas : float or dict[query_id → float], optional
        Planted α, used for every document when synthetic_gt_df is not
        given.  rank_correlation is then NaN (constant truth per query).
    method_name : str, optional
        Filter to this method only. If None, evaluate per-method.

    Raises
    ------
    ValueError
        If no ground truth is given, or any attributed document lacks it.

    Returns
    -------
    dict[method_name → {metric_name → {mean, ci_low, ci_high, ...}}]
    plus "calibration_error" (float) and "calibration_bins" (list).
    """
    if synthetic_gt_df is None and true_alphas is None:
        raise ValueError(
            "evaluate_all_synthetic requires either synthetic_gt_df or "
            "true_alphas. Provide ground truth explicitly."
        )

    if method_name:
        attribution_df = attribution_df[
            attribution_df["method_name"] == method_name
        ]

    df = _per_document_truth(attribution_df, synthetic_gt_df, true_alphas)

    results: dict[str, dict] = {}

    for mname, mgroup in df.groupby("method_name"):
        per_query_error: dict[str, float] = {}
        per_query_corr: dict[str, float] = {}

        for qid, qgroup in mgroup.groupby("query_id"):
            predicted = qgroup["stage_1_attribution"].values
            truth = qgroup["true_stage_1"].values
            per_query_error[str(qid)] = attribution_error(predicted, truth)
            per_query_corr[str(qid)] = rank_correlation(predicted, truth)

        cal_err, cal_bins = calibration_error(
            mgroup["stage_1_attribution"].values,
            mgroup["true_stage_1"].values,
        )

        results[str(mname)] = {
            "attribution_error": bootstrap_ci(per_query_error),
            "rank_correlation": bootstrap_ci(per_query_corr),
            "calibration_error": cal_err,
            "calibration_bins": cal_bins,
        }

    return results


def evaluate_all_real(
    attribution_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
    method_name: str | None = None,
    top_k: int = 10,
    stage1_intervention: str = DEFAULT_STAGE1_INTERVENTION,
) -> dict[str, dict]:
    """
    Run all real-data metrics on attribution + intervention DataFrames.

    - stage_1_attribution is paired with *stage1_intervention* (open
      decision, see DEFAULT_STAGE1_INTERVENTION) and stage_2_attribution
      with remove_stage2.
    - Seed replicates are averaged before any metric is computed.
    - Every metric is per query with a bootstrap CI.

    Returns
    -------
    dict[method_name → {metric_name → {mean, ci_low, ci_high, ...}}]
    """
    if method_name:
        attribution_df = attribution_df[
            attribution_df["method_name"] == method_name
        ]

    interv = average_over_seeds(intervention_df)
    for itype in (stage1_intervention, STAGE2_INTERVENTION):
        _warn_if_degenerate(interv, itype)

    def _displacements(itype: str, name: str) -> pd.DataFrame:
        rows = interv[interv["intervention_type"] == itype]
        return rows[["query_id", "doc_id", "displacement"]].rename(
            columns={"displacement": name}
        )

    attr = _attribution_keys(attribution_df)
    merged = (
        attr.merge(_displacements(stage1_intervention, "d1"),
                   on=["query_id", "doc_id"], how="left")
        .merge(_displacements(STAGE2_INTERVENTION, "d2"),
               on=["query_id", "doc_id"], how="left")
    )

    results: dict[str, dict] = {}

    for mname, mgroup in merged.groupby("method_name"):
        corr_s1: dict[str, float] = {}
        corr_s2: dict[str, float] = {}
        recovery: dict[str, float] = {}

        for qid, g in mgroup.groupby("query_id"):
            qid = str(qid)

            g1 = g.dropna(subset=["d1"])
            if len(g1) >= 2:
                corr_s1[qid] = displacement_correlation(
                    g1["stage_1_attribution"].values, g1["d1"].values
                )

            g2 = g.dropna(subset=["d2"])
            if len(g2) >= 2:
                corr_s2[qid] = displacement_correlation(
                    g2["stage_2_attribution"].values, g2["d2"].values
                )

            g12 = g.dropna(subset=["d1", "d2"])
            if len(g12) >= 1:
                recovery[qid] = recovery_error(
                    g12["stage_1_attribution"].values,
                    g12["d1"].values,
                    g12["d2"].values,
                )

        mgroup_attr = attr[attr["method_name"] == mname]
        results[str(mname)] = {
            "displacement_correlation_stage1": bootstrap_ci(corr_s1),
            "displacement_correlation_stage2": bootstrap_ci(corr_s2),
            "comprehensiveness": bootstrap_ci(comprehensiveness(
                mgroup_attr, interv, top_k, stage1_intervention
            )),
            "sufficiency": bootstrap_ci(sufficiency(mgroup_attr, interv, top_k)),
            "recovery_error": bootstrap_ci(recovery),
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
                mean_val = value['mean']
                ci_lo = value.get('ci_low', 0)
                ci_hi = value.get('ci_high', 0)
                if np.isnan(mean_val):
                    print(f"    {metric_name:35s}  NaN  (no data)")
                else:
                    print(
                        f"    {metric_name:35s}  "
                        f"{mean_val:.4f}  "
                        f"[{ci_lo:.4f}, {ci_hi:.4f}]"
                    )
            elif isinstance(value, (int, float)):
                print(f"    {metric_name:35s}  {value:.4f}")
            # Skip non-numeric values like calibration_bins
    print()
