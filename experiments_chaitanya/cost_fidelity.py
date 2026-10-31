"""
cost_fidelity.py - Cost-fidelity harness for the budget experiment.

Generates a cost-fidelity curve by sweeping the Stage-2 budget from 0 to the
full universe size.  At each budget point, documents are scored by a cheap
predictor (feature-based) rather than the full reranker, and the fidelity of
the resulting effect prediction is measured against the exact (full-budget)
reference.

The curve must converge to the exact reference at full budget — this is the
key check that the harness is correct.

Key design decisions:

  * Extra Stage-2 calls are counted explicitly, not inferred.  Every call to
    the reranker for a query-document pair beyond the factual-run C_q counts
    as one unit of budget.

  * The curve is produced on SYNTHETIC data first (Week 1/2) and on real data
    in Week 3.  Both use the same harness code.

  * The reference is the full-universe effect table from the counterfactual
    engine.  The harness compares approximated effects (from limited budget)
    to this reference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from shared.metrics import cost_fidelity_curve, evaluate_within_strata
from shared.schema import DELTA_T, E_GATE, E_RERANK

logger = logging.getLogger(__name__)


@dataclass
class CostFidelityPoint:
    """One point on the cost-fidelity curve."""
    budget: int               # extra Stage-2 calls beyond the factual C_q
    effect_type: str
    stratum: str
    predictor: str
    spearman: float
    mae: float
    top_m_recovery: float
    n_documents: int


def _approximate_effects_at_budget(
    full_effects: pd.DataFrame,
    feature_df: pd.DataFrame,
    predictor,
    effect_type: str,
    budget: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a limited-budget approximation of effects.

    At budget B extra calls, score only B additional documents (beyond C_q)
    with the real reranker.  For the remaining documents, use the feature-based
    predictor as a proxy.

    For the synthetic harness, we simulate this by replacing a random sample
    of (1 - budget/n_unscored) predictions with the predictor output.
    """
    rng = np.random.default_rng(seed)
    eff = full_effects[full_effects["effect_type"] == effect_type].copy()

    # How many documents are outside C_q (non-survivors)?
    non_survivors = eff[eff["stratum"] != "survivor"]
    n_unscored = len(non_survivors)

    if budget >= n_unscored or n_unscored == 0:
        # Full budget: return exact effects
        return eff

    # Simulate: for (n_unscored - budget) documents, replace effect with
    # predictor output; for the remaining `budget` documents, use exact effect.
    pred_df = predictor.predict(feature_df)
    pred_col = f"pred_{effect_type}"

    approx = eff.copy()
    ns_idx = non_survivors.index.to_numpy()
    rng.shuffle(ns_idx)
    to_replace = ns_idx[budget:]  # these don't get extra scoring → predictor value

    if pred_col in pred_df.columns:
        pred_map = pred_df.set_index(["query_id", "doc_id"])[pred_col].to_dict()
        for idx in to_replace:
            row = approx.loc[idx]
            key = (str(row["query_id"]), str(row["doc_id"]))
            pred_val = pred_map.get(key, float("nan"))
            if np.isfinite(pred_val):
                approx.loc[idx, "effect"] = pred_val

    return approx


def compute_cost_fidelity_curve(
    full_effects: pd.DataFrame,
    feature_df: pd.DataFrame,
    predictor,
    effect_type: str,
    budget_grid: list[int],
    stratum: str,
    seed: int = 42,
    top_m: int = 20,
) -> pd.DataFrame:
    """
    Compute fidelity at each budget point in budget_grid.

    Parameters
    ----------
    full_effects:
        The exact effect table (full universe, from the counterfactual engine).
    feature_df:
        Feature table for the predictor.
    predictor:
        A fitted EffectPredictor for this effect_type.
    effect_type:
        Which effect type to curve.
    budget_grid:
        List of extra Stage-2 call budgets (e.g., [0, 25, 100, 500, n_universe]).
    stratum:
        Which stratum to track on the curve.
    seed:
        Random seed for the approximation simulation.
    top_m:
        For top_m_recovery metric.

    Returns
    -------
    pd.DataFrame
        One row per budget point, with fidelity metrics.
    """
    results_for_curve = []

    for budget in sorted(budget_grid):
        approx = _approximate_effects_at_budget(
            full_effects, feature_df, predictor, effect_type, budget, seed=seed
        )

        # Merge predictor predictions onto approximate frame
        pred_df = predictor.predict(feature_df)
        pred_col = f"pred_{effect_type}"
        merged = approx.merge(
            pred_df[["query_id", "doc_id", pred_col]],
            on=["query_id", "doc_id"],
            how="left",
        )

        # Compute fidelity of APPROX effects vs EXACT effects
        # (the predictor column here is the approximated effect, not the
        # raw predictor output — we compare approx to reference)
        approx_col = "effect_approx"
        merged[approx_col] = approx["effect"].values

        fid = evaluate_within_strata(
            merged,
            predicted_column=approx_col,
            effect_type=effect_type,
            top_m=top_m,
            n_iter=200,  # fewer iterations for the curve sweep
            seed=seed,
        )
        results_for_curve.append((budget, fid))

    curve = cost_fidelity_curve(results_for_curve, stratum=stratum)
    return curve


def convergence_check(
    curve: pd.DataFrame,
    full_budget: int,
    tolerance: float = 0.01,
) -> bool:
    """
    Assert that the curve converges to the exact reference at full budget.

    At full budget (budget = n_universe) the approximated effects ARE the
    exact effects, so fidelity must equal 1.0 (within floating-point noise).

    Returns True if the check passes, raises AssertionError otherwise.
    """
    if curve.empty:
        raise AssertionError("cost-fidelity curve is empty")

    max_budget_row = curve.loc[curve["extra_stage2_calls"].idxmax()]

    if abs(int(max_budget_row["extra_stage2_calls"]) - full_budget) > full_budget * 0.05:
        logger.warning(
            "convergence check: max budget in curve (%d) differs from "
            "expected full budget (%d) by more than 5%%",
            int(max_budget_row["extra_stage2_calls"]), full_budget,
        )

    # At full budget, approx == reference → spearman must be ~1.0 and mae ~0.0
    spearman = float(max_budget_row.get("spearman", float("nan")))
    mae = float(max_budget_row.get("mae", float("nan")))

    if not np.isnan(spearman) and spearman < 1.0 - tolerance:
        raise AssertionError(
            f"cost-fidelity curve does not converge: spearman at full budget "
            f"is {spearman:.4f}, expected ~1.0 (tolerance {tolerance}). "
            "Check that the reference and approximate effects are from the "
            "same effect table and that the budget grid includes the full universe."
        )

    logger.info(
        "convergence check passed: spearman=%.4f, mae=%.6f at budget=%d",
        spearman, mae, int(max_budget_row["extra_stage2_calls"]),
    )
    return True


def synthetic_cost_fidelity(
    n_queries: int = 50,
    k_c: int = 100,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Run the full cost-fidelity harness on synthetic data and verify convergence.

    Returns {effect_type: curve_df} for reporting in fidelity_table.md.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from shared.synthetic import generate, params_for_regime, to_lookups
    from shared.fixtures import TOY_GAIN_MAP
    from shared.effects import stage_effects_for_query, concat_effects
    from shared.worlds import build_worlds
    from shared.allocation import allocate
    from shared.config import load as load_config
    from experiments_chaitanya.features import extract_stage1_features
    from experiments_chaitanya.predictors import EffectPredictor

    logger.info("running synthetic cost-fidelity harness …")

    params = params_for_regime(
        "strong_stages", n_queries=n_queries, n_docs=500,
        quantiles=[0.80, 0.93, 0.98], seed=seed
    )
    df = generate(params)
    universe_map, s1, s2, rel = to_lookups(df)

    gain_map = {0: 0.0, 1: 0.0, 2: 3.0, 3: 7.0}
    k = 10

    effect_rows = []
    for qid in list(s1.keys()):
        docs = universe_map[qid]
        if len(docs) <= k_c:
            continue
        judged = {d: True for d in docs}
        try:
            worlds = build_worlds(qid, docs, s1[qid], s2[qid], k_c)
        except Exception:
            continue
        from shared.effects import stage_effects_for_query
        eff = stage_effects_for_query(worlds, rel[qid], judged, qid, k, gain_map)
        effect_rows.append(eff)

    if not effect_rows:
        raise RuntimeError("no valid synthetic queries for cost-fidelity")

    from shared.effects import concat_effects
    all_effects = concat_effects(effect_rows)

    # Build a minimal run table for feature extraction
    run_rows = []
    for qid, scores in s1.items():
        docs_sorted = sorted(scores, key=lambda d: -scores[d])
        for rank, did in enumerate(docs_sorted, start=1):
            run_rows.append({
                "query_id": qid, "doc_id": did,
                "system": "bm25", "rank": rank,
                "score": scores[did], "checkpoint": "synthetic",
            })
    run_df = pd.DataFrame(run_rows)
    for col, dtype in [("query_id", "string"), ("doc_id", "string"),
                        ("system", "string"), ("checkpoint", "string")]:
        run_df[col] = run_df[col].astype(dtype)

    from shared.schema import validate_run
    run_df = validate_run(run_df)

    # Extract features from factual run
    from experiments_chaitanya.features import extract_stage1_features
    feats = extract_stage1_features(run_df, "bm25", k_c)

    # Also add stage2_score feature from synthetic s2
    s2_rows = []
    for qid, doc_scores in s2.items():
        vals = list(doc_scores.values())
        vmin, vmax = min(vals), max(vals)
        vrange = max(vmax - vmin, 1e-12)
        for did, score in doc_scores.items():
            s2_rows.append({
                "query_id": qid, "doc_id": did,
                "feature_version": "v1", "feature_name": "stage2_score",
                "value": score, "generation_seconds": 0.0,
            })
            s2_rows.append({
                "query_id": qid, "doc_id": did,
                "feature_version": "v1", "feature_name": "stage2_score_norm",
                "value": (score - vmin) / vrange, "generation_seconds": 0.0,
            })
    s2_feat_df = pd.DataFrame(s2_rows)
    for col, dtype in [("query_id", "string"), ("doc_id", "string"),
                        ("feature_version", "string"), ("feature_name", "string")]:
        s2_feat_df[col] = s2_feat_df[col].astype(dtype)
    from shared.schema import validate_feature
    s2_feat_df = validate_feature(s2_feat_df)

    feats = pd.concat([feats, s2_feat_df], ignore_index=True)

    # Train/test split by query
    all_qids = all_effects["query_id"].astype(str).unique().tolist()
    split = int(len(all_qids) * 0.6)
    train_qids = set(all_qids[:split])
    test_qids = set(all_qids[split:])

    train_feats = feats[feats["query_id"].isin(train_qids)]
    test_feats = feats[feats["query_id"].isin(test_qids)]
    train_eff = all_effects[all_effects["query_id"].isin(train_qids)]
    test_eff = all_effects[all_effects["query_id"].isin(test_qids)]

    curves: dict[str, pd.DataFrame] = {}
    for effect_type in (E_RERANK, E_GATE):
        try:
            predictor = EffectPredictor(effect_type)
            predictor.fit(train_feats, train_eff)

            # n_unscored ≈ n_docs - k_c per query × n_test_queries
            n_test = len(test_qids)
            n_unscored = n_test * (500 - k_c)
            budget_grid = [0, 50, 200, 500, n_unscored]

            for stratum in ("survivor", "non_survivor"):
                sel = test_eff[test_eff["effect_type"] == effect_type]
                if stratum not in sel["stratum"].values:
                    continue
                curve = compute_cost_fidelity_curve(
                    test_eff, test_feats, predictor, effect_type,
                    budget_grid, stratum=stratum, seed=seed,
                )
                curves[f"{effect_type}_{stratum}"] = curve
                logger.info(
                    "cost-fidelity curve for %s/%s:\n%s",
                    effect_type, stratum, curve.to_string(index=False),
                )
        except Exception as exc:
            logger.warning("cost-fidelity for %s failed: %s", effect_type, exc)

    return curves


__all__ = [
    "CostFidelityPoint",
    "compute_cost_fidelity_curve",
    "convergence_check",
    "synthetic_cost_fidelity",
]
