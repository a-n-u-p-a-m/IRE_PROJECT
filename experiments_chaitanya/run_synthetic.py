"""
run_synthetic.py - Run all of Chaitanya's pipeline on synthetic data.

Produces:
  1. Feature table from synthetic factual run
  2. Effect tables (E_rerank, E_gate, Delta_t) from the synthetic engine
  3. Predictor fits on training queries
  4. Stratified fidelity tables with all three baselines
  5. Cost-fidelity curves for each effect type / stratum

This script uses ONLY shared/ modules and synthetic data — no GPU, no index,
no real data.  Every number it produces is provisional and marked as such.
Run it before pushing to verify the whole Chaitanya pipeline end-to-end.

Usage::

    python -m experiments_chaitanya.run_synthetic \
        --output-dir experiments_chaitanya/synthetic_results \
        --n-queries 100 \
        --k-c 100

Results are written as CSV files (not Parquet — these are provisional results,
not the final tables that go in the effects/ directory).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config import load_config
from shared.effects import concat_effects, stage_effects_for_query, budget_effects_for_query
from shared.fixtures import smoke_test
from shared.schema import DELTA_T, E_GATE, E_RERANK, validate_feature, validate_run
from shared.synthetic import (
    apply_gate_stress,
    generate,
    params_for_regime,
    to_lookups,
    alternative_ranks,
)
from shared.worlds import build_worlds, build_t_grid

logger = logging.getLogger(__name__)


def build_run_table(
    s1: dict[str, dict[str, float]],
    system: str,
    checkpoint: str = "synthetic",
) -> pd.DataFrame:
    """Convert synthetic s1 scores to a validated run table."""
    rows = []
    for qid, doc_scores in s1.items():
        docs_sorted = sorted(doc_scores, key=lambda d: -doc_scores[d])
        for rank, did in enumerate(docs_sorted, start=1):
            rows.append({
                "query_id": str(qid),
                "doc_id": str(did),
                "system": str(system),
                "rank": rank,
                "score": float(doc_scores[did]),
                "checkpoint": str(checkpoint),
            })
    df = pd.DataFrame(rows)
    for col, dtype in [("query_id", "string"), ("doc_id", "string"),
                        ("system", "string"), ("checkpoint", "string")]:
        df[col] = df[col].astype(dtype)
    return validate_run(df)


def run_synthetic_pipeline(
    n_queries: int = 100,
    n_docs: int = 500,
    k_c: int = 100,
    k: int = 10,
    t_grid: tuple = (1.0, 0.75, 0.5),
    output_dir: str = "experiments_chaitanya/synthetic_results",
    seed: int = 42,
    regime: str = "strong_stages",
) -> dict[str, pd.DataFrame]:
    """
    End-to-end synthetic pipeline for Chaitanya's deliverables.

    Returns a dict of result DataFrames.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gain_map = {0: 0.0, 1: 0.0, 2: 3.0, 3: 7.0}

    logger.info("generating synthetic data: %d queries × %d docs, k_c=%d, regime=%s",
                n_queries, n_docs, k_c, regime)

    params = params_for_regime(
        regime, n_queries=n_queries, n_docs=n_docs,
        quantiles=[0.80, 0.93, 0.98], seed=seed,
    )
    df = generate(params)
    if regime == "gate_stress":
        df = apply_gate_stress(df, k_c, n_planted=10, seed=seed)
    universe_map, s1, s2, rel = to_lookups(df)
    alt_ranks = alternative_ranks(df)

    # Build run table for feature extraction
    run_df = build_run_table(s1, system="bm25")

    # Compute effects for each query
    effect_rows = []
    valid_qids = []
    for qid in sorted(s1.keys()):
        docs = universe_map[qid]
        if len(docs) <= k_c:
            continue
        judged = {d: True for d in docs}
        try:
            worlds = build_worlds(qid, docs, s1[qid], s2[qid], k_c)
            t_worlds = build_t_grid(
                worlds, s1[qid], s2[qid], budget_b=k_c,
                t_grid=list(t_grid),
                policy="boundary",
                alternative_rank=alt_ranks.get(qid),
                seed=seed,
            )
            stage_eff = stage_effects_for_query(
                worlds, rel[qid], judged, qid, k, gain_map
            )
            budget_eff = budget_effects_for_query(
                worlds, t_worlds, rel[qid], judged, qid, k, gain_map
            )
            effect_rows.extend([stage_eff, budget_eff])
            valid_qids.append(qid)
        except Exception as exc:
            logger.debug("skipping query %s: %s", qid, exc)

    if not effect_rows:
        raise RuntimeError("No valid queries for synthetic run")

    all_effects = concat_effects(effect_rows)
    logger.info("computed effects for %d queries", len(valid_qids))

    # Train/val/test split by query (60/20/20)
    rng = np.random.default_rng(seed)
    all_q = sorted(valid_qids)
    rng.shuffle(all_q)
    n_train = int(0.6 * len(all_q))
    n_val = int(0.2 * len(all_q))
    train_qids = set(all_q[:n_train])
    val_qids = set(all_q[n_train:n_train + n_val])
    test_qids = set(all_q[n_train + n_val:])

    logger.info(
        "split: train=%d, val=%d, test=%d",
        len(train_qids), len(val_qids), len(test_qids),
    )

    # Feature extraction
    from experiments_chaitanya.features import (
        extract_stage1_features,
        extract_stage2_features,
    )

    s1_feats = extract_stage1_features(run_df, "bm25", k_c)

    # Simulate stage2 features from synthetic s2 scores (no cache needed)
    s2_rows = []
    for qid in valid_qids:
        doc_scores = s2[qid]
        vals = list(doc_scores.values())
        vmin, vmax = min(vals), max(vals)
        vrange = max(vmax - vmin, 1e-12)
        for did, score in doc_scores.items():
            s2_rows.append({
                "query_id": str(qid), "doc_id": str(did),
                "feature_version": "v1", "feature_name": "stage2_score",
                "value": float(score), "generation_seconds": 0.0,
            })
            s2_rows.append({
                "query_id": str(qid), "doc_id": str(did),
                "feature_version": "v1", "feature_name": "stage2_score_norm",
                "value": float((score - vmin) / vrange),
                "generation_seconds": 0.0,
            })
    s2_feat_df = pd.DataFrame(s2_rows)
    for col, dtype in [("query_id", "string"), ("doc_id", "string"),
                        ("feature_version", "string"), ("feature_name", "string")]:
        s2_feat_df[col] = s2_feat_df[col].astype(dtype)
    s2_feat_df = validate_feature(s2_feat_df)

    feats = pd.concat([s1_feats, s2_feat_df], ignore_index=True)

    train_feats = feats[feats["query_id"].isin(train_qids)]
    test_feats = feats[feats["query_id"].isin(test_qids)]
    train_eff = all_effects[all_effects["query_id"].isin(train_qids)]
    test_eff = all_effects[all_effects["query_id"].isin(test_qids)]

    # Fit predictors
    from experiments_chaitanya.predictors import fit_all_predictors, evaluate_all_predictors
    predictors = fit_all_predictors(train_feats, train_eff)
    logger.info("fitted %d predictors", len(predictors))

    # Evaluate with all baselines
    fidelity_table = evaluate_all_predictors(
        predictors, test_feats, test_eff, train_eff,
        n_iter=200, seed=seed, top_m=20,
    )
    logger.info("fidelity table:\n%s", fidelity_table.to_string(index=False))

    # Save results
    results = {
        "effects": all_effects,
        "features": feats,
        "fidelity_table": fidelity_table,
        "train_query_ids": pd.DataFrame({"query_id": sorted(train_qids)}),
        "test_query_ids": pd.DataFrame({"query_id": sorted(test_qids)}),
    }

    for name, frame in results.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        logger.info("saved %s (%d rows) → %s", name, len(frame), path)

    # Mark results as provisional
    (out / "PROVISIONAL.md").write_text(
        "# PROVISIONAL — Synthetic Results\n\n"
        "These results are from synthetic data only.  They are provisional "
        "and will be replaced by real-data results in Week 3.\n"
    )

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run synthetic pipeline for Chaitanya")
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--n-docs", type=int, default=500)
    parser.add_argument("--k-c", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--regime", default="strong_stages")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir",
                        default="experiments_chaitanya/synthetic_results")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke_test() first")
    args = parser.parse_args()

    if args.smoke_test:
        logger.info("running smoke test …")
        assert smoke_test(), "smoke test failed"
        logger.info("smoke test passed")

    run_synthetic_pipeline(
        n_queries=args.n_queries,
        n_docs=args.n_docs,
        k_c=args.k_c,
        k=args.k,
        regime=args.regime,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
