"""
run_real_predictors.py - Week 4: fit and evaluate predictors on real effect tables.

This is Chaitanya's Week-4 deliverable — the prediction and cost section of
the report.  It is the real-data counterpart of run_synthetic.py.

Prerequisites
-------------
* ``cache/stage2.sqlite`` (merged by merge_cache.py)
* ``effects/real/`` populated by ``python -m experiments_shubham.real --system bm25``
  and ``python -m experiments_shubham.real --system dpr``
* ``data/query_split.json`` (train/val/test split by query, never by document)
* ``config/base.yaml``

Outputs (all in ``effects/real_predictors/``)
----------------------------------------------
fidelity_bm25.csv / fidelity_dpr.csv
    Stratified fidelity table: Spearman ρ, MAE, top-M recovery for each
    predictor × baseline × effect-type × stratum on HELD-OUT TEST queries.

cost_fidelity_bm25.csv / cost_fidelity_dpr.csv
    One row per (budget, effect_type, stratum, predictor); budgets sweep
    from 0 to full universe.  Must reach 1.0 Spearman at full budget.

predictor_manifest.json
    Provenance: config fingerprint, query split sizes, effect source dir,
    compute time (s), git commit.

Usage::

    python -m experiments_chaitanya.run_real_predictors \\
        --config config/base.yaml \\
        --effects-dir effects/real \\
        --out-dir effects/real_predictors

Add ``--system bm25`` or ``--system dpr`` to run only one system.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SYSTEM_DEFAULT = None   # run both


def _load_query_split(split_path: Path | str) -> dict[str, set]:
    """
    Load train/val/test query-ID split.

    Accepts either:
    * A JSON file: ``{"train": [...], "val": [...], "test": [...]}``
    * A directory containing ``train.txt``, ``val.txt``, ``test.txt``
      (one query_id per line, no header).
    """
    import json
    p = Path(split_path)
    if p.is_file() and p.suffix == ".json":
        raw = json.loads(p.read_text())
        return {k: set(str(x) for x in v) for k, v in raw.items()}
    if p.is_dir():
        result: dict[str, set] = {}
        for split in ("train", "val", "test"):
            f = p / f"{split}.txt"
            if f.exists():
                result[split] = {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}
        if result:
            return result
    raise FileNotFoundError(
        f"Cannot find query split at {split_path}. "
        "Expected a JSON file with keys train/val/test, or a directory with "
        "train.txt / val.txt / test.txt."
    )


def _effect_files_for_system(effects_dir: Path, system: str) -> list[Path]:
    """Return all effect parquet files for a given stage1 system."""
    # real.py writes one file per (k_c, policy) with naming:
    # e_rerank_{system}_kc{K}.parquet / e_gate_{system}_kc{K}.parquet / ...
    # or grouped in subdirectories — be flexible.
    hits = list(effects_dir.glob(f"*{system}*.parquet"))
    if not hits:
        # Try any parquet; filter by system column later
        hits = list(effects_dir.glob("*.parquet"))
    return hits


def _load_effects(effects_dir: Path, system: str, cfg) -> pd.DataFrame:
    """Load, filter, and concatenate effect tables for one system."""
    files = _effect_files_for_system(effects_dir, system)
    if not files:
        raise FileNotFoundError(
            f"No effect parquet files found for system={system!r} in {effects_dir}. "
            "Run `python -m experiments_shubham.real --system {system}` first."
        )
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if "stage1_system" in df.columns:
                df = df[df["stage1_system"] == system]
            elif "system" in df.columns:
                df = df[df["system"] == system]
            frames.append(df)
        except Exception as exc:
            logger.warning("skipping %s: %s", f, exc)
    if not frames:
        raise RuntimeError(f"No valid effect data loaded for system={system!r}")
    effects = pd.concat(frames, ignore_index=True)
    effects["query_id"] = effects["query_id"].astype(str)
    effects["doc_id"] = effects["doc_id"].astype(str)
    logger.info("loaded %d effect rows for system=%s from %d files",
                len(effects), system, len(frames))
    return effects


def _build_features_from_cache(
    effects: pd.DataFrame,
    cache_path: Path,
    cfg,
    system: str,
    universe_path: Path,
) -> pd.DataFrame:
    """
    Build feature table from the merged Stage-2 cache and the universe.

    Uses experiments_chaitanya.features.extract_stage1_features and
    extract_stage2_features; records generation_seconds and peak memory.
    """
    import tracemalloc

    from shared.cache import Stage2Cache
    from shared.universe import load_frozen

    from experiments_chaitanya.features import (
        extract_stage1_features,
        extract_stage2_features,
    )

    # Load universe and run table
    universe = load_frozen(str(universe_path), fingerprint=cfg.fingerprint())
    model = cfg.reranker_model
    cache = Stage2Cache(cache_path)

    # Build a minimal run table from the effect table (we need stage1 rank/score)
    run_cols = ["query_id", "doc_id", "stage1_system", "stage1_rank", "stage1_score",
                "k_c", "checkpoint"]
    available = [c for c in run_cols if c in effects.columns]
    run_df = effects[available].drop_duplicates(["query_id", "doc_id"]).copy()

    # Rename for feature extractor compatibility
    rename = {}
    if "stage1_system" in run_df.columns:
        rename["stage1_system"] = "system"
    if "stage1_rank" in run_df.columns:
        rename["stage1_rank"] = "rank"
    if "stage1_score" in run_df.columns:
        rename["stage1_score"] = "score"
    if rename:
        run_df = run_df.rename(columns=rename)
    if "system" not in run_df.columns:
        run_df["system"] = system
    if "checkpoint" not in run_df.columns:
        run_df["checkpoint"] = cfg.reranker_model

    k_c = cfg.candidate_depths[0]  # headline depth

    tracemalloc.start()
    t0 = time.perf_counter()
    s1_feats = extract_stage1_features(run_df, system, k_c)
    s2_feats = extract_stage2_features(universe, cache, model, k_c)
    gen_secs = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    feats = pd.concat([s1_feats, s2_feats], ignore_index=True)
    logger.info(
        "features built: %d rows in %.1fs, peak RAM %.1f MB",
        len(feats), gen_secs, peak_bytes / 1e6,
    )
    return feats, gen_secs, peak_bytes


def _run_system(
    system: str,
    effects_dir: Path,
    out_dir: Path,
    cfg,
    cache_path: Path,
    universe_path: Path,
    query_split: dict[str, set],
    top_m: int,
    seed: int,
    n_bootstrap: int,
) -> dict:
    """Full predictor evaluation pipeline for one Stage-1 system."""
    from experiments_chaitanya.predictors import fit_all_predictors, evaluate_all_predictors
    from experiments_chaitanya.cost_fidelity import CostFidelityHarness
    from shared.provenance import run_manifest, write_manifest

    effects = _load_effects(effects_dir, system, cfg)

    train_qids = query_split.get("train", set())
    test_qids = query_split.get("test", set())
    if not train_qids or not test_qids:
        raise RuntimeError("Query split must contain 'train' and 'test' keys.")

    train_eff = effects[effects["query_id"].isin(train_qids)]
    test_eff = effects[effects["query_id"].isin(test_qids)]
    logger.info("system=%s  train=%d queries  test=%d queries",
                system, train_eff["query_id"].nunique(), test_eff["query_id"].nunique())

    # Build features (needs GPU cache)
    try:
        feats, gen_secs, peak_bytes = _build_features_from_cache(
            effects, cache_path, cfg, system, universe_path
        )
        feature_source = "real_cache"
    except Exception as exc:
        logger.warning(
            "Could not build features from cache (%s). "
            "Attempting to use stage1-only features as a fallback.", exc
        )
        from experiments_chaitanya.features import extract_stage1_features
        run_df = effects[["query_id", "doc_id"]].drop_duplicates().copy()
        if "stage1_rank" in effects.columns:
            run_df["rank"] = effects["stage1_rank"]
        if "stage1_score" in effects.columns:
            run_df["score"] = effects["stage1_score"]
        run_df["system"] = system
        run_df["checkpoint"] = cfg.reranker_model
        k_c = cfg.candidate_depths[0]
        feats = extract_stage1_features(run_df, system, k_c)
        gen_secs, peak_bytes = 0.0, 0
        feature_source = "stage1_only_fallback"

    train_feats = feats[feats["query_id"].isin(train_qids)]
    test_feats = feats[feats["query_id"].isin(test_qids)]

    # --- Fit predictors --------------------------------------------------
    logger.info("fitting predictors for system=%s …", system)
    t0 = time.perf_counter()
    predictors = fit_all_predictors(train_feats, train_eff)
    fit_secs = time.perf_counter() - t0
    logger.info("fitted %d predictors in %.1fs", len(predictors), fit_secs)

    # --- Evaluate: stratified fidelity table on test queries -------------
    logger.info("evaluating predictors on %d test queries …", len(test_qids))
    fidelity = evaluate_all_predictors(
        predictors, test_feats, test_eff, train_eff,
        n_iter=n_bootstrap, seed=seed, top_m=top_m,
    )
    fid_path = out_dir / f"fidelity_{system}.csv"
    fidelity.to_csv(fid_path, index=False)
    logger.info("fidelity table (%d rows) → %s", len(fidelity), fid_path)

    # --- Cost-fidelity curves --------------------------------------------
    logger.info("computing cost-fidelity curves for system=%s …", system)
    harness = CostFidelityHarness(predictors)
    cf_rows = harness.sweep_budgets(
        full_effects=test_eff,
        feature_df=test_feats,
        budgets=cfg.outsider_budgets + [len(test_eff)],
        seed=seed,
        top_m=top_m,
    )
    cf_path = out_dir / f"cost_fidelity_{system}.csv"
    cf_rows.to_csv(cf_path, index=False)
    logger.info("cost-fidelity (%d rows) → %s", len(cf_rows), cf_path)

    return {
        "system": system,
        "n_train_queries": len(train_qids),
        "n_test_queries": len(test_qids),
        "n_predictors": len(predictors),
        "feature_source": feature_source,
        "feature_gen_seconds": round(gen_secs, 2),
        "feature_peak_bytes": peak_bytes,
        "predictor_fit_seconds": round(fit_secs, 2),
        "fidelity_path": str(fid_path),
        "cost_fidelity_path": str(cf_path),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Week-4: fit and evaluate predictors on the real effect tables."
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--effects-dir", default="effects/real",
                        help="Directory written by real.py (default: effects/real)")
    parser.add_argument("--cache", default="cache/stage2.sqlite")
    parser.add_argument("--universe", default="data/universe.parquet")
    parser.add_argument("--query-split", default="data/query_split.json",
                        help="Train/val/test split (JSON or dir with *.txt files)")
    parser.add_argument("--out-dir", default="effects/real_predictors")
    parser.add_argument("--system", choices=["bm25", "dpr"], default=None,
                        help="Run only this system (default: both)")
    parser.add_argument("--top-m", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)

    from shared.config import load_config
    from shared.provenance import run_manifest, write_manifest

    cfg = load_config(args.config)
    effects_dir = Path(args.effects_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache)
    universe_path = Path(args.universe)

    if not effects_dir.exists():
        logger.error("effects dir %s not found. Run real.py first.", effects_dir)
        return 1
    if not cache_path.exists():
        logger.error("cache %s not found. Run merge_cache.py first.", cache_path)
        return 1

    # Load query split
    split_path = Path(args.query_split)
    if not split_path.exists():
        # Create a simple 60/20/20 split from the universe query IDs
        logger.warning(
            "No query split found at %s — generating a deterministic 60/20/20 split "
            "from the universe. For reproducibility, commit data/query_split.json.",
            split_path,
        )
        import pandas as pd
        universe = pd.read_parquet(universe_path)
        all_q = sorted(universe["query_id"].astype(str).unique())
        rng = np.random.default_rng(42)
        rng.shuffle(all_q := list(all_q))
        n = len(all_q)
        query_split = {
            "train": set(all_q[: int(0.6 * n)]),
            "val": set(all_q[int(0.6 * n): int(0.8 * n)]),
            "test": set(all_q[int(0.8 * n):]),
        }
        # Save for reproducibility
        import json
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps({k: sorted(v) for k, v in query_split.items()}, indent=1))
        logger.info("saved query split → %s", split_path)
    else:
        query_split = _load_query_split(split_path)

    logger.info("query split: train=%d  val=%d  test=%d",
                len(query_split.get("train", [])),
                len(query_split.get("val", [])),
                len(query_split.get("test", [])))

    systems = [args.system] if args.system else list(cfg.stage1_systems)
    t_wall = time.perf_counter()
    system_records = []
    for system in systems:
        try:
            record = _run_system(
                system=system,
                effects_dir=effects_dir,
                out_dir=out_dir,
                cfg=cfg,
                cache_path=cache_path,
                universe_path=universe_path,
                query_split=query_split,
                top_m=args.top_m,
                seed=args.seed,
                n_bootstrap=args.n_bootstrap,
            )
            system_records.append(record)
        except Exception as exc:
            logger.error("system %s failed: %s", system, exc, exc_info=True)
            return 1

    total_secs = time.perf_counter() - t_wall

    # Hashing: document the decision
    hashing_note = (
        "Hashing features were not included. The go/no-go criteria required a "
        "concrete, pre-specified method committed before any experiments, a "
        "representation source independent of the Stage-2 model, and a fidelity "
        "improvement over the membership-only baseline on held-out queries. "
        "At the Week-2 decision meeting none of the four criteria were fully "
        "satisfied (method, representation source, and bit-width were not "
        "pre-specified in the config freeze), so hashing.enabled remained false. "
        "All prediction results use conventional features only."
    )

    manifest = run_manifest(
        cfg.fingerprint(),
        owner="chaitanya",
        extra={
            "task": "week4_real_predictors",
            "effects_dir": str(effects_dir),
            "cache": str(cache_path),
            "universe": str(universe_path),
            "query_split": str(split_path),
            "systems": system_records,
            "total_wall_seconds": round(total_secs, 2),
            "hashing": {"enabled": False, "note": hashing_note},
            "compute_accounting": {
                "feature_gen_seconds": sum(
                    s.get("feature_gen_seconds", 0) for s in system_records
                ),
                "predictor_fit_seconds": sum(
                    s.get("predictor_fit_seconds", 0) for s in system_records
                ),
                "total_seconds": round(total_secs, 2),
            },
        },
    )
    man_path = out_dir / "predictor_manifest.json"
    write_manifest(man_path, manifest)
    logger.info("predictor manifest → %s", man_path)
    logger.info("Done in %.1fs", total_secs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
