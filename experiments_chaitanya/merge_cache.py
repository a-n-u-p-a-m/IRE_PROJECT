"""
merge_cache.py - Merge three shard caches and write the provenance manifest.

This runner is the bridge between the distributed fill (three people, three
Kaggle sessions) and the downstream analysis (real.py, features.py,
predictors.py).  It is the only place that may read all three shard files.

What it does
------------
1. Opens or creates the primary cache at ``cache/stage2.sqlite``.
2. Merges each shard file (anupam, shubham, chaitanya) with
   ``Stage2Cache.merge_from()``, which is idempotent — re-running is safe.
3. Calls ``Stage2Cache.require_complete(universe, model)`` across all 97
   queries; raises if any (query, doc) pair is missing.
4. Asserts the total row count equals EXPECTED_ROWS = 872,455.
5. Reads each shard manifest, extracts its digest and query_ids, and writes
   ``cache/stage2_manifest.json`` recording which shards were merged, their
   digests, the total row count, and the config fingerprint.

Usage (run from repo root after all three shards are on disk)::

    python -m experiments_chaitanya.merge_cache \\
        --universe data/universe.parquet \\
        --config   config/base.yaml \\
        --out      cache/stage2_manifest.json

Shard file locations are read from the shard manifests in cache/; override
with --shard-anupam, --shard-shubham, --shard-chaitanya if needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_ROWS = 872_455
OWNERS = ("anupam", "shubham", "chaitanya")


def _load_shard_manifest(owner: str, cache_dir: Path) -> dict:
    p = cache_dir / f"stage2_shard_{owner}_manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Merge three Stage-2 shard caches and write a provenance manifest."
    )
    parser.add_argument("--universe", default="data/universe.parquet",
                        help="Frozen universe parquet (default: data/universe.parquet)")
    parser.add_argument("--config", default="config/base.yaml",
                        help="Config YAML (default: config/base.yaml)")
    parser.add_argument("--cache-dir", default="cache",
                        help="Directory holding shard SQLite files (default: cache)")
    parser.add_argument("--primary", default="cache/stage2.sqlite",
                        help="Output: primary merged cache (default: cache/stage2.sqlite)")
    parser.add_argument("--out", default="cache/stage2_manifest.json",
                        help="Output manifest path (default: cache/stage2_manifest.json)")
    parser.add_argument("--shard-anupam", default=None)
    parser.add_argument("--shard-shubham", default=None)
    parser.add_argument("--shard-chaitanya", default=None)
    parser.add_argument("--skip-completeness", action="store_true",
                        help="Skip require_complete (useful when universe not yet downloaded)")
    args = parser.parse_args(argv)

    from shared.cache import Stage2Cache
    from shared.config import load_config
    from shared.provenance import run_manifest, write_manifest

    cfg = load_config(args.config)
    model = cfg.reranker_model
    cache_dir = Path(args.cache_dir)

    shard_paths: dict[str, Path] = {}
    for owner in OWNERS:
        override = getattr(args, f"shard_{owner}", None)
        if override:
            shard_paths[owner] = Path(override)
        else:
            # Try the conventional name first, then fall back to the manifest
            conventional = cache_dir / f"stage2_shard_{owner}.sqlite"
            if conventional.exists():
                shard_paths[owner] = conventional
            else:
                man = _load_shard_manifest(owner, cache_dir)
                artifact = man.get("artifact")
                if artifact and Path(artifact).exists():
                    shard_paths[owner] = Path(artifact)
                else:
                    logger.warning(
                        "Shard for %s not found (tried %s and manifest.artifact=%s); "
                        "this shard will be SKIPPED — the merged cache will be incomplete.",
                        owner, conventional, artifact,
                    )

    primary = Stage2Cache(args.primary)

    shard_records = []
    for owner in OWNERS:
        path = shard_paths.get(owner)
        if path is None:
            continue
        logger.info("merging shard: %s (%s) …", owner, path)
        n_added = primary.merge_from(path, model)
        man = _load_shard_manifest(owner, cache_dir)
        shard_records.append({
            "owner": owner,
            "path": str(path),
            "n_added_this_merge": n_added,
            "digest": man.get("digest"),
            "query_ids": man.get("query_ids", []),
            "gpu": man.get("gpu"),
            "config_fingerprint": man.get("config_fingerprint"),
        })
        logger.info("  added %d rows from %s", n_added, owner)

    total = primary.count(model)
    logger.info("primary cache now has %d rows (model=%s)", total, model)

    if total != EXPECTED_ROWS:
        logger.error(
            "row count mismatch: expected %d, got %d. "
            "One or more shards may be missing or incomplete.",
            EXPECTED_ROWS, total,
        )
        # Don't return 1 yet — still run completeness check to surface which queries
        # are missing, then fail.

    if not args.skip_completeness:
        universe_path = args.universe
        if not Path(universe_path).exists():
            logger.error(
                "Universe not found at %s. Use --skip-completeness to bypass "
                "this check if you are on a machine without the data.",
                universe_path,
            )
            return 1

        logger.info("loading universe from %s …", universe_path)
        universe = pd.read_parquet(universe_path)
        query_ids = sorted(universe["query_id"].astype(str).unique())
        logger.info("checking completeness across %d queries …", len(query_ids))
        try:
            primary.require_complete(universe, model, query_ids)
            logger.info("completeness check PASSED — all pairs present")
        except AssertionError as exc:
            logger.error("completeness check FAILED: %s", exc)
            return 1

    completeness_checked = not args.skip_completeness
    # Write manifest
    manifest = run_manifest(
        cfg.fingerprint(),
        owner="chaitanya",   # merge is Chaitanya's responsibility (ROADMAP H4/H5)
        extra={
            "artifact": str(args.primary),
            "total_rows": total,
            "expected_rows": EXPECTED_ROWS,
            "row_count_ok": total == EXPECTED_ROWS,
            "completeness_checked": completeness_checked,
            "stage2_model": model,
            "shards": shard_records,
            "config_fingerprint": cfg.fingerprint(),
            "note": (
                "Primary Stage-2 cache; merged from three disjoint shards. "
                "Completeness verified against the frozen universe."
                if completeness_checked else
                "Primary Stage-2 cache; merged from three disjoint shards. "
                "Completeness check was SKIPPED (--skip-completeness). "
                "Row count matches expected but per-query coverage was not verified."
            ),
        },
    )
    out_path = write_manifest(args.out, manifest)
    logger.info("manifest written to %s", out_path)

    if total != EXPECTED_ROWS:
        logger.error("aborting: row count is wrong (%d != %d)", total, EXPECTED_ROWS)
        return 1

    logger.info("merge complete — primary cache has %d rows, manifest at %s",
                total, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "EXPECTED_ROWS", "OWNERS"]
