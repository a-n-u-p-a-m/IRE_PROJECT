"""
run_stage1.py - A Stage-1 run: top-M for every query, as a schema-valid run table.

    python -m experiments_anupam.run_stage1 --system bm25

Writes runs/<system>.parquet and runs/<system>_manifest.json. Mirrors
`experiments_shubham.run_dpr` so the two run tables are interchangeable
downstream.

Two refusals, both about depth. A universe source retrieved shallower than M
would make the frozen universe shallower than the config claims, and a Stage-1
system retrieved shallower than the largest K_c cannot fill its own candidate
set. Both are caught here rather than discovered when the universe is built.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from experiments_anupam.retrievers import PrecomputedRun, PyseriniBM25, Retriever
from shared.config import FrozenConfig, load_config
from shared.provenance import run_manifest, write_manifest
from shared.schema import load, save

logger = logging.getLogger(__name__)


def read_queries(path: str | Path) -> dict[str, str]:
    from experiments_anupam.data_prep import load_queries

    return load_queries(path)


def build_retriever(system: str, args) -> Retriever:
    """
    Construct the named system.

    `--from-run` re-exports an existing run table under this system name,
    carrying its checkpoint through. That is the path for a baseline run that
    was downloaded rather than reproduced.
    """
    if args.from_run:
        return PrecomputedRun(load(args.from_run, "run"), name=system)
    if system == "bm25":
        return PyseriniBM25(
            index=args.index or "msmarco-v1-passage", k1=args.k1, b=args.b,
            threads=args.threads,
        )
    raise SystemExit(
        f"unknown system {system!r}. This module owns BM25; the dense run lives "
        "in experiments_shubham.run_dpr. Any other system needs --from-run."
    )


def check_depth(cfg: FrozenConfig, system: str, depth: int) -> None:
    if system in cfg.universe_sources and depth < cfg.universe_depth_m:
        raise SystemExit(
            f"{system} is a universe source, so it must be retrieved to at least "
            f"M={cfg.universe_depth_m}; got {depth}. A shallower run would make "
            "the frozen universe shallower than the config claims."
        )
    if system in cfg.stage1_systems and depth < max(cfg.candidate_depths):
        raise SystemExit(
            f"{system} is used as Stage 1, so depth must be at least the largest "
            f"candidate depth K_c={max(cfg.candidate_depths)}; got {depth}."
        )


def run(
    system: str,
    queries_path: str,
    out_path: str,
    config_path: str,
    depth: int | None,
    args,
) -> Path:
    cfg = load_config(config_path)
    depth = depth or cfg.universe_depth_m
    check_depth(cfg, system, depth)

    queries = read_queries(queries_path)
    if args.limit_queries:
        queries = dict(list(queries.items())[: args.limit_queries])
        logger.warning(
            "running on %d queries only - a smoke test, not a reportable run",
            len(queries),
        )

    retriever = build_retriever(system, args)

    start = time.perf_counter()
    df = retriever.run(queries, depth)
    elapsed = time.perf_counter() - start

    out = save(df, out_path, "run")

    per_query = df.groupby("query_id", observed=True).size()
    shallow = per_query[per_query < depth]
    if not shallow.empty:
        logger.warning(
            "%d queries returned fewer than %d documents (min %d). Expected for "
            "short queries on a sparse index; recorded in the manifest.",
            len(shallow), depth, int(per_query.min()),
        )

    write_manifest(
        Path(out).with_name(f"{system}_manifest.json"),
        run_manifest(cfg.fingerprint(), owner="anupam", extra={
            "artifact": str(out),
            "system": system,
            "checkpoint": retriever.checkpoint,
            "n_queries": int(per_query.size),
            "n_rows": int(len(df)),
            "depth": depth,
            "depth_min": int(per_query.min()),
            "depth_median": int(per_query.median()),
            "queries_file": queries_path,
            "from_run": args.from_run,
            "seconds": round(elapsed, 1),
        }),
    )
    logger.info(
        "wrote %d rows (%d queries x up to %d) to %s in %.0fs",
        len(df), per_query.size, depth, out, elapsed,
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Stage-1 run to depth M")
    parser.add_argument("--system", default="bm25")
    parser.add_argument("--queries", default="data/queries.tsv")
    parser.add_argument("--out", default=None, help="default: runs/<system>.parquet")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--depth", type=int, default=None, help="default: universe depth M")
    parser.add_argument("--index", default=None, help="override the prebuilt index name")
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--from-run", default=None,
                        help="re-export an existing run table instead of retrieving")
    parser.add_argument("--limit-queries", type=int, default=None,
                        help="smoke test only; never use for a reported run")
    args = parser.parse_args()
    out = args.out or f"runs/{args.system}.parquet"
    run(args.system, args.queries, out, args.config, args.depth, args)


if __name__ == "__main__":
    main()
