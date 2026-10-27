"""
run_dpr.py - The DPR Stage-1 run: top-M for every query, as a schema-valid run table.

    python -m experiments_shubham.run_dpr \
        --index gdrive:IRE_PROJECT/dpr_index_cosdpr_distil     # or a local directory

Writes runs/dpr.parquet (system=dpr) and runs/dpr_manifest.json. The depth
defaults to the frozen universe depth M and may never be smaller, otherwise the
universe built from this run would be shallower than the config claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd

from experiments_shubham.dpr import CosDPREncoder, exact_search, open_shards
from shared.config import load_config
from shared.provenance import run_manifest, write_manifest
from shared.schema import save

logger = logging.getLogger(__name__)

SYSTEM = "dpr"


def read_queries(path: str | Path) -> dict[str, str]:
    queries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        qid, text = line.split("\t", 1)
        queries[qid] = text
    return queries


def checkpoint_label(encoder: CosDPREncoder, meta: dict) -> str:
    """Exact model weights plus a hash of the index settings."""
    meta_hash = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
    return f"{encoder.model_name}@{encoder.revision[:12]}|index:fp16-exact-{meta_hash}"


def run(index: str, queries_path: str, out_path: str, config_path: str,
        depth: int | None, device: str) -> Path:
    cfg = load_config(config_path)
    depth = depth or cfg.universe_depth_m
    if depth < cfg.universe_depth_m:
        raise ValueError(f"depth {depth} is below the universe depth M={cfg.universe_depth_m}")

    queries = read_queries(queries_path)
    shards = open_shards(index)
    encoder = CosDPREncoder(shards.meta["model_name"], device=device)
    qids = list(queries)

    start = time.perf_counter()
    q_emb = encoder.encode([queries[q] for q in qids])
    hits = exact_search(q_emb, shards, k=depth)
    elapsed = time.perf_counter() - start

    checkpoint = checkpoint_label(encoder, shards.meta)
    rows = [(qid, doc, SYSTEM, rank, score, checkpoint)
            for qid, ranked in zip(qids, hits)
            for rank, (doc, score) in enumerate(ranked, start=1)]
    df = pd.DataFrame(rows, columns=["query_id", "doc_id", "system", "rank", "score", "checkpoint"])
    out = save(df, out_path, "run")

    manifest = run_manifest(cfg.fingerprint(), owner="shubham", extra={
        "artifact": str(out), "system": SYSTEM, "checkpoint": checkpoint,
        "index_source": shards.description, "index_meta": shards.meta,
        "n_shards": len(shards.numbers), "n_queries": len(qids), "depth": depth,
        "queries_file": queries_path, "search": "exact inner product, ties by doc_id",
        "seconds": round(elapsed, 1),
    })
    write_manifest(Path(out).with_name(f"{SYSTEM}_manifest.json"), manifest)
    logger.info("wrote %d rows (%d queries x %d) to %s in %.0fs",
                len(df), len(qids), depth, out, elapsed)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="DPR Stage-1 run to depth M")
    parser.add_argument("--index", required=True,
                        help="local index directory or rclone remote path")
    parser.add_argument("--queries", default="data/queries.tsv")
    parser.add_argument("--out", default="runs/dpr.parquet")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--depth", type=int, default=None, help="default: universe depth M")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.index, args.queries, args.out, args.config, args.depth, args.device)


if __name__ == "__main__":
    main()
