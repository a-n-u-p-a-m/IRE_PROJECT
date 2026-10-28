"""
shard_plan.py - Shard planning for the Week-3 cache fill (H4).

Divides the required (query, doc) pairs into three shards — one per team
member — such that each shard is a contiguous set of queries and shards are
roughly equal in total pair count.

The plan is deterministic: the same universe and the same shard_count always
produce the same split.  Output is written to a JSON file for reference.

Usage::

    python -m experiments_chaitanya.shard_plan \
        --universe data/universe.parquet \
        --queries data/queries.tsv \
        --output experiments_chaitanya/shard_plan.json

Or import and call directly::

    from experiments_chaitanya.shard_plan import build_shard_plan
    plan = build_shard_plan(universe, query_ids, n_shards=3)
    print(plan)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SHARD_OWNERS = ["anupam", "shubham", "chaitanya"]


def build_shard_plan(
    universe: pd.DataFrame,
    query_ids: list[str] | None = None,
    n_shards: int = 3,
    worlds_ab_first: bool = True,
) -> dict:
    """
    Divide query_ids into n_shards of approximately equal pair count.

    Parameters
    ----------
    universe:
        The frozen universe table (query_id, doc_id, sources).
    query_ids:
        The query IDs to shard.  If None, uses all queries in the universe.
    n_shards:
        Number of shards (default 3, one per team member).
    worlds_ab_first:
        If True, note that worlds A/B (C_q pairs) should be filled first,
        then worlds C/D.

    Returns
    -------
    dict
        Shard plan with structure::

            {
              "total_queries": N,
              "total_pairs": M,
              "worlds_ab_first": True/False,
              "shards": [
                {
                  "shard_id": 0,
                  "owner": "anupam",
                  "query_ids": [...],
                  "n_queries": K,
                  "n_pairs": P,
                },
                ...
              ]
            }
    """
    if query_ids is None:
        query_ids = universe["query_id"].astype(str).unique().tolist()

    # Count pairs per query
    pair_counts: dict[str, int] = {}
    for qid in query_ids:
        qid = str(qid)
        n = int((universe["query_id"].astype(str) == qid).sum())
        pair_counts[qid] = n

    # Sort queries by query_id for determinism
    sorted_qids = sorted(query_ids, key=str)
    total_pairs = sum(pair_counts[q] for q in sorted_qids)
    target_per_shard = total_pairs / n_shards

    # Greedy bin-packing: assign queries to shards until each shard is full
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    shard_pair_counts = [0] * n_shards

    for qid in sorted_qids:
        # Assign to the shard with the smallest current pair count
        min_shard = int(np.argmin(shard_pair_counts))
        shards[min_shard].append(qid)
        shard_pair_counts[min_shard] += pair_counts[qid]

    plan: dict = {
        "total_queries": len(sorted_qids),
        "total_pairs": total_pairs,
        "worlds_ab_first": worlds_ab_first,
        "note": (
            "Worlds A and B score only C_q per query (k_c pairs each). "
            "Worlds C and D require U_q or reference-subset pairs. "
            "Fill A/B first, then C and D after merging."
        ),
        "shards": [],
    }

    owners = SHARD_OWNERS[:n_shards]
    for i, (shard_qids, n_pairs) in enumerate(zip(shards, shard_pair_counts)):
        plan["shards"].append(
            {
                "shard_id": i,
                "owner": owners[i] if i < len(owners) else f"member_{i}",
                "query_ids": sorted(shard_qids),
                "n_queries": len(shard_qids),
                "n_pairs": n_pairs,
            }
        )

    # Log the plan
    logger.info(
        "shard plan: %d queries, %d total pairs → %d shards",
        len(sorted_qids), total_pairs, n_shards,
    )
    for s in plan["shards"]:
        logger.info(
            "  shard %d (%s): %d queries, %d pairs",
            s["shard_id"], s["owner"], s["n_queries"], s["n_pairs"],
        )

    return plan


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build the Week-3 cache fill shard plan."
    )
    parser.add_argument("--universe", default="data/universe.parquet",
                        help="Path to frozen universe parquet")
    parser.add_argument("--queries", default="data/queries.tsv",
                        help="Path to queries TSV (used to get query IDs)")
    parser.add_argument("--n-shards", type=int, default=3)
    parser.add_argument("--output", default="experiments_chaitanya/shard_plan.json")
    args = parser.parse_args()

    universe = pd.read_parquet(args.universe)

    # Get query IDs from the queries file
    import csv
    query_ids = []
    with open(args.queries, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row:
                query_ids.append(row[0].strip())

    plan = build_shard_plan(universe, query_ids, n_shards=args.n_shards)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(plan, indent=2))
    logger.info("shard plan written to %s", args.output)

    # Print summary
    print(f"\nShard plan: {plan['total_queries']} queries, "
          f"{plan['total_pairs']} total pairs")
    for s in plan["shards"]:
        print(f"  Shard {s['shard_id']} ({s['owner']}): "
              f"{s['n_queries']} queries, {s['n_pairs']} pairs")


if __name__ == "__main__":
    main()


__all__ = ["build_shard_plan", "SHARD_OWNERS"]
