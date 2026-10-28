"""
data_prep.py - Queries, qrels and the by-query split.

    python -m experiments_anupam.data_prep

Turns the raw files that `experiments_shubham.fetch_trec_dl` downloads into the
two artefacts the rest of the project reads:

    data/qrels.parquet      schema-valid, unjudged rows carry relevance = NA
    data/query_split.json   train / val / test, split BY QUERY

The split is a CONTRACT, not a convenience. Chaitanya fits predictors on
training queries only and Shubham's analysis splits the same way, so a change
to it after either has started invalidates every fidelity number. It is
deterministic from (sorted query ids, fractions, seed) and the materialised
file is committed anyway: recomputing it should give the same answer, and
committing it turns "should be identical" into "is identical".

Splitting by query rather than by query-document row is the whole point. Two
documents from one query are not independent observations; putting one in train
and one in test leaks the query's difficulty across the boundary.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from shared.config import load_config
from shared.provenance import run_manifest, write_manifest
from shared.qrels import read_trec_qrels
from shared.schema import save

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def load_queries(path: str | Path) -> dict[str, str]:
    """
    Read `query_id<TAB>query_text`.

    Duplicate ids are an error rather than a last-one-wins overwrite: a
    duplicated query would silently appear in two splits.
    """
    queries: dict[str, str] = {}
    for lineno, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        if "\t" not in line:
            raise ValueError(f"{path}:{lineno}: expected query_id<TAB>query_text")
        qid, text = line.split("\t", 1)
        qid, text = qid.strip(), text.strip()
        if qid in queries:
            raise ValueError(f"{path}:{lineno}: duplicate query id {qid!r}")
        if not text:
            raise ValueError(f"{path}:{lineno}: empty query text for {qid!r}")
        queries[qid] = text
    if not queries:
        raise ValueError(f"{path}: no queries read")
    return queries


def build_qrels(qrels_txt: str | Path, out_path: str | Path) -> Path:
    """Convert TREC qrels to the validated parquet the project reads."""
    df = read_trec_qrels(qrels_txt)
    out = save(df, out_path, "qrels")
    logger.info("wrote %d judgments to %s", len(df), out)
    return out


def check_gain_map_covers(qrels_path: str | Path, gain_map: dict[int, float]) -> None:
    """
    Every grade present in the judgments must appear in the frozen gain map.

    An unmapped grade would otherwise reach `utility.gain` at analysis time and
    raise there - deep inside a long run - instead of here, in a second.
    """
    from shared.schema import load

    grades = set(int(g) for g in load(qrels_path, "qrels")["relevance"].dropna().unique())
    missing = sorted(grades - set(gain_map))
    if missing:
        raise ValueError(
            f"relevance grades {missing} appear in {qrels_path} but not in the "
            f"frozen gain map {dict(gain_map)}. Add them to the config "
            "deliberately rather than letting them fall through to zero."
        )
    logger.info("gain map covers every observed grade %s", sorted(grades))


def split_queries(
    query_ids: list[str],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """
    Partition query ids into train / val / test.

    Deterministic: the ids are sorted before the seeded permutation, so the
    result does not depend on the order they arrived in. Sizes are allocated by
    largest remainder, so they sum to exactly len(query_ids) with no query
    dropped and none duplicated.
    """
    missing = set(SPLITS) - set(fractions)
    if missing:
        raise ValueError(f"split fractions missing {sorted(missing)}")
    total = sum(fractions[s] for s in SPLITS)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    ordered = sorted(set(query_ids))
    if len(ordered) != len(query_ids):
        raise ValueError("duplicate query ids passed to split_queries")
    n = len(ordered)

    exact = {s: fractions[s] * n for s in SPLITS}
    sizes = {s: int(exact[s]) for s in SPLITS}
    remainder = n - sum(sizes.values())
    # Largest fractional part first; ties by split name so it stays deterministic.
    for s in sorted(SPLITS, key=lambda s: (-(exact[s] - int(exact[s])), s))[:remainder]:
        sizes[s] += 1

    rng = np.random.default_rng(seed)
    shuffled = [ordered[int(i)] for i in rng.permutation(n)]

    out: dict[str, list[str]] = {}
    start = 0
    for s in SPLITS:
        out[s] = sorted(shuffled[start:start + sizes[s]])
        start += sizes[s]

    assert start == n
    return out


def validate_split(split: dict[str, list[str]], query_ids: list[str]) -> None:
    """Disjoint, complete, and no split left empty."""
    seen: set[str] = set()
    for s in SPLITS:
        ids = set(split[s])
        if len(ids) != len(split[s]):
            raise ValueError(f"split '{s}' contains a duplicate query id")
        overlap = seen & ids
        if overlap:
            raise ValueError(
                f"query ids appear in more than one split, e.g. {sorted(overlap)[:3]}. "
                "A query leaking across the boundary invalidates every held-out "
                "fidelity number."
            )
        seen |= ids
        if not ids:
            raise ValueError(f"split '{s}' is empty")
    if seen != set(query_ids):
        missing = sorted(set(query_ids) - seen)
        extra = sorted(seen - set(query_ids))
        raise ValueError(
            f"split does not cover the query set: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(extra)} unknown (e.g. {extra[:3]})"
        )


def write_split(path: str | Path, split: dict[str, list[str]], seed: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "n_queries": sum(len(split[s]) for s in SPLITS),
        "sizes": {s: len(split[s]) for s in SPLITS},
        "splits": {s: split[s] for s in SPLITS},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_split(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["splits"]


def split_of(split: dict[str, list[str]]) -> dict[str, str]:
    """{query_id: split name}, for joining onto a per-query table."""
    return {q: s for s in SPLITS for q in split[s]}


def run(
    queries_path: str,
    qrels_txt: str,
    qrels_out: str,
    split_out: str,
    config_path: str,
) -> dict[str, list[str]]:
    cfg = load_config(config_path)

    queries = load_queries(queries_path)
    logger.info("%d queries from %s", len(queries), queries_path)

    build_qrels(qrels_txt, qrels_out)
    check_gain_map_covers(qrels_out, cfg.gain_map)

    split = split_queries(list(queries), cfg.split_fractions, cfg.seed)
    validate_split(split, list(queries))
    write_split(split_out, split, cfg.seed)

    logger.info(
        "split by query: %s",
        ", ".join(f"{s}={len(split[s])}" for s in SPLITS),
    )

    write_manifest(
        Path(split_out).with_name("data_prep_manifest.json"),
        run_manifest(cfg.fingerprint(), owner="anupam", extra={
            "artifact": [qrels_out, split_out],
            "queries_file": queries_path,
            "qrels_source": qrels_txt,
            "n_queries": len(queries),
            "split_sizes": {s: len(split[s]) for s in SPLITS},
            "split_fractions": cfg.split_fractions,
            "split_seed": cfg.seed,
            "split_rule": "sorted query ids, seeded permutation, largest-remainder sizes",
            "gain_map": {str(k): v for k, v in cfg.gain_map.items()},
        }),
    )
    return split


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Queries, qrels and the by-query split")
    parser.add_argument("--queries", default="data/queries.tsv")
    parser.add_argument("--qrels-txt", default="data/qrels.txt")
    parser.add_argument("--qrels-out", default="data/qrels.parquet")
    parser.add_argument("--split-out", default="data/query_split.json")
    parser.add_argument("--config", default="config/base.yaml")
    args = parser.parse_args()
    run(args.queries, args.qrels_txt, args.qrels_out, args.split_out, args.config)


if __name__ == "__main__":
    main()
