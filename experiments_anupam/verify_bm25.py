"""
verify_bm25.py - Is the BM25 run actually BM25?

    python -m experiments_anupam.verify_bm25

Every other check in this repo verifies that a run is *well formed*: the schema
validates, the depth reaches M, the candidate sets sit inside U_q. A run built
against the wrong index, or with mangled query text, passes all of them. This
script is the one that checks the run is *right*, by reproducing the published
effectiveness of BM25 on TREC-DL 2019 and 2020 and refusing if it misses.

    k1=0.9, b=0.4 on msmarco-v1-passage
        DL19   nDCG@10 0.5058   R@1000 0.7501
        DL20   nDCG@10 0.4796   R@1000 0.7863
    https://castorini.github.io/pyserini/2cr/msmarco-v1-passage.html

**This module deliberately does not use `shared.utility`.** The project's
frozen gain map differs from trec_eval's nDCG in two independent ways: it
scores TREC-DL grade 1 ("Related") as 0, and its {2: 3, 3: 7} is an
exponential 2^g - 1 scale. Both are right for this project and wrong for
reproducing a published number. The official measures are implemented here
instead, as trec_eval actually defines them:

    nDCG@10     graded, gain = rel (LINEAR), discount = log2(rank + 1)
    R@1000      binary, relevant = grade >= 2

The grade >= 2 binarisation is the TREC Deep Learning track's own convention
for the set-based measures, and it is why nDCG and recall disagree about what
"relevant" means. Mixing the two conventions is the easiest way to get a number
that looks plausible and matches nothing.

Exits non-zero when a measure is outside tolerance, so it can gate a handoff.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from shared.schema import load

logger = logging.getLogger(__name__)

# The per-year judged query sets. Taken from the NIST qrels rather than the
# query files: a query with no judgments is not evaluated, and the published
# numbers are over the judged sets (43 and 54), not the 200 released per year.
QRELS_URLS = {
    "dl19": "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
    "dl20": "https://trec.nist.gov/data/deep/2020qrels-pass.txt",
}

EXPECTED = {
    "dl19": {"n_queries": 43, "ndcg@10": 0.5058, "recall@1000": 0.7501},
    "dl20": {"n_queries": 54, "ndcg@10": 0.4796, "recall@1000": 0.7863},
}

RECALL_THRESHOLD = 2          # TREC DL: grades 2 and 3 are relevant for recall


# ──────────────────────────────────────────────────────────────────────
# The official measures
# ──────────────────────────────────────────────────────────────────────

def dcg(grades: list[int]) -> float:
    """
    sum g / log2(i + 1), i one-indexed.

    LINEAR gain, which is what `trec_eval`'s `ndcg_cut` computes and therefore
    what produced the published figures. Several toolkits - and this project's
    own `shared.utility` gain map - use the exponential 2^g - 1 instead. The
    two disagree by several points on the same run, so the choice is not
    cosmetic: it was established here by probing `pytrec_eval` directly, and
    `test_verify_bm25.py` holds it in place against the real trec_eval code.
    """
    return float(sum(g / np.log2(i + 1.0)
                     for i, g in enumerate(grades, start=1)))


def ndcg_at_k(ranked: list[str], grade_of: dict[str, int], k: int) -> float:
    """
    Graded nDCG@k. Unjudged documents contribute 0, which is trec_eval's
    behaviour and the reason a sparsely judged pool depresses this measure.

    The ideal ranking comes from every grade the query has, not only the ones
    this run retrieved - otherwise a run that found nothing would score 1.0.
    """
    actual = dcg([int(grade_of.get(d, 0)) for d in ranked[:k]])
    ideal = dcg(sorted((int(g) for g in grade_of.values()), reverse=True)[:k])
    return (actual / ideal) if ideal > 0 else 0.0


def recall_at_k(ranked: list[str], grade_of: dict[str, int], k: int,
                threshold: int = RECALL_THRESHOLD) -> float:
    """Binary recall@k over grades >= threshold."""
    relevant = {d for d, g in grade_of.items() if int(g) >= threshold}
    if not relevant:
        return float("nan")
    return len(relevant & set(ranked[:k])) / len(relevant)


# ──────────────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────────────

def _get(url: str) -> bytes:
    # trec.nist.gov returns 403 for urllib's default User-Agent
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"})) as resp:
        return resp.read()


def year_query_ids(cache_dir: str | Path | None = None) -> dict[str, set[str]]:
    """
    {year: judged query ids}, from the NIST per-year qrels.

    The project's merged `qrels.parquet` cannot answer this - it is both years
    in one table - and the published numbers are per year, so the split has to
    come from somewhere. Cached on disk when a directory is given, because this
    check gets run repeatedly and the files never change.
    """
    out: dict[str, set[str]] = {}
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    for year, url in QRELS_URLS.items():
        local = cache / f"{year}qrels-pass.txt" if cache else None
        if local and local.exists():
            text = local.read_text(encoding="utf-8")
        else:
            logger.info("fetching %s", url)
            text = _get(url).decode("utf-8")
            if local:
                local.write_text(text, encoding="utf-8")
        ids = {line.split()[0] for line in text.splitlines() if line.strip()}
        out[year] = ids
        logger.info("%s: %d judged queries", year, len(ids))
    return out


def grades_by_query(qrels: pd.DataFrame) -> dict[str, dict[str, int]]:
    """{query_id: {doc_id: grade}} over every judgment."""
    out: dict[str, dict[str, int]] = {}
    for qid, grp in qrels.groupby("query_id", observed=True):
        out[str(qid)] = dict(
            zip(grp["doc_id"].astype(str), grp["relevance"].astype(int))
        )
    return out


def ranked_by_query(run: pd.DataFrame, system: str) -> dict[str, list[str]]:
    r = run[run["system"] == system]
    if r.empty:
        raise SystemExit(f"run contains no rows for system {system!r}")
    out: dict[str, list[str]] = {}
    for qid, grp in r.groupby("query_id", observed=True):
        out[str(qid)] = list(grp.sort_values("rank")["doc_id"].astype(str))
    return out


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate(
    ranked: dict[str, list[str]],
    grades: dict[str, dict[str, int]],
    query_ids: set[str],
    k_ndcg: int = 10,
    k_recall: int = 1000,
) -> dict:
    """Mean nDCG@k and R@k over the judged queries in `query_ids`."""
    evaluated = sorted(query_ids & set(grades))
    missing = sorted(q for q in evaluated if q not in ranked)
    if missing:
        logger.warning(
            "%d judged queries absent from the run (e.g. %s); scored as 0",
            len(missing), missing[:3],
        )

    nd, rc = [], []
    for q in evaluated:
        docs = ranked.get(q, [])
        nd.append(ndcg_at_k(docs, grades[q], k_ndcg))
        rc.append(recall_at_k(docs, grades[q], k_recall))

    return {
        "n_queries": len(evaluated),
        "n_missing_from_run": len(missing),
        f"ndcg@{k_ndcg}": float(np.mean(nd)) if nd else float("nan"),
        f"recall@{k_recall}": float(np.nanmean(rc)) if rc else float("nan"),
    }


def run(
    run_path: str,
    qrels_path: str,
    system: str = "bm25",
    tolerance: float = 0.01,
    cache_dir: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    run_df = load(run_path, "run")
    qrels = load(qrels_path, "qrels")

    checkpoint = sorted(set(run_df[run_df["system"] == system]["checkpoint"]
                            .dropna().astype(str)))
    logger.info("checking %s (%s)", system, ", ".join(checkpoint) or "no checkpoint")

    ranked = ranked_by_query(run_df, system)
    grades = grades_by_query(qrels)
    years = year_query_ids(cache_dir)

    rows, failures = [], []
    for year in sorted(EXPECTED):
        got = evaluate(ranked, grades, years[year])
        exp = EXPECTED[year]

        if got["n_queries"] != exp["n_queries"]:
            failures.append(
                f"{year}: evaluated {got['n_queries']} queries, expected "
                f"{exp['n_queries']}. The run or the qrels covers the wrong set."
            )

        for measure in ("ndcg@10", "recall@1000"):
            delta = got[measure] - exp[measure]
            rows.append({
                "year": year, "measure": measure,
                "observed": round(got[measure], 4),
                "published": exp[measure],
                "delta": round(delta, 4),
                "within_tolerance": bool(abs(delta) <= tolerance),
                "n_queries": got["n_queries"],
            })
            if abs(delta) > tolerance:
                failures.append(
                    f"{year} {measure}: {got[measure]:.4f} against a published "
                    f"{exp[measure]:.4f} (delta {delta:+.4f}, tolerance "
                    f"{tolerance}). A gap this size is a wrong index, wrong "
                    "query text or wrong parameters, not noise."
                )

    return pd.DataFrame(rows), failures


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Reproduce published BM25 effectiveness, or refuse"
    )
    parser.add_argument("--run", default="runs/bm25.parquet")
    parser.add_argument("--qrels", default="data/qrels.parquet")
    parser.add_argument("--system", default="bm25")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--cache-dir", default="data/.trec_cache",
                        help="where to keep the per-year qrels (gitignored)")
    parser.add_argument("--out", default=None, help="optional parquet of the table")
    args = parser.parse_args()

    table, failures = run(args.run, args.qrels, args.system,
                          args.tolerance, args.cache_dir)
    print()
    print(table.to_string(index=False))
    print()

    if args.out:
        table.to_parquet(args.out, index=False)

    if failures:
        print("BM25 verification FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"{args.system} reproduces published effectiveness within "
          f"{args.tolerance}. The run is sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
