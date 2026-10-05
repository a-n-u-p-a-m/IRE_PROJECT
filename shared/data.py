"""
data.py — Corpus, query, and qrel loading for MS MARCO + TREC-DL.

Handles downloading, caching, and providing a clean Python API for:
  - MS MARCO Passage v1 corpus  (8.8M passages)
  - TREC-DL 2019 queries  (43 queries with graded NIST judgments)
  - TREC-DL 2020 queries  (54 queries with graded NIST judgments)
  - MS MARCO dev-small     (6,980 queries, sparse binary labels)
  - TREC qrels             (graded relevance labels: 0/1/2/3)

All functions return plain Python dicts or DataFrames — no retriever-specific
objects.  Retriever wrappers (bm25.py, dpr.py, etc.) consume these.

Usage:
    from shared.data import load_corpus, load_queries, load_qrels

    corpus  = load_corpus()            # dict[str, str]  pid → passage text
    queries = load_queries("dl19")     # dict[str, str]  qid → query text
    qrels   = load_qrels("dl19")       # dict[str, dict[str, int]]
"""

from __future__ import annotations

import csv
import gzip
import logging
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Literal
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Paths and URLs
# ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.environ.get("BENCHMARK_DATA_DIR", "data"))

# MS MARCO Passage v1
_MSMARCO_COLLECTION_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
    "collection.tar.gz"
)
_MSMARCO_QUERIES_DEV_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
    "queries.dev.small.tsv"
)
_MSMARCO_QRELS_DEV_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
    "qrels.dev.small.tsv"
)

# TREC-DL 2019
_DL19_QUERIES_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
    "msmarco-test2019-queries.tsv.gz"
)
_DL19_QRELS_URL = (
    "https://trec.nist.gov/data/deep/2019qrels-pass.txt"
)

# TREC-DL 2020
_DL20_QUERIES_URL = (
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
    "msmarco-test2020-queries.tsv.gz"
)
_DL20_QRELS_URL = (
    "https://trec.nist.gov/data/deep/2020qrels-pass.txt"
)

QuerySet = Literal["dl19", "dl20", "dev_small"]

# ──────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> Path:
    """Download *url* to *dest* if *dest* does not already exist."""
    if dest.exists():
        logger.info("Already cached: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)
    # trec.nist.gov returns 403 for urllib's default User-Agent
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    # Write to a temp file first so an interrupted download isn't cached
    tmp = dest.with_name(dest.name + ".part")
    with urlopen(req) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.rename(dest)
    return dest


def _ensure_collection_tsv() -> Path:
    """Download and extract the MS MARCO passage collection to a .tsv."""
    tsv_path = DATA_DIR / "msmarco" / "collection.tsv"
    if tsv_path.exists():
        return tsv_path

    tar_path = DATA_DIR / "msmarco" / "collection.tar.gz"
    _download(_MSMARCO_COLLECTION_URL, tar_path)

    logger.info("Extracting %s …", tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=DATA_DIR / "msmarco", filter="data")

    if not tsv_path.exists():
        # The tar may contain "collection.tsv" nested — look for it
        candidates = list((DATA_DIR / "msmarco").rglob("collection.tsv"))
        if candidates:
            candidates[0].rename(tsv_path)
        else:
            raise FileNotFoundError(
                f"collection.tsv not found after extracting {tar_path}"
            )
    return tsv_path


# ──────────────────────────────────────────────────────────────────────
# Corpus
# ──────────────────────────────────────────────────────────────────────

def load_corpus(max_passages: int | None = None) -> dict[str, str]:
    """
    Load MS MARCO Passage v1 corpus.

    Returns
    -------
    dict[str, str]
        Mapping from passage ID (string) to passage text.

    Parameters
    ----------
    max_passages : int, optional
        Load only the first N passages (for development / debugging).
    """
    tsv_path = _ensure_collection_tsv()
    logger.info("Loading corpus from %s …", tsv_path)

    corpus: dict[str, str] = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if max_passages is not None and i >= max_passages:
                break
            pid, text = row[0], row[1]
            corpus[pid] = text

    logger.info("Loaded %d passages", len(corpus))
    return corpus


def load_passages(doc_ids) -> dict[str, str]:
    """
    Load only the given passages, streaming the collection instead of holding
    all 8.8M in memory.  Returns dict[pid → text]; unknown ids are omitted.
    """
    wanted = {str(d) for d in doc_ids}
    if not wanted:
        return {}

    passages: dict[str, str] = {}
    with open(_ensure_collection_tsv(), encoding="utf-8") as f:
        for line in f:
            pid, text = line.rstrip("\n").split("\t", 1)
            if pid in wanted:
                passages[pid] = text
                if len(passages) == len(wanted):
                    break

    logger.info("Loaded %d of %d requested passages", len(passages), len(wanted))
    return passages


# ──────────────────────────────────────────────────────────────────────
# Queries
# ──────────────────────────────────────────────────────────────────────

def _load_tsv_queries(path: Path) -> dict[str, str]:
    """Read a two-column (qid, text) TSV."""
    queries: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            qid, text = row[0], row[1]
            queries[qid] = text
    return queries


def _load_gzipped_tsv_queries(path: Path) -> dict[str, str]:
    """Read a gzipped two-column (qid, text) TSV."""
    queries: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            qid, text = row[0], row[1]
            queries[qid] = text
    return queries


def load_queries(
    query_set: QuerySet = "dl19",
    judged_only: bool = True,
) -> dict[str, str]:
    """
    Load queries for a given evaluation set.

    Parameters
    ----------
    query_set : {"dl19", "dl20", "dev_small"}
    judged_only : bool
        The TREC-DL query files contain 200 queries each, but only 43 (DL19)
        / 54 (DL20) have NIST judgments.  If True (default), keep only the
        judged queries so every member evaluates on the same 43/54.

    Returns
    -------
    dict[str, str]
        Mapping from query ID (string) to query text.
    """
    if query_set == "dl19":
        gz_path = DATA_DIR / "trec_dl" / "msmarco-test2019-queries.tsv.gz"
        _download(_DL19_QUERIES_URL, gz_path)
        queries = _load_gzipped_tsv_queries(gz_path)

    elif query_set == "dl20":
        gz_path = DATA_DIR / "trec_dl" / "msmarco-test2020-queries.tsv.gz"
        _download(_DL20_QUERIES_URL, gz_path)
        queries = _load_gzipped_tsv_queries(gz_path)

    elif query_set == "dev_small":
        tsv_path = DATA_DIR / "msmarco" / "queries.dev.small.tsv"
        _download(_MSMARCO_QUERIES_DEV_URL, tsv_path)
        queries = _load_tsv_queries(tsv_path)

    else:
        raise ValueError(f"Unknown query set: {query_set!r}")

    if judged_only and query_set in ("dl19", "dl20"):
        qrels = load_qrels(query_set)
        queries = {qid: text for qid, text in queries.items() if qid in qrels}

    logger.info("Loaded %d queries for %s", len(queries), query_set)
    return queries


# ──────────────────────────────────────────────────────────────────────
# Qrels  (relevance judgments)
# ──────────────────────────────────────────────────────────────────────

def load_qrels(
    query_set: QuerySet = "dl19",
) -> dict[str, dict[str, int]]:
    """
    Load relevance judgments.

    Parameters
    ----------
    query_set : {"dl19", "dl20", "dev_small"}

    Returns
    -------
    dict[str, dict[str, int]]
        qrels[query_id][doc_id] = relevance_label.
        - TREC-DL 2019/2020: graded 0–3.
        - dev-small: binary 0/1.
    """
    if query_set == "dl19":
        qrel_path = DATA_DIR / "trec_dl" / "2019qrels-pass.txt"
        _download(_DL19_QRELS_URL, qrel_path)
        return _parse_trec_qrels(qrel_path)

    elif query_set == "dl20":
        qrel_path = DATA_DIR / "trec_dl" / "2020qrels-pass.txt"
        _download(_DL20_QRELS_URL, qrel_path)
        return _parse_trec_qrels(qrel_path)

    elif query_set == "dev_small":
        qrel_path = DATA_DIR / "msmarco" / "qrels.dev.small.tsv"
        _download(_MSMARCO_QRELS_DEV_URL, qrel_path)
        return _parse_msmarco_qrels(qrel_path)

    else:
        raise ValueError(f"Unknown query set: {query_set!r}")


def _parse_trec_qrels(path: Path) -> dict[str, dict[str, int]]:
    """
    Parse TREC-format qrels: qid  0  docid  relevance
    (four whitespace-separated columns, second column is always 0).
    """
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            qrels[qid][docid] = rel

    logger.info("Loaded qrels from %s: %d queries", path, len(qrels))
    return dict(qrels)


def _parse_msmarco_qrels(path: Path) -> dict[str, dict[str, int]]:
    """
    Parse MS MARCO qrels: qid  0  docid  1
    (TSV, all relevant = 1).
    """
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            qid, docid, rel = row[0], row[2], int(row[3])
            qrels[qid][docid] = rel

    logger.info("Loaded qrels from %s: %d queries", path, len(qrels))
    return dict(qrels)


# ──────────────────────────────────────────────────────────────────────
# Convenience: attach relevance labels to a score matrix
# ──────────────────────────────────────────────────────────────────────

def attach_relevance_labels(
    score_df: "pd.DataFrame",
    qrels: dict[str, dict[str, int]],
) -> "pd.DataFrame":
    """
    Fill the `relevance_label` column of a score matrix using qrels.

    Unjudged (query_id, doc_id) pairs get pd.NA.
    """
    import pandas as pd

    def _lookup(row: pd.Series) -> int | None:
        qid, did = row["query_id"], row["doc_id"]
        return qrels.get(qid, {}).get(did, None)

    score_df = score_df.copy()
    labels = score_df.apply(_lookup, axis=1)
    score_df["relevance_label"] = pd.array(labels, dtype="Int64")
    return score_df


# ──────────────────────────────────────────────────────────────────────
# Query metadata (for Chaitanya's query-type slicing)
# ──────────────────────────────────────────────────────────────────────

def load_query_types(
    path: str | Path = "query_annotations/trec_dl_query_types.csv",
) -> dict[str, str]:
    """
    Load Chaitanya's hand-annotated query-type CSV.

    Expected format: query_id, query_type
    query_type ∈ {factoid, entity, comparison, procedural, ...}

    Returns dict[query_id → query_type].
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Query type annotations not found at %s", path)
        return {}

    qtypes: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qtypes[row["query_id"]] = row["query_type"]
    logger.info("Loaded %d query type annotations", len(qtypes))
    return qtypes


# ──────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────

def describe_dataset(query_set: QuerySet = "dl19") -> dict[str, int]:
    """Load and print basic stats for a query set (no corpus load)."""
    queries = load_queries(query_set)
    qrels = load_qrels(query_set)

    n_judged_docs = sum(len(v) for v in qrels.values())
    rel_distribution: dict[int, int] = defaultdict(int)
    for docs in qrels.values():
        for rel in docs.values():
            rel_distribution[rel] += 1

    stats = {
        "query_set": query_set,
        "n_queries": len(queries),
        "n_queries_with_qrels": len(qrels),
        "n_judged_docs": n_judged_docs,
        "relevance_distribution": dict(sorted(rel_distribution.items())),
    }

    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TREC-DL 2019 ===")
    describe_dataset("dl19")
    print("\n=== TREC-DL 2020 ===")
    describe_dataset("dl20")
