"""
retrievers.py - Stage-1 retriever adapters.

Every first-stage system exposes the same interface, so the cross-paradigm
experiment changes only which object is constructed. Pyserini is imported
lazily inside the constructor, so this module costs nothing to import and the
tests run with no index, no JDK and no download.

Also here: `scores_from_run`, which spreads a retriever's scores over the whole
frozen universe. A universe document this retriever never returned still needs
an s1 value or it cannot be ordered at all, and the only defensible claim is
that it ranks below everything the retriever did return.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import pandas as pd

from shared.schema import validate_run

logger = logging.getLogger(__name__)


class Retriever(ABC):
    """A first-stage retriever producing scored, ranked candidates."""

    name: str
    checkpoint: str

    @abstractmethod
    def search(self, query_id: str, query_text: str, k: int) -> list[tuple[str, float]]:
        """[(doc_id, score)] sorted by score descending, length <= k."""

    def run(self, queries: Mapping[str, str], depth: int) -> pd.DataFrame:
        """Produce a schema-valid `run` table over a query set."""
        rows = []
        for i, (qid, qtext) in enumerate(queries.items(), start=1):
            if i == 1 or i % 25 == 0:
                logger.info("%s: query %d/%d", self.name, i, len(queries))
            for rank, (doc_id, score) in enumerate(
                self.search(qid, qtext, depth), start=1
            ):
                rows.append(
                    {
                        "query_id": str(qid),
                        "doc_id": str(doc_id),
                        "system": self.name,
                        "rank": rank,
                        "score": float(score),
                        "checkpoint": self.checkpoint,
                    }
                )
        return validate_run(pd.DataFrame(rows))


class PyseriniBM25(Retriever):
    """
    Sparse lexical retrieval over a prebuilt Lucene index.

    k1 and b are part of the checkpoint label rather than buried in a manifest
    field, because two BM25 runs at different parameters are different systems
    and must never be mistaken for the same one.
    """

    name = "bm25"

    def __init__(
        self,
        index: str = "msmarco-v1-passage",
        k1: float = 0.9,
        b: float = 0.4,
        threads: int = 1,
    ):
        from pyserini.search.lucene import LuceneSearcher

        self.index, self.k1, self.b, self.threads = index, k1, b, threads
        self.searcher = LuceneSearcher.from_prebuilt_index(index)
        self.searcher.set_bm25(k1, b)
        self.checkpoint = f"pyserini:{index}|bm25-k1={k1}-b={b}"
        logger.info("BM25 ready (%s)", self.checkpoint)

    def search(self, query_id: str, query_text: str, k: int):
        hits = self.searcher.search(query_text, k=k)
        return [(h.docid, float(h.score)) for h in hits]


class PrecomputedRun(Retriever):
    """
    A retriever backed by an existing run table.

    Two uses: re-exporting a baseline run that was downloaded rather than
    reproduced, and exercising everything downstream without an index. The
    checkpoint is carried through from the source table, so a precomputed run
    never launders its provenance.
    """

    def __init__(self, run: pd.DataFrame, name: str):
        self.name = name
        r = validate_run(run)
        r = r[r["system"] == name]
        if r.empty:
            raise ValueError(f"run contains no rows for system {name!r}")

        checkpoints = sorted(set(r["checkpoint"].dropna().astype(str)))
        if len(checkpoints) != 1:
            raise ValueError(
                f"run for {name!r} mixes {len(checkpoints)} checkpoints "
                f"{checkpoints[:3]}; one run table is one system at one checkpoint"
            )
        self.checkpoint = checkpoints[0]

        self._by_query: dict[str, list[tuple[str, float]]] = {}
        for qid, grp in r.groupby("query_id", observed=True):
            ordered = grp.sort_values("rank")
            self._by_query[str(qid)] = list(
                zip(ordered["doc_id"].astype(str), ordered["score"].astype(float))
            )

    def search(self, query_id: str, query_text: str, k: int):
        return self._by_query.get(str(query_id), [])[:k]


def scores_from_run(
    run: pd.DataFrame,
    system: str,
    universe: Mapping[str, Sequence[str]],
    missing: str = "floor",
) -> dict[str, dict[str, float]]:
    """
    Build {query: {doc: s1}} over the frozen universe from a run table.

    A universe document may be absent from this retriever's own top-M because
    another source contributed it. Two policies:

        floor   give it a score strictly below this retriever's observed
                minimum for that query, so it sorts beneath everything the
                retriever actually returned while keeping a deterministic order
                among its peers. This asserts only what the retriever really
                said: "not in my top-M".
        error   refuse, forcing the caller to score them explicitly.

    'floor' is the honest default, and the policy goes in the run manifest.
    Note what it is NOT: a claim that the document is irrelevant, or a score
    comparable across queries.
    """
    if missing not in {"floor", "error"}:
        raise ValueError(f"unknown missing-score policy {missing!r}")

    r = validate_run(run)
    r = r[r["system"] == system]
    if r.empty:
        raise ValueError(f"run contains no rows for system {system!r}")

    by_query = {str(q): g for q, g in r.groupby("query_id", observed=True)}
    out: dict[str, dict[str, float]] = {}

    for qid, docs in universe.items():
        qid = str(qid)
        grp = by_query.get(qid)
        known: dict[str, float] = (
            dict(zip(grp["doc_id"].astype(str), grp["score"].astype(float)))
            if grp is not None else {}
        )
        absent = [d for d in docs if d not in known]
        if absent:
            if missing == "error":
                raise KeyError(
                    f"query {qid}: {len(absent)} universe documents have no "
                    f"{system} score, e.g. {absent[:5]}"
                )
            floor = (min(known.values()) - 1.0) if known else 0.0
            for d in absent:
                known[d] = floor
        out[qid] = {d: known[d] for d in docs}

    return out


def floor_counts(
    run: pd.DataFrame, system: str, universe: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    """
    How many universe documents each query had to floor.

    Worth reporting beside any result built on floored scores: if a retriever
    floors most of U_q, its candidate set is doing far less work than the
    cross-paradigm comparison assumes.
    """
    r = validate_run(run)
    r = r[r["system"] == system]
    by_query = {
        str(q): set(g["doc_id"].astype(str)) for q, g in r.groupby("query_id", observed=True)
    }
    rows = []
    for qid, docs in universe.items():
        retrieved = by_query.get(str(qid), set())
        n_floored = sum(1 for d in docs if d not in retrieved)
        rows.append(
            {
                "query_id": str(qid),
                "system": system,
                "universe_size": len(docs),
                "n_retrieved": len(docs) - n_floored,
                "n_floored": n_floored,
                "floored_fraction": n_floored / len(docs) if docs else float("nan"),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "Retriever", "PyseriniBM25", "PrecomputedRun",
    "scores_from_run", "floor_counts",
]
