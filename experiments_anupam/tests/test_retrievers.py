"""
test_retrievers.py - The retriever interface, run tables and the floor policy.

No index, no JDK, no network: `PyseriniBM25` imports pyserini inside its
constructor, so this module is importable and testable without any of it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from experiments_anupam.retrievers import (
    PrecomputedRun,
    Retriever,
    floor_counts,
    scores_from_run,
)
from experiments_anupam.tests.conftest import make_run
from shared.schema import validate_run


class FakeRetriever(Retriever):
    """Returns documents in a fixed order with descending scores."""

    name = "fake"
    checkpoint = "fake/model@deadbeef|settings:v1"

    def __init__(self, docs_by_query):
        self._docs = docs_by_query

    def search(self, query_id, query_text, k):
        return [(d, float(100 - i)) for i, d in enumerate(self._docs[query_id][:k])]


# ── the interface ─────────────────────────────────────────────────────

def test_run_produces_a_schema_valid_table():
    r = FakeRetriever({"q1": list("abcde"), "q2": list("fghij")})
    df = r.run({"q1": "text one", "q2": "text two"}, depth=5)
    validate_run(df)
    assert len(df) == 10
    assert set(df["system"]) == {"fake"}
    assert set(df["checkpoint"]) == {"fake/model@deadbeef|settings:v1"}


def test_ranks_are_contiguous_and_one_indexed():
    r = FakeRetriever({"q1": list("abcde")})
    df = r.run({"q1": "t"}, depth=5)
    assert list(df["rank"]) == [1, 2, 3, 4, 5]


def test_depth_truncates():
    r = FakeRetriever({"q1": list("abcdefghij")})
    assert len(r.run({"q1": "t"}, depth=3)) == 3


def test_checkpoint_is_recorded_on_every_row():
    """A run table that cannot say which checkpoint produced it is unusable."""
    r = FakeRetriever({"q1": list("ab")})
    df = r.run({"q1": "t"}, depth=2)
    assert df["checkpoint"].notna().all()


# ── precomputed runs ──────────────────────────────────────────────────

def test_precomputed_run_replays_a_table(two_source_runs):
    r = PrecomputedRun(two_source_runs["bm25"], name="bm25")
    hits = r.search("q0", "unused", 3)
    assert len(hits) == 3
    assert [d for d, _ in hits] == list(
        two_source_runs["bm25"].query("query_id == 'q0'").sort_values("rank")["doc_id"][:3]
    )


def test_precomputed_run_carries_the_checkpoint_through(two_source_runs):
    """A precomputed run must not launder its provenance."""
    r = PrecomputedRun(two_source_runs["bm25"], name="bm25")
    assert r.checkpoint == "pyserini:test|bm25-k1=0.9-b=0.4"


def test_precomputed_run_rejects_mixed_checkpoints():
    df = pd.concat([
        make_run("bm25", {"q1": ["a", "b"]}, "index|k1=0.9"),
        make_run("bm25", {"q2": ["c", "d"]}, "index|k1=1.2"),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="mixes 2 checkpoints"):
        PrecomputedRun(df, name="bm25")


def test_precomputed_run_rejects_an_unknown_system(two_source_runs):
    with pytest.raises(ValueError, match="no rows for system"):
        PrecomputedRun(two_source_runs["bm25"], name="splade")


def test_precomputed_run_returns_empty_for_an_unseen_query(two_source_runs):
    r = PrecomputedRun(two_source_runs["bm25"], name="bm25")
    assert r.search("not_a_query", "unused", 5) == []


# ── scores over the universe ──────────────────────────────────────────

def _universe(two_source_runs):
    """Every document either source returned, per query."""
    uni = {}
    for run in two_source_runs.values():
        for qid, grp in run.groupby("query_id", observed=True):
            uni.setdefault(str(qid), set()).update(grp["doc_id"].astype(str))
    return {q: sorted(d) for q, d in uni.items()}


def test_scores_cover_every_universe_document(two_source_runs):
    uni = _universe(two_source_runs)
    s1 = scores_from_run(two_source_runs["bm25"], "bm25", uni)
    for q, docs in uni.items():
        assert set(s1[q]) == set(docs)


def test_floored_documents_sort_below_everything_retrieved(two_source_runs):
    """
    The floor asserts only what the retriever actually said: not in my top-M.
    It must never let an unretrieved document outrank a retrieved one.
    """
    uni = _universe(two_source_runs)
    bm25 = two_source_runs["bm25"]
    s1 = scores_from_run(bm25, "bm25", uni)

    for q, docs in uni.items():
        retrieved = set(bm25.query("query_id == @q")["doc_id"].astype(str))
        floored = [d for d in docs if d not in retrieved]
        assert floored, "the fixture no longer exercises the floor"
        assert max(s1[q][d] for d in floored) < min(s1[q][d] for d in retrieved)


def test_floored_documents_share_one_value(two_source_runs):
    """All equally unranked, so they tie and the doc_id tie-break orders them."""
    uni = _universe(two_source_runs)
    bm25 = two_source_runs["bm25"]
    s1 = scores_from_run(bm25, "bm25", uni)
    q = sorted(uni)[0]
    retrieved = set(bm25.query("query_id == @q")["doc_id"].astype(str))
    floored = {s1[q][d] for d in uni[q] if d not in retrieved}
    assert len(floored) == 1


def test_error_policy_refuses_to_invent_scores(two_source_runs):
    uni = _universe(two_source_runs)
    with pytest.raises(KeyError, match="no bm25 score"):
        scores_from_run(two_source_runs["bm25"], "bm25", uni, missing="error")


def test_unknown_policy_is_rejected(two_source_runs):
    with pytest.raises(ValueError, match="unknown missing-score policy"):
        scores_from_run(two_source_runs["bm25"], "bm25", _universe(two_source_runs),
                        missing="guess")


def test_unknown_system_is_rejected(two_source_runs):
    with pytest.raises(ValueError, match="no rows for system"):
        scores_from_run(two_source_runs["bm25"], "splade", _universe(two_source_runs))


def test_a_query_absent_from_the_run_floors_to_zero(two_source_runs):
    """
    A retriever that returned nothing for a query has no observed minimum, so
    every document floors to the same value rather than crashing.
    """
    uni = _universe(two_source_runs)
    uni["unseen"] = ["x", "y", "z"]
    s1 = scores_from_run(two_source_runs["bm25"], "bm25", uni)
    assert set(s1["unseen"].values()) == {0.0}


# ── reporting ─────────────────────────────────────────────────────────

def test_floor_counts_reports_the_floored_fraction(two_source_runs):
    """
    Worth reporting beside any cross-paradigm result: a retriever that floors
    most of U_q is doing far less work than the comparison assumes.
    """
    uni = _universe(two_source_runs)
    rep = floor_counts(two_source_runs["bm25"], "bm25", uni)
    assert set(rep["query_id"]) == set(uni)
    assert (rep["n_retrieved"] + rep["n_floored"] == rep["universe_size"]).all()
    assert (rep["floored_fraction"] > 0).all()
    assert (rep["floored_fraction"] < 1).all()
