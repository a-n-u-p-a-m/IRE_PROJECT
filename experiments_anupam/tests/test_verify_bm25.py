"""
test_verify_bm25.py - The official measures, checked against trec_eval itself.

The point of `verify_bm25` is to compare against a published number, so its
metric implementation has to match the one that produced that number. Asserting
it against my own arithmetic would only prove I am consistent. Where
`pytrec_eval` (a binding over the real trec_eval C code) is installed, these
tests check against it on random data; where it is not, they skip and the
hand-computed cases still run.

    pip install pytrec_eval-terrier     # to enable the cross-check
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments_anupam import verify_bm25 as vb

try:
    import pytrec_eval as _pte
except ImportError:                                     # pragma: no cover
    _pte = None

needs_pte = pytest.mark.skipif(_pte is None, reason="pytrec_eval not installed")


# ── hand-computed cases ───────────────────────────────────────────────

def test_dcg_uses_linear_gain_like_trec_eval():
    # grades 3, 0, 2 -> 3/log2(2) + 0/log2(3) + 2/log2(4)
    assert vb.dcg([3, 0, 2]) == pytest.approx(3.0 / 1.0 + 0.0 + 2.0 / 2.0)


def test_dcg_is_not_the_exponential_variant():
    """
    Guards the one decision in this module that is easy to get wrong and
    impossible to notice: 2^g - 1 would give 8.5 here against trec_eval's 4.0,
    and the script would then "verify" the run against a number it cannot
    reproduce. Established by probing pytrec_eval, pinned here.
    """
    exponential = 7.0 / 1.0 + 0.0 + 3.0 / 2.0
    assert vb.dcg([3, 0, 2]) != pytest.approx(exponential)


def test_perfect_ranking_scores_one():
    grades = {"a": 3, "b": 2, "c": 1}
    assert vb.ndcg_at_k(["a", "b", "c"], grades, 10) == pytest.approx(1.0)


def test_reversed_ranking_scores_less():
    grades = {"a": 3, "b": 2, "c": 1}
    assert vb.ndcg_at_k(["c", "b", "a"], grades, 10) < 0.8


def test_unjudged_documents_score_zero_not_skipped():
    """
    trec_eval treats an unretrieved judgment and an unjudged retrieval the same
    way: the slot earns nothing. This is exactly why a sparsely judged universe
    depresses nDCG, which is the bias world C is exposed to.
    """
    grades = {"a": 3}
    assert vb.ndcg_at_k(["x", "y", "a"], grades, 10) < 1.0
    assert vb.ndcg_at_k(["x", "y", "z"], grades, 10) == 0.0


def test_ideal_comes_from_all_grades_not_the_retrieved_ones():
    """A run that found one relevant document of five must not score 1.0."""
    grades = {f"d{i}": 3 for i in range(5)}
    assert vb.ndcg_at_k(["d0"], grades, 10) == pytest.approx(0.33916, abs=1e-4)


def test_no_judgments_gives_zero_not_a_crash():
    assert vb.ndcg_at_k(["a", "b"], {}, 10) == 0.0


def test_recall_binarises_at_grade_two():
    """TREC DL's own convention: grade 1 is 'Related', not relevant."""
    grades = {"a": 1, "b": 2, "c": 3}
    assert vb.recall_at_k(["a"], grades, 10) == pytest.approx(0.0)
    assert vb.recall_at_k(["b"], grades, 10) == pytest.approx(0.5)
    assert vb.recall_at_k(["b", "c"], grades, 10) == pytest.approx(1.0)


def test_recall_is_nan_when_nothing_is_relevant():
    """Undefined, not zero: averaging zeros in would bias the mean downwards."""
    assert np.isnan(vb.recall_at_k(["a"], {"a": 1, "b": 0}, 10))


def test_recall_respects_the_cutoff():
    grades = {"a": 3, "b": 3}
    assert vb.recall_at_k(["a", "b"], grades, 1) == pytest.approx(0.5)


# ── the cross-check against trec_eval ─────────────────────────────────

def _random_case(seed: int, n_queries: int = 12, n_docs: int = 60):
    rng = np.random.default_rng(seed)
    qrels, run = {}, {}
    for q in (f"q{i}" for i in range(n_queries)):
        n_judged = int(rng.integers(1, 25))
        docs = [f"d{d}" for d in rng.choice(n_docs, size=n_judged, replace=False)]
        qrels[q] = {d: int(rng.integers(0, 4)) for d in docs}
        # Retrieve a mix of judged and unjudged, so unjudged handling is exercised
        ranked = [f"d{d}" for d in rng.permutation(n_docs)[:30]]
        run[q] = {d: float(30 - i) for i, d in enumerate(ranked)}
    return qrels, run


@needs_pte
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ndcg_at_10_matches_trec_eval(seed):
    qrels, run = _random_case(seed)
    ev = _pte.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    theirs = ev.evaluate(run)

    for q in qrels:
        ranked = [d for d, _ in sorted(run[q].items(), key=lambda kv: -kv[1])]
        mine = vb.ndcg_at_k(ranked, qrels[q], 10)
        assert mine == pytest.approx(theirs[q]["ndcg_cut_10"], abs=1e-9), (
            f"query {q}: mine {mine}, trec_eval {theirs[q]['ndcg_cut_10']}"
        )


@needs_pte
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recall_at_1000_matches_trec_eval_with_the_same_binarisation(seed):
    """
    trec_eval binarises at >= 1 by default, so it is told to use >= 2 via
    `relevance_level`. If these two ever disagree, the convention is what
    changed - and the published number uses >= 2.
    """
    qrels, run = _random_case(seed)
    ev = _pte.RelevanceEvaluator(
        qrels, {"recall.1000"}, relevance_level=vb.RECALL_THRESHOLD
    )
    theirs = ev.evaluate(run)

    for q in qrels:
        ranked = [d for d, _ in sorted(run[q].items(), key=lambda kv: -kv[1])]
        mine = vb.recall_at_k(ranked, qrels[q], 1000)
        expected = theirs[q]["recall_1000"]
        if np.isnan(mine):
            assert expected == pytest.approx(0.0), "nan should mean no relevant docs"
        else:
            assert mine == pytest.approx(expected, abs=1e-9)


# ── the gate behaviour ────────────────────────────────────────────────

def _tiny_inputs(tmp_path, monkeypatch, shuffle: bool):
    """A run that is perfect, or deliberately degraded, over two fake years."""
    from shared.schema import save

    rng = np.random.default_rng(0)
    qrels_rows, run_rows = [], []
    years = {"dl19": [f"a{i}" for i in range(43)],
             "dl20": [f"b{i}" for i in range(54)]}

    for qids in years.values():
        for q in qids:
            docs = [f"{q}_d{i}" for i in range(20)]
            for i, d in enumerate(docs):
                qrels_rows.append({"query_id": q, "doc_id": d,
                                   "relevance": 3 if i < 5 else 0, "judged": True})
            order = list(docs)
            if shuffle:
                rng.shuffle(order)
            for rank, d in enumerate(order, start=1):
                run_rows.append({"query_id": q, "doc_id": d, "system": "bm25",
                                 "rank": rank, "score": float(100 - rank),
                                 "checkpoint": "test|bm25"})

    run_p = save(pd.DataFrame(run_rows), tmp_path / "bm25.parquet", "run")
    qrels_p = save(pd.DataFrame(qrels_rows), tmp_path / "qrels.parquet", "qrels")
    monkeypatch.setattr(vb, "year_query_ids",
                        lambda cache_dir=None: {k: set(v) for k, v in years.items()})
    return str(run_p), str(qrels_p)


def test_a_perfect_run_fails_against_the_published_numbers(tmp_path, monkeypatch):
    """
    Sanity in the other direction. The published figures are ~0.50, so a run
    that scores 1.0 is as wrong as one that scores 0.1 - this script checks
    agreement with a reference, not that bigger is better.
    """
    run_p, qrels_p = _tiny_inputs(tmp_path, monkeypatch, shuffle=False)
    table, failures = vb.run(run_p, qrels_p, "bm25", tolerance=0.01)
    assert failures
    assert not table["within_tolerance"].any()


def test_the_query_count_is_checked_independently(tmp_path, monkeypatch):
    """
    Wrong query set is a distinct failure from wrong effectiveness, and it has
    to be named separately: 43 queries scoring correctly is still a broken run
    if 54 were expected.
    """
    run_p, qrels_p = _tiny_inputs(tmp_path, monkeypatch, shuffle=False)
    monkeypatch.setattr(vb, "year_query_ids",
                        lambda cache_dir=None: {"dl19": {"a0", "a1"}, "dl20": {"b0"}})
    _, failures = vb.run(run_p, qrels_p, "bm25", tolerance=1.0)
    assert any("expected 43" in f for f in failures)
    assert any("expected 54" in f for f in failures)


def test_tolerance_is_honoured(tmp_path, monkeypatch):
    run_p, qrels_p = _tiny_inputs(tmp_path, monkeypatch, shuffle=False)
    _, failures = vb.run(run_p, qrels_p, "bm25", tolerance=1.0)
    assert not failures, "a tolerance of 1.0 should accept anything"


def test_unknown_system_is_refused(tmp_path, monkeypatch):
    run_p, qrels_p = _tiny_inputs(tmp_path, monkeypatch, shuffle=False)
    with pytest.raises(SystemExit, match="no rows for system"):
        vb.run(run_p, qrels_p, "splade")


def test_queries_missing_from_the_run_are_reported_not_hidden(tmp_path, monkeypatch):
    """A run covering half the queries must not quietly average over the half."""
    from shared.schema import load, save

    run_p, qrels_p = _tiny_inputs(tmp_path, monkeypatch, shuffle=False)
    run_df = load(run_p, "run")
    save(run_df[~run_df["query_id"].astype(str).str.startswith("a")],
         run_p, "run")

    ranked = vb.ranked_by_query(load(run_p, "run"), "bm25")
    grades = vb.grades_by_query(load(qrels_p, "qrels"))
    got = vb.evaluate(ranked, grades, {f"a{i}" for i in range(43)})
    assert got["n_missing_from_run"] == 43
    assert got["ndcg@10"] == 0.0


# ── the year split ────────────────────────────────────────────────────

def test_year_query_ids_caches_and_does_not_refetch(tmp_path, monkeypatch):
    calls = []

    def fake_get(url):
        calls.append(url)
        return b"1 0 d1 3\n1 0 d2 0\n2 0 d3 2\n"

    monkeypatch.setattr(vb, "_get", fake_get)
    first = vb.year_query_ids(tmp_path)
    assert first == {"dl19": {"1", "2"}, "dl20": {"1", "2"}}
    assert len(calls) == 2

    second = vb.year_query_ids(tmp_path)
    assert second == first
    assert len(calls) == 2, "the cached files should not be refetched"
