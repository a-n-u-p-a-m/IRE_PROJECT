"""
test_chaitanya.py - Tests for experiments_chaitanya modules.

All tests are GPU-free and require no real data.  They use in-memory fakes,
synthetic data and the shared toy fixture so they can run on any machine.

Run with::

    pytest tests/test_chaitanya.py -v
"""
from __future__ import annotations

import sys
import os
import math

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────────────────────
# passage_fetcher tests
# ──────────────────────────────────────────────────────────────────────

class TestDictPassageFetcher:
    def test_fetch_existing(self):
        from experiments_chaitanya.passage_fetcher import DictPassageFetcher
        f = DictPassageFetcher({"d1": "hello world", "d2": "another passage"})
        assert f.fetch("d1") == "hello world"
        assert f.fetch("d2") == "another passage"

    def test_fetch_missing_raises(self):
        from experiments_chaitanya.passage_fetcher import (
            DictPassageFetcher, PassageFetchError
        )
        f = DictPassageFetcher({"d1": "text"})
        with pytest.raises(PassageFetchError):
            f.fetch("missing_doc")

    def test_fetch_many_preserves_order(self):
        from experiments_chaitanya.passage_fetcher import DictPassageFetcher
        store = {f"d{i}": f"text {i}" for i in range(5)}
        f = DictPassageFetcher(store)
        ids = ["d3", "d0", "d4", "d1"]
        texts = f.fetch_many(ids)
        assert texts == [store[d] for d in ids]

    def test_fetch_many_raises_on_missing(self):
        from experiments_chaitanya.passage_fetcher import (
            DictPassageFetcher, PassageFetchError
        )
        f = DictPassageFetcher({"d1": "text"})
        with pytest.raises(PassageFetchError):
            f.fetch_many(["d1", "missing"])

    def test_len(self):
        from experiments_chaitanya.passage_fetcher import DictPassageFetcher
        f = DictPassageFetcher({"a": "t", "b": "u"})
        assert len(f) == 2

    def test_make_fetcher_dict_backend(self):
        from experiments_chaitanya.passage_fetcher import make_fetcher
        f = make_fetcher("dict", passage_store={"d1": "hello"})
        assert f.fetch("d1") == "hello"

    def test_make_fetcher_invalid_backend(self):
        from experiments_chaitanya.passage_fetcher import make_fetcher
        with pytest.raises(ValueError, match="backend"):
            make_fetcher("unknown_backend")

    def test_make_fetcher_dict_without_store_raises(self):
        from experiments_chaitanya.passage_fetcher import make_fetcher
        with pytest.raises(ValueError, match="passage_store"):
            make_fetcher("dict")


# ──────────────────────────────────────────────────────────────────────
# reranker tests (no GPU required — only tests the interface and stub)
# ──────────────────────────────────────────────────────────────────────

class TestListwiseStub:
    def test_stub_raises_on_instantiation(self):
        from experiments_chaitanya.reranker import (
            ListwiseCrossEncoderStub, ListwiseRerankerError
        )
        with pytest.raises(ListwiseRerankerError):
            ListwiseCrossEncoderStub()


class TestPointwiseCrossEncoderInterface:
    def test_checkpoint_returns_model_name(self):
        from experiments_chaitanya.reranker import PointwiseCrossEncoder
        r = PointwiseCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
        assert r.checkpoint == "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def test_score_raises_on_mismatched_lengths(self):
        from experiments_chaitanya.reranker import PointwiseCrossEncoder
        r = PointwiseCrossEncoder()
        with pytest.raises(ValueError, match="same length"):
            r.score("query", ["p1", "p2"], ["d1"])

    def test_score_empty_returns_empty(self):
        from experiments_chaitanya.reranker import PointwiseCrossEncoder
        r = PointwiseCrossEncoder()
        result = r.score("query", [], [])
        assert result == {}


# ──────────────────────────────────────────────────────────────────────
# features tests
# ──────────────────────────────────────────────────────────────────────

def _make_synthetic_run(n_queries: int = 5, n_docs: int = 20, k_c: int = 5) -> pd.DataFrame:
    """Build a minimal validated run table for testing."""
    from shared.schema import validate_run
    rows = []
    for q in range(n_queries):
        qid = f"q{q:03d}"
        scores = np.random.default_rng(q).uniform(0, 1, n_docs)
        sorted_docs = np.argsort(-scores)
        for rank, idx in enumerate(sorted_docs, start=1):
            rows.append({
                "query_id": qid, "doc_id": f"d{idx:05d}",
                "system": "bm25", "rank": rank,
                "score": float(scores[idx]), "checkpoint": "synthetic",
            })
    df = pd.DataFrame(rows)
    for col, dtype in [("query_id", "string"), ("doc_id", "string"),
                        ("system", "string"), ("checkpoint", "string")]:
        df[col] = df[col].astype(dtype)
    return validate_run(df)


class TestStage1Features:
    def test_basic_extraction(self):
        from experiments_chaitanya.features import extract_stage1_features
        run = _make_synthetic_run()
        feats = extract_stage1_features(run, "bm25", k_c=5)
        assert not feats.empty
        assert "feature_name" in feats.columns
        expected_names = {"stage1_rank", "stage1_score", "stage1_rank_norm",
                          "stage1_percentile", "is_survivor"}
        assert expected_names.issubset(set(feats["feature_name"].unique()))

    def test_is_survivor_correct(self):
        from experiments_chaitanya.features import extract_stage1_features
        run = _make_synthetic_run(n_queries=2, n_docs=10, k_c=3)
        feats = extract_stage1_features(run, "bm25", k_c=3)
        survivors = feats[feats["feature_name"] == "is_survivor"]
        assert survivors["value"].isin([0.0, 1.0]).all()
        # Rank <= k_c → survivor = 1
        rank_feats = feats[feats["feature_name"] == "stage1_rank"]
        for _, r in rank_feats.iterrows():
            qid, did = r["query_id"], r["doc_id"]
            rank = r["value"]
            surv = feats[
                (feats["query_id"] == qid) &
                (feats["doc_id"] == did) &
                (feats["feature_name"] == "is_survivor")
            ]["value"].iloc[0]
            expected = 1.0 if rank <= 3 else 0.0
            assert surv == expected, f"rank={rank}, survivor={surv}, expected={expected}"

    def test_schema_valid(self):
        from experiments_chaitanya.features import extract_stage1_features
        from shared.schema import validate_feature
        run = _make_synthetic_run()
        feats = extract_stage1_features(run, "bm25", k_c=5)
        # validate_feature should not raise
        validate_feature(feats)


class TestPassageLengthFeatures:
    def test_basic(self):
        from experiments_chaitanya.features import extract_passage_length_features
        store = {"d1": "hello world", "d2": "a b c d e f g h i j"}
        feats = extract_passage_length_features(
            store,
            query_ids=["q0"],
            doc_ids_per_query={"q0": ["d1", "d2"]},
        )
        assert not feats.empty
        names = set(feats["feature_name"].unique())
        assert "passage_words" in names
        assert "passage_words_norm" in names

        words_d1 = float(feats[
            (feats["doc_id"] == "d1") & (feats["feature_name"] == "passage_words")
        ]["value"].iloc[0])
        assert words_d1 == 2.0

    def test_missing_doc_not_included(self):
        from experiments_chaitanya.features import extract_passage_length_features
        # d3 is not in the store; should be silently skipped (0 words)
        store = {"d1": "hello"}
        feats = extract_passage_length_features(
            store, ["q0"], {"q0": ["d1", "d3"]}
        )
        # d3 gets 0 words from the empty string
        d3_rows = feats[feats["doc_id"] == "d3"]
        if not d3_rows.empty:
            words = float(d3_rows[d3_rows["feature_name"] == "passage_words"]["value"].iloc[0])
            assert words == 0.0


# ──────────────────────────────────────────────────────────────────────
# shard_plan tests
# ──────────────────────────────────────────────────────────────────────

class TestShardPlan:
    def _make_universe(self, n_queries: int = 9, n_docs: int = 100) -> pd.DataFrame:
        rows = []
        for q in range(n_queries):
            for d in range(n_docs):
                rows.append({
                    "query_id": f"q{q:03d}",
                    "doc_id": f"d{d:05d}",
                    "sources": "bm25",
                })
        df = pd.DataFrame(rows)
        df["query_id"] = df["query_id"].astype("string")
        df["doc_id"] = df["doc_id"].astype("string")
        df["sources"] = df["sources"].astype("string")
        return df

    def test_total_pairs_correct(self):
        from experiments_chaitanya.shard_plan import build_shard_plan
        uni = self._make_universe(9, 100)
        plan = build_shard_plan(uni, n_shards=3)
        total = sum(s["n_pairs"] for s in plan["shards"])
        assert total == plan["total_pairs"]
        assert plan["total_pairs"] == 9 * 100

    def test_n_shards_correct(self):
        from experiments_chaitanya.shard_plan import build_shard_plan
        uni = self._make_universe(9, 100)
        plan = build_shard_plan(uni, n_shards=3)
        assert len(plan["shards"]) == 3

    def test_no_query_in_two_shards(self):
        from experiments_chaitanya.shard_plan import build_shard_plan
        uni = self._make_universe(9, 100)
        plan = build_shard_plan(uni, n_shards=3)
        all_qids = []
        for s in plan["shards"]:
            all_qids.extend(s["query_ids"])
        # No duplicates
        assert len(all_qids) == len(set(all_qids))

    def test_all_queries_covered(self):
        from experiments_chaitanya.shard_plan import build_shard_plan
        uni = self._make_universe(9, 100)
        plan = build_shard_plan(uni, n_shards=3)
        all_qids = set()
        for s in plan["shards"]:
            all_qids.update(s["query_ids"])
        expected = {f"q{i:03d}" for i in range(9)}
        assert all_qids == expected

    def test_deterministic(self):
        from experiments_chaitanya.shard_plan import build_shard_plan
        uni = self._make_universe()
        p1 = build_shard_plan(uni, n_shards=3)
        p2 = build_shard_plan(uni, n_shards=3)
        for s1, s2 in zip(p1["shards"], p2["shards"]):
            assert s1["query_ids"] == s2["query_ids"]


# ──────────────────────────────────────────────────────────────────────
# smoke_test integration
# ──────────────────────────────────────────────────────────────────────

class TestSmokeTest:
    def test_shared_smoke_test_passes(self):
        from shared.fixtures import smoke_test
        assert smoke_test() is True


# ──────────────────────────────────────────────────────────────────────
# Synthetic pipeline smoke test
# ──────────────────────────────────────────────────────────────────────

class TestSyntheticPipeline:
    def test_run_synthetic_pipeline_smoke(self, tmp_path):
        """End-to-end synthetic run on tiny data — no GPU, no disk I/O."""
        from experiments_chaitanya.run_synthetic import run_synthetic_pipeline
        results = run_synthetic_pipeline(
            n_queries=20,
            n_docs=150,
            k_c=50,
            k=5,
            output_dir=str(tmp_path / "synth_results"),
            seed=0,
        )
        assert "effects" in results
        assert "features" in results
        assert "fidelity_table" in results
        assert not results["effects"].empty
        assert not results["features"].empty

    def test_fidelity_table_has_effect_types(self, tmp_path):
        from experiments_chaitanya.run_synthetic import run_synthetic_pipeline
        from shared.schema import E_RERANK, E_GATE
        results = run_synthetic_pipeline(
            n_queries=20, n_docs=150, k_c=50, k=5,
            output_dir=str(tmp_path / "synth2"),
            seed=1,
        )
        fid = results["fidelity_table"]
        # May be empty if no test queries, but should not raise
        if not fid.empty:
            assert "effect_type" in fid.columns
