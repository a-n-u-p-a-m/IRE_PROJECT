"""
test_dpr.py — DPR retriever: resumable shard encoding and exact search.

Uses a deterministic fake encoder, so no model download is needed.
Run with:  pytest tests/test_dpr.py -v
"""

import numpy as np
import pytest

from retrievers.dpr import EMBEDDING_DIM, DPRRetriever, encode_corpus, exact_search


class FakeEncoder:
    """Deterministic unit vectors derived from the text; counts calls."""

    def __init__(self):
        self.n_texts = 0

    def encode(self, texts, max_length=None):
        self.n_texts += len(texts)
        out = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            v = np.random.default_rng(abs(hash(t)) % 2**32).standard_normal(EMBEDDING_DIM)
            out[i] = v / np.linalg.norm(v)
        return out


@pytest.fixture
def collection(tmp_path):
    path = tmp_path / "collection.tsv"
    path.write_text("".join(f"{i}\tpassage number {i} about topic {i % 7}\n" for i in range(250)))
    return path


@pytest.fixture
def index_dir(tmp_path, collection):
    out = tmp_path / "index"
    encode_corpus(out, collection, encoder=FakeEncoder(), shard_size=100, batch_size=32)
    return out


def _all_embeddings(index_dir):
    ids, embs = [], []
    for n in range(3):
        ids += (index_dir / f"ids_{n:05d}.txt").read_text().split("\n")[:-1]
        embs.append(np.load(index_dir / f"emb_{n:05d}.npy").astype(np.float32))
    return ids, np.vstack(embs)


class TestEncodeCorpus:

    def test_shards_cover_corpus_in_order(self, index_dir):
        ids, emb = _all_embeddings(index_dir)
        assert ids == [str(i) for i in range(250)]
        assert emb.shape == (250, EMBEDDING_DIM)
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-2)

    def test_embedding_matches_its_passage(self, index_dir):
        # Length-sorted batching must not scramble rows
        ids, emb = _all_embeddings(index_dir)
        expected = FakeEncoder().encode(["passage number 123 about topic 4"])[0]
        assert np.allclose(emb[ids.index("123")], expected, atol=1e-2)

    def test_resume_only_encodes_missing_shards(self, index_dir, collection):
        (index_dir / "emb_00001.npy").unlink()
        enc = FakeEncoder()
        encode_corpus(index_dir, collection, encoder=enc, shard_size=100, batch_size=32)
        assert enc.n_texts == 100
        assert (index_dir / "emb_00001.npy").exists()

    def test_changed_settings_rejected(self, index_dir, collection):
        with pytest.raises(ValueError, match="different settings"):
            encode_corpus(index_dir, collection, encoder=FakeEncoder(), shard_size=50)


class TestExactSearch:

    def test_matches_brute_force(self, index_dir):
        ids, emb = _all_embeddings(index_dir)
        q = FakeEncoder().encode(["query a", "query b", "query c"])
        doc_ids, scores = exact_search(q, index_dir, k=20, device="cpu")
        brute = q @ emb.T
        for i in range(3):
            top = np.argsort(-brute[i])[:20]
            assert doc_ids[i] == [ids[j] for j in top]
            assert np.allclose(scores[i], brute[i][top], atol=1e-5)

    def test_k_larger_than_corpus(self, index_dir):
        doc_ids, scores = exact_search(FakeEncoder().encode(["q"]), index_dir, k=1000, device="cpu")
        assert len(doc_ids[0]) == 250
        assert np.all(np.diff(scores[0]) <= 0)

    def test_missing_shard_rejected(self, index_dir):
        (index_dir / "emb_00001.npy").unlink()
        with pytest.raises(ValueError, match="missing shards"):
            exact_search(FakeEncoder().encode(["q"]), index_dir, k=5, device="cpu")


class TestRetriever:

    def test_search_and_batch_search_agree(self, index_dir):
        r = DPRRetriever({"retriever": {"index_dir": str(index_dir), "device": "cpu"}},
                         encoder=FakeEncoder())
        batch = r.batch_search({"q1": "topic 3", "q2": "topic 5"}, k=10)
        single = r.search("topic 3", k=10)
        assert [d for d, _ in batch["q1"]] == [d for d, _ in single]
        assert np.allclose([s for _, s in batch["q1"]], [s for _, s in single], atol=1e-6)
        assert len(batch["q2"]) == 10
        assert all(isinstance(d, str) and isinstance(s, float) for d, s in batch["q2"])

    def test_env_overrides_index_dir(self, index_dir, monkeypatch):
        monkeypatch.setenv("DPR_INDEX_DIR", str(index_dir))
        r = DPRRetriever({"retriever": {"index_dir": "does/not/exist"}}, encoder=FakeEncoder())
        assert r.index_dir == index_dir
