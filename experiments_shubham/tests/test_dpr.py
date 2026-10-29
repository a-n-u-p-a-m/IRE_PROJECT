"""
test_dpr.py - Exact search, tie-breaking and the run table, with a fake encoder.

No model or index needed. Run from the repo root:

    python -m pytest experiments_shubham
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

try:
    import torch  # noqa: F401
except ImportError:
    pytest.skip("exact search runs on torch", allow_module_level=True)

from experiments_shubham import run_dpr  # noqa: E402
from experiments_shubham.dpr import LocalShards, encode_corpus, exact_search  # noqa: E402
from shared.schema import validate_run  # noqa: E402


class FakeEncoder:
    """Deterministic unit vectors per text; counts calls."""

    model_name = "fake/model"
    revision = "0123456789abcdef"

    def __init__(self, *args, **kwargs):
        self.n_texts = 0

    def encode(self, texts, max_length=None):
        self.n_texts += len(texts)
        out = np.empty((len(texts), 768), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(768)
            out[i] = v / np.linalg.norm(v)
        return out


@pytest.fixture
def collection(tmp_path):
    path = tmp_path / "collection.tsv"
    path.write_text("".join(f"{i}\tpassage {i} about topic {i % 7}\n" for i in range(250)))
    return path


@pytest.fixture
def index_dir(tmp_path, collection):
    out = tmp_path / "index"
    encode_corpus(out, collection, model_name="fake/model", shard_size=100,
                  batch_size=32, encoder=FakeEncoder())
    return out


def _all(index_dir):
    ids, embs = [], []
    for n in range(3):
        ids += (index_dir / f"ids_{n:05d}.txt").read_text().split("\n")[:-1]
        embs.append(np.load(index_dir / f"emb_{n:05d}.npy").astype(np.float32))
    return ids, np.vstack(embs)


def test_exact_search_matches_brute_force(index_dir):
    ids, emb = _all(index_dir)
    q = FakeEncoder().encode(["query a", "query b"])
    hits = exact_search(q, LocalShards(index_dir), k=30, margin=5)
    brute = q @ emb.T
    for i in range(2):
        expected = sorted(zip(ids, brute[i].astype(np.float64)), key=lambda t: (-t[1], t[0]))[:30]
        assert [d for d, _ in hits[i]] == [d for d, _ in expected]
        assert np.allclose([s for _, s in hits[i]], [s for _, s in expected], atol=1e-6)


def test_ties_break_by_doc_id(tmp_path):
    # Three identical vectors: the order must be by ascending doc_id
    d = tmp_path / "tie_index"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"model_name": "fake/model"}))
    emb = np.zeros((5, 768), dtype=np.float16)
    emb[:, 0] = [1.0, 0.5, 1.0, 1.0, 0.2]
    np.save(d / "emb_00000.npy", emb)
    (d / "ids_00000.txt").write_text("".join(f"{x}\n" for x in ["z9", "a1", "m5", "b2", "c3"]))
    q = np.zeros((1, 768), dtype=np.float32)
    q[0, 0] = 1.0
    hits = exact_search(q, LocalShards(d), k=4, margin=1)
    assert [doc for doc, _ in hits[0]] == ["b2", "m5", "z9", "a1"]


def test_unresolvable_boundary_tie_raises(tmp_path):
    d = tmp_path / "flat_index"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"model_name": "fake/model"}))
    np.save(d / "emb_00000.npy", np.ones((10, 768), dtype=np.float16))
    (d / "ids_00000.txt").write_text("".join(f"d{i}\n" for i in range(10)))
    with pytest.raises(ValueError, match="tie with the depth"):
        exact_search(np.ones((1, 768), dtype=np.float32), LocalShards(d), k=3, margin=2)


def test_missing_shard_rejected(index_dir):
    (index_dir / "emb_00001.npy").unlink()
    with pytest.raises(ValueError, match="missing shards"):
        LocalShards(index_dir)


def test_encode_is_resumable(index_dir, collection):
    (index_dir / "emb_00001.npy").unlink()
    enc = FakeEncoder()
    encode_corpus(index_dir, collection, model_name="fake/model", shard_size=100,
                  batch_size=32, encoder=enc)
    assert enc.n_texts == 100


def test_run_table_is_schema_valid(index_dir, tmp_path, monkeypatch):
    # The 250-passage toy index is smaller than M=5000, so every passage comes back
    queries = tmp_path / "queries.tsv"
    queries.write_text("q1\tfirst query\nq2\tsecond query\n")
    monkeypatch.setattr(run_dpr, "CosDPREncoder", FakeEncoder)

    out = run_dpr.run(str(index_dir), str(queries), str(tmp_path / "runs" / "dpr.parquet"),
                      "config/base.yaml", depth=None, device="cpu")
    df = validate_run(pd.read_parquet(out))
    assert set(df["system"]) == {"dpr"} and len(df) == 2 * 250
    assert df.groupby("query_id")["rank"].max().tolist() == [250, 250]
    assert df["checkpoint"].str.startswith("fake/model@0123456789ab|index:fp16-exact-").all()
    manifest = json.loads((tmp_path / "runs" / "dpr_manifest.json").read_text())
    assert manifest["owner"] == "shubham" and manifest["depth"] == 5000


def test_depth_below_universe_depth_refused(index_dir, tmp_path):
    queries = tmp_path / "queries.tsv"
    queries.write_text("q1\tfirst query\n")
    with pytest.raises(ValueError, match="below the universe depth"):
        run_dpr.run(str(index_dir), str(queries), str(tmp_path / "x.parquet"),
                    "config/base.yaml", depth=100, device="cpu")
