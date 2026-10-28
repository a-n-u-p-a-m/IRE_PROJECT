"""Shared fixtures for experiments_anupam's tests. No index, no JDK, no network."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from shared.config import load_config


@pytest.fixture(scope="session")
def small_cfg_path(tmp_path_factory):
    """The frozen config shrunk so a universe can be built in a unit test."""
    with open("config/base.yaml", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["universe"]["depth_m"] = 40
    raw["universe"]["max_size"] = 200
    raw["cascade"]["candidate_depths"] = [5, 10, 20]
    raw["cascade"]["final_cutoff_k"] = 5
    raw["cascade"]["cutoff_sensitivity"] = [3]
    path = tmp_path_factory.mktemp("cfg") / "small.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


@pytest.fixture(scope="session")
def small_cfg(small_cfg_path):
    return load_config(small_cfg_path)


def make_run(system: str, docs_by_query: dict[str, list[str]],
             checkpoint: str = "test:index|v1") -> pd.DataFrame:
    """A schema-valid run table with descending scores and contiguous ranks."""
    rows = []
    for qid, docs in docs_by_query.items():
        for rank, doc in enumerate(docs, start=1):
            rows.append({"query_id": qid, "doc_id": doc, "system": system,
                         "rank": rank, "score": float(1000 - rank),
                         "checkpoint": checkpoint})
    return pd.DataFrame(rows)


@pytest.fixture
def two_source_runs():
    """
    Two retrievers over 8 queries, 40 documents deep, sharing part of each
    ranking so the union is genuinely broader than either source.
    """
    queries = [f"q{i}" for i in range(8)]
    a, b = {}, {}
    for i, q in enumerate(queries):
        a[q] = [f"d{(i * 7 + j) % 120:03d}" for j in range(40)]
        b[q] = [f"d{(i * 7 + j + 20) % 120:03d}" for j in range(40)]
    return {
        "bm25": make_run("bm25", a, "pyserini:test|bm25-k1=0.9-b=0.4"),
        "dpr": make_run("dpr", b, "fake/dpr@abc123|index:test"),
    }
