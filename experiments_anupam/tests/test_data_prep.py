"""
test_data_prep.py - Queries, qrels and the by-query split.

    python -m pytest experiments_anupam

The split is a contract: Chaitanya fits on training queries only and Shubham
splits the same way, so these tests are about determinism, disjointness and
completeness rather than about any particular partition.
"""

from __future__ import annotations

import json

import pytest

from experiments_anupam.data_prep import (
    SPLITS,
    build_qrels,
    check_gain_map_covers,
    load_queries,
    read_split,
    split_of,
    split_queries,
    validate_split,
    write_split,
)

TREC_DL_GAINS = {0: 0.0, 1: 0.0, 2: 3.0, 3: 7.0}


# ── queries ───────────────────────────────────────────────────────────

def test_load_queries(tmp_path):
    p = tmp_path / "queries.tsv"
    p.write_text("1\twhat is a cascade\n2\tsecond query here\n\n", encoding="utf-8")
    q = load_queries(p)
    assert q == {"1": "what is a cascade", "2": "second query here"}


def test_duplicate_query_id_is_an_error(tmp_path):
    """Last-one-wins would put the same query in two splits."""
    p = tmp_path / "queries.tsv"
    p.write_text("1\tfirst\n1\tsecond\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate query id"):
        load_queries(p)


def test_malformed_and_empty_lines_are_rejected(tmp_path):
    p = tmp_path / "queries.tsv"
    p.write_text("1 no tab here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected query_id"):
        load_queries(p)

    p.write_text("1\t\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty query text"):
        load_queries(p)


# ── qrels ─────────────────────────────────────────────────────────────

def test_build_qrels_round_trip(tmp_path):
    from shared.schema import load

    src = tmp_path / "qrels.txt"
    src.write_text("1 0 d1 3\n1 0 d2 0\n2 0 d3 2\n", encoding="utf-8")
    out = build_qrels(src, tmp_path / "qrels.parquet")
    df = load(out, "qrels")
    assert len(df) == 3
    assert bool(df["judged"].all())


def test_gain_map_must_cover_every_observed_grade(tmp_path):
    """
    An unmapped grade would otherwise raise inside `utility.gain` deep in a
    long run instead of here, in a second.
    """
    src = tmp_path / "qrels.txt"
    src.write_text("1 0 d1 3\n1 0 d2 4\n", encoding="utf-8")
    out = build_qrels(src, tmp_path / "qrels.parquet")
    with pytest.raises(ValueError, match=r"grades \[4\]"):
        check_gain_map_covers(out, TREC_DL_GAINS)


def test_gain_map_check_passes_on_trec_dl_grades(tmp_path):
    src = tmp_path / "qrels.txt"
    src.write_text("1 0 d1 0\n1 0 d2 1\n1 0 d3 2\n1 0 d4 3\n", encoding="utf-8")
    out = build_qrels(src, tmp_path / "qrels.parquet")
    check_gain_map_covers(out, TREC_DL_GAINS)


# ── the split ─────────────────────────────────────────────────────────

FRACTIONS = {"train": 0.5, "val": 0.2, "test": 0.3}


def test_split_is_deterministic_and_order_independent():
    ids = [f"q{i}" for i in range(97)]
    a = split_queries(ids, FRACTIONS, seed=42)
    b = split_queries(list(reversed(ids)), FRACTIONS, seed=42)
    assert a == b, "the split must not depend on the order the ids arrived in"


def test_split_changes_with_the_seed():
    ids = [f"q{i}" for i in range(97)]
    assert split_queries(ids, FRACTIONS, 42) != split_queries(ids, FRACTIONS, 43)


def test_split_is_disjoint_and_complete():
    ids = [f"q{i}" for i in range(97)]
    split = split_queries(ids, FRACTIONS, seed=42)
    validate_split(split, ids)
    assert sum(len(split[s]) for s in SPLITS) == 97
    assert set().union(*(set(split[s]) for s in SPLITS)) == set(ids)


@pytest.mark.parametrize("n", [10, 13, 97, 100, 101])
def test_sizes_sum_exactly_at_awkward_counts(n):
    """
    Largest-remainder allocation: no query dropped, none duplicated, whatever
    the rounding does. 97 is the real case.
    """
    ids = [f"q{i}" for i in range(n)]
    split = split_queries(ids, FRACTIONS, seed=7)
    assert sum(len(split[s]) for s in SPLITS) == n
    validate_split(split, ids)


def test_real_query_count_gives_sensible_sizes():
    ids = [f"q{i}" for i in range(97)]
    split = split_queries(ids, FRACTIONS, seed=42)
    sizes = {s: len(split[s]) for s in SPLITS}
    assert sizes["train"] >= sizes["test"] >= sizes["val"]
    assert all(v > 0 for v in sizes.values())


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_queries(["a", "b"], {"train": 0.5, "val": 0.2, "test": 0.1}, 1)


def test_missing_split_name_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        split_queries(["a", "b"], {"train": 0.5, "test": 0.5}, 1)


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate query ids"):
        split_queries(["a", "a", "b"], FRACTIONS, 1)


def test_validate_catches_a_leaking_query():
    ids = ["a", "b", "c", "d"]
    leaky = {"train": ["a", "b"], "val": ["b"], "test": ["c", "d"]}
    with pytest.raises(ValueError, match="more than one split"):
        validate_split(leaky, ids)


def test_validate_catches_an_incomplete_split():
    ids = ["a", "b", "c", "d"]
    short = {"train": ["a"], "val": ["b"], "test": ["c"]}
    with pytest.raises(ValueError, match="does not cover"):
        validate_split(short, ids)


def test_validate_catches_an_empty_split():
    ids = ["a", "b"]
    with pytest.raises(ValueError, match="is empty"):
        validate_split({"train": ["a", "b"], "val": [], "test": []}, ids)


# ── persistence ───────────────────────────────────────────────────────

def test_split_file_round_trip(tmp_path):
    ids = [f"q{i}" for i in range(97)]
    split = split_queries(ids, FRACTIONS, seed=42)
    path = write_split(tmp_path / "query_split.json", split, seed=42)
    assert read_split(path) == split

    payload = json.loads(path.read_text())
    assert payload["seed"] == 42
    assert payload["n_queries"] == 97
    assert payload["sizes"] == {s: len(split[s]) for s in SPLITS}


def test_split_of_maps_every_query():
    ids = [f"q{i}" for i in range(20)]
    split = split_queries(ids, FRACTIONS, seed=3)
    mapping = split_of(split)
    assert set(mapping) == set(ids)
    assert set(mapping.values()) <= set(SPLITS)


def test_committed_split_is_reproducible_from_the_file(tmp_path):
    """
    The file is committed AND the split is deterministic. This checks the two
    agree, which is what turns "should be identical" into "is identical".
    """
    ids = [f"q{i}" for i in range(97)]
    path = write_split(tmp_path / "s.json", split_queries(ids, FRACTIONS, 42), 42)
    assert read_split(path) == split_queries(ids, FRACTIONS, 42)
