"""
Universe, qrels, cache and the schema validators.

Every refusal in this file corresponds to a way the counterfactual frame
breaks silently - producing results that still run, still look coherent, and
mean nothing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from shared.cache import Stage2Cache
from shared.qrels import (
    attach_judgments,
    coverage_report,
    judged_flags,
    judged_non_survivor_counts,
    judged_subset,
    read_trec_qrels,
    relevance_lookup,
)
from shared.schema import (
    DELTA_T,
    E_GATE,
    NON_SURVIVOR,
    RETAINED,
    SURVIVOR,
    SchemaError,
    empty,
    load,
    save,
    validate,
    validate_effect,
    validate_qrels,
    validate_run,
    validate_stage2_cache,
    validate_universe,
)
from shared.universe import (
    UniverseError,
    build_universe,
    candidate_overlap,
    check_containment,
    freeze,
    load_frozen,
    read_reference_subset,
    sample_reference_subset,
    universe_sets,
    write_reference_subset,
)

# The frozen map for TREC Deep Learning passage ranking: grade 1 is "Related"
# and earns nothing. Spelled out here rather than read from the committed
# config so that editing the config cannot silently rewrite what these tests
# assert about grade 1.
TREC_DL_GAINS = {0: 0.0, 1: 0.0, 2: 3.0, 3: 7.0}


def _run(system, docs_by_query, checkpoint="v1"):
    rows = []
    for qid, docs in docs_by_query.items():
        for rank, doc in enumerate(docs, start=1):
            rows.append({"query_id": qid, "doc_id": doc, "system": system,
                         "rank": rank, "score": 10.0 - rank,
                         "checkpoint": checkpoint})
    return pd.DataFrame(rows)


@pytest.fixture
def two_runs():
    return {
        "bm25": _run("bm25", {"q1": list("abcd"), "q2": list("wx")}),
        "dpr": _run("dpr", {"q1": list("cdef"), "q2": list("xy")}),
    }


# ── run schema ────────────────────────────────────────────────────────

def test_run_dtypes_are_coerced(two_runs):
    out = validate_run(two_runs["bm25"])
    assert out["query_id"].dtype == "string"
    assert out["rank"].dtype == "Int64"


def test_run_rejects_zero_indexed_and_non_contiguous_ranks(two_runs):
    df = two_runs["bm25"].copy()
    df.loc[0, "rank"] = 0
    with pytest.raises(SchemaError, match="1-indexed"):
        validate_run(df)
    df = two_runs["bm25"].copy()
    df.loc[2, "rank"] = 99
    with pytest.raises(SchemaError, match="contiguous"):
        validate_run(df)


def test_run_rejects_duplicate_keys(two_runs):
    df = pd.concat([two_runs["bm25"], two_runs["bm25"]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate keys"):
        validate_run(df)


# ── universe ──────────────────────────────────────────────────────────

def test_union_is_broader_than_either_source(two_runs):
    sets = universe_sets(build_universe(two_runs, depth_m=4))
    assert set(sets["q1"]) == set("abcdef")
    assert set(sets["q2"]) == set("wxy")


def test_sources_are_recorded_per_document(two_runs):
    uni = build_universe(two_runs, depth_m=4).set_index(["query_id", "doc_id"])
    assert uni.loc[("q1", "c"), "sources"] == "bm25,dpr"
    assert uni.loc[("q1", "a"), "sources"] == "bm25"


def test_max_size_keeps_the_best_ranked(two_runs):
    sets = universe_sets(build_universe(two_runs, depth_m=4, max_size=3))
    assert len(sets["q1"]) == 3
    assert {"a", "c"}.issubset(set(sets["q1"]))


def test_empty_sources_is_an_error():
    with pytest.raises(UniverseError, match="no source runs"):
        build_universe({}, depth_m=10)


def test_frozen_universe_refuses_a_fingerprint_mismatch(two_runs, tmp_path):
    uni = build_universe(two_runs, depth_m=4)
    path = tmp_path / "universe.parquet"
    freeze(uni, path, "abc123456789")
    assert len(load_frozen(path, "abc123456789")) == len(uni)
    with pytest.raises(UniverseError, match="not comparable"):
        load_frozen(path, "deadbeefcafe")


def test_containment_catches_a_retriever_that_was_not_a_source(two_runs):
    uni = build_universe({"bm25": two_runs["bm25"]}, depth_m=4)
    check_containment(two_runs["bm25"], uni, 2, "bm25")
    with pytest.raises(UniverseError, match="outside the frozen universe"):
        check_containment(two_runs["dpr"], uni, 4, "dpr")


def test_candidate_overlap_reports_jaccard():
    a = {"q1": {"x", "y", "z"}}
    b = {"q1": {"y", "z", "w"}}
    out = candidate_overlap(a, b, "bm25", "dpr", 3)
    assert out.iloc[0]["jaccard"] == pytest.approx(2 / 4)
    assert out.iloc[0]["n_a_only"] == 1


# ── reference subset ──────────────────────────────────────────────────

def _wide_universe(n_queries=40):
    rows = []
    for q in range(n_queries):
        size = 10 + q * 3           # deliberately varied |U_q|
        for d in range(size):
            rows.append({"query_id": f"q{q:02d}", "doc_id": f"d{d:03d}",
                         "sources": "bm25"})
    return validate_universe(pd.DataFrame(rows))


def test_reference_sampling_is_reproducible():
    uni = _wide_universe()
    a = sample_reference_subset(uni, 10, "random", seed=3)
    b = sample_reference_subset(uni, 10, "random", seed=3)
    c = sample_reference_subset(uni, 10, "random", seed=4)
    assert a == b and a != c and len(a) == 10


def test_stratified_sampling_spans_the_size_range():
    """
    Convenience selection would over-sample small universes, which is exactly
    where the retrievers agree and gate effects are smallest. Stratifying
    guards against that.
    """
    uni = _wide_universe()
    sizes = uni.groupby("query_id", observed=True).size()
    chosen = sample_reference_subset(uni, 10, "stratified_universe_size",
                                     seed=1, n_strata=5)
    picked = sizes.loc[chosen]
    assert len(chosen) == 10
    assert picked.max() > sizes.median(), "the subset never reaches large universes"
    assert picked.min() < sizes.median(), "the subset never reaches small universes"


def test_requesting_everything_returns_everything():
    uni = _wide_universe(5)
    assert len(sample_reference_subset(uni, 99, "random")) == 5


def test_unknown_sampling_strategy_is_refused():
    with pytest.raises(ValueError, match="unknown reference-subset strategy"):
        sample_reference_subset(_wide_universe(), 5, "whatever_is_easiest")


def test_reference_subset_round_trip(tmp_path):
    ids = ["q01", "q05", "q09"]
    path = write_reference_subset(tmp_path / "ref.txt", ids)
    assert read_reference_subset(path) == ids


# ── qrels ─────────────────────────────────────────────────────────────

def test_trec_qrels_round_trip(tmp_path):
    p = tmp_path / "qrels.txt"
    p.write_text("q1 0 a 3\nq1 0 b 0\nq2 0 x 1\n", encoding="utf-8")
    qr = read_trec_qrels(p)
    assert len(qr) == 3 and bool(qr["judged"].all())


def test_malformed_qrels_line_is_reported(tmp_path):
    p = tmp_path / "qrels.txt"
    p.write_text("q1 0 a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 4"):
        read_trec_qrels(p)


def test_unjudged_universe_documents_get_na_not_zero(two_runs, tmp_path):
    uni = build_universe(two_runs, depth_m=4)
    p = tmp_path / "qrels.txt"
    p.write_text("q1 0 a 3\nq1 0 e 2\n", encoding="utf-8")
    attached = attach_judgments(uni, read_trec_qrels(p)).set_index(
        ["query_id", "doc_id"]
    )
    assert attached.loc[("q1", "a"), "relevance"] == 3
    assert bool(attached.loc[("q1", "b"), "judged"]) is False
    assert pd.isna(attached.loc[("q1", "b"), "relevance"])


def test_relevance_lookup_omits_unjudged_documents(two_runs, tmp_path):
    """
    Forcing callers to treat a missing key as unknown is what stops a default
    of zero slipping in.
    """
    uni = build_universe(two_runs, depth_m=4)
    p = tmp_path / "qrels.txt"
    p.write_text("q1 0 a 3\n", encoding="utf-8")
    attached = attach_judgments(uni, read_trec_qrels(p))
    lookup = relevance_lookup(attached)
    assert lookup[("q1", "a")] == 3
    assert ("q1", "b") not in lookup
    assert judged_flags(attached)[("q1", "b")] is False


def test_qrels_validator_refuses_a_graded_unjudged_row():
    df = pd.DataFrame(
        {
            "query_id": ["q1"], "doc_id": ["d1"],
            "relevance": pd.array([0], dtype="Int64"),
            "judged": pd.array([False], dtype="boolean"),
        }
    )
    with pytest.raises(SchemaError, match="Never silently convert"):
        validate_qrels(df)


def test_coverage_and_non_survivor_counts(two_runs, tmp_path):
    uni = build_universe(two_runs, depth_m=4)
    p = tmp_path / "qrels.txt"
    p.write_text("q1 0 a 3\nq1 0 e 2\nq1 0 f 0\n", encoding="utf-8")
    attached = attach_judgments(uni, read_trec_qrels(p))

    cov = coverage_report(uni, attached, TREC_DL_GAINS).set_index("query_id")
    assert cov.loc["q1", "n_judged"] == 3
    assert cov.loc["q1", "n_gainful"] == 2

    counts = judged_non_survivor_counts(
        attached, {"q1": {"a", "b"}}, "bm25", 2, TREC_DL_GAINS
    ).set_index("query_id")
    assert counts.loc["q1", "n_judged_non_survivors"] == 2    # e and f
    assert counts.loc["q1", "n_gainful_non_survivors"] == 1   # e only
    assert len(judged_subset(attached)) == 3


def test_grade_one_is_judged_but_not_a_gate_failure(two_runs, tmp_path):
    """
    The TREC-DL trap, and the reason both functions take a gain map.

    Grade 1 is "Related": on topic, does not answer the query. The frozen map
    scores it 0.0, so a grade-1 document contributes nothing wherever it is
    ranked and excluding one costs no utility - it is judged, but it is not a
    gate failure. Counting `relevance > 0` instead overstated the gate-failure
    supply by 4x to 9x on the real collection, and the headline K_c was about
    to be chosen from that number.
    """
    uni = build_universe(two_runs, depth_m=4)
    p = tmp_path / "qrels.txt"
    p.write_text("q1 0 a 3\nq1 0 e 1\nq1 0 f 1\n", encoding="utf-8")
    attached = attach_judgments(uni, read_trec_qrels(p))

    cov = coverage_report(uni, attached, TREC_DL_GAINS).set_index("query_id")
    assert cov.loc["q1", "n_judged"] == 3, "all three were judged"
    assert cov.loc["q1", "n_gainful"] == 1, "only grade 3 carries gain"

    counts = judged_non_survivor_counts(
        attached, {"q1": {"a", "b"}}, "bm25", 2, TREC_DL_GAINS
    ).set_index("query_id")
    assert counts.loc["q1", "n_judged_non_survivors"] == 2
    assert counts.loc["q1", "n_gainful_non_survivors"] == 0, (
        "excluding two grade-1 documents is not a gate failure"
    )

    # A collection whose grade 1 really does mean relevant gets the other answer
    from shared.utility import EXPONENTIAL_GAIN
    counts2 = judged_non_survivor_counts(
        attached, {"q1": {"a", "b"}}, "bm25", 2, EXPONENTIAL_GAIN
    ).set_index("query_id")
    assert counts2.loc["q1", "n_gainful_non_survivors"] == 2


def test_an_unmapped_grade_raises_rather_than_scoring_zero(two_runs, tmp_path):
    """
    A grade absent from the map must not fall through to "not relevant": that
    would silently shrink the gate-failure supply for a whole collection.
    """
    uni = build_universe(two_runs, depth_m=4)
    p = tmp_path / "qrels.txt"
    p.write_text("q1 0 a 4\n", encoding="utf-8")
    attached = attach_judgments(uni, read_trec_qrels(p))

    with pytest.raises(KeyError, match="not in the frozen gain map"):
        coverage_report(uni, attached, TREC_DL_GAINS)


# ── cache ─────────────────────────────────────────────────────────────

def test_cache_round_trip_and_resume(tmp_path):
    cache = Stage2Cache(tmp_path / "s2.sqlite")
    cache.put_many("minilm", [("q1", "a", 0.5), ("q1", "b", 0.25)])
    assert cache.get_query("q1", "minilm") == {"a": 0.5, "b": 0.25}
    assert cache.missing("q1", ["a", "b", "c"], "minilm") == ["c"]
    assert cache.count("minilm") == 2 and cache.models() == ["minilm"]
    assert cache.queries("minilm") == ["q1"]


def test_cache_keys_on_model_so_rerankers_cannot_collide(tmp_path):
    cache = Stage2Cache(tmp_path / "s2.sqlite")
    cache.put_many("minilm", [("q1", "a", 0.5)])
    cache.put_many("monot5", [("q1", "a", -2.0)])
    assert cache.get_query("q1", "minilm")["a"] == 0.5
    assert cache.get_query("q1", "monot5")["a"] == -2.0


def test_cache_writes_are_idempotent(tmp_path):
    cache = Stage2Cache(tmp_path / "s2.sqlite")
    cache.put_many("minilm", [("q1", "a", 0.5)])
    cache.put_many("minilm", [("q1", "a", 0.9)])
    assert cache.count("minilm") == 1
    assert cache.get_query("q1", "minilm")["a"] == 0.9


def test_shard_merge_is_idempotent(tmp_path):
    """Three members fill disjoint query shards, then merge."""
    main = Stage2Cache(tmp_path / "main.sqlite")
    s0 = Stage2Cache(tmp_path / "s0.sqlite")
    s1 = Stage2Cache(tmp_path / "s1.sqlite")
    s0.put_many("m", [("q0", "a", 1.0), ("q0", "b", 2.0)])
    s1.put_many("m", [("q1", "a", 3.0)])

    main.merge_from(s0, "m")
    main.merge_from(s1, "m")
    main.merge_from(s1, "m")            # re-merge changes nothing
    assert main.count("m") == 3
    assert main.get_query("q0", "m") == {"a": 1.0, "b": 2.0}


def test_incomplete_cache_blocks_the_no_gate_world(two_runs, tmp_path):
    uni = build_universe(two_runs, depth_m=4)
    cache = Stage2Cache(tmp_path / "s2.sqlite")
    cache.put_many("minilm", [("q1", "a", 0.5)])

    rep = cache.coverage_report(uni, "minilm").set_index("query_id")
    assert rep.loc["q1", "n_missing"] == 5
    with pytest.raises(RuntimeError, match="no-gate world requires"):
        cache.require_complete(uni, "minilm")


def test_completeness_can_be_checked_on_the_reference_subset_only(two_runs, tmp_path):
    uni = build_universe(two_runs, depth_m=4)
    cache = Stage2Cache(tmp_path / "s2.sqlite")
    for d in universe_sets(uni)["q2"]:
        cache.put_many("minilm", [("q2", d, 0.1)])
    cache.require_complete(uni, "minilm", query_ids=["q2"])   # must not raise
    with pytest.raises(RuntimeError):
        cache.require_complete(uni, "minilm")


def test_cache_export_is_schema_compliant(tmp_path):
    cache = Stage2Cache(tmp_path / "s2.sqlite")
    cache.put_many("minilm", [("q1", "a", 0.5)])
    validate_stage2_cache(cache.to_frame("minilm"))


# ── effect schema edges ───────────────────────────────────────────────

def _effect_row(**over):
    row = {
        "query_id": "q1", "doc_id": "d1", "stage1_system": "bm25",
        "effect_type": E_GATE, "k_c": 100, "k": 10, "utility": "dcg",
        "budget_b": None, "t": None, "policy": None, "seed": None,
        "stratum": SURVIVOR, "judged": True,
        "rank_treatment": 1, "rank_control": 3,
        "utility_treatment": 7.0, "utility_control": 3.5, "effect": 3.5,
    }
    row.update(over)
    return pd.DataFrame([row])


def test_a_consistent_gate_row_validates():
    validate_effect(_effect_row())


def test_unknown_effect_type_is_rejected():
    with pytest.raises(SchemaError, match="unknown"):
        validate_effect(_effect_row(effect_type="vibes"))


def test_delta_t_row_without_its_budget_is_rejected():
    with pytest.raises(SchemaError, match="budget"):
        validate_effect(
            _effect_row(effect_type=DELTA_T, stratum=RETAINED, budget_b=None,
                        t=None)
        )


def test_delta_t_row_with_a_foreign_stratum_is_rejected():
    with pytest.raises(SchemaError, match="unknown strata"):
        validate_effect(
            _effect_row(effect_type=DELTA_T, stratum=SURVIVOR, budget_b=10, t=0.5)
        )


def test_parquet_round_trip_keeps_na(tmp_path):
    df = _effect_row(
        stratum=NON_SURVIVOR, rank_treatment=None, utility_treatment=0.0,
        utility_control=2.0, effect=-2.0,
    )
    path = tmp_path / "effect.parquet"
    save(df, path, "effect")
    back = load(path, "effect")
    assert pd.isna(back.loc[0, "rank_treatment"])
    assert back.loc[0, "effect"] == pytest.approx(-2.0)


@pytest.mark.parametrize(
    "kind", ["run", "universe", "qrels", "stage2_cache", "effect", "feature"]
)
def test_empty_tables_are_typed_and_validate(kind):
    df = empty(kind)
    assert df.empty
    validate(df, kind)


def test_load_names_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="qrels"):
        load(tmp_path / "nope.parquet", "qrels")
