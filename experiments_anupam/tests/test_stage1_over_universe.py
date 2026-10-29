"""
test_stage1_over_universe.py - Extending Stage-1 to U_q, and the four checks.

The checks are the point of the module, so most of these tests feed each one a
case built to break it. The top-K_c agreement check gets the most attention:
it is the invariant that decides whether C_q reconstructed from the universe is
the candidate set the cascade actually used, and it fails quietly rather than
loudly if the tie-break drifts.
"""

from __future__ import annotations

import json

import pytest

from experiments_anupam import build_universe as bu
from experiments_anupam import stage1_over_universe as s1u
from shared.config import load_config
from shared.schema import load, save, validate_run
from shared.universe import load_frozen, universe_sets


def _universe(two_source_runs):
    uni: dict[str, set[str]] = {}
    for run in two_source_runs.values():
        for qid, grp in run.groupby("query_id", observed=True):
            uni.setdefault(str(qid), set()).update(grp["doc_id"].astype(str))
    return {q: sorted(d) for q, d in uni.items()}


def _table(two_source_runs, system="bm25"):
    return s1u.build_table(
        two_source_runs[system], system, _universe(two_source_runs),
        "test|cp", "doc_id",
    )


# ── the table ─────────────────────────────────────────────────────────

def test_table_is_a_valid_run_over_the_whole_universe(two_source_runs):
    uni = _universe(two_source_runs)
    table = _table(two_source_runs)
    validate_run(table)
    assert len(table) == sum(len(v) for v in uni.values())


def test_ranks_are_contiguous_per_query(two_source_runs):
    """`validate_run` demands it, and a gap would make rank-within-U_q
    ambiguous exactly where the outsider policies read it."""
    table = _table(two_source_runs)
    for _, grp in table.groupby("query_id", observed=True):
        assert sorted(grp["rank"]) == list(range(1, len(grp) + 1))


def test_retrieved_documents_keep_their_original_order(two_source_runs):
    table = _table(two_source_runs)
    src = two_source_runs["bm25"]
    for qid, grp in src.groupby("query_id", observed=True):
        want = list(grp.sort_values("rank")["doc_id"].astype(str))
        got = list(
            table[table["query_id"] == str(qid)].sort_values("rank")["doc_id"]
        )[: len(want)]
        assert got == want


def test_floored_documents_come_last_and_tie(two_source_runs):
    uni = _universe(two_source_runs)
    table = _table(two_source_runs)
    src = two_source_runs["bm25"]
    for qid, grp in table.groupby("query_id", observed=True):
        retrieved = set(src[src["query_id"] == str(qid)]["doc_id"].astype(str))
        floored = grp[~grp["doc_id"].astype(str).isin(retrieved)]
        assert len(floored) == len(uni[str(qid)]) - len(retrieved)
        assert floored["score"].nunique() == 1
        assert floored["rank"].min() > len(retrieved)


def test_checkpoint_is_carried_through(two_source_runs):
    table = _table(two_source_runs)
    assert set(table["checkpoint"]) == {"test|cp"}


# ── check 1: top-K_c agreement ────────────────────────────────────────

def test_top_k_agreement_holds_for_a_well_formed_table(two_source_runs):
    table = _table(two_source_runs)
    assert s1u.check_top_k_agreement(
        table, two_source_runs["bm25"], "bm25", [5, 10, 20]
    ) == []


def test_top_k_disagreement_is_caught(two_source_runs):
    """
    The failure this check exists for. Here the reconstructed table swaps two
    documents across the K_c=5 boundary - which is what a drifting tie-break
    would do - so C_q from U_q is no longer C_q from the run.
    """
    table = _table(two_source_runs).copy()
    q = sorted(set(table["query_id"]))[0]
    sel = table["query_id"] == q
    ranks = table.loc[sel, "rank"]
    i5 = ranks[ranks == 5].index[0]
    i6 = ranks[ranks == 6].index[0]
    table.loc[[i5, i6], "rank"] = [6, 5]

    problems = s1u.check_top_k_agreement(
        table, two_source_runs["bm25"], "bm25", [5, 10, 20]
    )
    assert any("K_c=5" in p for p in problems)
    assert any("never used" in p or "cascade used" in p for p in problems)


def test_agreement_is_checked_at_every_configured_depth(two_source_runs):
    """A swap deep in the ranking must not pass merely because K_c=5 is fine."""
    table = _table(two_source_runs).copy()
    q = sorted(set(table["query_id"]))[0]
    sel = table["query_id"] == q
    ranks = table.loc[sel, "rank"]
    a = ranks[ranks == 18].index[0]
    b = ranks[ranks == 25].index[0]
    table.loc[[a, b], "rank"] = [25, 18]

    problems = s1u.check_top_k_agreement(
        table, two_source_runs["bm25"], "bm25", [5, 10, 20]
    )
    assert problems
    assert not any("K_c=5" in p for p in problems)
    assert any("K_c=20" in p for p in problems)


# ── check 2: coverage ─────────────────────────────────────────────────

def test_coverage_check_passes_on_the_real_table(two_source_runs):
    table = _table(two_source_runs)
    assert s1u.check_covers_universe(table, _universe(two_source_runs), "bm25") == []


def test_a_missing_document_is_caught(two_source_runs):
    uni = _universe(two_source_runs)
    table = _table(two_source_runs)
    assert s1u.check_covers_universe(table.iloc[1:], uni, "bm25")


def test_a_missing_query_is_caught(two_source_runs):
    uni = _universe(two_source_runs)
    table = _table(two_source_runs)
    q = sorted(set(table["query_id"]))[0]
    problems = s1u.check_covers_universe(
        table[table["query_id"] != q], uni, "bm25"
    )
    assert any("queries against" in p for p in problems)


# ── check 3: floor ordering ───────────────────────────────────────────

def test_floor_ordering_holds(two_source_runs):
    table = _table(two_source_runs)
    assert s1u.check_floor_ordering(table, two_source_runs["bm25"], "bm25") == []


def test_a_floored_document_outranking_a_retrieved_one_is_caught(two_source_runs):
    """
    Not reachable through `build_table`, but it is the one thing the floor
    policy promises, so the check is kept honest with a hand-built violation.
    """
    table = _table(two_source_runs).copy()
    src = two_source_runs["bm25"]
    q = sorted(set(table["query_id"]))[0]
    retrieved = set(src[src["query_id"] == q]["doc_id"].astype(str))
    sel = (table["query_id"] == q) & (~table["doc_id"].astype(str).isin(retrieved))
    table.loc[sel, "score"] = 1e9

    assert s1u.check_floor_ordering(table, src, "bm25")


# ── check 4: outsider-policy supply ───────────────────────────────────

def test_policy_supply_is_reported_per_depth(two_source_runs):
    table = _table(two_source_runs)
    problems, stats = s1u.check_policy_supply(
        table, two_source_runs["bm25"], "bm25",
        two_source_runs["dpr"], "dpr", [5, 10],
    )
    assert set(stats) == {"k_c=5", "k_c=10"}
    for v in stats.values():
        assert v["boundary_ranked_supply_min"] >= 0
        assert v["alternative_supply_min"] is not None


def test_a_budget_beyond_the_ranked_depth_is_flagged(two_source_runs):
    """
    The point of this check. At K_c=35 against a depth-40 run only 5 genuinely
    ranked documents sit below the cutoff, so a budget of 35 would force the
    boundary policy to draw from the floored block - where every score is tied
    and the selection is the tie-break, not the policy.
    """
    table = _table(two_source_runs)
    problems, stats = s1u.check_policy_supply(
        table, two_source_runs["bm25"], "bm25",
        two_source_runs["dpr"], "dpr", [35],
    )
    assert stats["k_c=35"]["boundary_ranked_supply_min"] == 5
    assert any("floored block" in p for p in problems)


def test_no_other_system_leaves_alternative_supply_unreported(two_source_runs):
    table = _table(two_source_runs)
    _, stats = s1u.check_policy_supply(
        table, two_source_runs["bm25"], "bm25", None, None, [5],
    )
    assert stats["k_c=5"]["alternative_supply_min"] is None


# ── end to end ────────────────────────────────────────────────────────

def _setup(tmp_path, small_cfg_path, two_source_runs):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for s, df in two_source_runs.items():
        save(df, runs_dir / f"{s}.parquet", "run")
    uni = bu.run(small_cfg_path, str(runs_dir),
                 str(tmp_path / "universe.parquet"), None, False)
    return str(runs_dir), str(uni)


def test_end_to_end_writes_tables_report_and_manifest(
    tmp_path, small_cfg_path, two_source_runs, monkeypatch
):
    runs_dir, uni = _setup(tmp_path, small_cfg_path, two_source_runs)
    monkeypatch.chdir(tmp_path)          # so the analysis/ report lands here

    extra, problems = s1u.run(small_cfg_path, runs_dir, uni, str(tmp_path / "out"))
    assert not problems, problems

    out = tmp_path / "out"
    for system in ("bm25", "dpr"):
        assert (out / f"stage1_over_universe_{system}.parquet").exists()
        assert (tmp_path / "analysis" / f"stage1_floor_{system}.parquet").exists()

    manifest = json.loads(
        (out / "stage1_over_universe_manifest.json").read_text()
    )
    assert manifest["owner"] == "anupam"
    assert set(manifest["systems"]) == {"bm25", "dpr"}
    assert 0.0 < manifest["systems"]["bm25"]["floored_fraction_median"] < 1.0


def test_the_written_table_replays_as_a_run(
    tmp_path, small_cfg_path, small_cfg, two_source_runs, monkeypatch
):
    """
    It must be interchangeable with a real run table, because downstream code
    reads it through the same loader.
    """
    from experiments_anupam.retrievers import PrecomputedRun

    runs_dir, uni = _setup(tmp_path, small_cfg_path, two_source_runs)
    monkeypatch.chdir(tmp_path)
    s1u.run(small_cfg_path, runs_dir, uni, str(tmp_path / "out"))

    table = load(tmp_path / "out" / "stage1_over_universe_bm25.parquet", "run")
    r = PrecomputedRun(table, name="bm25")
    sizes = universe_sets(load_frozen(uni, small_cfg.fingerprint()))
    q = sorted(sizes)[0]
    assert len(r.search(q, "unused", 10 ** 6)) == len(sizes[q])


def test_a_run_with_mixed_checkpoints_is_refused(
    tmp_path, small_cfg_path, two_source_runs
):
    """
    A table that cannot say which checkpoint produced it must not be extended
    over the universe - the extension would inherit the ambiguity.
    """
    runs_dir, uni = _setup(tmp_path, small_cfg_path, two_source_runs)

    # Half the run re-scored under a second checkpoint, which is what a
    # partially regenerated run actually looks like.
    mixed = load(f"{runs_dir}/bm25.parquet", "run").copy()
    half = mixed.index[: len(mixed) // 2]
    mixed.loc[half, "checkpoint"] = "pyserini:test|bm25-k1=1.2-b=0.75"
    save(mixed, f"{runs_dir}/bm25.parquet", "run")

    with pytest.raises(SystemExit, match="2 checkpoints"):
        s1u.run(small_cfg_path, runs_dir, uni, str(tmp_path / "out"))


def test_a_fingerprint_mismatch_refuses_before_any_work(
    tmp_path, small_cfg_path, two_source_runs
):
    import yaml

    from shared.universe import UniverseError

    runs_dir, uni = _setup(tmp_path, small_cfg_path, two_source_runs)
    with open(small_cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["seed"] = raw["seed"] + 1          # any change moves the fingerprint
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump(raw))
    assert load_config(other).fingerprint() != load_config(small_cfg_path).fingerprint()

    with pytest.raises(UniverseError, match="not comparable"):
        s1u.run(str(other), runs_dir, uni, str(tmp_path / "out"))
