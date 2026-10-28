"""
test_feasibility_report.py - The Week-2 gate, and the three findings it exists
to catch.

Each test below builds a universe whose judgments are placed to trigger exactly
one finding, because the value of this report is entirely in whether it fires
when it should and stays quiet when it should not. A report that warns about
everything is as useless as one that warns about nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments_anupam import build_universe as bu
from experiments_anupam import feasibility_report as fr
from experiments_anupam.tests.conftest import make_run
from shared.config import load_config
from shared.schema import save

GAINS = {0: 0.0, 1: 0.0, 2: 3.0, 3: 7.0}


# ── fixtures: a frozen universe plus qrels we control precisely ───────

def _cfg(tmp_path, small_cfg_path, **overrides):
    with open(small_cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for section, kv in overrides.items():
        raw[section].update(kv)
    p = tmp_path / "fcfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return str(p)


def _runs(n_queries=8, depth=40, overlap=20, n_docs=120):
    """Two systems sharing `overlap` of their top-`depth` per query."""
    a, b = {}, {}
    for i in range(n_queries):
        q = f"q{i}"
        a[q] = [f"d{(i * 7 + j) % n_docs:03d}" for j in range(depth)]
        b[q] = [f"d{(i * 7 + j + (depth - overlap)) % n_docs:03d}"
                for j in range(depth)]
    return {"bm25": make_run("bm25", a, "pyserini:test|bm25-k1=0.9-b=0.4"),
            "dpr": make_run("dpr", b, "fake/dpr@abc123|index:test")}


def _write(tmp_path, cfg_path, runs, qrels_rows):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(exist_ok=True)
    for s, df in runs.items():
        save(df, runs_dir / f"{s}.parquet", "run")
    uni = bu.run(cfg_path, str(runs_dir), str(tmp_path / "universe.parquet"),
                 None, False)
    # A document the two runs share gets judged twice when both are passed to
    # `_judge`; the qrels schema refuses duplicate (query_id, doc_id) keys, and
    # rightly so - one document has one judgment.
    df = pd.DataFrame(qrels_rows).drop_duplicates(
        subset=["query_id", "doc_id"], keep="first")
    qrels = save(df, tmp_path / "qrels.parquet", "qrels")
    return str(runs_dir), str(uni), str(qrels)


def _judge(runs, system, grades_by_rank, n_queries=8):
    """
    Judge documents by their rank in one system's run.

    `grades_by_rank` maps a 0-indexed rank to a grade, which is how judgment
    placement is controlled: putting every grade inside the top-5 makes C_q
    dense and U_q sparse, which is the real TREC pooling pattern.
    """
    run = runs[system]
    rows = []
    for i in range(n_queries):
        q = f"q{i}"
        docs = list(run[run["query_id"] == q].sort_values("rank")["doc_id"])
        for rank, grade in grades_by_rank.items():
            if rank < len(docs):
                rows.append({"query_id": q, "doc_id": docs[rank],
                             "relevance": grade, "judged": True})
    return rows


# ── the happy path ────────────────────────────────────────────────────

def test_clean_case_reports_no_findings(tmp_path, small_cfg_path):
    """
    Judgments spread through the universe, plenty of gain-carrying documents
    outside the candidate set, low candidate overlap. Nothing should fire.
    """
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [20]})
    runs = _runs(overlap=20)
    # Judged at every other rank, so C_q and U_q are judged to similar depth
    # and both systems keep gain-carrying documents outside their top-20.
    rows = _judge(runs, "bm25", {i: (3 if i % 4 else 2) for i in range(0, 40, 2)})
    rows += _judge(runs, "dpr", {i: (3 if i % 4 else 2) for i in range(0, 40, 2)})
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)

    summary, found = fr.run(cfg_path, runs_dir, uni, qrels, str(tmp_path / "out"))
    assert summary["headline_k_c"] == 20
    assert summary["by_k_c"][20]["usable_for_headline_gate"]
    assert not found, list(found)


def test_outputs_and_manifest_are_written(tmp_path, small_cfg_path):
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    runs = _runs(overlap=4)
    rows = _judge(runs, "bm25", {0: 3, 20: 3, 30: 3})
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)
    out = tmp_path / "out"

    summary, _ = fr.run(cfg_path, runs_dir, uni, qrels, str(out))
    for name in ("feasibility_overlap", "feasibility_density",
                 "feasibility_non_survivors"):
        assert (out / f"{name}.parquet").exists(), name

    manifest = json.loads((out / "feasibility_manifest.json").read_text())
    assert manifest["owner"] == "anupam"
    assert manifest["total_universe_pairs"] == summary["total_universe_pairs"]
    assert "headline_k_c_rule" in manifest


def test_the_stage2_ceiling_is_the_universe_not_a_multiple_of_it(
    tmp_path, small_cfg_path
):
    """
    The number the whole cost argument rests on. Every world re-ranks a subset
    of U_q and the cache keys on (query, doc), so the bill is |U_q| summed -
    once - not once per world, per K_c or per t.
    """
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5, 10]})
    runs = _runs(overlap=4)
    rows = _judge(runs, "bm25", {0: 3, 20: 3, 30: 3})
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)

    summary, _ = fr.run(cfg_path, runs_dir, uni, qrels, str(tmp_path / "out"))
    from shared.universe import load_frozen, universe_sizes
    cfg = load_config(cfg_path)
    expected = int(universe_sizes(load_frozen(uni, cfg.fingerprint())).sum())
    assert summary["total_universe_pairs"] == expected


# ── finding 1: no leverage ────────────────────────────────────────────

def test_identical_candidate_sets_are_flagged_as_no_leverage(
    tmp_path, small_cfg_path
):
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    runs = _runs(overlap=40)                    # the two runs are identical
    rows = _judge(runs, "bm25", {0: 3, 20: 3, 30: 3})
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)

    summary, found = fr.run(cfg_path, runs_dir, uni, qrels, str(tmp_path / "out"))
    assert summary["by_k_c"][5]["median_candidate_jaccard"] == pytest.approx(1.0)
    assert "no_leverage" in found


def test_disjoint_candidate_sets_are_not_flagged(tmp_path, small_cfg_path):
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    runs = _runs(overlap=0)
    rows = _judge(runs, "bm25", {0: 3, 20: 3, 30: 3})
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)

    _, found = fr.run(cfg_path, runs_dir, uni, qrels, str(tmp_path / "out"))
    assert "no_leverage" not in found


# ── finding 2: the judgment-placement bias on E_gate ──────────────────

def test_judgments_concentrated_in_the_head_are_flagged(tmp_path, small_cfg_path):
    """
    The real TREC pooling pattern: everything judged sits in the top few ranks,
    so C_q is densely judged and the rest of U_q is not. World C promotes from
    the sparse region and earns zero for it, which inflates E_gate.
    """
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    runs = _runs(overlap=4)
    rows = _judge(runs, "bm25", {0: 3, 1: 2, 2: 3, 3: 1, 4: 2})   # all inside K_c=5
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)

    summary, found = fr.run(cfg_path, runs_dir, uni, qrels, str(tmp_path / "out"))
    bm25 = summary["by_k_c"][5]["by_system"]["bm25"]
    assert bm25["median_density_ratio"] > fr.DENSITY_GAP_LIMIT
    assert "judgment_placement" in found
    assert any("bm25" in w for w in found["judgment_placement"]["where"])


def test_evenly_spread_judgments_are_not_flagged(tmp_path, small_cfg_path):
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [20]})
    runs = _runs(overlap=20)
    rows = _judge(runs, "bm25", {i: (3 if i % 2 else 2) for i in range(0, 40, 2)})
    rows += _judge(runs, "dpr", {i: (3 if i % 2 else 2) for i in range(0, 40, 2)})
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)

    _, found = fr.run(cfg_path, runs_dir, uni, qrels, str(tmp_path / "out"))
    assert "judgment_placement" not in found


# ── finding 3: no gate-failure supply ─────────────────────────────────

def test_no_gainful_non_survivors_is_flagged(tmp_path, small_cfg_path):
    """
    Every gain-carrying document is already inside C_q, so nothing the gate
    excluded could have contributed utility. There is no gate failure to
    measure at this depth however large the effects look.
    """
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [20]})
    runs = _runs(overlap=4)
    rows = _judge(runs, "bm25", {0: 3, 1: 3, 2: 2})       # all inside K_c=20
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)

    summary, found = fr.run(cfg_path, runs_dir, uni, qrels, str(tmp_path / "out"))
    assert summary["by_k_c"][20]["by_system"]["bm25"][
        "median_gainful_non_survivors"] == 0
    assert not summary["by_k_c"][20]["usable_for_headline_gate"]
    assert "gate_failure_supply" in found


def test_headline_k_c_picks_the_largest_usable_depth(tmp_path, small_cfg_path):
    """
    The declared rule. Judgments at rank 30 are outside C_q at K_c=5 and 10 but
    inside it at 35, so the two shallower depths are usable and the deepest is
    not - the rule must pick 10, not 35.
    """
    cfg_path = _cfg(tmp_path, small_cfg_path,
                    cascade={"candidate_depths": [5, 10, 35]})
    runs = _runs(overlap=4)
    rows = _judge(runs, "bm25", {0: 3, 30: 3, 31: 3})
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)

    summary, _ = fr.run(cfg_path, runs_dir, uni, qrels, str(tmp_path / "out"))
    assert summary["by_k_c"][5]["usable_for_headline_gate"]
    assert summary["by_k_c"][10]["usable_for_headline_gate"]
    assert not summary["by_k_c"][35]["usable_for_headline_gate"]
    assert summary["headline_k_c"] == 10


def test_no_usable_depth_at_all_is_stated_plainly(tmp_path, small_cfg_path):
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [20]})
    runs = _runs(overlap=4)
    rows = _judge(runs, "bm25", {0: 3, 1: 2})
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)

    summary, found = fr.run(cfg_path, runs_dir, uni, qrels, str(tmp_path / "out"))
    assert summary["headline_k_c"] is None
    assert "no_usable_depth" in found


# ── the gain-map vs shared-definition gap ─────────────────────────────

def test_grade_one_documents_are_not_counted_as_gate_failures(
    tmp_path, small_cfg_path
):
    """
    Grade-1 non-survivors must not register as gate failures. `shared.qrels`
    now applies the gain map itself, so with grade 1 the only thing outside
    C_q there is no supply at all - and the report must say so rather than
    reporting three excluded documents.

    The companion test, that `shared.qrels` agrees, is
    `tests/test_data_layer.py::test_grade_one_is_judged_but_not_a_gate_failure`.
    """
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [20]})
    runs = _runs(overlap=20)
    # Judged by bm25's ranks only: grade 3 inside its C_q at K_c=20, and the
    # three documents outside it are all grade 1. (With overlap=20, dpr's head
    # starts at bm25's rank 20, so judging dpr's head too would put grade-3
    # documents outside bm25's candidate set and defeat the point.)
    rows = _judge(runs, "bm25", {0: 3, 1: 3, 25: 1, 26: 1, 30: 1})
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)

    summary, found = fr.run(cfg_path, runs_dir, uni, qrels, str(tmp_path / "out"))
    bm25 = summary["by_k_c"][20]["by_system"]["bm25"]
    assert bm25["median_gainful_non_survivors"] == 0, (
        "three judged documents sit outside C_q, but all are grade 1"
    )
    assert not summary["by_k_c"][20]["usable_for_headline_gate"]
    assert "gate_failure_supply" in found
    assert any("bm25" in w for w in found["gate_failure_supply"]["where"])


def test_findings_are_grouped_by_kind_not_repeated_per_depth(
    tmp_path, small_cfg_path
):
    """
    The first version emitted one message per (K_c, system), turning two
    distinct problems into twelve lines of near-identical prose. One entry per
    kind, listing everywhere it fires.
    """
    cfg_path = _cfg(tmp_path, small_cfg_path,
                    cascade={"candidate_depths": [5, 10, 20]})
    runs = _runs(overlap=4)
    rows = _judge(runs, "bm25", {0: 3, 1: 2, 2: 3})     # head-only judgments
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)

    _, found = fr.run(cfg_path, runs_dir, uni, qrels, str(tmp_path / "out"))
    assert found, "this fixture should trip at least one finding"
    for kind, v in found.items():
        assert isinstance(v["where"], list) and v["where"], kind
        assert len(v["explain"]) > 40, f"{kind} needs a real explanation"
    # 3 depths x 2 systems would be 6+ messages ungrouped; kinds are far fewer
    assert len(found) <= 4


def test_relevant_doc_sets_follows_the_configured_gain_map(tmp_path):
    """
    Not a hardcoded threshold: change the gain map and the definition of
    relevant changes with it.
    """
    qrels = pd.DataFrame([
        {"query_id": "q", "doc_id": "a", "relevance": 1, "judged": True},
        {"query_id": "q", "doc_id": "b", "relevance": 2, "judged": True},
    ])
    assert fr.relevant_doc_sets(qrels, GAINS)["q"] == {"b"}
    assert fr.relevant_doc_sets(qrels, {0: 0.0, 1: 1.0, 2: 3.0, 3: 7.0})["q"] == \
        {"a", "b"}


def test_unjudged_rows_are_excluded(tmp_path):
    qrels = pd.DataFrame([
        {"query_id": "q", "doc_id": "a", "relevance": 3, "judged": False},
        {"query_id": "q", "doc_id": "b", "relevance": 3, "judged": True},
    ])
    assert fr.relevant_doc_sets(qrels, GAINS)["q"] == {"b"}
    assert fr.judged_doc_sets(qrels)["q"] == {"b"}


# ── plumbing ──────────────────────────────────────────────────────────

def test_candidate_sets_truncate_at_k_c(two_source_runs):
    cands = fr.candidate_sets(two_source_runs["bm25"], "bm25", 5)
    assert all(len(v) == 5 for v in cands.values())


def test_unknown_system_is_refused(two_source_runs):
    with pytest.raises(SystemExit, match="no rows for system"):
        fr.candidate_sets(two_source_runs["bm25"], "splade", 5)


def test_density_ratio_survives_a_zero_coverage_query(tmp_path, small_cfg_path):
    """
    A query with no judgments at all must give nan rather than dividing by
    zero and poisoning the median for every other query.
    """
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    runs = _runs(overlap=4)
    rows = _judge(runs, "bm25", {0: 3, 20: 3, 30: 3}, n_queries=4)   # q4..q7 unjudged
    runs_dir, uni, qrels = _write(tmp_path, cfg_path, runs, rows)

    summary, _ = fr.run(cfg_path, runs_dir, uni, qrels, str(tmp_path / "out"))
    assert np.isfinite(
        summary["by_k_c"][5]["by_system"]["bm25"]["median_density_ratio"])


# ── the headline-K_c selection rule ───────────────────────────────────

def _fake_summary(supply_by_k: dict[int, dict[str, float]]) -> dict:
    """A by_k_c summary carrying only what the selection rule reads."""
    return {
        k: {
            "by_system": {
                s: {"median_gainful_non_survivors": v} for s, v in sys_supply.items()
            },
            "usable_for_headline_gate": all(
                v >= fr.MIN_GATE_FAILURES for v in sys_supply.values()
            ),
        }
        for k, sys_supply in supply_by_k.items()
    }


def test_headline_prefers_supply_not_depth():
    """
    The regression guard for the rule I got wrong first time.

    Supply falls as K_c grows, because a deeper candidate set swallows more of
    the judged pool. A rule that takes the LARGEST usable depth therefore picks
    the depth with least signal - which is what the original rule did on the
    real collection, selecting K_c=1000 at a median supply of 1.
    """
    summary = _fake_summary({
        100: {"bm25": 6.0, "dpr": 6.0},
        500: {"bm25": 3.0, "dpr": 3.0},
        1000: {"bm25": 2.0, "dpr": 1.0},
    })
    k, rule = fr.select_headline_k_c(summary)
    assert k == 100, "the deepest usable depth is not the right answer"
    assert "greatest supply" in rule


def test_headline_follows_supply_upward_when_the_pool_is_deeper():
    """
    Proof the rule is a rule and not a dressed-up constant: give a collection
    whose supply grows with depth and the headline moves with it.
    """
    summary = _fake_summary({
        100: {"bm25": 2.0, "dpr": 2.0},
        500: {"bm25": 9.0, "dpr": 8.0},
        1000: {"bm25": 4.0, "dpr": 4.0},
    })
    assert fr.select_headline_k_c(summary)[0] == 500


def test_headline_uses_the_weakest_system_not_the_average():
    """
    The cross-paradigm comparison needs both systems, so a depth is only as
    good as its weaker one. Averaging would prefer 1000 here (6.5 against 5.0).
    """
    summary = _fake_summary({
        500: {"bm25": 5.0, "dpr": 5.0},
        1000: {"bm25": 12.0, "dpr": 1.0},
    })
    assert fr.select_headline_k_c(summary)[0] == 500


def test_ties_break_towards_the_deeper_depth():
    """Equal supply: prefer the deeper depth, which is the more realistic one."""
    summary = _fake_summary({
        100: {"bm25": 4.0, "dpr": 4.0},
        500: {"bm25": 4.0, "dpr": 4.0},
    })
    assert fr.select_headline_k_c(summary)[0] == 500


def test_degenerate_depths_are_excluded_before_supply_is_compared():
    """
    The non-degeneracy floor is a gate, not a tie-break. A depth where one
    system has no supply at all cannot win on the other system's abundance.
    """
    summary = _fake_summary({
        100: {"bm25": 30.0, "dpr": 0.0},
        500: {"bm25": 2.0, "dpr": 2.0},
    })
    assert not summary[100]["usable_for_headline_gate"]
    assert fr.select_headline_k_c(summary)[0] == 500


def test_no_usable_depth_returns_none_with_the_rule_still_stated():
    summary = _fake_summary({100: {"bm25": 0.0, "dpr": 0.0}})
    k, rule = fr.select_headline_k_c(summary)
    assert k is None
    assert "non-degeneracy floor" in rule
