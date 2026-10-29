"""
test_allocations.py - The t-allocations, the audit, and the reference subset.

The audit is the deliverable, so most of these feed it a broken allocation and
check it complains. Three invariants get the most attention, because each one
would corrupt a different downstream claim if it slipped:

  * spending exactly B, or the t curve compares compute rather than budget;
  * the four strata partitioning U_q, or the Delta_t sign rules apply to a
    partition that is not one;
  * t=1 being the plain cascade, or every Delta_t is relative to a policy that
    never ran.
"""

from __future__ import annotations

import json

import pytest
import yaml

from experiments_anupam import allocations as al
from experiments_anupam import build_universe as bu
from experiments_anupam import reference_subset as rs
from experiments_anupam import stage1_over_universe as s1u
from shared.config import load_config
from shared.schema import save


def _cfg(tmp_path, small_cfg_path, **sections):
    with open(small_cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for name, kv in sections.items():
        raw.setdefault(name, {}).update(kv)
    p = tmp_path / "alloc_cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return str(p)


def _prepare(tmp_path, cfg_path, two_source_runs):
    """Universe + stage1-over-universe tables, as the runner expects them."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(exist_ok=True)
    for s, df in two_source_runs.items():
        save(df, runs_dir / f"{s}.parquet", "run")
    uni = bu.run(cfg_path, str(runs_dir), str(tmp_path / "universe.parquet"),
                 None, False)
    tables = tmp_path / "tables"
    s1u.run(cfg_path, str(runs_dir), str(uni), str(tables))
    return str(uni), str(tables)


# ── the audit, on hand-built allocations ──────────────────────────────

def _alloc(retained, removed, selected, unselected):
    return {
        "retained": list(retained), "removed": list(removed),
        "selected": list(selected), "unselected": list(unselected),
        "submitted": list(retained) + list(selected),
    }


def test_a_correct_allocation_passes():
    uq = {"a", "b", "c", "d"}
    a = _alloc(["a"], ["b"], ["c"], ["d"])
    assert al.audit_one(a, uq, budget=2, t=0.5) == []


def test_a_short_allocation_is_caught():
    """Spends 1 of a budget of 2, so this t point is cheaper than the others."""
    uq = {"a", "b", "c", "d"}
    a = _alloc(["a"], ["b"], [], ["c", "d"])
    problems = al.audit_one(a, uq, budget=2, t=0.5)
    assert any("spend" in p for p in problems)


def test_a_gap_in_the_partition_is_caught():
    """'d' belongs to no stratum, so no Delta_t sign rule applies to it."""
    uq = {"a", "b", "c", "d"}
    a = _alloc(["a"], ["b"], ["c"], [])
    problems = al.audit_one(a, uq, budget=2, t=0.5)
    assert any("strata cover 3" in p for p in problems)


def test_an_overlap_in_the_partition_is_caught():
    """'c' is both promoted and not, so two sign rules would apply to it."""
    uq = {"a", "b", "c"}
    a = _alloc(["a"], ["b"], ["c"], ["c"])
    problems = al.audit_one(a, uq, budget=2, t=0.5)
    assert any("overlaps" in p for p in problems)


def test_a_duplicate_inside_one_stratum_is_caught():
    uq = {"a", "b"}
    a = _alloc(["a", "a"], [], [], ["b"])
    assert any("duplicate" in p for p in al.audit_one(a, uq, budget=2, t=1.0))


def test_a_document_outside_the_universe_is_caught():
    uq = {"a", "b"}
    a = _alloc(["a"], ["b"], ["zzz"], [])
    assert any("against" in p for p in al.audit_one(a, uq, budget=2, t=0.5))


# ── seeds ─────────────────────────────────────────────────────────────

def test_only_the_random_policy_consumes_a_seed():
    """
    Three seeds on a deterministic policy would write three identical
    allocations and make the audit look three times as thorough as it is.
    """
    assert al.seeds_for("random", (42, 43, 44)) == (42, 43, 44)
    assert al.seeds_for("boundary", (42, 43, 44)) == (42,)
    assert al.seeds_for("alternative", (42, 43, 44)) == (42,)


# ── end to end ────────────────────────────────────────────────────────

def test_end_to_end_audits_clean(tmp_path, small_cfg_path, two_source_runs):
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    uni, tables = _prepare(tmp_path, cfg_path, two_source_runs)

    extra, problems = al.run(cfg_path, uni, tables, str(tmp_path / "out"))
    assert not problems, problems
    assert extra["all_allocations_valid"]
    assert extra["n_allocations"] > 0


def test_outputs_and_manifest_are_written(tmp_path, small_cfg_path, two_source_runs):
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    uni, tables = _prepare(tmp_path, cfg_path, two_source_runs)
    out = tmp_path / "out"

    extra, _ = al.run(cfg_path, uni, tables, str(out))
    assert (out / "allocation_audit.parquet").exists()
    assert (out / "allocation_promoted.parquet").exists()
    manifest = json.loads((out / "allocation_manifest.json").read_text())
    assert manifest["owner"] == "anupam"
    assert manifest["n_allocations"] == extra["n_allocations"]


def test_t_one_is_the_conventional_cascade(tmp_path, small_cfg_path, two_source_runs):
    """
    With B = K_c, t=1 must put every call inside C_q and promote nothing. It is
    the baseline every Delta_t is measured against.
    """
    import pandas as pd

    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    uni, tables = _prepare(tmp_path, cfg_path, two_source_runs)
    out = tmp_path / "out"
    al.run(cfg_path, uni, tables, str(out))

    audit = pd.read_parquet(out / "allocation_audit.parquet")
    at1 = audit[audit["t"] == 1.0]
    assert not at1.empty
    assert (at1["n_selected"] == 0).all()
    assert (at1["n_retained"] == at1["budget_b"]).all()


def test_the_random_control_varies_with_its_seed(
    tmp_path, small_cfg_path, two_source_runs
):
    """
    A control that chose the same documents under every seed would not be
    sampling, and could not separate a policy effect from the effect of
    reranking more documents at all.
    """
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    uni, tables = _prepare(tmp_path, cfg_path, two_source_runs)

    extra, _ = al.run(cfg_path, uni, tables, str(tmp_path / "out"))
    assert extra["random_seed_sensitivity"] is not None
    assert extra["random_seed_sensitivity"] > 0.0


def test_promoted_counts_follow_the_budget_split(
    tmp_path, small_cfg_path, two_source_runs
):
    """n_selected must equal B - floor(t*B) at every t, or the split is wrong."""
    import math

    import pandas as pd

    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    uni, tables = _prepare(tmp_path, cfg_path, two_source_runs)
    out = tmp_path / "out"
    al.run(cfg_path, uni, tables, str(out))

    audit = pd.read_parquet(out / "allocation_audit.parquet")
    for _, r in audit.iterrows():
        b, t = int(r["budget_b"]), float(r["t"])
        assert r["n_selected"] == b - math.floor(t * b)


def test_missing_stage1_table_names_the_command(
    tmp_path, small_cfg_path, two_source_runs
):
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    uni, _ = _prepare(tmp_path, cfg_path, two_source_runs)
    with pytest.raises(SystemExit, match="stage1_over_universe"):
        al.run(cfg_path, uni, str(tmp_path / "empty"), str(tmp_path / "out"))


# ── reference subset ──────────────────────────────────────────────────

def _power(fingerprint, median, group=None, continuous=None):
    return {
        "config_fingerprint": fingerprint,
        "realistic_universe": {"value": median},
        "recommendation": {
            "k_c": 100, "t": 0.5, "target_power": 0.8,
            "group": {"smallest_per_setting": group or {}},
            "continuous": {"smallest_per_setting": continuous or {}},
        },
    }


def test_reference_subset_is_every_query_when_nothing_smaller_powers(
    tmp_path, small_cfg_path, small_cfg, two_source_runs
):
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    uni, _ = _prepare(tmp_path, cfg_path, two_source_runs)
    cfg = load_config(cfg_path)

    from shared.universe import load_frozen, universe_sizes
    median = int(universe_sizes(load_frozen(uni, cfg.fingerprint())).median())

    pw = tmp_path / "power.json"
    pw.write_text(json.dumps(_power(
        cfg.fingerprint(), median,
        group={f"alternative@{median}": 8, f"boundary@{median}": None},
        continuous={f"alternative@{median}": 8},
    )))

    extra = rs.run(cfg_path, uni, str(pw), str(tmp_path / "ref.txt"))
    assert extra["n_reference_queries"] == 8
    assert extra["power_verdicts"]["by_test"]["group"]["confirmatory"] == ["alternative"]
    assert extra["power_verdicts"]["by_test"]["group"]["descriptive"] == ["boundary"]


def test_a_power_result_from_another_freeze_is_refused(
    tmp_path, small_cfg_path, two_source_runs
):
    """A reference size from a different freeze answers a different question."""
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    uni, _ = _prepare(tmp_path, cfg_path, two_source_runs)
    pw = tmp_path / "power.json"
    pw.write_text(json.dumps(_power("000000000000", 60)))

    with pytest.raises(SystemExit, match="different freeze"):
        rs.run(cfg_path, uni, str(pw), str(tmp_path / "ref.txt"))


def test_a_power_result_for_another_universe_size_is_refused(
    tmp_path, small_cfg_path, two_source_runs
):
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    uni, _ = _prepare(tmp_path, cfg_path, two_source_runs)
    cfg = load_config(cfg_path)
    pw = tmp_path / "power.json"
    pw.write_text(json.dumps(_power(cfg.fingerprint(), 999999)))

    with pytest.raises(SystemExit, match="does not describe the universe"):
        rs.run(cfg_path, uni, str(pw), str(tmp_path / "ref.txt"))


def test_a_missing_power_result_names_the_command(
    tmp_path, small_cfg_path, two_source_runs
):
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    uni, _ = _prepare(tmp_path, cfg_path, two_source_runs)
    with pytest.raises(SystemExit, match="power_h7"):
        rs.run(cfg_path, uni, str(tmp_path / "nope.json"), str(tmp_path / "r.txt"))


def test_a_size_beyond_the_query_set_is_refused(
    tmp_path, small_cfg_path, two_source_runs
):
    """
    H7 needing more queries than the collection has is a finding, not a budget
    problem: it cannot be confirmatory here at any cost.
    """
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    uni, _ = _prepare(tmp_path, cfg_path, two_source_runs)
    cfg = load_config(cfg_path)
    from shared.universe import load_frozen, universe_sizes
    median = int(universe_sizes(load_frozen(uni, cfg.fingerprint())).median())

    pw = tmp_path / "power.json"
    pw.write_text(json.dumps(_power(
        cfg.fingerprint(), median, group={f"alternative@{median}": 500}
    )))
    with pytest.raises(SystemExit, match="cannot be confirmatory"):
        rs.run(cfg_path, uni, str(pw), str(tmp_path / "ref.txt"))


def test_a_subset_smaller_than_the_query_set_refuses_to_sample_implicitly(
    tmp_path, small_cfg_path, two_source_runs
):
    """
    Taking the head of the id list would bias towards whatever order the ids
    happen to be in. Sampling needs the declared seeded strategy.
    """
    cfg_path = _cfg(tmp_path, small_cfg_path, cascade={"candidate_depths": [5]})
    uni, _ = _prepare(tmp_path, cfg_path, two_source_runs)
    cfg = load_config(cfg_path)
    from shared.universe import load_frozen, universe_sizes
    median = int(universe_sizes(load_frozen(uni, cfg.fingerprint())).median())

    pw = tmp_path / "power.json"
    pw.write_text(json.dumps(_power(
        cfg.fingerprint(), median, group={f"alternative@{median}": 3}
    )))
    with pytest.raises(SystemExit, match="declared seeded strategy"):
        rs.run(cfg_path, uni, str(pw), str(tmp_path / "ref.txt"))
