"""
test_build_universe.py - Building and freezing U_q, and the refusals.

Each refusal below corresponds to a way the counterfactual frame breaks
silently - producing a universe that still loads, still looks coherent, and
makes every effect computed against it meaningless.
"""

from __future__ import annotations

import json

import pytest

from experiments_anupam import build_universe as bu
from experiments_anupam.tests.conftest import make_run
from shared.config import load_config
from shared.schema import load, save
from shared.universe import (
    check_containment,
    load_frozen,
    universe_sets,
    universe_sizes,
)


def _write_runs(tmp_path, runs):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(exist_ok=True)
    for system, df in runs.items():
        save(df, runs_dir / f"{system}.parquet", "run")
    return str(runs_dir)


def _build(tmp_path, cfg_path, runs, qrels=None, force=False):
    return bu.run(
        config_path=cfg_path,
        runs_dir=_write_runs(tmp_path, runs),
        out_path=str(tmp_path / "data" / "universe.parquet"),
        qrels_path=qrels,
        force=force,
    )


# ── the happy path ────────────────────────────────────────────────────

def test_builds_and_freezes(tmp_path, small_cfg_path, small_cfg, two_source_runs):
    out = _build(tmp_path, small_cfg_path, two_source_runs)
    uni = load_frozen(out, fingerprint=small_cfg.fingerprint())

    sizes = universe_sizes(uni)
    assert sizes.size == 8
    assert int(sizes.min()) > max(small_cfg.candidate_depths)

    sidecar = out.with_suffix(".frozen")
    assert sidecar.read_text().strip() == small_cfg.fingerprint()


def test_union_is_broader_than_either_source(tmp_path, small_cfg_path, two_source_runs):
    out = _build(tmp_path, small_cfg_path, two_source_runs)
    sets = universe_sets(load(out, "universe"))
    bm25_only = set(two_source_runs["bm25"].query("query_id == 'q0'")["doc_id"])
    assert len(sets["q0"]) > len(bm25_only)


def test_sources_are_recorded_per_document(tmp_path, small_cfg_path, two_source_runs):
    out = _build(tmp_path, small_cfg_path, two_source_runs)
    uni = load(out, "universe")
    labels = set(uni["sources"])
    assert "bm25,dpr" in labels, "no document was contributed by both sources"
    assert {"bm25", "dpr"} & labels, "no document is unique to a single source"


def test_manifest_records_the_source_checkpoints(tmp_path, small_cfg_path, two_source_runs):
    """
    "Which runs built this universe" must be answerable from the artefact, not
    from a chat message.
    """
    out = _build(tmp_path, small_cfg_path, two_source_runs)
    manifest = json.loads(out.with_name("universe_manifest.json").read_text())
    assert manifest["owner"] == "anupam"
    assert set(manifest["sources"]) == {"bm25", "dpr"}
    assert manifest["source_checkpoints"]["bm25"] == ["pyserini:test|bm25-k1=0.9-b=0.4"]
    assert manifest["total_universe_pairs"] > 0
    assert manifest["universe_size_min"] > 0


# ── the refusals ──────────────────────────────────────────────────────

def test_refuses_to_overwrite_a_frozen_universe(tmp_path, small_cfg_path, two_source_runs):
    """
    Every effect computed against the old universe would become incomparable,
    so rebuilding is a deliberate act.
    """
    _build(tmp_path, small_cfg_path, two_source_runs)
    with pytest.raises(SystemExit, match="not rebuilt"):
        _build(tmp_path, small_cfg_path, two_source_runs)


def test_force_allows_a_deliberate_rebuild(tmp_path, small_cfg_path, two_source_runs):
    _build(tmp_path, small_cfg_path, two_source_runs)
    assert _build(tmp_path, small_cfg_path, two_source_runs, force=True).exists()


def test_missing_source_run_is_named(tmp_path, small_cfg_path, two_source_runs):
    with pytest.raises(SystemExit, match="missing source run"):
        _build(tmp_path, small_cfg_path, {"bm25": two_source_runs["bm25"]})


def test_shallow_source_is_refused(tmp_path, small_cfg_path, two_source_runs):
    """A source below M would make the universe shallower than the config claims."""
    shallow = two_source_runs["dpr"][two_source_runs["dpr"]["rank"] <= 10]
    with pytest.raises(SystemExit, match="below M="):
        _build(tmp_path, small_cfg_path,
               {"bm25": two_source_runs["bm25"], "dpr": shallow})


def test_ragged_query_sets_are_refused(tmp_path, small_cfg_path, two_source_runs):
    """Sources covering different queries would give a ragged U_q."""
    dropped = two_source_runs["dpr"][two_source_runs["dpr"]["query_id"] != "q0"]
    with pytest.raises(SystemExit, match="different query sets"):
        _build(tmp_path, small_cfg_path,
               {"bm25": two_source_runs["bm25"], "dpr": dropped})


def test_config_refuses_a_depth_below_the_largest_k_c(tmp_path, small_cfg_path):
    """
    Caught at config load, before any run is read: M below the largest K_c
    means C_q == U_q and E_gate is identically zero.
    """
    import yaml

    from shared.config import ConfigError

    with open(small_cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["cascade"]["candidate_depths"] = [5, 10, 60]      # 60 > M = 40
    path = tmp_path / "wide_kc.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError, match="identically zero"):
        load_config(path)


def test_universe_capped_below_k_c_is_refused(tmp_path, small_cfg_path, two_source_runs):
    """
    The case config cannot catch: M clears the largest K_c, but `max_size`
    caps |U_q| below it. The config checks the retrieval depth, not the cap,
    so this one has to be caught here - after the union is built and its real
    size is known.
    """
    import yaml

    with open(small_cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["universe"]["max_size"] = 15          # below candidate_depths max of 20
    path = tmp_path / "tight_cap.yaml"
    path.write_text(yaml.safe_dump(raw))

    load_config(path)                          # config itself is satisfied
    with pytest.raises(SystemExit, match="identically zero"):
        _build(tmp_path, str(path), two_source_runs)


def test_frozen_universe_refuses_a_fingerprint_mismatch(
    tmp_path, small_cfg_path, two_source_runs
):
    from shared.universe import UniverseError

    out = _build(tmp_path, small_cfg_path, two_source_runs)
    with pytest.raises(UniverseError, match="not comparable"):
        load_frozen(out, fingerprint="deadbeefcafe")


def test_a_regenerated_stage1_run_fails_containment(
    tmp_path, small_cfg_path, small_cfg, two_source_runs
):
    """
    The real containment scenario, and the reason the check exists.

    The universe is frozen from one set of runs. Later - a different index, a
    re-retrieval, a changed parameter - bm25 is regenerated over documents the
    frozen universe never saw. Every candidate set from that run now escapes
    U_q, and every effect computed from it would be relative to a universe that
    does not contain its own candidates. The check must fire before any scoring.
    """
    from shared.universe import UniverseError

    out = _build(tmp_path, small_cfg_path, two_source_runs)
    frozen = load_frozen(out, fingerprint=small_cfg.fingerprint())

    regenerated = make_run(
        "bm25",
        {q: [f"zzz{i:03d}" for i in range(40)]
         for q in sorted(set(two_source_runs["bm25"]["query_id"]))},
        "pyserini:other-index|bm25-k1=0.9-b=0.4",
    )

    with pytest.raises(UniverseError, match="outside the frozen universe"):
        check_containment(regenerated, frozen, max(small_cfg.candidate_depths), "bm25")


def test_partial_escape_is_caught_too(
    tmp_path, small_cfg_path, small_cfg, two_source_runs
):
    """
    A wholly alien run is the easy case. The dangerous one is a run that mostly
    agrees with the universe and smuggles a few documents in at the top, which
    is what a re-indexed collection actually looks like.
    """
    from shared.universe import UniverseError

    out = _build(tmp_path, small_cfg_path, two_source_runs)
    frozen = load_frozen(out, fingerprint=small_cfg.fingerprint())

    bm25 = two_source_runs["bm25"].copy()
    bm25.loc[bm25.index[0], "doc_id"] = "zzz999"      # rank 1 of the first query
    with pytest.raises(UniverseError, match="outside the frozen universe"):
        check_containment(bm25, frozen, max(small_cfg.candidate_depths), "bm25")


def test_the_in_build_containment_check_cannot_fire(small_cfg):
    """
    Documents why the loop at the end of `build_universe.run` is defence and
    not a real gate: the config refuses any Stage-1 system that is not also a
    universe source, so inside one build every candidate set is a subset of a
    union that was built from that same run. The check is kept in case that
    config rule is ever loosened - this test is what will notice if it is.
    """
    assert set(small_cfg.stage1_systems) <= set(small_cfg.universe_sources)


# ── coverage report ───────────────────────────────────────────────────

def test_coverage_report_is_written_when_qrels_exist(
    tmp_path, small_cfg_path, two_source_runs
):
    from shared.qrels import read_trec_qrels

    uni_docs = sorted(set(two_source_runs["bm25"]["doc_id"]))[:6]
    lines = "".join(f"q0 0 {d} {i % 4}\n" for i, d in enumerate(uni_docs))
    qrels_txt = tmp_path / "qrels.txt"
    qrels_txt.write_text(lines, encoding="utf-8")
    qrels_pq = tmp_path / "qrels.parquet"
    save(read_trec_qrels(qrels_txt), qrels_pq, "qrels")

    out = _build(tmp_path, small_cfg_path, two_source_runs, qrels=str(qrels_pq))
    assert out.with_name("universe_coverage.parquet").exists()

    manifest = json.loads(out.with_name("universe_manifest.json").read_text())
    assert "coverage_median" in manifest
    assert manifest["n_judged_median"] >= 0


def test_missing_qrels_is_not_fatal(tmp_path, small_cfg_path, two_source_runs):
    """The universe does not depend on judgments; coverage is a report."""
    out = _build(tmp_path, small_cfg_path, two_source_runs,
                 qrels=str(tmp_path / "nope.parquet"))
    assert out.exists()
    manifest = json.loads(out.with_name("universe_manifest.json").read_text())
    assert "coverage_median" not in manifest


# ── end to end with the real config shape ─────────────────────────────

def test_real_config_shape_loads(two_source_runs):
    """
    The committed config must name exactly the systems these runs provide, or
    the Week-2 build will fail on something this suite never exercised.
    """
    cfg = load_config("config/base.yaml")
    assert set(cfg.universe_sources) == {"bm25", "dpr"}
    assert set(cfg.stage1_systems) == {"bm25", "dpr"}
    assert set(two_source_runs) == set(cfg.universe_sources)
