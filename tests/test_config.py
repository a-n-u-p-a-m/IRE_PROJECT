"""
The design freeze.

Each refusal below corresponds to a configuration that would still run and
still produce plausible-looking numbers that mean nothing. Catching them at
load time is the cheapest place to catch them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shared.config import ConfigError, load_config
from shared.provenance import assert_same_freeze, run_manifest, write_manifest


def _write(tmp_path: Path, raw: dict) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(p)


def _raw(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_base_config_loads(config_path):
    cfg = load_config(config_path)
    assert cfg.universe_depth_m == 5000
    assert cfg.final_cutoff_k == 10
    assert cfg.unjudged_policy == "unknown"
    assert cfg.reranker_pointwise is True
    assert set(cfg.stage1_systems).issubset(set(cfg.universe_sources))
    assert 1.0 in cfg.t_grid


def test_tau_query_is_frozen(config_path):
    """
    tau gates group membership for H7, so it is the parameter most exposed to
    post-hoc tuning. It belongs in the freeze, under its own name, distinct
    from the document-level threshold.
    """
    cfg = load_config(config_path)
    assert cfg.tau_query > 0
    assert cfg.tau_document > 0
    assert hasattr(cfg, "tau_query") and hasattr(cfg, "tau_document")


def test_trec_dl_gain_map_is_the_frozen_default(config_path):
    cfg = load_config(config_path)
    assert cfg.gain_map[1] == 0.0, (
        "TREC DL grade 1 means 'Related', not relevant; a nonzero gain here "
        "inflates every effect in which a merely-related passage moves"
    )
    assert cfg.gain_map[2] == 3.0 and cfg.gain_map[3] == 7.0


def test_headline_and_attribution_gain_maps_are_both_recorded(config_path):
    """
    The headline NDCG may use official conventions while the attribution
    utility does not. Both must be recorded so the difference is stated rather
    than discovered.
    """
    cfg = load_config(config_path)
    assert cfg.headline_gain_map != cfg.gain_map
    assert cfg.headline_gain_map[1] == 1.0


def test_fingerprint_is_stable_and_sensitive(config_path, tmp_path):
    cfg = load_config(config_path)
    assert cfg.fingerprint() == load_config(config_path).fingerprint()
    assert len(cfg.fingerprint()) == 12

    raw = _raw(config_path)
    raw["diagnosis"]["tau_query"] = 0.5
    assert load_config(_write(tmp_path, raw)).fingerprint() != cfg.fingerprint()


def test_budget_helper_follows_b_equals_kc(config_path, tmp_path):
    cfg = load_config(config_path)
    assert cfg.budget_for(100) == 100
    raw = _raw(config_path)
    raw["budget"]["b_equals_kc"] = False
    raw["budget"]["b_absolute"] = 64
    assert load_config(_write(tmp_path, raw)).budget_for(100) == 64


# ── refusals ──────────────────────────────────────────────────────────

def test_stage1_system_must_be_a_universe_source(config_path, tmp_path):
    raw = _raw(config_path)
    raw["cascade"]["stage1_systems"] = ["bm25", "colbert"]
    with pytest.raises(ConfigError, match="not universe sources"):
        load_config(_write(tmp_path, raw))


def test_universe_must_be_deeper_than_the_candidate_set(config_path, tmp_path):
    raw = _raw(config_path)
    raw["universe"]["depth_m"] = 1000
    raw["cascade"]["candidate_depths"] = [100, 1000]
    with pytest.raises(ConfigError, match="identically zero"):
        load_config(_write(tmp_path, raw))


def test_cutoff_cannot_exceed_the_candidate_depth(config_path, tmp_path):
    raw = _raw(config_path)
    raw["cascade"]["candidate_depths"] = [5, 100]
    with pytest.raises(ConfigError, match="exceeds the smallest candidate"):
        load_config(_write(tmp_path, raw))


def test_t_grid_must_contain_the_baseline(config_path, tmp_path):
    raw = _raw(config_path)
    raw["budget"]["t_grid"] = [0.9, 0.5]
    with pytest.raises(ConfigError, match="must contain 1.0"):
        load_config(_write(tmp_path, raw))


def test_unset_absolute_budget_is_refused(config_path, tmp_path):
    raw = _raw(config_path)
    raw["budget"]["b_equals_kc"] = False
    raw["budget"]["b_absolute"] = None
    with pytest.raises(ConfigError, match="b_absolute is unset"):
        load_config(_write(tmp_path, raw))


def test_alternative_policy_needs_a_real_source(config_path, tmp_path):
    raw = _raw(config_path)
    raw["budget"]["alternative_source"] = "a_retriever_we_never_ran"
    with pytest.raises(ConfigError, match="not a universe source"):
        load_config(_write(tmp_path, raw))


def test_treating_unjudged_as_non_relevant_is_refused(config_path, tmp_path):
    raw = _raw(config_path)
    raw["qrels"]["unjudged"] = "nonrelevant"
    with pytest.raises(ConfigError, match="fabricate the evidence"):
        load_config(_write(tmp_path, raw))


def test_non_monotone_gain_map_is_refused(config_path, tmp_path):
    raw = _raw(config_path)
    raw["utility"]["gain_map"] = {0: 0.0, 1: 5.0, 2: 1.0, 3: 7.0}
    with pytest.raises(ConfigError, match="non-decreasing"):
        load_config(_write(tmp_path, raw))


def test_convenience_reference_sampling_is_refused(config_path, tmp_path):
    raw = _raw(config_path)
    raw["reference_subset"]["strategy"] = "smallest_universes_first"
    with pytest.raises(ConfigError, match="Convenience selection is refused"):
        load_config(_write(tmp_path, raw))


def test_document_collection_needs_declared_segmentation(config_path, tmp_path):
    """
    A cross-encoder cannot consume a long document in one 512-token input, and
    the PASSAGE-level call count is what must be budgeted - not the document
    count.
    """
    raw = _raw(config_path)
    raw["dataset"]["granularity"] = "document"
    raw["dataset"]["segmentation"] = None
    with pytest.raises(ConfigError, match="segmentation declared"):
        load_config(_write(tmp_path, raw))


def test_document_collection_with_segmentation_is_accepted(config_path, tmp_path):
    raw = _raw(config_path)
    raw["dataset"]["granularity"] = "document"
    raw["dataset"]["segmentation"] = {"window": 256, "stride": 128,
                                      "aggregate": "max"}
    cfg = load_config(_write(tmp_path, raw))
    assert cfg.segmentation["aggregate"] == "max"


def test_hashing_enabled_without_a_method_is_refused(config_path, tmp_path):
    raw = _raw(config_path)
    raw["hashing"]["enabled"] = True
    raw["hashing"]["method"] = None
    with pytest.raises(ConfigError, match="no method is named"):
        load_config(_write(tmp_path, raw))


def test_missing_section_and_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="missing sections"):
        load_config(_write(tmp_path, {"seed": 1}))
    with pytest.raises(ConfigError, match="config not found"):
        load_config(tmp_path / "absent.yaml")


# ── provenance ────────────────────────────────────────────────────────

def test_manifests_detect_a_mixed_design_freeze(tmp_path, config_path):
    cfg = load_config(config_path)
    a = write_manifest(tmp_path / "a.json", run_manifest(cfg.fingerprint(), "member1"))
    b = write_manifest(tmp_path / "b.json", run_manifest(cfg.fingerprint(), "member2"))
    assert assert_same_freeze(a, b) == cfg.fingerprint()

    c = write_manifest(tmp_path / "c.json", run_manifest("deadbeefcafe", "member3"))
    with pytest.raises(ValueError, match="different design freezes"):
        assert_same_freeze(a, c)


def test_manifest_records_the_owner(config_path):
    cfg = load_config(config_path)
    rec = run_manifest(cfg.fingerprint(), "member2", {"experiment": "E3"})
    assert rec["owner"] == "member2"
    assert rec["experiment"] == "E3"
    assert rec["config_fingerprint"] == cfg.fingerprint()
