"""
test_compare.py - BM25 against DPR one axis at a time, on two synthetic
"systems" written out as effect tables.

    python -m pytest experiments_shubham/tests/test_compare.py
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments_shubham import analysis, compare
from experiments_shubham.engine import (
    EffectSettings,
    run_grid,
    synthetic_frame,
    synthetic_inputs,
)

SYSTEMS = ("bm25", "dpr")


@pytest.fixture(scope="module")
def effects(small_cfg, tmp_path_factory):
    """Headline grid at two depths, top-k and DCG@5 at one, for two systems."""
    root = tmp_path_factory.mktemp("effects")
    inputs = {"bm25": synthetic_inputs(synthetic_frame(small_cfg, "weak_retriever", n_queries=6)),
              "dpr": synthetic_inputs(synthetic_frame(small_cfg, "strong_stages", n_queries=6))}
    base = EffectSettings.from_config(small_cfg, "x")
    runs = [("dcg", 10, [50, 100]), ("topk", 10, [50]), ("dcg", 5, [50])]
    for utility, k, depths in runs:
        out = root / compare.effects_dir_name(small_cfg, utility, k, subset=False)
        for system in SYSTEMS:
            run_grid(inputs[system], small_cfg,
                     replace(base, stage1_system=system, utility=utility, k=k), out,
                     candidate_depths=depths)
    return root


def test_frozen_settings_change_one_thing_each(small_cfg):
    assert compare.frozen_settings(small_cfg) == [("dcg", 10), ("topk", 10), ("rbp", 10),
                                                  ("dcg", 5), ("dcg", 20)]


def test_one_axis_keeps_only_controlled_comparisons(small_cfg, effects, tmp_path):
    result = compare.run(effects, small_cfg, tmp_path, k_c=50, systems=SYSTEMS)
    wide = result["one_axis"]
    blocks = set(zip(wide["varied"], wide["setting"]))
    assert blocks == {("headline", ""), ("k_c", "100"), ("utility", "topk"), ("k", "5"),
                      ("condensed", "True")}
    # rbp and DCG@20 were never run: reported as missing, not invented
    assert len(result["missing"]) == 2
    # nothing two axes away, e.g. top-k condensed or DCG@5 at K_c 100
    assert not ((wide["utility"] == "topk") & wide["condensed"]).any()
    assert not ((wide["k"] == 5) & (wide["k_c"] == 100)).any()
    for name in ("compare_long.csv", "compare_one_axis.csv", "compare_manifest.json"):
        assert (tmp_path / name).exists()


def test_headline_numbers_match_the_diagnosis(small_cfg, effects, tmp_path):
    wide = compare.run(effects, small_cfg, tmp_path, k_c=50, systems=SYSTEMS)["one_axis"]
    head = wide[wide["varied"] == "headline"].set_index("metric")
    for system in SYSTEMS:
        stage = analysis.load_stage(effects / "real", system, 50)
        diag = analysis.diagnose(stage, small_cfg.tau_query)
        assert head.loc["n_gate_sensitive", system] == (diag["label"] == "gate_sensitive").sum()
        assert head.loc["g_q_mean", system] == pytest.approx(diag["g_q"].mean())
        assert head.loc["n_queries", system] == 6
    diff = head.loc["g_q_mean", "bm25"] - head.loc["g_q_mean", "dpr"]
    assert head.loc["g_q_mean", "bm25_minus_dpr"] == pytest.approx(diff)
    deltas = [m for m in head.index if m.startswith("delta_")]
    assert len(deltas) == len(small_cfg.outsider_policies) * (len(small_cfg.t_grid) - 1)


def test_utility_block_is_measured_in_its_own_unit(small_cfg, effects, tmp_path):
    wide = compare.run(effects, small_cfg, tmp_path, k_c=50, systems=SYSTEMS)["one_axis"]
    topk = wide[(wide["varied"] == "utility")].set_index("metric")
    # top-k utility counts gain-carrying documents in the top 10
    for system in SYSTEMS:
        assert 0 <= topk.loc["utility_actual_mean", system] <= 10
    stage = analysis.load_stage(effects / "real_topk_k10", "bm25", 50)
    per_query = analysis.diagnose(stage, small_cfg.tau_query)["dcg_actual"]
    assert (per_query == per_query.round()).all()
