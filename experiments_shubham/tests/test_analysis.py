"""
test_analysis.py - The effect-table analysis agrees with shared.diagnosis, and
the whole run works on a small synthetic grid.

    python -m pytest experiments_shubham
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from experiments_shubham import analysis
from experiments_shubham.engine import (
    EffectSettings,
    QueryInputs,
    run_budget,
    run_grid,
    run_worlds,
    synthetic_frame,
    synthetic_inputs,
)
from shared import fixtures as fx
from shared.diagnosis import QUERY_LABELS, query_delta_dcg, query_diagnosis
from shared.worlds import build_t_grid, build_worlds


@pytest.fixture(scope="module")
def grid(small_cfg):
    queries = synthetic_inputs(synthetic_frame(small_cfg, "weak_retriever", n_queries=8))
    return queries, EffectSettings.from_config(small_cfg, "synthetic")


@pytest.fixture
def toy_stage():
    c = fx.toy_cascade()
    q = QueryInputs(c["query_id"], tuple(c["universe"]), c["s1"], c["s2"],
                    c["relevances"], c["judged"], c["alternative_rank"])
    return run_worlds([q], fx.TOY_K_C, EffectSettings("toy", fx.TOY_K, fx.TOY_GAIN_MAP))


def test_diagnosis_from_effects_matches_shared(small_cfg, grid):
    queries, settings = grid
    ours = analysis.diagnose(run_worlds(queries, 50, settings), small_cfg.tau_query)
    ours = ours.set_index("query_id")
    for q in queries:
        w = build_worlds(q.query_id, q.universe, q.s1, q.s2, 50)
        ref = query_diagnosis(w, q.relevance, settings.k, settings.gain_map,
                              small_cfg.tau_query)
        row = ours.loc[q.query_id]
        assert row["label"] == ref["label"]
        for col, key in (("g_q", "g_q"), ("r_q", "r_q"), ("rerank_signed", "rerank_signed"),
                         ("dcg_actual", "dcg_actual")):
            assert row[col] == pytest.approx(ref[key], abs=1e-9)


def test_budget_outcomes_match_shared(small_cfg, grid):
    queries, settings = grid
    budget = run_budget(queries, 50, 50, small_cfg.t_grid, "boundary", [42], settings)
    out = analysis.budget_outcomes({"boundary": budget}).set_index(["query_id", "t"])
    for q in queries:
        w = build_worlds(q.query_id, q.universe, q.s1, q.s2, 50)
        tw = build_t_grid(w, q.s1, q.s2, 50, small_cfg.t_grid, "boundary",
                          alternative_rank=q.alternative_rank, seed=42)
        for t, delta in query_delta_dcg(tw, q.relevance, settings.k,
                                        settings.gain_map).items():
            if t != 1.0:
                assert out.loc[(q.query_id, t), "delta_dcg"] == pytest.approx(delta, abs=1e-9)


def test_random_policy_outcome_is_the_mean_over_seeds(small_cfg, grid):
    queries, settings = grid
    seeds = list(small_cfg.budget_seeds)
    budget = run_budget(queries, 50, 50, (1.0, 0.5), "random", seeds, settings)
    out = analysis.budget_outcomes({"random": budget})
    per_seed = budget.groupby(["query_id", "seed"])["effect"].sum().groupby("query_id").mean()
    merged = out.set_index("query_id")["delta_dcg"]
    assert np.allclose(merged.loc[per_seed.index.astype(str)], per_seed.to_numpy())


def test_distribution_sizes_add_up(toy_stage):
    dist = analysis.distributions(toy_stage)
    assert dist["n"].sum() == len(toy_stage)
    judged = analysis.distributions(toy_stage.assign(judged=False), judged_only=True)
    assert judged.empty


def test_gating_failures_lead_with_the_toy_failure(toy_stage):
    ex = analysis.gating_failure_examples(toy_stage)
    assert ex.iloc[0]["doc_id"] == "e" and ex.iloc[0]["e_gate"] == pytest.approx(-7.0)


def test_additivity_figure_shows_the_toy_residual(toy_stage, tmp_path):
    info = analysis.additivity_figure(toy_stage, tmp_path / "a.png")
    assert (tmp_path / "a.png").stat().st_size > 0
    # document a: E_rerank -7, E_gate 0, actual utility 0 - off the diagonal
    assert info["share_off_diagonal"] > 0 and info["max_abs_residual"] >= 7.0


def test_run_end_to_end(small_cfg, grid, tmp_path):
    queries, settings = grid
    run_grid(queries, small_cfg, settings, tmp_path / "effects")
    result = analysis.run(tmp_path / "effects", "synthetic", 50, small_cfg,
                          tmp_path / "out", n_permutations=200)
    for name in ("distributions", "diagnosis", "budget_outcomes", "h7", "labels",
                 "gating_failures"):
        assert (tmp_path / "out" / f"{name}.csv").exists()
    assert (tmp_path / "out" / "additivity.png").exists()

    h7 = result["tables"]["h7"]
    n_t = sum(t != 1.0 for t in small_cfg.t_grid)
    assert len(h7) == len(small_cfg.outsider_policies) * n_t
    assert (h7["n_queries"] == len(queries)).all()
    assert set(result["label_counts"]) <= set(QUERY_LABELS)
    assert json.dumps(result["additivity"])
    assert isinstance(result["tables"]["labels"], pd.DataFrame)
