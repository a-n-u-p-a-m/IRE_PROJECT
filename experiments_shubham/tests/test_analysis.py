"""
test_analysis.py - The effect-table analysis agrees with shared.diagnosis, and
the whole run works on a small synthetic grid.

    python -m pytest experiments_shubham
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from experiments_shubham import analysis
from experiments_shubham.engine import (
    MODES,
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
from shared.schema import SchemaError
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


def test_diagnosis_refuses_a_frame_that_mixes_modes(toy_stage):
    both = pd.concat([toy_stage, toy_stage.assign(condensed=True)], ignore_index=True)
    with pytest.raises(SchemaError, match="mixes condensed"):
        analysis.diagnose(both, 0.05)
    with pytest.raises(SchemaError, match="mixes condensed"):
        analysis.budget_outcomes({"boundary": both})


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


def test_unjudged_at_k_counts_each_worlds_top_k(grid):
    queries, settings = grid
    rng = np.random.default_rng(3)
    queries = [replace(q, judged={d: True for d in q.universe if rng.random() < 0.3})
               for q in queries]
    for k_c in (20, 50):
        got = analysis.unjudged_at_k(run_worlds(queries, k_c, settings))
        got = got.set_index(["query_id", "world"])
        assert len(got) == 3 * len(queries)
        for q in queries:
            w = build_worlds(q.query_id, q.universe, q.s1, q.s2, k_c)
            for world, ranking in (("A", w.pi_act), ("B", w.pi_no2), ("C", w.pi_no1)):
                top = ranking[:settings.k]
                row = got.loc[(q.query_id, world)]
                assert row["n_top"] == len(top)
                assert row["n_unjudged"] == sum(d not in q.judged for d in top)

        condensed = analysis.unjudged_at_k(run_worlds(queries, k_c, settings,
                                                      condensed=True))
        assert (condensed["n_unjudged"] == 0).all()


def test_run_end_to_end(small_cfg, grid, tmp_path):
    queries, settings = grid
    run_grid(queries, small_cfg, settings, tmp_path / "effects")
    result = analysis.run(tmp_path / "effects", "synthetic", 50, small_cfg,
                          tmp_path / "out", n_permutations=200)
    for name in ("distributions", "diagnosis", "budget_outcomes", "h7", "labels",
                 "gating_failures", "unjudged_at_k"):
        assert (tmp_path / "out" / f"{name}.csv").exists()
    assert (tmp_path / "out" / "additivity.png").exists()
    assert (tmp_path / "out" / "additivity_condensed.png").exists()
    for df in result["tables"].values():
        assert set(df["condensed"]) == set(MODES)

    h7 = result["tables"]["h7"]
    n_t = sum(t != 1.0 for t in small_cfg.t_grid)
    assert len(h7) == len(MODES) * len(small_cfg.outsider_policies) * n_t
    assert (h7["n_queries"] == len(queries)).all()
    assert set(result["label_counts"]) == {"uncondensed", "condensed"}
    for counts in result["label_counts"].values():
        assert set(counts) <= set(QUERY_LABELS)
    assert json.dumps(result["additivity"])
    assert isinstance(result["tables"]["labels"], pd.DataFrame)


def _diag(gate_sensitive, queries=("q1", "q2", "q3", "q4", "q5")):
    return pd.DataFrame({"query_id": list(queries), "condensed": False,
                         "label": ["gate_sensitive" if q in gate_sensitive
                                   else "rerank_sensitive" for q in queries]})


def test_label_contrast_is_paired():
    out = analysis.label_contrast(_diag({"q1", "q2", "q3"}), _diag({"q3", "q4"}))
    assert (out["n_a"], out["n_b"], out["both"], out["only_a"], out["only_b"]) == (3, 2, 1, 2, 1)
    assert out["p_one_sided"] == pytest.approx(0.5)        # P(X >= 2 | n = 3)
    assert out["p_two_sided"] == pytest.approx(1.0)
    five_none = analysis.label_contrast(_diag({"q1", "q2", "q3", "q4", "q5"}), _diag(set()))
    assert five_none["p_one_sided"] == pytest.approx(0.5 ** 5)


def test_label_contrast_refuses_unpaired_queries():
    with pytest.raises(ValueError, match="same queries"):
        analysis.label_contrast(_diag({"q1"}), _diag({"q1"}, queries=("q1", "q2")))


def test_contrast_reads_both_systems_in_each_mode(small_cfg, grid, tmp_path):
    from shared.schema import save
    queries, settings = grid
    weaker = synthetic_inputs(synthetic_frame(small_cfg, "strong_stages", n_queries=8))
    for system, qs in (("a", queries), ("b", weaker)):
        st = replace(settings, stage1_system=system)
        for condensed in MODES:
            save(run_worlds(qs, 50, st, condensed=condensed),
                 tmp_path / f"{analysis.table_name(system, 50, None, condensed)}.parquet",
                 "effect")
    out = analysis.contrast(tmp_path, "a", "b", 50, small_cfg)
    assert list(out["condensed"]) == list(MODES)
    assert (out["n_queries"] == len(queries)).all()
    for _, row in out.iterrows():
        diag = [analysis.diagnose(analysis.load_stage(tmp_path, s, 50, row["condensed"]),
                                  small_cfg.tau_query) for s in ("a", "b")]
        assert row["n_a"] == (diag[0]["label"] == "gate_sensitive").sum()
        assert row["n_b"] == (diag[1]["label"] == "gate_sensitive").sum()
