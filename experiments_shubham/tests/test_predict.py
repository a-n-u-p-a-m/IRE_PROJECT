"""
test_predict.py - Predictors and probes against the reference effects.

The four ways the first version went wrong are each pinned here: a Stage-2
score leaking in for a non-survivor, a missing prediction silently becoming the
true value, a per-query budget spent as a total, and non-survivors left with no
prediction at all.

    python -m pytest experiments_shubham/tests/test_predict.py
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from experiments_shubham import predict
from experiments_shubham.engine import (
    EffectSettings,
    run_grid,
    run_worlds,
    synthetic_frame,
    synthetic_inputs,
)
from shared.ordering import order_by
from shared.schema import E_GATE, NON_SURVIVOR, SURVIVOR

K_C = 50


@pytest.fixture(scope="module")
def queries(small_cfg):
    return synthetic_inputs(synthetic_frame(small_cfg, "weak_retriever", n_queries=8))


@pytest.fixture(scope="module")
def split(queries):
    ids = sorted(q.query_id for q in queries)
    return {"train": set(ids[:5]), "val": set(), "test": set(ids[5:])}


def _features(q, cfg):
    return predict.query_features(q, K_C, cfg.final_cutoff_k, cfg.universe_depth_m)


def test_features_never_read_a_non_candidate_stage2_score(small_cfg, queries):
    q = queries[0]
    cands = set(order_by(q.universe, q.s1)[:K_C])
    scrambled = {d: (s if d in cands else -1e6 * (i + 1))
                 for i, (d, s) in enumerate(q.s2.items())}
    before = _features(q, small_cfg)
    after = _features(dataclasses.replace(q, s2=scrambled), small_cfg)
    pd.testing.assert_frame_equal(before, after)


def test_stage2_features_exist_for_survivors_only(small_cfg, queries):
    f = _features(queries[0], small_cfg)
    surv = f[f["survivor"]]
    assert len(surv) == K_C
    assert surv[["s2_score", "s2_rank", "displacement"]].notna().all().all()
    assert f.loc[~f["survivor"], ["s2_score", "s2_rank", "displacement"]].isna().all().all()
    assert f.loc[~f["survivor"], list(predict.SHARED_FEATURES)].notna().all().all()
    rank = np.exp(surv["s1_log_rank"]).round().astype(int)
    assert (surv["displacement"] == rank - surv["s2_rank"]).all()


def test_probing_every_outsider_reproduces_the_engine(small_cfg, queries):
    settings = EffectSettings.from_config(small_cfg, "synthetic")
    stage = run_worlds(queries, K_C, settings)
    gate = stage[stage["effect_type"] == E_GATE].set_index(["query_id", "doc_id"])["effect"]
    for q in queries:
        approx, calls = predict.probe_effects(
            q, K_C, settings.k, settings.gain_map, settings.utility, "boundary",
            len(q.universe), small_cfg.seed)
        assert calls == len(q.universe) - K_C
        ref = gate.loc[q.query_id]
        got = pd.Series({d: approx.get(d, 0.0) for d in ref.index})
        assert np.allclose(got.to_numpy(), ref.to_numpy(), atol=1e-12)


def test_zero_budget_probes_nothing_and_budgets_are_per_query(small_cfg, queries):
    q = queries[0]
    s = EffectSettings.from_config(small_cfg, "synthetic")
    assert predict.probe_effects(q, K_C, s.k, s.gain_map, s.utility, "boundary", 0, 42) == ({}, 0)
    for policy in ("boundary", "alternative", "random"):
        _, calls = predict.probe_effects(q, K_C, s.k, s.gain_map, s.utility, policy, 20, 42)
        assert calls == 20
    _, calls = predict.probe_effects(q, K_C, s.k, s.gain_map, s.utility, "boundary",
                                     10 ** 6, 42)
    assert calls == len(q.universe) - K_C


def test_fit_predict_per_stratum():
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame({
        "query_id": [f"q{i % 10}" for i in range(n)],
        "stratum": [SURVIVOR] * 100 + [NON_SURVIVOR] * 95 + ["rare"] * 5,
        "judged": True, "x": rng.normal(size=n), "y": rng.normal(size=n)})
    frame["effect"] = np.where(frame["stratum"] == SURVIVOR, 2 * frame["x"], 0.5)
    train = {f"q{i}" for i in range(7)}
    pred, info = predict.fit_predict(
        frame, {SURVIVOR: ("x",), NON_SURVIVOR: ("y",), "rare": ("x",)}, train, "ridge", 0)
    assert np.isfinite(pred).all()
    surv = frame["stratum"] == SURVIVOR
    assert np.corrcoef(pred[surv], frame.loc[surv, "effect"])[0, 1] > 0.99
    assert info[NON_SURVIVOR]["fitted"] == "train_mean"           # constant target
    assert info["rare"]["fitted"] == "train_mean"                  # too few rows
    pred2, _ = predict.fit_predict(frame, {SURVIVOR: ("x",)}, train, "ridge", 0)
    assert (pred2[~surv] == 0.0).all()                             # no model -> 0


@pytest.fixture(scope="module")
def effects_dir(small_cfg, queries, tmp_path_factory):
    out = tmp_path_factory.mktemp("effects")
    settings = EffectSettings.from_config(small_cfg, "synthetic")
    run_grid(queries, small_cfg, settings, out, candidate_depths=[K_C])
    return out


def test_every_non_survivor_gets_a_prediction(small_cfg, queries, split, effects_dir):
    stage = predict._read_effects(effects_dir / "synthetic_stage_kc50.parquet")
    feats = predict.feature_table(queries, K_C, small_cfg.final_cutoff_k,
                                  small_cfg.universe_depth_m)
    test, _ = predict.predictor_block(stage[stage["effect_type"] == E_GATE], feats,
                                      E_GATE, split, small_cfg.seed)
    assert set(test["query_id"]) == split["test"]
    for kind in predict.MODELS:
        assert test[f"pred_{kind}"].notna().all()
    nonsurv = test["stratum"] == NON_SURVIVOR
    assert test.loc[nonsurv, "pred_displacement"].isna().all()
    assert test.loc[~nonsurv, "pred_displacement"].notna().all()
    for col in predict.BASELINES:
        assert col in test


def test_run_system_end_to_end(small_cfg, queries, split, effects_dir):
    res = predict.run_system("synthetic", small_cfg, K_C, queries, effects_dir, split,
                             n_iter=20)
    fid = res["fidelity"]
    assert {"pred_ridge", "pred_gbm", "pred_zero", "pred_stratum_mean"} <= set(fid["predictor"])
    assert set(fid["effect_type"]) == {"e_rerank", "e_gate", "delta_t"}
    assert set(fid.dropna(subset=["policy"])["policy"]) == set(small_cfg.outsider_policies)
    assert res["cost"]["exact_at_full_budget_max_abs_diff"] < 1e-9
    probe = res["query_level"]
    # b = 0 probes nothing, so every policy gives the same all-zero G_q in a scope
    zero = probe[(probe["budget"] == 0) & probe["seed"].notna()]
    for _, blk in zero.groupby("scope"):
        assert np.allclose(blk["gq_mae"], blk["gq_mae"].iloc[0])
        assert (blk["gs_predicted"] == 0).all()
    assert (probe.dropna(subset=["seed"])["extra_calls_per_query"]
            <= max(predict._budgets(small_cfg))).all()
    s = predict.summary({"fidelity": fid, "query_level": probe})
    assert "synthetic" in s and s["synthetic"]["fidelity"]
