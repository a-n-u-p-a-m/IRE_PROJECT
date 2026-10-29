"""
test_engine.py - The world and budget runners against the hand-checked toy
cascade, and on every synthetic regime in strict mode.

    python -m pytest experiments_shubham
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

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
from shared.config import load_config
from shared.ordering import MissingScoreError
from shared.schema import DELTA_T, E_GATE, E_RERANK, load
from shared.synthetic import REGIMES

KEY = ["query_id", "doc_id", "effect_type", "t"]


@pytest.fixture
def toy_query():
    c = fx.toy_cascade()
    return QueryInputs(c["query_id"], tuple(c["universe"]), c["s1"], c["s2"],
                       c["relevances"], c["judged"], c["alternative_rank"])


@pytest.fixture
def toy_settings():
    return EffectSettings("toy", fx.TOY_K, fx.TOY_GAIN_MAP)


@pytest.fixture(scope="module")
def cfg():
    return load_config("config/base.yaml")


def _effect(df, effect_type, doc, t=None):
    sel = (df["effect_type"] == effect_type) & (df["doc_id"] == doc)
    if t is not None:
        sel &= df["t"] == t
    assert sel.sum() == 1
    return float(df.loc[sel, "effect"].iloc[0])


def test_world_runner_matches_hand_computation(toy_query, toy_settings):
    eff = run_worlds([toy_query], fx.TOY_K_C, toy_settings)
    assert len(eff) == 2 * len(toy_query.universe)
    assert _effect(eff, E_RERANK, "a") == pytest.approx(-7.0)
    assert _effect(eff, E_GATE, "e") == pytest.approx(-7.0)
    assert _effect(eff, E_GATE, "b") == pytest.approx(3 - 3 / math.log2(3))


def test_runners_agree_with_the_shared_fixture(toy_query, toy_settings):
    stage = run_worlds([toy_query], fx.TOY_K_C, toy_settings)
    budget = run_budget([toy_query], fx.TOY_K_C, fx.TOY_BUDGET, (1.0, 0.75, 0.5),
                        "boundary", [42], toy_settings)
    ours = pd.concat([stage, budget]).set_index(KEY)["effect"].sort_index()
    ref = fx.toy_effects().set_index(KEY)["effect"].sort_index()
    pd.testing.assert_series_equal(ours, ref)


def test_budget_runner_matches_hand_computation(toy_query, toy_settings):
    eff = run_budget([toy_query], fx.TOY_K_C, fx.TOY_BUDGET, (1.0, 0.5),
                     "boundary", [42], toy_settings)
    assert _effect(eff, DELTA_T, "e", 0.5) == pytest.approx(7.0)
    assert _effect(eff, DELTA_T, "b", 0.5) == pytest.approx(3 / math.log2(3) - 3)
    assert not (eff["t"] == 1.0).any(), "t=1 is the baseline, not an effect row"


def test_only_the_random_policy_repeats_over_seeds(toy_query, toy_settings):
    seeds = [42, 43, 44]
    boundary = run_budget([toy_query], fx.TOY_K_C, fx.TOY_BUDGET, (1.0, 0.5),
                          "boundary", seeds, toy_settings)
    random = run_budget([toy_query], fx.TOY_K_C, fx.TOY_BUDGET, (1.0, 0.5),
                        "random", seeds, toy_settings)
    assert boundary["seed"].isna().all() and len(boundary) == 10
    assert sorted(random["seed"].unique()) == seeds and len(random) == 30


def test_missing_stage2_score_raises(toy_query, toy_settings):
    # An incomplete cache must crash, never silently shrink the no-gate world
    s2 = {d: v for d, v in toy_query.s2.items() if d != "i"}
    q = QueryInputs(toy_query.query_id, toy_query.universe, toy_query.s1, s2,
                    toy_query.relevance, toy_query.judged)
    with pytest.raises(MissingScoreError):
        run_worlds([q], fx.TOY_K_C, toy_settings)


def test_gate_that_admits_everything_is_refused(toy_query, toy_settings):
    with pytest.raises(ValueError, match="not smaller than"):
        run_worlds([toy_query], len(toy_query.universe), toy_settings)


def test_t_grid_without_baseline_is_refused(toy_query, toy_settings):
    with pytest.raises(ValueError, match="must contain 1.0"):
        run_budget([toy_query], fx.TOY_K_C, fx.TOY_BUDGET, (0.75, 0.5),
                   "boundary", [42], toy_settings)


def test_budget_smaller_than_kc_is_a_candidate_only_baseline(toy_query, toy_settings):
    eff = run_budget([toy_query], fx.TOY_K_C, 2, (1.0, 0.5), "boundary", [42],
                     toy_settings)
    assert set(eff["budget_b"]) == {2}


@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_every_regime_passes_strict_checks(cfg, regime):
    queries = synthetic_inputs(
        synthetic_frame(cfg, regime, n_queries=3, n_docs=300, stress_kc=100))
    settings = EffectSettings.from_config(cfg, "synthetic")
    stage = run_worlds(queries, 100, settings)
    assert len(stage) == 3 * 300 * 2
    for policy in cfg.outsider_policies:
        run_budget(queries, 100, 100, cfg.t_grid, policy, cfg.budget_seeds, settings)


def test_grid_writes_validated_tables_and_manifest(cfg, tmp_path):
    queries = synthetic_inputs(synthetic_frame(cfg, "strong_stages", n_queries=2,
                                               n_docs=1100))
    written = run_grid(queries, cfg, EffectSettings.from_config(cfg, "synthetic"),
                       tmp_path)

    tables = {n: p for n, p in written.items() if p.suffix == ".parquet"}
    n_kc, n_policies = len(cfg.candidate_depths), len(cfg.outsider_policies)
    assert len(tables) == len(MODES) * n_kc * (1 + n_policies)
    for name, path in tables.items():
        eff = load(path, "effect", strict=True)
        assert (eff["condensed"] == name.endswith("_condensed")).all()

    manifest = json.loads(written["manifest"].read_text())
    assert manifest["config_fingerprint"] == cfg.fingerprint()
    assert manifest["owner"] == "shubham" and manifest["strict"] is True
    assert manifest["budgets"] == {str(k): k for k in cfg.candidate_depths}
    assert manifest["condensed_modes"] == [False, True]


# ── condensed mode ────────────────────────────────────────────────────

def _with_unjudged(queries, share, seed=7):
    """The same queries with a random share of documents unjudged."""
    rng = np.random.default_rng(seed)
    out = []
    for q in queries:
        judged = {d: bool(rng.random() >= share) for d in q.universe}
        out.append(replace(q, judged=judged, relevance={
            d: r for d, r in q.relevance.items() if judged[d]}))
    return out


def _recount(plain, queries, settings):
    """
    Condensed ranks and effects derived from the uncondensed table alone: a
    judged document's condensed rank is its position among the judged
    documents of the same uncondensed ranking.
    """
    gain = {(q.query_id, d): settings.gain_map.get(int(r), 0.0)
            for q in queries for d, r in q.relevance.items()}
    df = plain[plain["judged"].fillna(False)].copy()
    g = np.array([gain.get(key, 0.0) for key in zip(df["query_id"], df["doc_id"])])
    groups = df.groupby(["query_id", "effect_type", "t", "seed"], dropna=False,
                        observed=True)
    utility = {}
    for col in ("rank_treatment", "rank_control"):
        r = groups[col].rank(method="first").to_numpy(float)
        df[col] = r
        inside = ~np.isnan(r) & (r <= settings.k)
        utility[col] = np.where(inside, g / np.log2(1.0 + np.where(inside, r, 1.0)), 0.0)
    df["effect"] = utility["rank_treatment"] - utility["rank_control"]
    return df


def _by_key(df):
    keys = ["query_id", "doc_id", "effect_type", "t", "seed"]
    out = df.assign(t=df["t"].fillna(-1.0), seed=df["seed"].astype("Float64").fillna(-1))
    return out.set_index(keys).sort_index()


@pytest.mark.parametrize("policy", [None, "boundary", "random"])
def test_condensed_effects_recount_the_uncondensed_ranks(cfg, policy):
    queries = _with_unjudged(synthetic_inputs(synthetic_frame(
        cfg, "weak_retriever", n_queries=3, n_docs=400, stress_kc=100)), share=0.6)
    settings = EffectSettings.from_config(cfg, "synthetic")
    assert settings.utility == "dcg"

    def run(condensed):
        if policy is None:
            return run_worlds(queries, 100, settings, condensed=condensed)
        return run_budget(queries, 100, 100, cfg.t_grid, policy, cfg.budget_seeds,
                          settings, condensed=condensed)

    plain, cond = run(False), run(True)
    got, want = _by_key(cond), _by_key(_recount(plain, queries, settings))
    assert got.index.equals(want.index)          # exactly the judged documents
    for col in ("rank_treatment", "rank_control"):
        assert np.array_equal(got[col].astype("Float64").fillna(0).to_numpy(float),
                              np.nan_to_num(want[col].to_numpy(float)))
    assert np.allclose(got["effect"].to_numpy(float), want["effect"].to_numpy(float),
                       atol=1e-12)
    # not vacuous: condensing really moved some judged documents
    before = _by_key(plain).loc[got.index, "rank_control"].astype("Float64")
    assert (before.fillna(0) != got["rank_control"].astype("Float64").fillna(0)).any()


def test_condensing_changes_nothing_when_everything_is_judged(cfg):
    queries = synthetic_inputs(synthetic_frame(cfg, "weak_retriever", n_queries=2,
                                               n_docs=300, stress_kc=100))
    settings = EffectSettings.from_config(cfg, "synthetic")
    plain = run_worlds(queries, 100, settings)
    cond = run_worlds(queries, 100, settings, condensed=True)
    pd.testing.assert_frame_equal(plain.drop(columns="condensed"),
                                  cond.drop(columns="condensed"))
