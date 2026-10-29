"""
test_power_h7.py - The pool simulator and the size recommendation.

    python -m pytest experiments_shubham
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from experiments_shubham import power_h7
from shared.diagnosis import GATE_SENSITIVE, QUERY_LABELS, RERANK_SENSITIVE


def test_pool_has_one_row_per_query_depth_policy_and_t(small_cfg):
    pool = power_h7.simulate_pool(small_cfg, pool_size=3)
    n_t = sum(t != 1.0 for t in small_cfg.t_grid)
    expected = 3 * len(small_cfg.candidate_depths) * len(small_cfg.outsider_policies) * n_t
    assert len(pool) == expected
    assert set(pool["label"]) <= set(QUERY_LABELS)
    assert set(pool["n_docs"]) == {small_cfg.synthetic["n_docs_per_query"]}
    # a query's diagnosis does not depend on the policy or t it is paired with
    per_query = pool.groupby(["query", "k_c"])["label"].nunique()
    assert (per_query == 1).all()


def test_pool_is_reproducible(small_cfg):
    a = power_h7.simulate_pool(small_cfg, pool_size=2, seed=3)
    b = power_h7.simulate_pool(small_cfg, pool_size=2, seed=3)
    pd.testing.assert_frame_equal(a, b)


def _pool(small_cfg, gate_mean):
    """Gate-sensitive queries gain gate_mean more, and G_q tracks the gain."""
    rng = np.random.default_rng(0)
    rows = []
    for k_c in small_cfg.candidate_depths:
        for policy in small_cfg.outsider_policies:
            for t in small_cfg.t_grid:
                if t == 1.0:
                    continue
                for i in range(200):
                    gate = i % 2 == 0
                    delta = rng.normal(gate_mean if gate else 0.0, 1.0)
                    g_q = (delta if gate_mean else 0.0) + rng.normal(0.0, 1.0)
                    rows.append({
                        "n_docs": 300, "query": i, "k_c": k_c, "policy": policy, "t": t,
                        "label": GATE_SENSITIVE if gate else RERANK_SENSITIVE,
                        "g_q": g_q, "delta_dcg": delta,
                    })
    return pd.DataFrame(rows)


def _power(small_cfg, gate_mean):
    pool = _pool(small_cfg, gate_mean)
    sizes = small_cfg.power["candidate_sizes"]
    return pd.concat([power_h7.group_power(pool, small_cfg, sizes),
                      power_h7.correlation_power(pool, small_cfg, sizes)])


def test_a_clear_separation_gets_a_size_from_both_tests(small_cfg):
    rec = power_h7.recommend(_power(small_cfg, gate_mean=3.0), small_cfg)
    for test in ("group", "continuous"):
        assert rec[test]["reference_size"] in small_cfg.power["candidate_sizes"]


def test_no_separation_gets_no_size(small_cfg):
    rec = power_h7.recommend(_power(small_cfg, gate_mean=0.0), small_cfg)
    assert rec["group"]["reference_size"] is None
    assert rec["continuous"]["reference_size"] is None


def test_untestable_subsets_count_as_misses(small_cfg):
    # Only 1% gate-sensitive: almost no subset of 6 can be tested at all
    pool = _pool(small_cfg, gate_mean=3.0)
    pool["label"] = RERANK_SENSITIVE
    pool.loc[pool["query"] < 2, "label"] = GATE_SENSITIVE
    table = power_h7.group_power(pool, small_cfg, [6])
    # The conditional figure is the optimistic one: it throws the untestable
    # draws away instead of scoring them as the H7 failures they are.
    assert (
        table["unconditional_power"] <= table["power_given_testable"].fillna(1.0)
    ).all()
    assert (table["unconditional_power"] < 0.2).all()


def test_headline_k_c_comes_from_the_feasibility_report(small_cfg, tmp_path):
    report = tmp_path / "feasibility_manifest.json"
    assert power_h7.headline_k_c(small_cfg, report)[0] == max(small_cfg.candidate_depths)

    report.write_text(json.dumps({"config_fingerprint": small_cfg.fingerprint(),
                                  "headline_k_c": 20}))
    assert power_h7.headline_k_c(small_cfg, report) == (20, str(report))

    report.write_text(json.dumps({"config_fingerprint": "000000000000",
                                  "headline_k_c": 20}))
    with pytest.raises(ValueError, match="different freezes"):
        power_h7.headline_k_c(small_cfg, report)


def test_realistic_universe_comes_from_the_frozen_universe(tmp_path):
    record = tmp_path / "universe_manifest.json"
    assert power_h7.realistic_universe(record)[0] == power_h7.FALLBACK_UNIVERSE
    record.write_text(json.dumps({"universe_size_median": 9097}))
    assert power_h7.realistic_universe(record) == (9097, str(record))
