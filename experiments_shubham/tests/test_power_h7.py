"""
test_power_h7.py - The pool simulator and the size recommendation.

    python -m pytest experiments_shubham
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments_shubham import power_h7
from shared.diagnosis import GATE_SENSITIVE, QUERY_LABELS, RERANK_SENSITIVE


def test_pool_has_one_row_per_query_depth_policy_and_t(small_cfg):
    pool = power_h7.simulate_pool(small_cfg, pool_size=3)
    n_t = sum(t != 1.0 for t in small_cfg.t_grid)
    expected = 3 * len(small_cfg.candidate_depths) * len(small_cfg.outsider_policies) * n_t
    assert len(pool) == expected
    assert set(pool["label"]) <= set(QUERY_LABELS)
    # a query's diagnosis does not depend on the policy or t it is paired with
    per_query = pool.groupby(["query", "k_c"])["label"].nunique()
    assert (per_query == 1).all()


def test_pool_is_reproducible(small_cfg):
    a = power_h7.simulate_pool(small_cfg, pool_size=2, seed=3)
    b = power_h7.simulate_pool(small_cfg, pool_size=2, seed=3)
    pd.testing.assert_frame_equal(a, b)


def _pool(small_cfg, gate_mean):
    rng = np.random.default_rng(0)
    rows = []
    for k_c in small_cfg.candidate_depths:
        for policy in small_cfg.outsider_policies:
            for t in small_cfg.t_grid:
                if t == 1.0:
                    continue
                for i in range(200):
                    gate = i % 2 == 0
                    rows.append({
                        "query": i, "k_c": k_c, "policy": policy, "t": t,
                        "label": GATE_SENSITIVE if gate else RERANK_SENSITIVE,
                        "delta_dcg": rng.normal(gate_mean if gate else 0.0, 1.0),
                    })
    return pd.DataFrame(rows)


def test_a_clear_separation_gets_a_size(small_cfg):
    power = power_h7.power_tables(_pool(small_cfg, gate_mean=3.0), small_cfg)
    rec = power_h7.recommend(power, small_cfg)
    assert rec["reference_size"] in small_cfg.power["candidate_sizes"]


def test_no_separation_gets_no_size(small_cfg):
    power = power_h7.power_tables(_pool(small_cfg, gate_mean=0.0), small_cfg)
    rec = power_h7.recommend(power, small_cfg)
    assert rec["reference_size"] is None
