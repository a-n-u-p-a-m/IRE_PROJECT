"""
test_synthetic_e1.py - The independent re-derivation, and E1 on a small grid.

    python -m pytest experiments_shubham
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments_shubham import synthetic_e1 as e1
from experiments_shubham.engine import synthetic_frame
from shared import fixtures as fx
from shared.schema import DELTA_T, E_GATE, E_RERANK


def _toy_ref():
    c = fx.toy_cascade()
    docs = np.array(c["universe"])
    q = e1.RefQuery(
        c["query_id"], docs, np.array([c["s1"][d] for d in docs]),
        np.array([c["s2"][d] for d in docs]),
        np.array([fx.TOY_GAIN_MAP[c["relevances"][d]] for d in docs]),
        np.array([c["alternative_rank"][d] for d in docs]), np.empty(0, dtype=np.int64),
    )
    q.stage1 = e1._order(q, q.s1, np.arange(len(docs)))
    return q


def _value(df, doc, **where):
    sel = df["doc_id"] == doc
    for col, v in where.items():
        sel &= df[col] == v
    assert sel.sum() == 1
    return float(df.loc[sel, "effect"].iloc[0])


def test_rederivation_matches_hand_computation():
    q = _toy_ref()
    stage = e1.ref_stage([q], fx.TOY_K_C, fx.TOY_K)
    assert _value(stage, "a", effect_type=E_RERANK) == pytest.approx(-7.0)
    assert _value(stage, "e", effect_type=E_GATE) == pytest.approx(-7.0)
    budget = e1.ref_budget([q], fx.TOY_K_C, fx.TOY_BUDGET, (1.0, 0.75, 0.5),
                           "boundary", [42], fx.TOY_K)
    assert _value(budget, "e", t=0.5) == pytest.approx(7.0)


def test_rederivation_matches_the_shared_fixture():
    q = _toy_ref()
    stage = e1.ref_stage([q], fx.TOY_K_C, fx.TOY_K)
    budget = e1.ref_budget([q], fx.TOY_K_C, fx.TOY_BUDGET, (1.0, 0.75, 0.5),
                           "boundary", [42], fx.TOY_K).assign(effect_type=DELTA_T)
    ref = fx.toy_effects()
    ref = ref.assign(t=ref["t"].fillna(-1.0), doc_id=ref["doc_id"].astype(str),
                     effect_type=ref["effect_type"].astype(str))
    ours = pd.concat([stage.assign(t=-1.0), budget])
    key = ["doc_id", "effect_type", "t"]
    merged = ours.merge(ref, on=key, suffixes=("", "_fx"))
    assert len(merged) == len(ref) == len(ours)
    assert np.allclose(merged["effect"], merged["effect_fx"], atol=1e-12)


@pytest.mark.parametrize("variant", list(e1.VARIANTS))
def test_engine_is_exactly_recovered(small_cfg, variant):
    recovery, stats = e1.run_variant(small_cfg, variant, None, None)
    assert len(recovery) == 3 * (1 + len(small_cfg.outsider_policies))
    for rec in recovery:
        assert rec["unmatched_rows"] == 0
        assert rec["rank_mismatches"] == 0 and rec["stratum_mismatches"] == 0
        assert rec["max_abs_diff"] <= e1.TOLERANCE


def test_planted_documents_are_the_moved_relevant_ones(small_cfg):
    stressed = synthetic_frame(small_cfg, "gate_stress")
    control = synthetic_frame(small_cfg, "gate_stress", stress=False)
    planted = e1.planted_documents(stressed, control)
    per_query = pd.Series([q for q, _ in planted]).value_counts()
    assert per_query.max() <= small_cfg.synthetic["gate_stress_planted"]
    rel = control.set_index(["query_id", "doc_id"])["relevance"]
    assert all(rel.loc[key] > 0 for key in planted)


def test_policy_checks_are_untestable_when_every_outsider_is_promoted(small_cfg):
    stats = pd.concat(
        [pd.DataFrame(e1.run_variant(small_cfg, v, None, None)[1]) for v in e1.VARIANTS],
        ignore_index=True)
    # With 120 documents and K_c = 100, D_q has 20 members but the smallest t
    # sends 50 outsiders: every policy promotes everything.
    checks = {c["check"]: c for c in e1.evaluate(stats, small_cfg, n_docs=120)}
    for key in ("gate_stress_far_boundary_misses", "budget_stress_retained_lose_more"):
        assert not checks[key]["passed"]
        assert checks[key]["detail"].startswith("not testable")
    assert set(checks) == set(e1.EXPECTATIONS)
