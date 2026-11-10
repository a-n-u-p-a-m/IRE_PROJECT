"""
test_gating_examples.py - The frozen selection rule and the rendered examples,
on the hand-checked toy cascade and a synthetic grid.

    python -m pytest experiments_shubham/tests/test_gating_examples.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from experiments_shubham import gating_examples as ge
from experiments_shubham.engine import (
    EffectSettings,
    QueryInputs,
    run_worlds,
    synthetic_frame,
    synthetic_inputs,
)
from shared import fixtures as fx
from shared.effects import gating_failures


@pytest.fixture(scope="module")
def stage(small_cfg):
    df = synthetic_frame(small_cfg, "gate_stress", n_queries=6)
    return run_worlds(synthetic_inputs(df), 50, EffectSettings.from_config(small_cfg, "bm25"))


def test_toy_failure_is_selected():
    c = fx.toy_cascade()
    q = QueryInputs(c["query_id"], tuple(c["universe"]), c["s1"], c["s2"],
                    c["relevances"], c["judged"], c["alternative_rank"])
    toy = run_worlds([q], fx.TOY_K_C, EffectSettings("toy", fx.TOY_K, fx.TOY_GAIN_MAP))
    picked = ge.select(toy)
    worst = gating_failures(toy).iloc[0]
    assert list(picked["doc_id"]) == [worst["doc_id"]]
    assert picked["e_gate"].iloc[0] == pytest.approx(worst["effect"])


def test_rule_is_worst_per_query_largest_losses_first(stage):
    picked = ge.select(stage, n=3)
    fails = gating_failures(stage)
    assert picked["query_id"].is_unique and len(picked) == 3
    for row in picked.itertuples():
        mine = fails[fails["query_id"] == row.query_id]
        assert row.e_gate == pytest.approx(mine["effect"].min())
    worst_per_query = fails.groupby("query_id")["effect"].min().sort_values()
    assert list(picked["e_gate"]) == pytest.approx(list(worst_per_query.iloc[:3]))
    top = stage[(stage["effect_type"] == "e_rerank") & (stage["rank_treatment"] == 1)]
    tops = dict(zip(top["query_id"], top["doc_id"]))
    assert all(tops[r.query_id] == r.actual_top_doc_id for r in picked.itertuples())


def test_render_flags_passages_stage1_never_returned():
    ex = pd.DataFrame([{"system": "bm25", "k_c": 100, "query_id": "q1", "doc_id": "d9",
                        "rank_without_gate": 1, "e_gate": -7.0, "actual_top_doc_id": "d1",
                        "query": "what is famvir", "grade": 3, "stage1_rank": 9203,
                        "stage1_retrieved": False, "depth_m": 5000, "actual_top_grade": 1}])
    text = ge.render_markdown(ex, {"d9": "Famvir treats shingles.", "d1": "Famvir is a drug."})
    assert "not in BM25's top 5,000" in text and "Famvir treats shingles." in text
    assert "grade 3" in text and "grade 1" in text
    with pytest.raises(KeyError, match="d1"):
        ge.render_markdown(ex, {"d9": "only one"})
