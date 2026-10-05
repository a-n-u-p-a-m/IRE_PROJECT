"""
test_attribution_sanity.py — Attribution methods on hand-crafted pipelines.

Run with:  pytest tests/test_attribution_sanity.py -v
"""

import numpy as np
import pandas as pd
import pytest

from attribution import stage_shapley
from attribution.stage_shapley import compute_attribution, shapley_two_stage


def _score_matrix(rows):
    """rows: (doc_id, stage_1_rank, final_rank) for query q1."""
    return pd.DataFrame({
        "query_id": "q1",
        "doc_id": [d for d, _, _ in rows],
        "stage_1_rank": [s for _, s, _ in rows],
        "stage_1_score": [1.0 / s for _, s, _ in rows],
        "in_candidate_set": True,
        "stage_2_rank": [f for _, _, f in rows],
        "stage_2_score": [1.0 / f for _, _, f in rows],
        "final_rank": [f for _, _, f in rows],
        "relevance_label": None,
    })


@pytest.fixture
def pipeline():
    #           doc  stage-1 rank  final rank
    return _score_matrix([
        ("same",     1,  1),    # reranker kept it in place
        ("promoted", 50, 2),    # reranker lifted it from deep in Stage 1
        ("demoted",  3,  5),    # reranker pushed it down
        ("outside",  8,  20),   # not in the final top-k
    ])


def _by_doc(attr):
    return attr.set_index("doc_id")


class TestStageShapley:

    def test_efficiency(self):
        v1, v12, v2 = np.array([0.2, 1.0]), np.array([1.0, 0.5]), np.array([0.0, 0.3])
        phi1, phi2 = shapley_two_stage(v1, v12, v2)
        assert np.allclose(phi1 + phi2, v12)

    def test_unchanged_rank_is_all_stage_1(self, pipeline):
        a = _by_doc(compute_attribution(pipeline, top_k=10))
        assert a.loc["same", "stage_1_attribution"] == pytest.approx(1.0)
        assert a.loc["same", "shapley_stage_2"] == pytest.approx(0.0)

    def test_promotion_credits_stage_2(self, pipeline):
        # rr: v1 = 1/50, v12 = 1/2 -> share_1 = (v1 + v12) / (2 v12) = 0.52
        a = _by_doc(compute_attribution(pipeline, top_k=10))
        assert a.loc["promoted", "stage_1_attribution"] == pytest.approx(0.52)
        assert a.loc["promoted", "stage_2_attribution"] == pytest.approx(0.48)

    def test_demotion_gives_stage_2_no_credit(self, pipeline):
        a = _by_doc(compute_attribution(pipeline, top_k=10))
        assert a.loc["demoted", "shapley_stage_2"] < 0              # raw value is negative
        assert a.loc["demoted", "stage_1_attribution"] == pytest.approx(1.0)

    def test_stage_1_floor_without_stage2_alone(self, pipeline):
        # Stage 1 is a necessary player when v({S2}) = 0
        a = compute_attribution(pipeline, top_k=10)
        assert (a["stage_1_attribution"] >= 0.5).all()

    def test_only_top_k_and_schema(self, pipeline):
        a = compute_attribution(pipeline, top_k=10)
        assert set(a["doc_id"]) == {"same", "promoted", "demoted"}
        assert (a["method_name"] == "stage_shapley").all()
        assert np.allclose(a["stage_1_attribution"] + a["stage_2_attribution"], 1.0)
        assert a["gating_attribution"].isna().all()

    @pytest.mark.parametrize("value, expected", [("dcg", (1 / np.log2(51) + 1 / np.log2(3)) / (2 / np.log2(3))),
                                                 ("topk", 0.5)])
    def test_alternative_value_functions(self, pipeline, value, expected):
        a = _by_doc(compute_attribution(pipeline, top_k=10, value=value))
        assert a.loc["promoted", "stage_1_attribution"] == pytest.approx(expected)

    def test_unknown_value_function(self, pipeline):
        with pytest.raises(ValueError, match="value must be one of"):
            compute_attribution(pipeline, value="ndcg")

    def test_stage2_alone_can_make_stage_2_dominant(self, pipeline):
        # If the reranker alone ranks "promoted" first, Stage 2 earns most credit
        alone = pd.DataFrame({"query_id": "q1", "doc_id": ["same", "promoted", "demoted"],
                              "rank": [2, 1, 3]})
        a = _by_doc(compute_attribution(pipeline, top_k=10, stage2_alone=alone))
        assert a.loc["promoted", "stage_2_attribution"] > 0.9
        assert a.loc["promoted", "stage_1_attribution"] < 0.5

    def test_stage2_alone_from_remove_gating(self, pipeline):
        gating = pd.DataFrame({"query_id": "q1", "doc_id": ["same", "promoted", "demoted"],
                               "intervention_type": "remove_gating", "seed": None,
                               "rank_before": [1, 2, 5], "rank_after": [1, 4, 6],
                               "displacement": [0, 2, 1]})
        alone = stage_shapley.stage2_alone_from_intervention(gating)
        assert list(alone.columns) == ["query_id", "doc_id", "rank"]
        compute_attribution(pipeline, top_k=10, stage2_alone=alone)   # runs end to end

    def test_stage2_alone_missing_document(self, pipeline):
        alone = pd.DataFrame({"query_id": "q1", "doc_id": ["same"], "rank": [1]})
        with pytest.raises(ValueError, match="no rank for 2"):
            compute_attribution(pipeline, top_k=10, stage2_alone=alone)
