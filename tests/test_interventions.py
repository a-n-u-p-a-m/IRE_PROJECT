"""
test_interventions.py — remove_stage2 and remove_gating on hand-made pipelines.

Run with:  pytest tests/test_interventions.py -v
"""

import pandas as pd
import pytest

from interventions import remove_gating, remove_stage2
from shared.schema import VALID_INTERVENTIONS


@pytest.fixture
def score_matrix() -> pd.DataFrame:
    """One query, 4 candidates plus a padding row.

    Stage 1 order: a, b, c, d.  Reranker order: c, a, d, b.
    """
    return pd.DataFrame({
        "query_id":         ["q1"] * 5,
        "doc_id":           ["a", "b", "c", "d", "__PAD_4__"],
        "stage_1_rank":     [1, 2, 3, 4, 5],
        "stage_1_score":    [0.9, 0.8, 0.7, 0.6, -1e9],
        "in_candidate_set": [True, True, True, True, False],
        "stage_2_rank":     [2, 4, 1, 3, None],
        "stage_2_score":    [5.0, 1.0, 9.0, 3.0, None],
        "final_rank":       [2, 4, 1, 3, 5],
        "relevance_label":  [None] * 5,
    })


class FakeRetriever:
    """Stage 1 top-k for q1 is a, b, c, d, e, f (in that order)."""

    def search(self, query, k):
        return [(d, 1.0 - i / 10) for i, d in enumerate("abcdef")][:k]


class FakeReranker:
    """Scores taken from a fixed table; records what it was asked to score."""

    TABLE = {"passage e": 7.0, "passage f": 0.5}

    def __init__(self):
        self.scored = []

    def score(self, query, texts):
        self.scored += list(texts)
        return [self.TABLE[t] for t in texts]


def _by_doc(df):
    return df.set_index("doc_id")


class TestRemoveStage2:

    def test_ranks_and_displacement(self, score_matrix):
        df = _by_doc(remove_stage2.run(score_matrix))
        assert list(df.loc["c", ["rank_before", "rank_after", "displacement"]]) == [1, 3, 2]
        assert list(df.loc["b", ["rank_before", "rank_after", "displacement"]]) == [4, 2, -2]

    def test_padding_excluded_and_schema_valid(self, score_matrix):
        df = remove_stage2.run(score_matrix)
        assert set(df["doc_id"]) == {"a", "b", "c", "d"}
        assert set(df["intervention_type"]) <= VALID_INTERVENTIONS
        assert df["seed"].isna().all()


class TestRemoveGating:

    def _run(self, score_matrix, k=6):
        reranker = FakeReranker()
        df = remove_gating.run(
            score_matrix, FakeRetriever(), reranker, {"q1": "query"},
            load_passages=lambda ids: {d: f"passage {d}" for d in ids}, expanded_k=k,
        )
        return _by_doc(df), reranker

    def test_new_competitor_pushes_documents_down(self, score_matrix):
        # Expanded reranked order: c 9, e 7 (new), a 5, d 3, b 1, f 0.5 (new)
        df, _ = self._run(score_matrix)
        assert set(df.index) == {"a", "b", "c", "d"}          # rows only for original docs
        assert df.loc["c", "rank_after"] == 1 and df.loc["c", "displacement"] == 0
        assert df.loc["a", "rank_after"] == 3 and df.loc["a", "displacement"] == 1
        assert df.loc["b", "rank_after"] == 5 and df.loc["b", "displacement"] == 1

    def test_only_new_candidates_are_scored(self, score_matrix):
        _, reranker = self._run(score_matrix)
        assert sorted(reranker.scored) == ["passage e", "passage f"]

    def test_no_new_candidates_means_no_movement(self, score_matrix):
        df, reranker = self._run(score_matrix, k=4)
        assert (df["displacement"] == 0).all()
        assert reranker.scored == []

    def test_requires_reranked_score_matrix(self, score_matrix):
        sm = score_matrix.copy()
        sm.loc[0, "stage_2_score"] = None
        with pytest.raises(ValueError, match="no Stage 2 scores"):
            remove_gating.run(sm, FakeRetriever(), FakeReranker(), {"q1": "query"},
                              load_passages=lambda ids: {})
