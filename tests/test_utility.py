"""
The document-level outcome function, and the gain-map trap.

The TREC DL mapping is the headline case. Grade 1 means "Related" - on topic,
does not answer the query - so the usual 2^rel-1 awards it a gain of 1. Because
every estimand here is a DIFFERENCE of utilities, that error does not cancel:
it inflates every effect in which a merely-related passage moves.
`test_trec_dl_grade_one_earns_nothing` pins the correct behaviour, and
`test_wrong_gain_map_inflates_effects` demonstrates what the wrong one costs.
"""

from __future__ import annotations

import math

import pytest

from shared.utility import (
    EXPONENTIAL_GAIN,
    check_aggregation,
    dcg_at_k,
    document_utility,
    gain,
    ideal_dcg_at_k,
    is_monotone,
    ndcg_at_k,
    rank_of,
    ranking_utilities,
    ranks_of,
)

TREC_DL = {0: 0.0, 1: 0.0, 2: 3.0, 3: 7.0}


def test_trec_dl_grade_one_earns_nothing():
    assert gain(0, TREC_DL) == 0.0
    assert gain(1, TREC_DL) == 0.0
    assert gain(2, TREC_DL) == 3.0
    assert gain(3, TREC_DL) == 7.0


def test_exponential_map_differs_exactly_at_grade_one():
    assert EXPONENTIAL_GAIN[1] == 1.0
    assert {g: EXPONENTIAL_GAIN[g] for g in (0, 2, 3)} == {
        g: TREC_DL[g] for g in (0, 2, 3)
    }


def test_wrong_gain_map_inflates_effects():
    """
    A related-but-not-relevant passage moving from rank 3 to rank 1 produces a
    nonzero effect under the exponential map and exactly zero under the TREC DL
    map. One of those is a measurement; the other is an artefact.
    """
    treatment = document_utility(1, 1, 10, EXPONENTIAL_GAIN)
    control = document_utility(3, 1, 10, EXPONENTIAL_GAIN)
    assert treatment - control > 0.4

    treatment = document_utility(1, 1, 10, TREC_DL)
    control = document_utility(3, 1, 10, TREC_DL)
    assert treatment - control == 0.0


def test_unmapped_grade_is_an_error_not_a_zero():
    """
    Letting an unmapped grade fall through to zero is how a collection with a
    4-point scale silently becomes a 3-point one.
    """
    with pytest.raises(KeyError, match="not in the frozen gain map"):
        gain(4, TREC_DL)


def test_unjudged_contributes_nothing_but_is_not_non_relevant():
    assert document_utility(1, None, 10, TREC_DL) == 0.0
    assert document_utility(1, 0, 10, TREC_DL) == 0.0


def test_dcg_contribution():
    assert document_utility(1, 3, 10, TREC_DL) == pytest.approx(7.0)
    assert document_utility(3, 2, 10, TREC_DL) == pytest.approx(3.0 / math.log2(4))


def test_below_cutoff_and_absent_are_both_zero():
    assert document_utility(11, 3, 10, TREC_DL) == 0.0
    assert document_utility(None, 3, 10, TREC_DL) == 0.0
    assert document_utility(0, 3, 10, TREC_DL) == 0.0


def test_topk_is_binary_on_positive_gain():
    assert document_utility(1, 2, 10, TREC_DL, "topk") == 1.0
    assert document_utility(1, 1, 10, TREC_DL, "topk") == 0.0   # grade 1 earns 0
    assert document_utility(11, 3, 10, TREC_DL, "topk") == 0.0


def test_rbp_decays_geometrically():
    a = document_utility(1, 2, 10, TREC_DL, "rbp", persistence=0.8)
    b = document_utility(2, 2, 10, TREC_DL, "rbp", persistence=0.8)
    assert b == pytest.approx(0.8 * a)


def test_aggregation_identity_holds():
    ranking = ["a", "b", "c", "d", "e"]
    rel = {"a": 3, "b": 0, "c": 1, "d": 2, "e": 3}
    for k in (3, 5, 10):
        assert check_aggregation(ranking, rel, k, TREC_DL)


def test_aggregation_identity_with_unjudged_documents():
    ranking = ["a", "b", "c"]
    rel = {"a": 2}
    assert check_aggregation(ranking, rel, 3, TREC_DL)
    per_doc = ranking_utilities(ranking, rel, 3, TREC_DL)
    assert sum(per_doc.values()) == pytest.approx(dcg_at_k(ranking, rel, 3, TREC_DL))


def test_ndcg_is_bounded_and_zero_without_relevant_documents():
    rel = {"a": 2, "b": 0}
    assert 0.0 <= ndcg_at_k(["a", "b"], rel, 2, TREC_DL) <= 1.0
    assert ndcg_at_k(["b"], {"b": 0}, 2, TREC_DL) == 0.0
    assert ideal_dcg_at_k({"a": 3}, 1, TREC_DL) == pytest.approx(7.0)


@pytest.mark.parametrize("kind", ["dcg", "topk", "rbp"])
def test_every_supported_utility_is_monotone_in_rank(kind):
    assert is_monotone(kind)
    values = [document_utility(r, 3, 10, TREC_DL, kind) for r in range(1, 11)]
    assert all(a >= b for a, b in zip(values, values[1:])), (
        f"{kind} is not non-increasing in rank, so the survivor invariant "
        "E_gate >= 0 would not follow from it"
    )


def test_unknown_utility_is_rejected():
    with pytest.raises(ValueError, match="unknown utility"):
        document_utility(1, 2, 10, TREC_DL, "ndcg_at_1000")


def test_rank_helpers():
    assert rank_of(["x", "y", "z"], "x") == 1
    assert rank_of(["x", "y"], "q") is None
    assert ranks_of(["x", "y", "z"]) == {"x": 1, "y": 2, "z": 3}
