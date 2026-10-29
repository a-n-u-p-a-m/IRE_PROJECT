"""
Budget splitting and the three outsider policies.

The policies are not interchangeable and the differences are the experiment.
`random` in particular is the control: without it, a boundary or alternative
result cannot be separated from the effect of simply reranking more documents.
"""

from __future__ import annotations

import pytest

from shared.allocation import (
    ALTERNATIVE,
    BOUNDARY,
    RANDOM,
    AllocationError,
    allocate,
    select_outsiders,
    split_budget,
    stratum_map,
    validate_allocation,
)
from shared.schema import REMOVED, RETAINED, SELECTED_OUTSIDER, UNSELECTED_OUTSIDER


def test_budget_split_always_totals_b():
    for b in (1, 4, 10, 100, 1000):
        for t in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
            n_cand, n_out = split_budget(b, t)
            assert n_cand + n_out == b
            assert n_cand == int(t * b) or n_cand == int(t * b)  # floor


def test_budget_split_endpoints():
    assert split_budget(10, 1.0) == (10, 0)
    assert split_budget(10, 0.0) == (0, 10)
    assert split_budget(10, 0.5) == (5, 5)
    assert split_budget(10, 0.75) == (7, 3)      # floor(7.5) = 7


def test_invalid_budget_or_t_is_refused():
    with pytest.raises(AllocationError, match="budget B must be positive"):
        split_budget(0, 0.5)
    with pytest.raises(AllocationError, match="t must lie"):
        split_budget(10, 1.5)


# ── policies ──────────────────────────────────────────────────────────

def test_boundary_policy_takes_the_documents_just_below_the_cutoff():
    discarded = [f"d{i:02d}" for i in range(10)]
    s1 = {d: float(-i) for i, d in enumerate(discarded)}
    assert select_outsiders(discarded, 3, BOUNDARY, s1=s1) == discarded[:3]


def test_alternative_policy_follows_the_other_retrievers_opinion():
    discarded = [f"d{i:02d}" for i in range(10)]
    alt = {d: 10 - i for i, d in enumerate(discarded)}      # reversed opinion
    assert select_outsiders(discarded, 3, ALTERNATIVE, alternative_rank=alt) == [
        "d09", "d08", "d07"
    ]


def test_alternative_policy_places_unranked_documents_last():
    discarded = ["x", "y", "z"]
    alt = {"z": 1}
    chosen = select_outsiders(discarded, 2, ALTERNATIVE, alternative_rank=alt)
    assert chosen[0] == "z"
    assert chosen[1] in {"x", "y"}


def test_random_policy_is_reproducible_and_seed_dependent():
    discarded = [f"d{i:02d}" for i in range(20)]
    a = select_outsiders(discarded, 5, RANDOM, seed=1)
    b = select_outsiders(discarded, 5, RANDOM, seed=1)
    c = select_outsiders(discarded, 5, RANDOM, seed=2)
    assert a == b
    assert a != c
    assert set(a).issubset(set(discarded))


def test_random_policy_ignores_input_order():
    """The seed must fully determine the sample, not the order it arrived in."""
    discarded = [f"d{i:02d}" for i in range(20)]
    forward = select_outsiders(discarded, 5, RANDOM, seed=7)
    backward = select_outsiders(list(reversed(discarded)), 5, RANDOM, seed=7)
    assert forward == backward


def test_zero_or_empty_budget_selects_nothing():
    assert select_outsiders(["a", "b"], 0, BOUNDARY, s1={"a": 1.0, "b": 0.0}) == []
    assert select_outsiders([], 5, BOUNDARY, s1={}) == []


def test_oversized_request_returns_everything():
    discarded = ["a", "b", "c"]
    assert select_outsiders(discarded, 10, BOUNDARY, s1={d: 1.0 for d in discarded}) \
        == discarded


def test_policies_need_their_inputs():
    with pytest.raises(AllocationError, match="needs Stage-1 scores"):
        select_outsiders(["a", "b"], 1, BOUNDARY)
    with pytest.raises(AllocationError, match="needs alternative_rank"):
        select_outsiders(["a", "b"], 1, ALTERNATIVE)
    with pytest.raises(AllocationError, match="unknown outsider policy"):
        select_outsiders(["a", "b"], 1, "vibes")


# ── whole allocations ─────────────────────────────────────────────────

def _setup():
    candidates = [f"c{i}" for i in range(6)]            # already in Stage-1 order
    discarded = [f"x{i}" for i in range(10)]
    s1 = {d: float(-i) for i, d in enumerate(discarded)}
    return candidates, discarded, s1


def test_allocation_partitions_every_document():
    candidates, discarded, s1 = _setup()
    alloc = allocate(candidates, discarded, 6, 0.5, BOUNDARY, s1=s1)
    seen = (
        set(alloc["retained"]) | set(alloc["removed"])
        | set(alloc["selected"]) | set(alloc["unselected"])
    )
    assert seen == set(candidates) | set(discarded)
    assert len(alloc["retained"]) + len(alloc["removed"]) == len(candidates)
    assert len(alloc["selected"]) + len(alloc["unselected"]) == len(discarded)


def test_allocation_validates():
    candidates, discarded, s1 = _setup()
    for t in (1.0, 0.75, 0.5, 0.0):
        alloc = allocate(candidates, discarded, 6, t, BOUNDARY, s1=s1)
        validate_allocation(alloc, 6, t)


def test_allocation_refuses_a_budget_larger_than_the_candidate_set():
    candidates, discarded, s1 = _setup()
    with pytest.raises(AllocationError, match="asks for"):
        allocate(candidates, discarded, 20, 1.0, BOUNDARY, s1=s1)


def test_stratum_map_covers_all_four_groups():
    candidates, discarded, s1 = _setup()
    alloc = allocate(candidates, discarded, 6, 0.5, BOUNDARY, s1=s1)
    strata = stratum_map(alloc)
    assert set(strata.values()) == {
        RETAINED, REMOVED, SELECTED_OUTSIDER, UNSELECTED_OUTSIDER
    }
    assert strata[candidates[0]] == RETAINED
    assert strata[candidates[-1]] == REMOVED
    assert strata["x0"] == SELECTED_OUTSIDER


def test_t_one_retains_everything_and_promotes_nothing():
    candidates, discarded, s1 = _setup()
    alloc = allocate(candidates, discarded, 6, 1.0, BOUNDARY, s1=s1)
    assert alloc["retained"] == candidates
    assert alloc["removed"] == []
    assert alloc["selected"] == []


def test_t_zero_replaces_the_whole_candidate_set():
    candidates, discarded, s1 = _setup()
    alloc = allocate(candidates, discarded, 6, 0.0, BOUNDARY, s1=s1)
    assert alloc["retained"] == []
    assert alloc["removed"] == candidates
    assert len(alloc["selected"]) == 6


def test_validate_catches_an_allocation_that_double_counts():
    alloc = {
        "retained": ["a"], "removed": [], "selected": ["a"], "unselected": [],
        "submitted": ["a", "a"],
    }
    with pytest.raises(AllocationError):
        validate_allocation(alloc, 2, 0.5)


def test_a_short_allocation_is_caught_when_the_pool_was_not_exhausted():
    """
    The regression guard for a check that used to be vacuous.

    `validate_allocation` capped the expectation at `min(n_out, len(selected))`
    - the amount TAKEN - so a short allocation always matched its own shortfall
    and passed. Here one outsider was available in D_q and was not taken, which
    makes this t point cheaper than the others and turns the budget comparison
    into a compute comparison.
    """
    short = {
        "retained": ["a"], "removed": ["b"],
        "selected": [], "unselected": ["c", "d"],
        "submitted": ["a"],
    }
    with pytest.raises(AllocationError, match="should submit 2"):
        validate_allocation(short, 2, 0.5)


def test_an_exhausted_discarded_pool_is_still_accepted():
    """
    The legitimate shortfall the cap exists for. D_q holds one document and the
    budget asks for one outsider, so taking all of it spends everything
    available - that is not a short allocation.
    """
    ok = {
        "retained": ["a"], "removed": ["b"],
        "selected": ["c"], "unselected": [],
        "submitted": ["a", "c"],
    }
    validate_allocation(ok, 2, 0.5)


def test_an_exhausted_candidate_pool_is_still_accepted():
    ok = {
        "retained": ["a"], "removed": [],
        "selected": ["c"], "unselected": ["d"],
        "submitted": ["a", "c"],
    }
    validate_allocation(ok, 2, 0.5)
