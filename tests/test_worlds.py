"""
The four worlds, the degeneracy result, and the regime where it stops holding.

`test_stage1_order_inside_candidates_is_irrelevant` is the executable form of
the result that forced this whole reformulation: once C_q is fixed, the final
ranking is invariant to s1, because a pointwise cross-encoder never reads it.
Any analysis treating Stage-1 rank within the candidate set as informative
about the final ranking measures something that does not exist.

`test_budget_makes_stage1_rank_causally_active` is its complement and the
motivation for world D. Under a fixed budget, Stage-1 rank decides who survives
the cut, so the inert variable becomes active again.
"""

from __future__ import annotations

import pytest

from shared.ordering import MissingScoreError, order_by
from shared.worlds import (
    Worlds,
    build_t_grid,
    build_t_world,
    build_worlds,
    shuffle_invariance_holds,
    verify_isolation,
    verify_t_baseline,
)


def test_worlds_match_hand_computation(worlds):
    assert worlds.candidates == ("a", "b", "c", "d")
    assert worlds.discarded == ("e", "f", "g", "h", "i", "j")
    assert worlds.pi_act == ("b", "c", "d", "a")
    assert worlds.pi_no2 == ("a", "b", "c", "d")
    assert worlds.pi_no1 == ("e", "b", "i", "c", "g", "d", "f", "j", "a", "h")


def test_no_reranker_world_preserves_the_candidate_set(worlds):
    """
    World B must rank C_q, not U_q. Sorting the universe by s1 would remove the
    reranker and the gate at once, confounding the two estimands.
    """
    assert set(worlds.pi_no2) == set(worlds.candidates)
    verify_isolation(worlds)


def test_no_gate_world_only_adds_competitors(worlds):
    assert worlds.candidate_set.issubset(set(worlds.pi_no1))
    assert len(worlds.pi_no1) == len(worlds.universe)


def test_stage1_order_inside_candidates_is_irrelevant(toy):
    """
    Reverse the Stage-1 scores WITHIN C_q, keeping every candidate above every
    non-candidate. C_q is unchanged, so pi_act must be bit-identical. Only
    world B moves.
    """
    base = build_worlds(toy["query_id"], toy["universe"], toy["s1"], toy["s2"],
                        toy["k_c"])
    shuffled = dict(toy["s1"])
    cand = list(base.candidates)
    for doc, value in zip(cand, sorted(toy["s1"][d] for d in cand)):
        shuffled[doc] = value        # a now has the lowest score inside C_q

    perturbed = build_worlds(toy["query_id"], toy["universe"], shuffled,
                             toy["s2"], toy["k_c"])

    assert set(perturbed.candidates) == set(base.candidates)
    assert perturbed.pi_act == base.pi_act, (
        "the final ranking changed when only the within-candidate Stage-1 order "
        "changed; Stage 1 has acquired a causal channel it must not have"
    )
    assert perturbed.pi_no1 == base.pi_no1
    assert perturbed.pi_no2 != base.pi_no2


def test_membership_is_the_only_stage1_channel(toy):
    """The complement: change membership and the final ranking does move."""
    base = build_worlds(toy["query_id"], toy["universe"], toy["s1"], toy["s2"],
                        toy["k_c"])
    promoted = dict(toy["s1"])
    promoted["e"] = 100.0            # e enters C_q, d drops out
    perturbed = build_worlds(toy["query_id"], toy["universe"], promoted,
                             toy["s2"], toy["k_c"])

    assert "e" in perturbed.candidates
    assert perturbed.pi_act != base.pi_act
    assert perturbed.pi_act[0] == "e"


def test_budget_makes_stage1_rank_causally_active(toy):
    """
    Under a fixed budget the inert variable turns on.

    At K_c the within-candidate order does not matter. At t < 1 it decides who
    is retained, so permuting s1 inside C_q changes which documents reach
    Stage 2 and therefore changes the ranking. This is what world D is for.
    """
    base_worlds = build_worlds(toy["query_id"], toy["universe"], toy["s1"],
                               toy["s2"], toy["k_c"])
    reversed_s1 = dict(toy["s1"])
    cand = list(base_worlds.candidates)
    for doc, value in zip(cand, sorted(toy["s1"][d] for d in cand)):
        reversed_s1[doc] = value
    perturbed_worlds = build_worlds(toy["query_id"], toy["universe"],
                                    reversed_s1, toy["s2"], toy["k_c"])

    base_t = build_t_world(base_worlds, toy["s1"], toy["s2"],
                           toy["budget_b"], 0.5, "boundary")
    pert_t = build_t_world(perturbed_worlds, reversed_s1, toy["s2"],
                           toy["budget_b"], 0.5, "boundary")

    assert set(base_t.submitted) != set(pert_t.submitted), (
        "the budgeted submission did not change when the within-candidate "
        "Stage-1 order was reversed; world D would then be measuring nothing "
        "the gate does not already measure"
    )


def test_ties_break_deterministically_and_identically():
    """
    Non-deterministic tie-breaking appears downstream as spurious nonzero
    effects and as survivors with E_gate < 0, so ordering must be a pure
    function of (score, doc_id).
    """
    universe = [f"d{i}" for i in range(10)]
    flat = {d: 1.0 for d in universe}

    first = build_worlds("q", universe, flat, flat, 4)
    second = build_worlds("q", list(reversed(universe)), flat, flat, 4)

    assert first.candidates == second.candidates
    assert first.pi_act == second.pi_act
    assert first.pi_no1 == second.pi_no1 == tuple(sorted(universe))


def test_shuffle_invariance(toy):
    """
    Pre-run validation check: permuting the input order must not change the
    resulting ranking. A pointwise scorer cannot be order dependent, and the
    whole E_gate derivation rests on that.
    """
    assert shuffle_invariance_holds(toy["universe"], toy["s2"], permutations=8)


def test_missing_score_is_an_error_not_a_zero():
    with pytest.raises(MissingScoreError, match="no score"):
        order_by(["a", "b"], {"a": 1.0})


def test_gate_that_admits_everything_is_refused():
    universe = [f"d{i}" for i in range(5)]
    flat = {d: 1.0 for d in universe}
    with pytest.raises(ValueError, match="identically zero"):
        build_worlds("q", universe, flat, flat, 5)


def test_verify_isolation_catches_a_leaky_world_b():
    broken = Worlds(
        query_id="q", k_c=2,
        candidates=("a", "b"), discarded=("c",),
        pi_act=("a", "b"),
        pi_no2=("a", "b", "c"),          # leaked a non-candidate
        pi_no1=("a", "b", "c"),
    )
    with pytest.raises(AssertionError, match="does not preserve C_q"):
        verify_isolation(broken)


# ── world D ───────────────────────────────────────────────────────────

def test_t_equals_one_reproduces_the_actual_cascade(worlds, t_worlds):
    """
    B == K_c, so t=1 must BE the actual cascade. If B differed, t=1 would only
    be a candidate-only budget baseline and the design constraint requires that
    distinction to be stated wherever results are reported.
    """
    verify_t_baseline(t_worlds, worlds)
    assert t_worlds[1.0].is_conventional_baseline
    assert t_worlds[1.0].pi_t == worlds.pi_act
    assert set(t_worlds[1.0].submitted) == worlds.candidate_set


def test_t_world_hand_computation(toy, worlds):
    tw = build_t_world(worlds, toy["s1"], toy["s2"], toy["budget_b"], 0.5,
                       "boundary")
    # floor(0.5 * 4) = 2 retained by Stage-1 rank, 2 boundary outsiders.
    assert set(tw.submitted) == {"a", "b", "e", "f"}
    assert tw.pi_t == ("e", "b", "f", "a")
    assert tw.strata["a"] == "retained"
    assert tw.strata["c"] == "removed"
    assert tw.strata["e"] == "selected_outsider"
    assert tw.strata["i"] == "unselected_outsider"


def test_t_world_spends_the_whole_budget(toy, worlds):
    for t in (1.0, 0.75, 0.5, 0.25, 0.0):
        tw = build_t_world(worlds, toy["s1"], toy["s2"], toy["budget_b"], t,
                           "boundary")
        assert len(tw.submitted) == toy["budget_b"], (
            f"t={t} submitted {len(tw.submitted)} documents at B="
            f"{toy['budget_b']}; a short allocation turns a budget comparison "
            "into a compute comparison"
        )


def test_alternative_policy_reaches_a_document_boundary_cannot(toy, worlds):
    """
    The deep outsider i has high s2 but low s1, so the boundary policy never
    reaches it. A second retriever that ranks it first does.
    """
    boundary = build_t_world(worlds, toy["s1"], toy["s2"], toy["budget_b"],
                             0.5, "boundary")
    alternative = build_t_world(
        worlds, toy["s1"], toy["s2"], toy["budget_b"], 0.5, "alternative",
        alternative_rank=toy["alternative_rank"],
    )
    assert "i" not in boundary.submitted
    assert "i" in alternative.submitted


def test_t_grid_builds_every_point(toy, worlds):
    grid = build_t_grid(worlds, toy["s1"], toy["s2"], toy["budget_b"],
                        (1.0, 0.75, 0.5), "boundary")
    assert set(grid) == {1.0, 0.75, 0.5}
    verify_t_baseline(grid, worlds)


def test_missing_t_one_is_refused(toy, worlds):
    grid = build_t_grid(worlds, toy["s1"], toy["s2"], toy["budget_b"],
                        (0.75, 0.5), "boundary")
    with pytest.raises(AssertionError, match="no t=1 world"):
        verify_t_baseline(grid, worlds)
