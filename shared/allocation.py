"""
allocation.py - Budget-constrained candidate allocation (proposal section 4).

World D sends a FIXED budget of B documents to the reranker and varies where
they come from. For a given t:

    floor(t*B)  documents retained from the Stage-1 candidate set C_q
    B - that    documents promoted from the discarded set D_q

The retained portion is the top floor(t*B) of C_q by Stage-1 rank. Note what
that means. The degeneracy result says Stage-1 rank WITHIN C_q is causally
inert for the final ranking at a fixed K_c, because a pointwise reranker never
reads it. Under a budget constraint that inertness ends: rank now decides who
survives the cut. The t-experiment is precisely the regime in which the inert
variable becomes causally active again.

Even here the rank is only a SELECTION rule. It is never passed to Stage 2,
which still scores each query-document pair independently.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .ordering import order_by

BOUNDARY = "boundary"
ALTERNATIVE = "alternative"
RANDOM = "random"
POLICIES = (BOUNDARY, ALTERNATIVE, RANDOM)


class AllocationError(ValueError):
    """Raised when a budget allocation is impossible or ill-specified."""


def split_budget(total_budget: int, t: float) -> tuple[int, int]:
    """
    (n_from_candidates, n_from_discarded) for budget B at allocation t.

    floor(t*B) from C_q and the remainder from D_q, so the two always sum to
    exactly B - which `validate_allocation` re-checks, because an allocation
    that does not spend the whole budget makes the t curve incomparable
    across its own points.
    """
    if total_budget < 1:
        raise AllocationError(f"budget B must be positive, got {total_budget}")
    if not 0.0 <= t <= 1.0:
        raise AllocationError(f"t must lie in [0, 1], got {t}")
    n_cand = int(math.floor(t * total_budget))
    return n_cand, total_budget - n_cand


def select_outsiders(
    discarded: Sequence[str],
    n: int,
    policy: str,
    s1: Mapping[str, float] | None = None,
    alternative_rank: Mapping[str, int] | None = None,
    seed: int = 42,
) -> list[str]:
    """
    Choose `n` documents from the discarded set D_q under one policy.

    boundary
        The n documents immediately below the Stage-1 cutoff. The cheapest
        realistic policy, and the one most likely to miss a valuable document
        that Stage 1 buried far down - a false-negative rate to measure, not a
        reason to drop the policy.
    alternative
        The n documents ranked highest by a DIFFERENT retriever. Asks a second
        opinion about what the primary retriever discarded.
    random
        Uniform sample. The control: without it, a boundary or alternative
        result cannot be separated from the effect of simply reranking more
        documents.

    Returns documents in the order the policy prefers them.
    """
    discarded = list(discarded)
    if n <= 0 or not discarded:
        return []
    if n >= len(discarded):
        return discarded

    if policy == BOUNDARY:
        if s1 is None:
            raise AllocationError("policy 'boundary' needs Stage-1 scores")
        return list(order_by(discarded, s1)[:n])

    if policy == ALTERNATIVE:
        if not alternative_rank:
            raise AllocationError(
                "policy 'alternative' needs alternative_rank, the rank each "
                "document received from the other retriever"
            )
        # Unranked documents sort last, deterministically by doc_id.
        ordered = sorted(
            discarded, key=lambda d: (alternative_rank.get(d, 1 << 30), d)
        )
        return ordered[:n]

    if policy == RANDOM:
        rng = np.random.default_rng(seed)
        pool = sorted(discarded)          # sort first so the seed fully determines it
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[int(i)] for i in idx]

    raise AllocationError(f"unknown outsider policy {policy!r}; expected {POLICIES}")


def allocate(
    candidates: Sequence[str],
    discarded: Sequence[str],
    total_budget: int,
    t: float,
    policy: str,
    s1: Mapping[str, float] | None = None,
    alternative_rank: Mapping[str, int] | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Build one t-allocation.

    `candidates` must already be in Stage-1 order, as `worlds.build_worlds`
    returns them.

    Returns the retained, removed and selected lists, which are exactly the
    first three membership strata of section 4.4; everything left in D_q is
    the fourth.
    """
    candidates = list(candidates)
    discarded = list(discarded)
    n_cand, n_out = split_budget(total_budget, t)

    if n_cand > len(candidates):
        raise AllocationError(
            f"t={t} asks for {n_cand} documents from a candidate set of "
            f"{len(candidates)}. Either B exceeds K_c or the candidate set is "
            "short for this query."
        )

    retained = candidates[:n_cand]
    removed = candidates[n_cand:]
    selected = select_outsiders(
        discarded, n_out, policy, s1=s1, alternative_rank=alternative_rank, seed=seed
    )
    unselected = [d for d in discarded if d not in set(selected)]

    return {
        "retained": retained,
        "removed": removed,
        "selected": selected,
        "unselected": unselected,
        "submitted": retained + selected,
    }


def validate_allocation(
    alloc: Mapping[str, Sequence[str]], total_budget: int, t: float
) -> None:
    """
    Check the allocation spends exactly B and partitions the universe.

    Pre-run validation check: "validate that every t allocation totals exactly
    B". A short allocation would make one point of the t curve cheaper than
    the others, which silently turns a budget comparison into a compute
    comparison.
    """
    submitted = list(alloc["submitted"])
    n_cand, n_out = split_budget(total_budget, t)
    expected = min(n_cand, len(alloc["retained"])) + min(
        n_out, len(alloc["selected"])
    )

    if len(submitted) != expected:
        raise AllocationError(
            f"allocation submits {len(submitted)} documents but should submit "
            f"{expected} at t={t}, B={total_budget}"
        )
    if len(set(submitted)) != len(submitted):
        raise AllocationError("allocation submits the same document twice")

    overlap = set(alloc["retained"]) & set(alloc["selected"])
    if overlap:
        raise AllocationError(
            f"{len(overlap)} documents appear as both retained candidates and "
            "promoted outsiders; C_q and D_q must be disjoint"
        )
    if set(alloc["selected"]) & set(alloc["unselected"]):
        raise AllocationError("a discarded document is both selected and unselected")


def stratum_of(doc_id: str, alloc: Mapping[str, Sequence[str]]) -> str:
    """Which of the four membership strata a document falls into."""
    from .schema import REMOVED, RETAINED, SELECTED_OUTSIDER, UNSELECTED_OUTSIDER

    if doc_id in set(alloc["retained"]):
        return RETAINED
    if doc_id in set(alloc["removed"]):
        return REMOVED
    if doc_id in set(alloc["selected"]):
        return SELECTED_OUTSIDER
    return UNSELECTED_OUTSIDER


def stratum_map(alloc: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """{doc_id: stratum} for every document the allocation knows about."""
    from .schema import REMOVED, RETAINED, SELECTED_OUTSIDER, UNSELECTED_OUTSIDER

    out: dict[str, str] = {}
    for d in alloc["retained"]:
        out[d] = RETAINED
    for d in alloc["removed"]:
        out[d] = REMOVED
    for d in alloc["selected"]:
        out[d] = SELECTED_OUTSIDER
    for d in alloc["unselected"]:
        out[d] = UNSELECTED_OUTSIDER
    return out


__all__ = [
    "BOUNDARY", "ALTERNATIVE", "RANDOM", "POLICIES", "AllocationError",
    "split_budget", "select_outsiders", "allocate", "validate_allocation",
    "stratum_of", "stratum_map",
]
