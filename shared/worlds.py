"""
worlds.py - The four counterfactual worlds (proposal section 2).

For a query with frozen universe U_q, Stage-1 scores s1 and Stage-2 scores s2:

    C_q      = TopKc(U_q, s1)           D_q = U_q \\ C_q
    A  pi_act   = Sort(C_q, s2)         the actual cascade
    B  pi_no2   = C_q in Stage-1 order  no reranker, GATE PRESERVED
    C  pi_no1   = Sort(U_q, s2)         no gate, reranker policy preserved
    D  pi_t     = Sort(submitted, s2)   fixed budget B, allocation t

Three details decide whether this isolates anything:

  * World B ranks C_q, NOT U_q. Sorting the universe by s1 would remove the
    reranker and the gate at once, confounding the two estimands.
  * Ties break deterministically and identically in all four worlds.
    Non-deterministic tie-breaking appears downstream as spurious nonzero
    effects and as survivors with E_gate < 0.
  * World D's baseline is pi at t=1, which equals pi_act ONLY when B = K_c.
    When it does not, t=1 is a candidate-only budget baseline and Delta_t
    measures something narrower than the proposal's headline claim. The code
    computes the t=1 world explicitly rather than reusing pi_act, and
    `TWorld.is_conventional_baseline` records which case applies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .allocation import allocate, stratum_map, validate_allocation
from .ordering import TIE_BREAK_DOC_ID, order_by


@dataclass(frozen=True)
class Worlds:
    """Worlds A, B and C for one query at one candidate depth."""

    query_id: str
    k_c: int
    candidates: tuple[str, ...]     # C_q in Stage-1 order
    discarded: tuple[str, ...]      # D_q in Stage-1 order
    pi_act: tuple[str, ...]         # world A
    pi_no2: tuple[str, ...]         # world B
    pi_no1: tuple[str, ...]         # world C

    @property
    def candidate_set(self) -> set[str]:
        return set(self.candidates)

    @property
    def universe(self) -> tuple[str, ...]:
        return self.candidates + self.discarded

    def is_survivor(self, doc_id: str) -> bool:
        return doc_id in self.candidate_set


@dataclass(frozen=True)
class TWorld:
    """World D at one (t, policy, seed) for one query."""

    query_id: str
    k_c: int
    budget_b: int
    t: float
    policy: str
    seed: int
    submitted: tuple[str, ...]
    pi_t: tuple[str, ...]
    strata: dict[str, str]
    is_conventional_baseline: bool

    @property
    def submitted_set(self) -> set[str]:
        return set(self.submitted)


def build_worlds(
    query_id: str,
    universe: Sequence[str],
    s1: Mapping[str, float],
    s2: Mapping[str, float],
    k_c: int,
    tie_break: str = TIE_BREAK_DOC_ID,
) -> Worlds:
    """Execute worlds A, B and C for one query at one candidate depth."""
    universe = list(universe)
    if k_c >= len(universe):
        raise ValueError(
            f"K_c={k_c} is not smaller than |U_q|={len(universe)} for query "
            f"{query_id}: the gate would admit everything and E_gate would be "
            "identically zero."
        )

    stage1_order = order_by(universe, s1, tie_break)
    candidates = stage1_order[:k_c]
    discarded = stage1_order[k_c:]

    return Worlds(
        query_id=str(query_id),
        k_c=int(k_c),
        candidates=candidates,
        discarded=discarded,
        pi_act=order_by(candidates, s2, tie_break),
        pi_no2=candidates,                        # already in Stage-1 order
        pi_no1=order_by(universe, s2, tie_break),
    )


def build_t_world(
    worlds: Worlds,
    s1: Mapping[str, float],
    s2: Mapping[str, float],
    budget_b: int,
    t: float,
    policy: str,
    alternative_rank: Mapping[str, int] | None = None,
    seed: int = 42,
    tie_break: str = TIE_BREAK_DOC_ID,
) -> TWorld:
    """
    Execute world D for one (t, policy, seed).

    `worlds` supplies C_q and D_q, so the t-world is always built against the
    same gate the other three worlds used.
    """
    alloc = allocate(
        candidates=worlds.candidates,
        discarded=worlds.discarded,
        total_budget=budget_b,
        t=t,
        policy=policy,
        s1=s1,
        alternative_rank=alternative_rank,
        seed=seed,
    )
    validate_allocation(alloc, budget_b, t)

    return TWorld(
        query_id=worlds.query_id,
        k_c=worlds.k_c,
        budget_b=int(budget_b),
        t=float(t),
        policy=policy,
        seed=int(seed),
        submitted=tuple(alloc["submitted"]),
        pi_t=order_by(alloc["submitted"], s2, tie_break),
        strata=stratum_map(alloc),
        is_conventional_baseline=(budget_b == worlds.k_c),
    )


def build_t_grid(
    worlds: Worlds,
    s1: Mapping[str, float],
    s2: Mapping[str, float],
    budget_b: int,
    t_grid: Sequence[float],
    policy: str,
    alternative_rank: Mapping[str, int] | None = None,
    seed: int = 42,
    tie_break: str = TIE_BREAK_DOC_ID,
) -> dict[float, TWorld]:
    """Every t in the grid for one policy and seed, keyed by t."""
    return {
        float(t): build_t_world(
            worlds, s1, s2, budget_b, t, policy, alternative_rank, seed, tie_break
        )
        for t in t_grid
    }


# ──────────────────────────────────────────────────────────────────────
# Isolation checks
# ──────────────────────────────────────────────────────────────────────

def verify_isolation(worlds: Worlds) -> None:
    """
    Assert the properties the design depends on.

      * World B ranks exactly C_q, so the Stage-2 intervention leaves the gate
        untouched;
      * World C contains every candidate, so the gate intervention only ADDS
        competitors and never removes one;
      * no world contains a duplicate.
    """
    if set(worlds.pi_act) != set(worlds.pi_no2):
        raise AssertionError(
            f"query {worlds.query_id}: world B does not preserve C_q. The "
            "no-reranker world must reorder the same candidate set, otherwise "
            "it removes the reranker and the gate at the same time."
        )
    if not worlds.candidate_set.issubset(set(worlds.pi_no1)):
        raise AssertionError(
            f"query {worlds.query_id}: world C does not contain every candidate. "
            "Removing the gate may only add documents."
        )
    for name, ranking in (
        ("pi_act", worlds.pi_act), ("pi_no2", worlds.pi_no2), ("pi_no1", worlds.pi_no1)
    ):
        if len(set(ranking)) != len(ranking):
            raise AssertionError(f"query {worlds.query_id}: {name} contains duplicates")
    if set(worlds.candidates) & set(worlds.discarded):
        raise AssertionError(
            f"query {worlds.query_id}: C_q and D_q overlap"
        )


def verify_t_baseline(t_worlds: Mapping[float, TWorld], worlds: Worlds) -> None:
    """
    Check that the t=1 world is the baseline the design intends.

    Pre-run validation check: t=1 must correspond to the complete
    candidate-only baseline when B = K_c. If B differs, this raises so the
    team states the distinction deliberately instead of discovering it in the
    numbers.
    """
    if 1.0 not in t_worlds:
        raise AssertionError(
            "no t=1 world was built; Delta_t has no baseline to measure against"
        )
    base = t_worlds[1.0]
    if base.is_conventional_baseline:
        if set(base.submitted) != worlds.candidate_set:
            raise AssertionError(
                f"query {worlds.query_id}: B equals K_c but the t=1 world does "
                "not submit exactly C_q, so it is not the actual cascade."
            )
        if base.pi_t != worlds.pi_act:
            raise AssertionError(
                f"query {worlds.query_id}: B equals K_c but the t=1 ranking "
                "differs from the actual cascade. Check that both worlds use "
                "the same Stage-2 scores and the same tie-break."
            )
    else:
        if len(base.submitted) != base.budget_b:
            raise AssertionError(
                f"query {worlds.query_id}: the t=1 world submits "
                f"{len(base.submitted)} documents at budget B={base.budget_b}"
            )


def shuffle_invariance_holds(
    universe: Sequence[str],
    s2: Mapping[str, float],
    permutations: int = 5,
    seed: int = 42,
    tie_break: str = TIE_BREAK_DOC_ID,
) -> bool:
    """
    Permuting the input order must not change the resulting ranking.

    Pre-run validation check. A pointwise reranker scores each pair
    independently, so the order documents arrive in cannot matter. If this
    fails, either the scorer is not pointwise or batching has introduced an
    order dependence - and the whole E_gate derivation rests on it not having.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    universe = list(universe)
    reference = order_by(universe, s2, tie_break)
    for _ in range(permutations):
        shuffled = list(rng.permutation(universe))
        if order_by(shuffled, s2, tie_break) != reference:
            return False
    return True


__all__ = [
    "Worlds", "TWorld", "build_worlds", "build_t_world", "build_t_grid",
    "verify_isolation", "verify_t_baseline", "shuffle_invariance_holds",
]
