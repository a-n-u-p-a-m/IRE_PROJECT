"""
fixtures.py - The shared toy cascade.

Section 14 requires each member to keep one GPU-free smoke test and one
minimal fixture. This is the common one, so all three are smoke-testing
against the same numbers and a disagreement means a real disagreement rather
than two different toys.

Small enough to verify by hand, and arranged so that every stratum of every
effect type contains at least one nonzero value - a fixture where the
interesting cases are all empty passes every test and proves nothing.

    doc  s1   s2    rel     role
    a    10   0.10   3      best by Stage 1, the reranker buries it
    b     9   0.90   2      survivor that reranking promotes
    c     8   0.50   0      survivor, no relevance
    d     7   0.30   1      survivor, relevant but below the cutoff
    e     6   0.95   3      THE GATE FAILURE: best s2 of all, discarded
    f     5   0.20   0      boundary outsider, no value
    g     4   0.40   1      mid outsider
    h     3   0.05   0      deep outsider
    i     2   0.60   2      deep outsider the boundary probe cannot see
    j     1   0.15   0      deepest

With K_c = 4, k = 2, B = 4 and the TREC DL gain map {0:0, 1:0, 2:3, 3:7}:

    C_q      = (a, b, c, d)        D_q = (e, f, g, h, i, j)
    pi_act   = (b, c, d, a)
    pi_no2   = (a, b, c, d)
    pi_no1   = (e, b, i, c, g, d, f, j, a, h)

    E_rerank(a) = -7            the reranker cost the best document everything
    E_gate(e)   = -7            the gate cost the collection its best document
    E_gate(b)   = +3 - 3/log2 3 the gate protected b from a stronger competitor

At t = 0.5 with the boundary policy the allocation is
retained (a, b), removed (c, d), selected (e, f), unselected (g, h, i, j),
giving Delta_t(e) = +7 and Delta_t(b) = 3/log2 3 - 3 < 0.
"""

from __future__ import annotations

from typing import Any

TOY_S1: dict[str, float] = {
    "a": 10.0, "b": 9.0, "c": 8.0, "d": 7.0, "e": 6.0,
    "f": 5.0, "g": 4.0, "h": 3.0, "i": 2.0, "j": 1.0,
}

TOY_S2: dict[str, float] = {
    "a": 0.10, "b": 0.90, "c": 0.50, "d": 0.30, "e": 0.95,
    "f": 0.20, "g": 0.40, "h": 0.05, "i": 0.60, "j": 0.15,
}

TOY_REL: dict[str, int] = {
    "a": 3, "b": 2, "c": 0, "d": 1, "e": 3,
    "f": 0, "g": 1, "h": 0, "i": 2, "j": 0,
}

# The TREC DL mapping: grade 1 ("Related") earns nothing.
TOY_GAIN_MAP: dict[int, float] = {0: 0.0, 1: 0.0, 2: 3.0, 3: 7.0}

TOY_K_C = 4
TOY_K = 2
TOY_BUDGET = 4          # equals K_c, so t=1 is the conventional cascade

# A second retriever's opinion, for the 'alternative' outsider policy. It
# ranks the deep outsider i highly, which the boundary policy never reaches.
TOY_ALTERNATIVE_RANK: dict[str, int] = {
    "i": 1, "e": 2, "b": 3, "g": 4, "a": 5,
    "c": 6, "f": 7, "d": 8, "j": 9, "h": 10,
}


def toy_cascade() -> dict[str, Any]:
    """Everything needed to build all four worlds for one query."""
    universe = list(TOY_S1)
    return {
        "query_id": "toy1",
        "universe": universe,
        "s1": dict(TOY_S1),
        "s2": dict(TOY_S2),
        "relevances": dict(TOY_REL),
        "judged": {d: True for d in universe},
        "gain_map": dict(TOY_GAIN_MAP),
        "k_c": TOY_K_C,
        "k": TOY_K,
        "budget_b": TOY_BUDGET,
        "alternative_rank": dict(TOY_ALTERNATIVE_RANK),
    }


def toy_worlds():
    """Worlds A, B and C for the toy cascade."""
    from .worlds import build_worlds

    c = toy_cascade()
    return build_worlds(c["query_id"], c["universe"], c["s1"], c["s2"], c["k_c"])


def toy_t_worlds(t_grid=(1.0, 0.75, 0.5), policy: str = "boundary", seed: int = 42):
    """World D across a t grid for the toy cascade."""
    from .worlds import build_t_grid

    c = toy_cascade()
    return build_t_grid(
        toy_worlds(), c["s1"], c["s2"], c["budget_b"], t_grid, policy,
        alternative_rank=c["alternative_rank"], seed=seed,
    )


def toy_effects():
    """The full effect table for the toy cascade: all three effect types."""
    from .effects import (
        budget_effects_for_query,
        concat_effects,
        stage_effects_for_query,
    )

    c = toy_cascade()
    w = toy_worlds()
    tw = toy_t_worlds()
    stage = stage_effects_for_query(
        w, c["relevances"], c["judged"], "toy", c["k"], c["gain_map"]
    )
    budget = budget_effects_for_query(
        w, tw, c["relevances"], c["judged"], "toy", c["k"], c["gain_map"]
    )
    return concat_effects([stage, budget])


def smoke_test() -> bool:
    """
    Run the whole engine on the toy cascade and check the known answers.

    Every member runs this before pushing anything. It needs no GPU, no index
    and no data, so there is never a reason to skip it.
    """
    from .effects import check_invariants
    from .schema import DELTA_T, E_GATE, E_RERANK
    from .strata import check_strata_valid
    from .worlds import verify_isolation, verify_t_baseline

    w = toy_worlds()
    verify_isolation(w)

    assert w.candidates == ("a", "b", "c", "d"), w.candidates
    assert w.pi_act == ("b", "c", "d", "a"), w.pi_act
    assert w.pi_no2 == ("a", "b", "c", "d"), w.pi_no2
    assert w.pi_no1[0] == "e", w.pi_no1

    tw = toy_t_worlds()
    verify_t_baseline(tw, w)
    assert tw[1.0].pi_t == w.pi_act, "B == K_c so t=1 must be the actual cascade"

    eff = toy_effects()
    check_invariants(eff)
    check_strata_valid(eff)

    rerank_a = eff[(eff["effect_type"] == E_RERANK) & (eff["doc_id"] == "a")]
    gate_e = eff[(eff["effect_type"] == E_GATE) & (eff["doc_id"] == "e")]
    assert abs(float(rerank_a["effect"].iloc[0]) + 7.0) < 1e-9
    assert abs(float(gate_e["effect"].iloc[0]) + 7.0) < 1e-9

    delta_e = eff[
        (eff["effect_type"] == DELTA_T)
        & (eff["doc_id"] == "e")
        & (eff["t"] == 0.5)
    ]
    assert abs(float(delta_e["effect"].iloc[0]) - 7.0) < 1e-9
    return True


__all__ = [
    "TOY_S1", "TOY_S2", "TOY_REL", "TOY_GAIN_MAP", "TOY_ALTERNATIVE_RANK",
    "TOY_K_C", "TOY_K", "TOY_BUDGET",
    "toy_cascade", "toy_worlds", "toy_t_worlds", "toy_effects", "smoke_test",
]
