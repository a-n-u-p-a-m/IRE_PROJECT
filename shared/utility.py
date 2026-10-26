"""
utility.py - The document-level outcome function U_d(pi).

Every effect in this project is a difference of utilities, so the utility is
the estimand's unit. It is a declared design choice, not a universal
definition of document value, which is why sensitivity under three outcomes is
reported.

The primary choice is the per-document contribution to DCG@k:

    U_d(pi) = G(rel_d) / log2(1 + rank_pi(d))   if rank_pi(d) <= k
            = 0                                  otherwise

Summing over documents recovers the system metric, sum_d U_d(pi) = DCG@k(pi),
so document-level effects aggregate back to a familiar effectiveness change.
`check_aggregation` tests that directly.

THE GAIN MAP IS EXPLICIT AND DATASET-SPECIFIC. In TREC Deep Learning passage
ranking grade 1 means "Related" - the passage is on topic but does not answer
the query - so the usual 2^rel-1 would award it a gain of 1. Because every
estimand here is a DIFFERENCE of utilities, that error does not cancel: it
inflates effects systematically wherever a merely-related passage moves. The
map lives in the frozen config and is passed in, never hard-coded here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

GainMap = Mapping[int, float]
UtilityKind = Literal["dcg", "topk", "rbp"]

# A sane default for collections whose grade 1 really does mean relevant.
# TREC DL is NOT such a collection; see the module docstring.
EXPONENTIAL_GAIN: dict[int, float] = {0: 0.0, 1: 1.0, 2: 3.0, 3: 7.0}


def gain(relevance: int | None, gain_map: GainMap) -> float:
    """
    Relevance gain under an explicit map.

    Unknown relevance (None) contributes nothing. That is a statement about
    measurable contribution, not a claim of non-relevance; restricting
    per-document analysis to the judged subset is what keeps the two apart.
    """
    if relevance is None:
        return 0.0
    try:
        grade = int(relevance)
    except (TypeError, ValueError):
        return 0.0
    if grade not in gain_map:
        raise KeyError(
            f"relevance grade {grade} is not in the frozen gain map "
            f"{dict(gain_map)}. Add it to the config deliberately rather than "
            "letting an unmapped grade fall through to zero."
        )
    return float(gain_map[grade])


def document_utility(
    rank: int | None,
    relevance: int | None,
    k: int,
    gain_map: GainMap,
    kind: UtilityKind = "dcg",
    persistence: float = 0.8,
) -> float:
    """
    Utility contributed by one document under one ranking.

    `rank` is 1-indexed, or None when the document does not appear in the
    ranking at all - a non-survivor in the actual world, or a document outside
    the submitted set in a t-world.
    """
    if rank is None or rank < 1 or rank > k:
        return 0.0

    if kind == "topk":
        g = gain(relevance, gain_map)
        return 1.0 if g > 0.0 else 0.0

    g = gain(relevance, gain_map)
    if g == 0.0:
        return 0.0

    if kind == "dcg":
        return g / math.log2(1.0 + rank)
    if kind == "rbp":
        # Moffat & Zobel: the user examines rank r with probability p^(r-1),
        # normalised by (1 - p) so a perfect ranking tends to 1.
        return g * (1.0 - persistence) * (persistence ** (rank - 1))

    raise ValueError(f"unknown utility kind {kind!r}")


def ranking_utilities(
    ranking: Sequence[str],
    relevances: Mapping[str, int],
    k: int,
    gain_map: GainMap,
    kind: UtilityKind = "dcg",
    persistence: float = 0.8,
) -> dict[str, float]:
    """Per-document utilities for a whole ranking (position 0 is rank 1)."""
    out: dict[str, float] = {}
    for idx, doc_id in enumerate(ranking, start=1):
        if idx > k:
            break
        out[doc_id] = document_utility(
            idx, relevances.get(doc_id), k, gain_map, kind, persistence
        )
    return out


def dcg_at_k(
    ranking: Sequence[str],
    relevances: Mapping[str, int],
    k: int,
    gain_map: GainMap,
) -> float:
    """System-level DCG@k under the same gain map."""
    return sum(
        gain(relevances.get(d), gain_map) / math.log2(1.0 + i)
        for i, d in enumerate(ranking[:k], start=1)
    )


def ideal_dcg_at_k(
    relevances: Mapping[str, int], k: int, gain_map: GainMap
) -> float:
    """Ideal DCG@k over the judged pool, for normalised reporting."""
    gains = sorted((gain(r, gain_map) for r in relevances.values()), reverse=True)
    return sum(g / math.log2(1.0 + i) for i, g in enumerate(gains[:k], start=1))


def ndcg_at_k(
    ranking: Sequence[str],
    relevances: Mapping[str, int],
    k: int,
    gain_map: GainMap,
) -> float:
    """NDCG@k. Returns 0.0 when the ideal is zero (no judged relevant docs)."""
    ideal = ideal_dcg_at_k(relevances, k, gain_map)
    if ideal <= 0.0:
        return 0.0
    return dcg_at_k(ranking, relevances, k, gain_map) / ideal


def check_aggregation(
    ranking: Sequence[str],
    relevances: Mapping[str, int],
    k: int,
    gain_map: GainMap,
    tol: float = 1e-9,
) -> bool:
    """
    Verify sum_d U_d(pi) == DCG@k(pi).

    Worth asserting in tests: if it ever fails, the per-document utility and
    the system metric have drifted apart, and the document-level effects no
    longer aggregate to anything interpretable.
    """
    per_doc = ranking_utilities(ranking, relevances, k, gain_map, "dcg")
    return abs(sum(per_doc.values()) - dcg_at_k(ranking, relevances, k, gain_map)) < tol


def ranks_of(ranking: Sequence[str]) -> dict[str, int]:
    """{doc_id: 1-indexed rank} for a whole ranking."""
    return {d: i for i, d in enumerate(ranking, start=1)}


def rank_of(ranking: Sequence[str], doc_id: str) -> int | None:
    """1-indexed rank, or None when absent."""
    try:
        return list(ranking).index(doc_id) + 1
    except ValueError:
        return None


def is_monotone(kind: UtilityKind) -> bool:
    """
    Whether utility is non-increasing in rank.

    The survivor invariant E_gate >= 0 relies on this together with pointwise
    Stage-2 scoring. All three supported utilities are monotone; the helper
    exists so a future non-monotone outcome cannot silently invalidate the
    invariant checks.
    """
    return kind in {"dcg", "topk", "rbp"}


def describe(kind: UtilityKind, k: int, persistence: float = 0.8) -> str:
    if kind == "dcg":
        return f"DCG@{k} contribution"
    if kind == "topk":
        return f"top-{k} inclusion"
    if kind == "rbp":
        return f"RBP(p={persistence})"
    return str(kind)


__all__ = [
    "GainMap", "UtilityKind", "EXPONENTIAL_GAIN",
    "gain", "document_utility", "ranking_utilities",
    "dcg_at_k", "ideal_dcg_at_k", "ndcg_at_k", "check_aggregation",
    "ranks_of", "rank_of", "is_monotone", "describe",
]
