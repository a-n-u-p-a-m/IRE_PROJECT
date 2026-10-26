"""
ordering.py - Deterministic ranking from scores.

Separated into its own module because every world, every probe and every
t-allocation sorts by this one function, and a single inconsistency here
surfaces downstream as spurious nonzero effects and as survivors with
E_gate < 0. It must be a pure function of (score, doc_id) and nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

TIE_BREAK_DOC_ID = "doc_id"


class MissingScoreError(KeyError):
    """Raised when a document to be ordered has no score."""


def order_by(
    doc_ids: Sequence[str],
    scores: Mapping[str, float],
    tie_break: str = TIE_BREAK_DOC_ID,
) -> tuple[str, ...]:
    """
    Sort descending by score, breaking ties by ascending doc_id.

    A missing score is an error, never a silent zero. A missing Stage-2 score
    means the cache is incomplete, which would silently truncate the no-gate
    world and produce plausible-looking wrong gate effects; a crash is better.
    """
    missing = [d for d in doc_ids if d not in scores]
    if missing:
        raise MissingScoreError(
            f"{len(missing)} documents have no score, e.g. {missing[:5]}. "
            "Populate the Stage-2 cache over the whole universe before "
            "constructing any world."
        )
    if tie_break != TIE_BREAK_DOC_ID:
        raise ValueError(
            f"unsupported tie_break {tie_break!r}; only {TIE_BREAK_DOC_ID!r} is "
            "implemented, deliberately, so ordering stays reproducible"
        )
    return tuple(sorted(doc_ids, key=lambda d: (-scores[d], d)))


def top_k(
    doc_ids: Sequence[str],
    scores: Mapping[str, float],
    k: int,
    tie_break: str = TIE_BREAK_DOC_ID,
) -> tuple[str, ...]:
    """The highest-scoring k documents, in order."""
    return order_by(doc_ids, scores, tie_break)[:k]


def rank_map(
    doc_ids: Sequence[str],
    scores: Mapping[str, float],
    tie_break: str = TIE_BREAK_DOC_ID,
) -> dict[str, int]:
    """{doc_id: 1-indexed rank} under this ordering."""
    return {d: i for i, d in enumerate(order_by(doc_ids, scores, tie_break), start=1)}


__all__ = ["order_by", "top_k", "rank_map", "TIE_BREAK_DOC_ID", "MissingScoreError"]
