"""
test_canary.py - The cross-machine consistency check.

No GPU and no model here: the canary's job is to turn a scorer's output into a
comparable fingerprint, and that logic is what can be wrong. A fake scorer
with controllable drift exercises the three verdicts the real check must
reach.
"""

from __future__ import annotations

from shared.canary import (
    CANARY_DOC_IDS,
    CANARY_PASSAGES,
    CANARY_QUERY,
    REFERENCE_DIGEST,
    REFERENCE_SCORES,
    canary_digest,
    canary_scores,
    compare_canaries,
)


class FakeScorer:
    """Returns declared scores, and asserts it was called the documented way."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores
        self.calls = 0

    def score(self, query, passages, doc_ids):
        self.calls += 1
        assert query == CANARY_QUERY
        assert list(passages) == list(CANARY_PASSAGES)
        assert list(doc_ids) == list(CANARY_DOC_IDS)
        return dict(self._scores)


BASE = {d: 10.0 - i for i, d in enumerate(CANARY_DOC_IDS)}


def test_inputs_are_parallel_and_frozen():
    assert len(CANARY_PASSAGES) == len(CANARY_DOC_IDS)
    assert len(set(CANARY_DOC_IDS)) == len(CANARY_DOC_IDS)
    assert isinstance(CANARY_PASSAGES, tuple), "must not be mutable"
    assert all(p.isascii() for p in CANARY_PASSAGES), (
        "non-ASCII would make the check sensitive to tokenizer handling, "
        "which is not what it is measuring"
    )


def test_scores_come_back_in_declared_order():
    out = canary_scores(FakeScorer(BASE))
    assert list(out) == list(CANARY_DOC_IDS)


def test_digest_is_stable_and_order_independent_of_dict_order():
    shuffled = dict(reversed(list(BASE.items())))
    assert canary_digest(FakeScorer(BASE)) == canary_digest(FakeScorer(shuffled))


def test_digest_changes_when_a_score_changes():
    drifted = dict(BASE)
    drifted[CANARY_DOC_IDS[0]] += 1e-5
    assert canary_digest(FakeScorer(BASE)) != canary_digest(FakeScorer(drifted))


def test_digest_tolerates_drift_below_the_rounding_place():
    """
    The digest rounds before hashing, so differences far below the rounding
    place must not produce a spurious mismatch and send anyone refilling.
    """
    tiny = dict(BASE)
    tiny[CANARY_DOC_IDS[0]] += 1e-9
    assert canary_digest(FakeScorer(BASE)) == canary_digest(FakeScorer(tiny))


def test_identical_stacks_are_mergeable():
    ok, why = compare_canaries(BASE, dict(BASE))
    assert ok and "identical" in why


def test_small_drift_is_mergeable_and_says_so():
    drifted = {d: v + 1e-6 for d, v in BASE.items()}
    ok, why = compare_canaries(drifted, BASE)
    assert ok
    assert "torch/cuda" in why, "a tolerated drift must still be recorded"


def test_reordering_blocks_the_merge():
    """
    The verdict that matters. Worlds are built by sorting on these scores, so
    a swap of two documents is a difference in every downstream ranking.
    """
    swapped = dict(BASE)
    a, b = CANARY_DOC_IDS[0], CANARY_DOC_IDS[1]
    swapped[a], swapped[b] = BASE[b], BASE[a]
    ok, why = compare_canaries(swapped, BASE)
    assert not ok
    assert "refilled on one machine" in why


def test_large_drift_without_reordering_still_blocks():
    """
    Ordering agreeing is necessary but not sufficient: a gap this large is not
    cuBLAS, it is a different model, precision or max_length.
    """
    drifted = {d: v + 0.5 for d, v in BASE.items()}
    ok, why = compare_canaries(drifted, BASE)
    assert not ok
    assert "model, precision and max_length" in why


def test_incomplete_canary_is_not_silently_mergeable():
    partial = {d: BASE[d] for d in CANARY_DOC_IDS[:-1]}
    ok, why = compare_canaries(partial, BASE)
    assert not ok and "incomplete" in why


# ── the recorded reference ────────────────────────────────────────────

def test_reference_digest_is_consistent_with_reference_scores():
    """
    The committed digest must be the one those committed scores produce.
    Otherwise a later member comparing against REFERENCE_DIGEST would be
    chasing a transcription error rather than a numerical difference.
    """
    assert canary_digest(FakeScorer(REFERENCE_SCORES)) == REFERENCE_DIGEST


def test_reference_covers_every_canary_pair():
    assert set(REFERENCE_SCORES) == set(CANARY_DOC_IDS)


def test_reference_compares_clean_against_itself():
    ok, why = compare_canaries(REFERENCE_SCORES, dict(REFERENCE_SCORES))
    assert ok and "identical" in why


def test_reference_ordering_is_semantically_sensible():
    """
    A guard against a silently broken canary. If the reranker ever stopped
    discriminating, the pairs would still produce a stable digest while
    measuring nothing. The exact-answer passage must outrank both unrelated
    ones, by a wide margin.
    """
    exact = REFERENCE_SCORES["canary00"]
    unrelated = max(REFERENCE_SCORES["canary03"], REFERENCE_SCORES["canary06"])
    assert exact > unrelated
    assert exact - unrelated > 10.0, "canary has stopped discriminating"


def test_reference_tripwire_is_far_above_float_drift():
    """
    `compare_canaries` disqualifies any reordering, so the smallest adjacent
    gap sets the sensitivity. It must sit far above float32 cuBLAS drift
    (~1e-5), or a clean merge would be blocked by noise.
    """
    order = sorted(CANARY_DOC_IDS, key=lambda d: (-REFERENCE_SCORES[d], d))
    gap = min(REFERENCE_SCORES[a] - REFERENCE_SCORES[b]
              for a, b in zip(order, order[1:]))
    assert gap > 1e-3, f"smallest adjacent gap {gap} is too tight to be safe"
