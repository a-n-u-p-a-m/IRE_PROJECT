"""
canary.py - A cross-machine consistency check for the Stage-2 scorer.

THE PROBLEM THIS EXISTS FOR. The Stage-2 cache is filled in three disjoint
shards on three machines. Because the shards are disjoint, no (query, doc)
pair is ever scored twice, so a numerical difference BETWEEN the machines is
invisible: each shard's digest covers only its own pairs, and `merge_from`
unions them without ever comparing like with like. The merged cache would then
hold scores from two slightly different functions while claiming to hold one.

That is not hypothetical. The first shard was filled under torch 2.11.0+cu130
on Python 3.13.16; a second machine on another platform will differ, and
float32 matmul is not bit-reproducible across cuBLAS versions. The expected
impact on DCG@10 is nil - the differences live far below the gap between
adjacent documents - but "expected" is exactly the kind of claim the rest of
this project refuses to make without measuring, and the frozen-universe
discipline does not permit an unverified join.

WHY THE PAIRS ARE HARD-CODED. The obvious canary - re-score a few pairs from
someone else's shard - does not work, and the reason is worth recording so it
is not attempted again. The passage fetcher is constructed with
`required_doc_ids` covering only the caller's OWN shard, so another member's
documents cannot be fetched without a second full pass over the collection.
Nothing about detecting numerical drift requires real MS MARCO passages: the
drift is a property of the model and the numerical stack, not of the text. So
the inputs are frozen here as literals, and the check costs milliseconds,
needs no corpus, no universe, no filled shard and no GPU.

USE. Every member runs `canary_digest(reranker)` on their fill machine and
posts the digest. All three must match.

  * identical digests - the stacks agree and the shards may be merged;
  * digests differ but `canary_scores` agree to ~1e-4 - tolerable drift,
    record both manifests' torch/cuda versions beside the merged digest and
    say so in the report;
  * any pair ORDERED differently - not tolerable. Ordering is what every
    world is built from, so the shards must be refilled on one machine.

The literals must never be edited: a changed input silently changes every
digest and destroys comparability with the digests already posted. They are
deliberately boring - short, ASCII, no tokenizer edge cases - because the
check is for the numerical stack, not for text handling.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

__all__ = [
    "CANARY_QUERY", "CANARY_PASSAGES", "CANARY_DOC_IDS",
    "REFERENCE_DIGEST", "REFERENCE_SCORES", "REFERENCE_ENV",
    "canary_scores", "canary_digest", "compare_canaries",
]

CANARY_QUERY = "what is the boiling point of water"

# Deliberately spans the relevance range: an exact answer, a near miss, a
# topical distractor and an unrelated passage. A stack difference that moved
# only the top score would be caught; so would one that reordered the middle.
CANARY_PASSAGES: tuple[str, ...] = (
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The boiling point of a liquid depends on the surrounding pressure.",
    "Ice melts at 0 degrees Celsius, which is the freezing point of water.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "At higher altitudes water boils at a lower temperature than 100 C.",
    "A kettle is a container used for boiling water in a kitchen.",
    "The Pacific Ocean is the largest and deepest of Earth's oceans.",
    "Steam is the gaseous phase of water, formed when water boils.",
)

CANARY_DOC_IDS: tuple[str, ...] = tuple(
    f"canary{i:02d}" for i in range(len(CANARY_PASSAGES))
)


# ── the recorded reference ────────────────────────────────────────────
#
# An OBSERVATION, not a definition of correctness: what this model produced on
# the environment below, during the shard-1 fill on Kaggle. It is committed so
# a later fill can be checked without anyone else being online - run the
# canary, compare to REFERENCE_DIGEST, done.
#
# Compare only against a matching REFERENCE_ENV. torch and cuda are what
# matter; numpy is listed because it is in the manifest, but it takes no part
# in the forward pass (the shard-1 and shard-2 fills differed on numpy -
# 2.1.3 against 2.4.6 - and agreed on everything that does matter).
REFERENCE_ENV: dict[str, str] = {
    "python": "3.13.16",
    "torch": "2.11.0+cu130",
    "cuda": "13.0",
    "gpu": "Tesla T4",
    "numpy": "2.1.3",
    "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "precision": "float32",
    "max_length": "512",
    "observed": "2026-11-09, shard anupam, Kaggle",
}

REFERENCE_SCORES: dict[str, float] = {
    "canary00": 5.095918,
    "canary01": 3.934693,
    "canary02": -2.580505,
    "canary03": -11.232469,
    "canary04": 3.759547,
    "canary05": -4.246465,
    "canary06": -11.252432,
    "canary07": -4.313988,
}

REFERENCE_DIGEST = "e1121c9d61e14fd5"


class _Scorer(Protocol):
    def score(self, query: str, passages, doc_ids) -> dict[str, float]: ...


def canary_scores(reranker: _Scorer) -> dict[str, float]:
    """{doc_id: score} over the frozen canary pairs, in declared order."""
    scores = reranker.score(
        CANARY_QUERY, list(CANARY_PASSAGES), list(CANARY_DOC_IDS)
    )
    return {d: float(scores[d]) for d in CANARY_DOC_IDS}


def canary_digest(reranker: _Scorer, places: int = 6) -> str:
    """
    A short fingerprint of the canary scores, for posting in a message.

    Rounded to `places` before hashing, so a digest match means agreement to
    that many decimals rather than bit-identity. Bit-identity across different
    cuBLAS versions is not achievable and is not what the check is asking for;
    agreement to 1e-6 is far tighter than any gap that could reorder a world.
    """
    scores = canary_scores(reranker)
    payload = "".join(f"{d}:{scores[d]:.{places}f}" for d in CANARY_DOC_IDS)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def compare_canaries(
    a: dict[str, float], b: dict[str, float], tol: float = 1e-4
) -> tuple[bool, str]:
    """
    Compare two members' canary scores and say whether the merge is sound.

    Returns (mergeable, explanation). The ordering check is the one that
    decides: scores feed `order_by`, and only a difference large enough to
    swap two documents can change any world.
    """
    missing = set(CANARY_DOC_IDS) - (set(a) & set(b))
    if missing:
        return False, f"canary is incomplete on one side: missing {sorted(missing)}"

    order_a = sorted(CANARY_DOC_IDS, key=lambda d: (-a[d], d))
    order_b = sorted(CANARY_DOC_IDS, key=lambda d: (-b[d], d))
    if order_a != order_b:
        return False, (
            "the two stacks ORDER the canary differently "
            f"({order_a} vs {order_b}). Every world is built by sorting on "
            "these scores, so the shards must be refilled on one machine."
        )

    worst = max(abs(a[d] - b[d]) for d in CANARY_DOC_IDS)
    if worst == 0.0:
        return True, "canary scores are identical; the stacks agree exactly"
    if worst <= tol:
        return True, (
            f"canary agrees on ordering, worst absolute difference {worst:.2e} "
            f"(<= {tol:.0e}). Mergeable; record both torch/cuda versions "
            "beside the merged digest."
        )
    return False, (
        f"canary ordering agrees but the worst difference is {worst:.2e}, "
        f"above the {tol:.0e} tolerance. Too large to attribute to cuBLAS; "
        "check that both sides use the same model, precision and max_length."
    )
