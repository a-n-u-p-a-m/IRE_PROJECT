"""
reranker.py - Pointwise Stage-2 reranker wrapper.

The reranker is the cross-encoder/ms-marco-MiniLM-L-12-v2 model.  It scores
one (query, passage) pair at a time from the model's perspective — the scores
are independent of the candidate set, which is the property that makes one
inference pass over U_q serve every downstream experiment.

Design decisions recorded here:

  * Lazy loading: torch and sentence-transformers are NOT imported at module
    top level.  A CPU-only machine can import this module without having torch
    installed; only the first call to score() triggers the actual import.

  * The listwise stub refuses to run.  A listwise reranker is out of scope
    (proposal § 3.4) and its score semantics break the E_gate derivation
    (scores would depend on which documents are in the candidate set).  The
    stub exists so the interface slot is occupied and cannot be filled
    accidentally with a listwise model.

  * Passage-length truncation is reported.  MS MARCO passages average ~55
    words; padding to 512 tokens wastes GPU throughput.  If the mean observed
    length is well below max_length, the throughput script uses it to estimate
    the effective inference cost.

  * Shuffle-invariance contract: score(q, d) must equal score(q, d) regardless
    of which other documents are passed in the same batch.  The integration
    test in tests/test_chaitanya.py checks this on small synthetic inputs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type annotations only — never executed unless the type-checker runs.
    import torch
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# The frozen model name from config/base.yaml stage2.model
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 64


class ListwiseRerankerError(RuntimeError):
    """Raised whenever any code tries to instantiate the listwise stub."""


class PointwiseCrossEncoder:
    """
    Wrapper around a SentenceTransformers CrossEncoder for pointwise scoring.

    Lazy-loads torch and sentence-transformers on first call so that this
    module is importable on CPU-only machines without GPU dependencies.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        precision: str = "float32",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.precision = precision
        self.device = device
        self._model: CrossEncoder | None = None

    # ── lazy load ────────────────────────────────────────────────────

    def _load(self) -> None:
        """Import and initialise the CrossEncoder on first use."""
        if self._model is not None:
            return
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers (and torch) must be installed to use "
                "the reranker.  Install them with:\n"
                "  pip install torch sentence-transformers\n"
                "For CPU-only development you can use the synthetic reranker "
                "in shared/synthetic.py instead."
            ) from exc

        logger.info("loading reranker %s", self.model_name)
        _device = self.device
        if _device is None:
            _device = "cuda" if torch.cuda.is_available() else "cpu"

        model = CrossEncoder(self.model_name, max_length=self.max_length,
                             device=_device)

        if self.precision == "float16" and _device != "cpu":
            # Half-precision cuts VRAM and inference time at negligible quality loss
            # on scoring tasks (not generation).
            model.model = model.model.half()
            logger.info("reranker cast to float16")
        elif self.precision not in {"float32", "float16"}:
            raise ValueError(
                f"precision must be 'float32' or 'float16', got {self.precision!r}"
            )

        self._model = model
        logger.info("reranker loaded on %s (max_length=%d, batch=%d, precision=%s)",
                    _device, self.max_length, self.batch_size, self.precision)

    # ── public API ───────────────────────────────────────────────────

    @property
    def checkpoint(self) -> str:
        """Model name used as the 'model' key in stage2_cache rows."""
        return self.model_name

    def score(
        self,
        query: str,
        passages: Sequence[str],
        doc_ids: Sequence[str],
    ) -> dict[str, float]:
        """
        Score a batch of passages against one query.

        Parameters
        ----------
        query:
            The query text.
        passages:
            Passage texts parallel to doc_ids.  Must be non-empty.
        doc_ids:
            Identifiers used as the cache keys.  Must be parallel to passages.

        Returns
        -------
        dict[str, float]
            {doc_id: score} for every doc_id in the input.

        Notes
        -----
        Score order is invariant to the order of (passage, doc_id) pairs —
        the cross-encoder scores each pair independently.  The integration test
        asserts this property explicitly.
        """
        if len(passages) != len(doc_ids):
            raise ValueError(
                f"passages ({len(passages)}) and doc_ids ({len(doc_ids)}) "
                "must have the same length"
            )
        if not passages:
            return {}

        self._load()

        pairs = [(query, p) for p in passages]
        t0 = time.perf_counter()
        scores = self._model.predict(  # type: ignore[union-attr]
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - t0
        logger.debug(
            "scored %d pairs in %.3fs (%.1f pairs/s)",
            len(pairs), elapsed, len(pairs) / max(elapsed, 1e-9),
        )

        return {str(did): float(s) for did, s in zip(doc_ids, scores)}

    def score_many(
        self,
        query: str,
        passages: Sequence[str],
        doc_ids: Sequence[str],
    ) -> dict[str, float]:
        """
        Like score(), but internally batches calls to respect self.batch_size.

        Use this for large candidate sets (e.g., the full U_q).  Identical
        results to score() — chunking is a throughput optimisation only.
        """
        if len(passages) != len(doc_ids):
            raise ValueError(
                "passages and doc_ids must be parallel sequences of equal length"
            )
        if not passages:
            return {}

        self._load()
        result: dict[str, float] = {}

        for start in range(0, len(passages), self.batch_size):
            chunk_p = passages[start: start + self.batch_size]
            chunk_d = doc_ids[start: start + self.batch_size]
            chunk_scores = self.score(query, chunk_p, chunk_d)
            result.update(chunk_scores)

        return result

    def verify_shuffle_invariance(
        self,
        query: str,
        passages: Sequence[str],
        doc_ids: Sequence[str],
        n_permutations: int = 3,
        seed: int = 42,
    ) -> bool:
        """
        Verify that scores are independent of input order.

        A pointwise cross-encoder must satisfy this: score(q, d) cannot depend
        on which other documents are passed in the same batch.  If this returns
        False, the E_gate derivation breaks.

        Returns True if invariance holds for all tested permutations.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        reference = self.score(query, list(passages), list(doc_ids))

        for _ in range(n_permutations):
            idx = rng.permutation(len(passages)).tolist()
            shuffled_p = [passages[i] for i in idx]
            shuffled_d = [doc_ids[i] for i in idx]
            result = self.score(query, shuffled_p, shuffled_d)
            for did in doc_ids:
                ref_s = reference[did]
                new_s = result[did]
                if abs(ref_s - new_s) > 1e-5:
                    logger.error(
                        "shuffle invariance FAILED: doc %s scored %.6f vs %.6f",
                        did, ref_s, new_s,
                    )
                    return False

        return True


class ListwiseCrossEncoderStub:
    """
    Stub that occupies the listwise slot and refuses to run.

    A listwise reranker's score for document d depends on the candidate set
    (the other documents in the window), so score(q, d | C_q) != score(q, d |
    U_q) and the one-pass-over-U_q engineering assumption breaks.  The
    proposal (§ 3.4) places listwise reranking out of scope.  This class exists
    so the code slot is filled and any attempt to accidentally use it fails
    loudly rather than silently.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise ListwiseRerankerError(
            "ListwiseCrossEncoderStub cannot be instantiated. "
            "Listwise reranking is out of scope (proposal § 3.4): its score "
            "semantics break the E_gate derivation because score(q, d | C_q) "
            "is not equal to score(q, d | U_q). Use PointwiseCrossEncoder."
        )

    def score(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise ListwiseRerankerError("listwise stub cannot score")


__all__ = [
    "PointwiseCrossEncoder",
    "ListwiseCrossEncoderStub",
    "ListwiseRerankerError",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_BATCH_SIZE",
]
