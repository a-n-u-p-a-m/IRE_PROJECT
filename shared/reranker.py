"""
reranker.py — Shared cross-encoder reranker wrapper.

Wraps sentence-transformers CrossEncoder so every team member uses
the exact same model, weights, truncation, and batching.

Primary model:   cross-encoder/ms-marco-MiniLM-L-12-v2
Extension model: castorini/monot5-base-msmarco  (for experiment E7)

Usage:
    from shared.reranker import Reranker

    reranker = Reranker()                    # MiniLM default
    scores   = reranker.score(query, docs)   # list[float]
    ranked   = reranker.rerank(query, docs, top_k=100)  # sorted
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import torch

logger = logging.getLogger(__name__)


@dataclass
class RankedDoc:
    """A document with its reranker score and rank."""
    doc_id: str
    text: str
    score: float
    rank: int  # 1-indexed


@dataclass
class RerankerConfig:
    """All reranker hyperparameters in one place for reproducibility."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    max_length: int = 512
    batch_size: int = 64
    device: str | None = None   # auto-detect if None
    dtype: str = "float32"       # "float16" for faster inference on GPU


class Reranker:
    """
    Thin wrapper around CrossEncoder with fixed configuration.

    All team members MUST use the same Reranker() (default config) for
    the primary experiments.  Only E7 (reranker strength) uses a
    different model_name.
    """

    def __init__(self, config: RerankerConfig | None = None) -> None:
        self.config = config or RerankerConfig()
        self._model = None  # lazy-loaded

    def _load_model(self) -> None:
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return

        from sentence_transformers import CrossEncoder

        device = self.config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            "Loading reranker %s on %s (max_length=%d)",
            self.config.model_name,
            device,
            self.config.max_length,
        )

        self._model = CrossEncoder(
            self.config.model_name,
            max_length=self.config.max_length,
            device=device,
        )

    @property
    def model(self):
        self._load_model()
        return self._model

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        """
        Score (query, document) pairs.

        Parameters
        ----------
        query : str
        documents : sequence of document texts

        Returns
        -------
        list[float]
            One score per document, in the same order as input.
        """
        if len(documents) == 0:
            return []

        pairs = [(query, doc) for doc in documents]
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        return [float(s) for s in raw_scores]

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[RankedDoc]:
        """
        Rerank candidate documents.

        Parameters
        ----------
        query : str
        candidates : sequence of (doc_id, doc_text) tuples
        top_k : int, optional
            Return only the top K.  None = return all, sorted.

        Returns
        -------
        list[RankedDoc]
            Sorted by reranker score (highest first), 1-indexed ranks.
        """
        if len(candidates) == 0:
            return []

        doc_ids = [c[0] for c in candidates]
        doc_texts = [c[1] for c in candidates]

        scores = self.score(query, doc_texts)

        # Build RankedDoc list, sort by score descending
        docs = [
            RankedDoc(doc_id=did, text=txt, score=s, rank=0)
            for did, txt, s in zip(doc_ids, doc_texts, scores)
        ]
        docs.sort(key=lambda d: d.score, reverse=True)

        # Assign 1-indexed ranks
        for i, doc in enumerate(docs):
            doc.rank = i + 1

        if top_k is not None:
            docs = docs[:top_k]

        return docs

    def rerank_from_score_matrix(
        self,
        query: str,
        query_id: str,
        candidates: Sequence[tuple[str, str]],
        corpus: dict[str, str],
    ) -> list[dict]:
        """
        Rerank and return rows ready for ScoreLogger.

        This is a convenience method that produces dicts matching the
        score-matrix schema.  The caller still needs to fill stage_1_*
        fields.

        Parameters
        ----------
        query : str
        query_id : str
        candidates : sequence of (doc_id, stage_1_score) from Stage 1
        corpus : dict mapping doc_id → passage text

        Returns
        -------
        list[dict]
            Each dict has: doc_id, stage_2_rank, stage_2_score.
        """
        pairs = []
        for doc_id, _ in candidates:
            text = corpus.get(doc_id, "")
            pairs.append((doc_id, text))

        ranked = self.rerank(query, pairs)

        return [
            {
                "doc_id": doc.doc_id,
                "stage_2_rank": doc.rank,
                "stage_2_score": doc.score,
            }
            for doc in ranked
        ]


# ──────────────────────────────────────────────────────────────────────
# MonoT5 reranker (extension, for E7 only)
# ──────────────────────────────────────────────────────────────────────

class MonoT5Reranker:
    """
    T5-based pointwise reranker (castorini/monot5-base-msmarco).

    Uses the Pygaggle/Castorini formulation:
      Input:  "Query: {query} Document: {document} Relevant:"
      Output: probability of generating "true" vs "false"
    """

    def __init__(
        self,
        model_name: str = "castorini/monot5-base-msmarco",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return

        from transformers import T5ForConditionalGeneration, T5Tokenizer

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading MonoT5 from %s on %s", self.model_name, device)
        self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(device).eval()
        self._device = device

        # Get token IDs for "true" and "false"
        self._true_id = self._tokenizer.encode("true", add_special_tokens=False)[0]
        self._false_id = self._tokenizer.encode("false", add_special_tokens=False)[0]

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Score documents. Returns P("true") for each (query, doc) pair."""
        self._load()

        prompts = [
            f"Query: {query} Document: {doc} Relevant:"
            for doc in documents
        ]

        all_scores: list[float] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self._device)

            with torch.no_grad():
                # Generate first token logits
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                logits = outputs.scores[0]  # (batch, vocab)
                true_logits = logits[:, self._true_id]
                false_logits = logits[:, self._false_id]

                # P(true) via softmax over [true, false]
                probs = torch.softmax(
                    torch.stack([true_logits, false_logits], dim=-1),
                    dim=-1,
                )[:, 0]

                all_scores.extend(probs.cpu().tolist())

        return all_scores

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[RankedDoc]:
        """Rerank candidates; same interface as Reranker.rerank."""
        doc_ids = [c[0] for c in candidates]
        doc_texts = [c[1] for c in candidates]

        scores = self.score(query, doc_texts)

        docs = [
            RankedDoc(doc_id=did, text=txt, score=s, rank=0)
            for did, txt, s in zip(doc_ids, doc_texts, scores)
        ]
        docs.sort(key=lambda d: d.score, reverse=True)
        for i, doc in enumerate(docs):
            doc.rank = i + 1

        if top_k is not None:
            docs = docs[:top_k]

        return docs
