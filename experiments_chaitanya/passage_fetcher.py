"""
passage_fetcher.py - MS MARCO passage text lookup.

Provides text for (query_id, doc_id) pairs so the reranker can score them.
The fetcher RAISES on a missing passage — never silently returns None or an
empty string — because a missed passage would produce a wrong score with no
observable symptom downstream.

Two backends are supported, selected at construction time:

  * ir_datasets (msmarco-passage/train or msmarco-passage) — the standard
    way to load the collection without having Pyserini or a local index.
    Install: pip install ir_datasets

  * dict / in-memory store — for tests, synthetic runs and CPU-only machines.
    Pass a pre-loaded {doc_id: text} mapping to the constructor.

Both expose the same interface so the cache filler does not need to know which
backend is active.
"""

from __future__ import annotations

import logging
from typing import Any, Union

logger = logging.getLogger(__name__)


class PassageFetchError(KeyError):
    """Raised when a requested doc_id is not found in the collection."""


class DictPassageFetcher:
    """
    In-memory passage fetcher for tests and synthetic runs.

    Parameters
    ----------
    passage_store:
        Mapping from doc_id (str) to passage text (str).  Any doc_id not
        present raises PassageFetchError.
    """

    def __init__(self, passage_store: dict[str, str]) -> None:
        self._store: dict[str, str] = passage_store

    def fetch(self, doc_id: str) -> str:
        """Return the passage text for doc_id or raise PassageFetchError."""
        doc_id = str(doc_id)
        try:
            return self._store[doc_id]
        except KeyError:
            raise PassageFetchError(
                f"doc_id {doc_id!r} not found in passage store. "
                "The reranker requires text for every document it scores; "
                "a missing passage would produce a wrong score with no "
                "observable symptom. Check that the passage store covers "
                "every document in U_q before calling the cache filler."
            ) from None

    def fetch_many(self, doc_ids: list[str]) -> list[str]:
        """
        Return passage texts for a list of doc_ids, preserving order.

        Raises PassageFetchError on the first missing doc_id.
        """
        return [self.fetch(d) for d in doc_ids]

    def __len__(self) -> int:
        return len(self._store)


class IrDatasetsPassageFetcher:
    """
    Passage fetcher backed by the ir_datasets MS MARCO passage collection.

    Lazy-loads the dataset on first access so that importing this module does
    not trigger a large download.  The dataset is cached by ir_datasets in
    ~/.ir_datasets; it is ~3 GB downloaded.

    Parameters
    ----------
    dataset_id:
        The ir_datasets identifier.  Default is 'msmarco-passage' which
        covers both DL 2019 and 2020 queries.
    cache_size:
        Number of passages to keep in an LRU memory cache.  The universe
        for our 97 queries is ~872,455 unique pairs, but many documents
        appear in multiple queries, so a modest cache cuts I/O significantly.
    """

    def __init__(
        self,
        dataset_id: str = "msmarco-passage",
        cache_size: int = 50_000,
    ) -> None:
        self.dataset_id = dataset_id
        self.cache_size = cache_size
        self._lookup: dict[str, str] | None = None
        self._loaded = False

    def _load(self) -> None:
        """Load the passage store into memory on first access."""
        if self._loaded:
            return
        try:
            import ir_datasets
        except ImportError as exc:
            raise ImportError(
                "ir_datasets must be installed to use IrDatasetsPassageFetcher.\n"
                "Install with: pip install ir_datasets\n"
                "Alternatively, use DictPassageFetcher with a pre-loaded store."
            ) from exc

        logger.info("loading passage collection %s via ir_datasets …", self.dataset_id)
        dataset = ir_datasets.load(self.dataset_id)

        # Build a complete doc_id → text lookup.  MS MARCO passage collection
        # has ~8.8 million passages; loading all into a dict uses ~3–4 GB RAM.
        # This is a one-time cost per process; subsequent calls hit the dict.
        self._lookup = {}
        for doc in dataset.docs_iter():
            self._lookup[doc.doc_id] = doc.text

        logger.info(
            "passage collection loaded: %d passages", len(self._lookup)
        )
        self._loaded = True

    def fetch(self, doc_id: str) -> str:
        """Return the passage text for doc_id or raise PassageFetchError."""
        self._load()
        doc_id = str(doc_id)
        assert self._lookup is not None
        try:
            return self._lookup[doc_id]
        except KeyError:
            raise PassageFetchError(
                f"doc_id {doc_id!r} not found in {self.dataset_id}. "
                "Check that this doc_id belongs to the MS MARCO passage "
                "collection and that the ir_datasets download completed "
                "without error."
            ) from None

    def fetch_many(self, doc_ids: list[str]) -> list[str]:
        """Return passage texts for a list of doc_ids, preserving order."""
        return [self.fetch(d) for d in doc_ids]

    def num_loaded(self) -> int:
        """Number of passages in the loaded lookup (0 if not yet loaded)."""
        return len(self._lookup) if self._lookup is not None else 0


def make_fetcher(
    backend: str = "ir_datasets",
    passage_store: dict[str, str] | None = None,
    dataset_id: str = "msmarco-passage",
) -> DictPassageFetcher | IrDatasetsPassageFetcher:
    """
    Factory: return the right fetcher for the given backend.

    Parameters
    ----------
    backend:
        'ir_datasets' — use IrDatasetsPassageFetcher (default, for real runs).
        'dict'        — use DictPassageFetcher (for tests / synthetic runs).
    passage_store:
        Required when backend='dict'.
    dataset_id:
        Used only when backend='ir_datasets'.
    """
    if backend == "dict":
        if passage_store is None:
            raise ValueError("passage_store is required when backend='dict'")
        return DictPassageFetcher(passage_store)
    elif backend == "ir_datasets":
        return IrDatasetsPassageFetcher(dataset_id=dataset_id)
    else:
        raise ValueError(
            f"backend must be 'dict' or 'ir_datasets', got {backend!r}"
        )


# Type alias used in the cache filler
AnyFetcher = Union[DictPassageFetcher, IrDatasetsPassageFetcher]


__all__ = [
    "PassageFetchError",
    "DictPassageFetcher",
    "IrDatasetsPassageFetcher",
    "make_fetcher",
    "AnyFetcher",
]
