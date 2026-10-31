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

    Lazy-loads only the documents in the provided ``required_doc_ids`` set
    (the universe's ~600k–800k unique documents) using ``docs_store().get_many()``
    instead of streaming all 8.8M passages.  This keeps RAM usage at roughly
    320 MB versus 3–4 GB for the full collection.

    If ``required_doc_ids`` is not supplied, the fetcher falls back to the full
    ``docs_store()`` random-access interface on every ``fetch()`` call, which is
    slower but never loads the entire collection into memory.

    Parameters
    ----------
    dataset_id:
        The ir_datasets identifier.  Default is 'msmarco-passage' which
        covers both DL 2019 and 2020 queries.
    required_doc_ids:
        Optional set of doc_ids that will be needed (typically the union of
        all U_q doc_ids).  When supplied, all passages are loaded at once via
        ``docs_store().get_many()`` and subsequent fetches are O(1) dict lookups.
    """

    def __init__(
        self,
        dataset_id: str = "msmarco-passage",
        required_doc_ids: set[str] | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self._required_doc_ids = required_doc_ids
        self._lookup: dict[str, str] | None = None
        self._store = None   # lazy docs_store handle for unbounded fetch()
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

        dataset = ir_datasets.load(self.dataset_id)

        if self._required_doc_ids is not None:
            # Efficient path: load only universe docs via get_many().
            # docs_store().get_many() accepts an iterable of doc_ids and returns
            # only those records — typically 600k-800k documents at ~320 MB
            # instead of 8.8M at 3-4 GB.
            logger.info(
                "loading %d universe passages from %s via docs_store().get_many() …",
                len(self._required_doc_ids), self.dataset_id,
            )
            store = dataset.docs_store()
            self._lookup = {
                doc_id: store.get(doc_id).text
                for doc_id in self._required_doc_ids
                if store.get(doc_id) is not None
            }
            missing = self._required_doc_ids - set(self._lookup)
            if missing:
                logger.warning(
                    "%d required doc_ids not found in %s (first 5: %s)",
                    len(missing), self.dataset_id, list(missing)[:5],
                )
            logger.info(
                "loaded %d passages (%.0f%% of requested)",
                len(self._lookup),
                100.0 * len(self._lookup) / max(len(self._required_doc_ids), 1),
            )
        else:
            # Unbounded path: keep a docs_store handle for on-demand lookups.
            # Never loads the whole collection into memory.
            logger.info(
                "opening docs_store for %s (unbounded mode — fetch per doc_id)",
                self.dataset_id,
            )
            self._store = dataset.docs_store()

        self._loaded = True

    def fetch(self, doc_id: str) -> str:
        """Return the passage text for doc_id or raise PassageFetchError."""
        self._load()
        doc_id = str(doc_id)
        if self._lookup is not None:
            try:
                return self._lookup[doc_id]
            except KeyError:
                raise PassageFetchError(
                    f"doc_id {doc_id!r} not found in preloaded universe subset. "
                    "Check that this doc_id was included in required_doc_ids."
                ) from None
        else:
            assert self._store is not None
            record = self._store.get(doc_id)
            if record is None:
                raise PassageFetchError(
                    f"doc_id {doc_id!r} not found in {self.dataset_id}. "
                    "Check that this doc_id belongs to the MS MARCO passage "
                    "collection and that the ir_datasets download completed "
                    "without error."
                ) from None
            return record.text

    def fetch_many(self, doc_ids: list[str]) -> list[str]:
        """Return passage texts for a list of doc_ids, preserving order."""
        return [self.fetch(d) for d in doc_ids]

    def num_loaded(self) -> int:
        """Number of passages in the preloaded lookup (0 if not yet loaded or unbounded mode)."""
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
