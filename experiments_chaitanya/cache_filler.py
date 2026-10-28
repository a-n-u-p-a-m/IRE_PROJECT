"""
cache_filler.py - Resumable, shardable Stage-2 cache filler.

This is the only piece of code that needs a GPU.  Everything else in the
project (the counterfactual engine, all metrics, all statistics) works from
the cached scores in shared/cache.py.

Design decisions:

  * Resumable.  Missing pairs are computed from cache.missing() before any
    scoring starts.  A killed job resumes from wherever it stopped: it re-
    computes the missing set and scores only those.  It never re-scores a pair
    that is already in the cache.

  * Shardable.  The caller supplies a list of query_ids; each member of the
    team fills a disjoint shard.  Shards are merged into the primary cache with
    Stage2Cache.merge_from() — that call is idempotent, so re-merging is safe.

  * Throughput-instrumented.  After scoring, the filler logs pairs/sec and
    mean sequence length so the cost model in docs/handoff_week1.md can be
    verified or updated.

  * Batch-writing.  Scores are written to the cache after each batch, not at
    the end.  The batch size matches the reranker's batch_size so GPU utilisation
    is not throttled by I/O.

Usage example (Colab, worlds A/B first)::

    from experiments_chaitanya.cache_filler import CacheFiller
    from experiments_chaitanya.reranker import PointwiseCrossEncoder
    from experiments_chaitanya.passage_fetcher import IrDatasetsPassageFetcher
    from shared.cache import Stage2Cache
    import pandas as pd

    universe = pd.read_parquet("data/universe.parquet")
    cache = Stage2Cache("cache/stage2.sqlite")
    reranker = PointwiseCrossEncoder()
    fetcher = IrDatasetsPassageFetcher()

    filler = CacheFiller(reranker=reranker, fetcher=fetcher, cache=cache)
    report = filler.fill_shard(
        universe=universe,
        queries=queries_file,     # list of query_id strings for this shard
        query_texts=query_texts,  # dict {query_id: text}
        world="AB",               # "AB" scores C_q; "CD" scores D_q / ref subset
    )
    print(report)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .passage_fetcher import AnyFetcher
    from .reranker import PointwiseCrossEncoder
    from shared.cache import Stage2Cache

logger = logging.getLogger(__name__)


@dataclass
class FillReport:
    """Summary produced by CacheFiller.fill_shard()."""
    model: str
    n_queries: int
    n_pairs_scored: int
    n_pairs_skipped: int          # already in cache — not re-scored
    total_tokens_approx: int      # approx sum of sequence lengths
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)

    @property
    def pairs_per_second(self) -> float:
        return self.n_pairs_scored / max(self.elapsed_seconds, 1e-9)

    @property
    def mean_seq_length(self) -> float:
        return (
            self.total_tokens_approx / max(self.n_pairs_scored, 1)
        )

    def __str__(self) -> str:
        lines = [
            f"Fill report — {self.model}",
            f"  Queries      : {self.n_queries}",
            f"  Pairs scored : {self.n_pairs_scored}",
            f"  Pairs skipped: {self.n_pairs_skipped} (already cached)",
            f"  Elapsed      : {self.elapsed_seconds:.1f}s",
            f"  Throughput   : {self.pairs_per_second:.1f} pairs/s",
            f"  Mean seq len : {self.mean_seq_length:.1f} tokens (approx)",
        ]
        if self.errors:
            lines.append(f"  Errors       : {len(self.errors)}")
            for e in self.errors[:3]:
                lines.append(f"    {e}")
        return "\n".join(lines)


class CacheFiller:
    """
    Scores (query, doc) pairs with the reranker and writes to the Stage-2 cache.

    Parameters
    ----------
    reranker:
        A PointwiseCrossEncoder instance.  Its .checkpoint property is used as
        the 'model' column in the cache.
    fetcher:
        A passage fetcher (DictPassageFetcher or IrDatasetsPassageFetcher).
    cache:
        The Stage2Cache to write to.
    query_batch_size:
        How many queries to process before logging progress.
    """

    def __init__(
        self,
        reranker: PointwiseCrossEncoder,
        fetcher: AnyFetcher,
        cache: Stage2Cache,
        query_batch_size: int = 10,
    ) -> None:
        self.reranker = reranker
        self.fetcher = fetcher
        self.cache = cache
        self.query_batch_size = query_batch_size

    def fill_shard(
        self,
        universe: pd.DataFrame,
        query_ids: list[str],
        query_texts: dict[str, str],
    ) -> FillReport:
        """
        Score all (query, doc) pairs in the universe for the given query_ids.

        Only pairs not already in the cache are scored; the rest are skipped.
        Results are written per-batch so the job is resumable after interruption.

        Parameters
        ----------
        universe:
            The frozen universe table (schema: query_id, doc_id, sources).
            Only rows matching query_ids are scored.
        query_ids:
            The list of query_ids assigned to this shard.
        query_texts:
            {query_id: query_text} for every query_id in query_ids.

        Returns
        -------
        FillReport
        """
        model = self.reranker.checkpoint
        universe_subset = universe[universe["query_id"].isin(query_ids)]

        n_queries = len(query_ids)
        n_pairs_scored = 0
        n_pairs_skipped = 0
        total_tokens = 0
        errors: list[str] = []
        t_start = time.perf_counter()

        for i, qid in enumerate(query_ids):
            qid = str(qid)
            if qid not in query_texts:
                msg = f"query_id {qid!r} not found in query_texts"
                logger.error(msg)
                errors.append(msg)
                continue

            query_text = query_texts[qid]
            doc_ids = (
                universe_subset[universe_subset["query_id"] == qid]["doc_id"]
                .astype(str)
                .tolist()
            )

            # Only score what is missing from the cache (resumable)
            missing_docs = self.cache.missing(qid, doc_ids, model)
            n_pairs_skipped += len(doc_ids) - len(missing_docs)

            if not missing_docs:
                logger.debug("query %s: all %d pairs already cached", qid, len(doc_ids))
                continue

            # Fetch passage texts — raises PassageFetchError on any missing doc
            try:
                passage_texts = self.fetcher.fetch_many(missing_docs)
            except Exception as exc:
                msg = f"query {qid}: passage fetch failed: {exc}"
                logger.error(msg)
                errors.append(msg)
                continue

            # Score and write in sub-batches matching reranker.batch_size
            for start in range(0, len(missing_docs), self.reranker.batch_size):
                batch_docs = missing_docs[start: start + self.reranker.batch_size]
                batch_texts = passage_texts[start: start + self.reranker.batch_size]

                scores = self.reranker.score(query_text, batch_texts, batch_docs)
                self.cache.put_many(
                    model,
                    [(qid, did, scores[did]) for did in batch_docs],
                )

                n_pairs_scored += len(batch_docs)
                # Approximate token count: words * 1.3 (average word-to-token ratio
                # for English text; good enough for throughput estimation).
                total_tokens += sum(
                    int(len(t.split()) * 1.3) for t in batch_texts
                )

            if (i + 1) % self.query_batch_size == 0 or i == n_queries - 1:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "progress %d/%d queries | %d pairs scored | %.1f pairs/s",
                    i + 1, n_queries, n_pairs_scored,
                    n_pairs_scored / max(elapsed, 1e-9),
                )

        elapsed = time.perf_counter() - t_start
        report = FillReport(
            model=model,
            n_queries=n_queries,
            n_pairs_scored=n_pairs_scored,
            n_pairs_skipped=n_pairs_skipped,
            total_tokens_approx=total_tokens,
            elapsed_seconds=elapsed,
            errors=errors,
        )
        logger.info(str(report))
        return report

    def verify_completeness(
        self,
        universe: pd.DataFrame,
        query_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Delegate to cache.coverage_report() and log missing pairs.

        Returns the per-query coverage DataFrame from Stage2Cache.
        """
        return self.cache.coverage_report(
            universe, self.reranker.checkpoint, query_ids
        )


def load_query_texts(queries_tsv: str) -> dict[str, str]:
    """
    Load query texts from the standard two-column TSV (query_id, text).

    Parameters
    ----------
    queries_tsv:
        Path to data/queries.tsv.

    Returns
    -------
    dict[str, str]
    """
    import csv
    result: dict[str, str] = {}
    with open(queries_tsv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                result[row[0].strip()] = row[1].strip()
    return result


__all__ = ["CacheFiller", "FillReport", "load_query_texts"]
