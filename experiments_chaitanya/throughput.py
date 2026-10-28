"""
throughput.py - Stage-2 reranker throughput measurement script.

Measures pairs/sec and mean sequence length at the FROZEN parameters from
config/base.yaml (max_length=512, batch_size=64, precision=float32).

Run on a machine with GPU (Colab T4 recommended) before committing to the
full cache fill.  The two key numbers this produces:

  pairs_per_second  — divides 872,455 (total unique pairs for the project)
                      to get the estimated GPU time
  mean_seq_length   — if this is much less than 512, reducing max_length
                      would cut inference time with no quality effect

Usage::

    # On Colab or a GPU machine:
    python -m experiments_chaitanya.throughput \
        --queries data/queries.tsv \
        --universe data/universe.parquet \
        --n-sample 500 \
        --output throughput_report.json

    # CPU-only dry run (synthetic passages, no GPU needed):
    python -m experiments_chaitanya.throughput --synthetic --n-sample 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Frozen config parameters — these MUST match config/base.yaml
FROZEN_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
FROZEN_MAX_LENGTH = 512
FROZEN_BATCH_SIZE = 64
FROZEN_PRECISION = "float32"

# MS MARCO passage average word count (from the literature; used for
# sequence-length estimation in the synthetic mode).
MSMARCO_AVG_WORDS = 55


def _make_synthetic_passages(n: int, seed: int = 42) -> tuple[list[str], list[str]]:
    """
    Generate n synthetic (doc_id, passage_text) pairs for dry-run mode.

    Passages are drawn from a fixed vocabulary so they look like English text
    but need no download.  Length is sampled from a realistic distribution
    centred on MSMARCO_AVG_WORDS.
    """
    rng = random.Random(seed)
    words = [
        "the", "a", "of", "to", "and", "in", "is", "for", "that", "with",
        "on", "are", "as", "at", "be", "by", "it", "this", "was", "from",
        "retrieval", "document", "passage", "query", "relevant", "score",
        "ranking", "information", "search", "system", "model", "neural",
        "dense", "sparse", "index", "candidate", "reranker", "effect",
    ]
    doc_ids: list[str] = []
    passages: list[str] = []
    for i in range(n):
        length = max(10, int(rng.gauss(MSMARCO_AVG_WORDS, 20)))
        text = " ".join(rng.choice(words) for _ in range(length))
        doc_ids.append(f"sd{i:07d}")
        passages.append(text)
    return doc_ids, passages


def measure_throughput(
    query_text: str,
    doc_ids: list[str],
    passages: list[str],
    model_name: str = FROZEN_MODEL,
    max_length: int = FROZEN_MAX_LENGTH,
    batch_size: int = FROZEN_BATCH_SIZE,
    precision: str = FROZEN_PRECISION,
    device: str | None = None,
    n_warmup_batches: int = 2,
) -> dict:
    """
    Score all passages against query_text and return a throughput report dict.

    The first n_warmup_batches are excluded from timing to avoid penalising
    JIT compilation / CUDA kernel initialisation.
    """
    from experiments_chaitanya.reranker import PointwiseCrossEncoder

    reranker = PointwiseCrossEncoder(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        precision=precision,
        device=device,
    )

    n_total = len(passages)
    if n_total == 0:
        raise ValueError("passages must be non-empty")

    # Warmup
    warmup_n = min(n_warmup_batches * batch_size, n_total)
    logger.info("warming up on %d pairs …", warmup_n)
    reranker.score(query_text, passages[:warmup_n], doc_ids[:warmup_n])

    # Timed run
    logger.info("timing %d pairs …", n_total)
    t0 = time.perf_counter()
    scores = reranker.score_many(query_text, passages, doc_ids)
    elapsed = time.perf_counter() - t0

    pairs_per_second = n_total / max(elapsed, 1e-9)

    # Sequence length in tokens (approx — word count * 1.3 heuristic)
    token_lengths = [len(p.split()) * 1.3 for p in passages]
    mean_seq_length = sum(token_lengths) / len(token_lengths)
    max_seq_length = max(token_lengths)

    # Project the full cache fill
    total_pairs_universe = 872_455
    estimated_hours = total_pairs_universe / pairs_per_second / 3600

    report = {
        "model": model_name,
        "max_length": max_length,
        "batch_size": batch_size,
        "precision": precision,
        "n_pairs": n_total,
        "elapsed_seconds": round(elapsed, 3),
        "pairs_per_second": round(pairs_per_second, 1),
        "mean_seq_length_tokens_approx": round(mean_seq_length, 1),
        "max_seq_length_tokens_approx": round(max_seq_length, 1),
        "estimated_hours_full_universe": round(estimated_hours, 2),
        "estimated_hours_full_universe_3way_split": round(estimated_hours / 3, 2),
        "note": (
            "Sequence lengths are estimates (word_count × 1.3). "
            "Actual tokenisation may differ by ±10%."
        ),
    }
    return report


def run_synthetic_benchmark(n_sample: int = 200, seed: int = 42) -> dict:
    """Dry run on synthetic passages — no GPU, no download needed."""
    doc_ids, passages = _make_synthetic_passages(n_sample, seed)
    query_text = "what is information retrieval"
    logger.info("running synthetic throughput benchmark (CPU, %d pairs)", n_sample)
    return measure_throughput(
        query_text=query_text,
        doc_ids=doc_ids,
        passages=passages,
        model_name=FROZEN_MODEL,
        max_length=FROZEN_MAX_LENGTH,
        batch_size=FROZEN_BATCH_SIZE,
        precision=FROZEN_PRECISION,
    )


def run_real_benchmark(
    queries_tsv: str,
    universe_parquet: str,
    n_sample: int = 500,
    seed: int = 42,
) -> dict:
    """
    Sample n_sample documents from the real universe and measure throughput.

    Samples uniformly across all 97 queries so mean sequence length reflects
    the real distribution, not a cherry-picked query.
    """
    import pandas as pd
    from experiments_chaitanya.cache_filler import load_query_texts
    from experiments_chaitanya.passage_fetcher import IrDatasetsPassageFetcher

    query_texts = load_query_texts(queries_tsv)
    universe = pd.read_parquet(universe_parquet)

    rng = random.Random(seed)
    sample = universe.sample(n=min(n_sample, len(universe)), random_state=seed)
    doc_ids = sample["doc_id"].astype(str).tolist()

    # Fetch real passages
    fetcher = IrDatasetsPassageFetcher()
    passages = fetcher.fetch_many(doc_ids)

    # Use the first query for all pairs (throughput is query-independent for
    # a pointwise model; using one query is fine)
    first_query = next(iter(query_texts.values()))

    return measure_throughput(
        query_text=first_query,
        doc_ids=doc_ids,
        passages=passages,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Measure Stage-2 reranker throughput at frozen parameters."
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="Run on synthetic passages (no GPU/download required)")
    parser.add_argument("--queries", default="data/queries.tsv",
                        help="Path to queries TSV (real run only)")
    parser.add_argument("--universe", default="data/universe.parquet",
                        help="Path to frozen universe parquet (real run only)")
    parser.add_argument("--n-sample", type=int, default=500,
                        help="Number of pairs to score")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Write JSON report to this path")
    args = parser.parse_args()

    if args.synthetic:
        report = run_synthetic_benchmark(n_sample=args.n_sample, seed=args.seed)
    else:
        report = run_real_benchmark(
            queries_tsv=args.queries,
            universe_parquet=args.universe,
            n_sample=args.n_sample,
            seed=args.seed,
        )

    print(json.dumps(report, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        logger.info("report written to %s", args.output)


if __name__ == "__main__":
    main()
