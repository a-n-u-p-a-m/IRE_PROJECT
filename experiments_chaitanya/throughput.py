"""
throughput.py - Stage-2 reranker throughput measurement script.

Measures pairs/sec and ACTUAL padded tensor width at the FROZEN parameters,
reading those parameters from config/base.yaml (stage2.model, stage2.max_length,
stage2.batch_size, stage2.precision) rather than hardcoding them.  This ensures
the report always describes the precision we are actually using.

Key numbers produced:
  pairs_per_second      — divide 872,455 (total unique pairs) to get GPU time
  mean_padded_width     — ACTUAL padded tensor width per batch (not an estimate);
                          if this is much less than max_length, the effective
                          throughput is higher than 512-token padding would imply,
                          and there is no need to lower max_length

Run on a machine with GPU (Colab T4 recommended) before committing to the
full cache fill.

Usage::

    # On Colab or a GPU machine (reads frozen params from config):
    python -m experiments_chaitanya.throughput \\
        --queries data/queries.tsv \\
        --universe data/universe.parquet \\
        --n-sample 500 \\
        --output cache/throughput_report.json

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

# Fallback constants — used ONLY when config cannot be loaded
_FALLBACK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
_FALLBACK_MAX_LENGTH = 512
_FALLBACK_BATCH_SIZE = 64
_FALLBACK_PRECISION = "float32"

# MS MARCO passage average word count (from the literature; used for
# sequence-length estimation in the synthetic mode).
MSMARCO_AVG_WORDS = 55


def _load_frozen_params() -> dict:
    """
    Read stage2 parameters from config/base.yaml.

    Returns a dict with keys: model, max_length, batch_size, precision.
    Falls back to the _FALLBACK_* constants if the config cannot be loaded,
    logging a warning so the caller knows the report may describe a stale config.
    """
    try:
        from shared.config import load_config
        cfg = load_config()
        s2 = cfg.get("stage2", {})
        params = {
            "model":      s2.get("model",     _FALLBACK_MODEL),
            "max_length": int(s2.get("max_length", _FALLBACK_MAX_LENGTH)),
            "batch_size": int(s2.get("batch_size", _FALLBACK_BATCH_SIZE)),
            "precision":  s2.get("precision", _FALLBACK_PRECISION),
        }
        logger.info("throughput params from config: %s", params)
        return params
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "could not load config (%s); falling back to hardcoded constants. "
            "The report may describe a stale precision — update config/base.yaml "
            "before reporting.",
            exc,
        )
        return {
            "model":      _FALLBACK_MODEL,
            "max_length": _FALLBACK_MAX_LENGTH,
            "batch_size": _FALLBACK_BATCH_SIZE,
            "precision":  _FALLBACK_PRECISION,
        }


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
    model_name: str | None = None,
    max_length: int | None = None,
    batch_size: int | None = None,
    precision: str | None = None,
    device: str | None = None,
    n_warmup_batches: int = 2,
) -> dict:
    """
    Score all passages against query_text and return a throughput report dict.

    Parameters not supplied are read from config/base.yaml (stage2.* section).
    The first n_warmup_batches are excluded from timing to avoid penalising
    JIT compilation / CUDA kernel initialisation.

    NEW: logs and reports the ACTUAL padded tensor width per batch by
    instrumenting the tokeniser directly, rather than using the word_count×1.3
    estimate.  This answers whether sentence-transformers pads to
    ``longest-in-batch`` (good, ~71 tokens) or to ``max_length`` (2.8× worse).
    """
    from experiments_chaitanya.reranker import PointwiseCrossEncoder

    # Read params from config if not explicitly overridden
    frozen = _load_frozen_params()
    model_name = model_name or frozen["model"]
    max_length  = max_length  if max_length  is not None else frozen["max_length"]
    batch_size  = batch_size  if batch_size  is not None else frozen["batch_size"]
    precision   = precision   or frozen["precision"]

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

    # Instrument the tokeniser to capture actual padded widths
    padded_widths: list[int] = []
    _original_tokenize = None

    try:
        import torch
        # sentence-transformers exposes the AutoTokenizer on the underlying module
        tokenizer = None
        if hasattr(reranker, "_model") and reranker._model is not None:
            inner = reranker._model
            if hasattr(inner, "tokenizer"):
                tokenizer = inner.tokenizer

        def _patched_tokenize(query_passages, **kwargs):
            result = _original_tokenize(query_passages, **kwargs)
            if hasattr(result, "input_ids"):
                w = result.input_ids.shape[-1]
                padded_widths.append(int(w))
            return result

        if tokenizer is not None:
            _original_tokenize = tokenizer.__call__
            tokenizer.__call__ = _patched_tokenize
    except Exception:
        pass  # instrumentation is best-effort; report falls back to estimate

    # Warmup (loads model weights on first call)
    warmup_n = min(n_warmup_batches * batch_size, n_total)
    logger.info("warming up on %d pairs …", warmup_n)
    reranker.score(query_text, passages[:warmup_n], doc_ids[:warmup_n])
    padded_widths.clear()   # discard warmup measurements

    # Timed run
    logger.info("timing %d pairs …", n_total)
    t0 = time.perf_counter()
    scores = reranker.score_many(query_text, passages, doc_ids)
    elapsed = time.perf_counter() - t0

    # Restore original tokenizer call
    if _original_tokenize is not None and tokenizer is not None:
        tokenizer.__call__ = _original_tokenize

    pairs_per_second = n_total / max(elapsed, 1e-9)

    # Actual padded width (from instrumentation) or word_count×1.3 fallback
    if padded_widths:
        mean_padded = sum(padded_widths) / len(padded_widths)
        max_padded = max(padded_widths)
        width_source = "actual_tokenizer_output"
        width_note = (
            f"Padding mode: longest-in-batch (mean {mean_padded:.0f} "
            f"vs max_length={max_length}); no truncation observed."
            if max_padded < max_length
            else f"WARNING: max padded width {max_padded} == max_length={max_length}; "
                 "some sequences may be truncated."
        )
    else:
        token_lengths = [len(p.split()) * 1.3 for p in passages]
        mean_padded = sum(token_lengths) / len(token_lengths)
        max_padded = max(token_lengths)
        width_source = "estimated_word_count_x1.3"
        width_note = (
            "Sequence lengths are estimates (word_count × 1.3); "
            "actual tokenisation may differ by ±10%."
        )

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
        "mean_padded_tensor_width": round(mean_padded, 1),
        "max_padded_tensor_width": round(max_padded, 1),
        "padded_width_source": width_source,
        "estimated_hours_full_universe": round(estimated_hours, 2),
        "estimated_hours_full_universe_3way_split": round(estimated_hours / 3, 2),
        "note": width_note,
    }
    logger.info("report: %s", json.dumps(report, indent=2))
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

    # Efficient: only load the sampled doc_ids, not all 8.8M passages
    fetcher = IrDatasetsPassageFetcher(required_doc_ids=set(doc_ids))
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
