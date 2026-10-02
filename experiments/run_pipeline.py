"""
run_pipeline.py — End-to-end pipeline runner: retrieve → rerank → log.

This script demonstrates how to wire a retriever with the shared reranker
and score logger to produce a schema-compliant score matrix.

Each team member writes their own retriever (bm25.py, dpr.py, etc.),
but they all call this common pipeline code to produce the final output.

Usage:
    python -m experiments.run_pipeline --retriever bm25 --query-set dl19
    python -m experiments.run_pipeline --retriever dpr  --query-set dl20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.data import load_corpus, load_queries, load_qrels, attach_relevance_labels
from shared.evaluation import compute_ndcg_from_score_matrix, bootstrap_ci
from shared.logger import ScoreLogger
from shared.reranker import Reranker, RerankerConfig
from shared.schema import save_parquet
from shared.utils import load_config, set_global_seed, setup_logging, Timer, TOP_K_DEFAULT

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Retriever registry  (each member registers theirs here)
# ──────────────────────────────────────────────────────────────────────

def _get_retriever(name: str, config: dict):
    """
    Factory: return a retriever object given its name.

    Each retriever must implement:
        .search(query: str, k: int) -> list[tuple[str, float]]
            Returns list of (doc_id, score), sorted by score descending.
    """
    if name == "bm25":
        from retrievers.bm25 import BM25Retriever
        return BM25Retriever(config)
    elif name == "dpr":
        from retrievers.dpr import DPRRetriever
        return DPRRetriever(config)
    elif name == "colbert":
        from retrievers.colbert import ColBERTRetriever
        return ColBERTRetriever(config)
    elif name == "splade":
        from retrievers.splade import SPLADERetriever
        return SPLADERetriever(config)
    else:
        raise ValueError(f"Unknown retriever: {name}")


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(
    retriever_name: str,
    query_set: str,
    config_path: str = "config/base.yaml",
    retriever_config_path: str | None = None,
    top_k: int = TOP_K_DEFAULT,
    skip_reranker: bool = False,
    output_path: str | None = None,
    max_queries: int | None = None,
) -> Path:
    """
    Run the full Stage 1 → Stage 2 pipeline.

    Parameters
    ----------
    retriever_name : str
        One of: bm25, dpr, colbert, splade
    query_set : str
        One of: dl19, dl20, dev_small
    config_path : str
        Path to base.yaml
    retriever_config_path : str, optional
        Path to retriever-specific config (e.g., config/bm25.yaml)
    top_k : int
        Number of candidates from Stage 1 (default 1000)
    skip_reranker : bool
        If True, produce Stage-1-only results (no reranking baseline)
    output_path : str, optional
        Override output Parquet path
    max_queries : int, optional
        Process only N queries (for debugging)

    Returns
    -------
    Path to the output Parquet file.
    """
    t = Timer()

    # ── Load config ──
    base_config = load_config(config_path)
    retriever_config = {}
    if retriever_config_path:
        retriever_config = load_config(retriever_config_path)

    set_global_seed(base_config.get("seed", 42))

    # ── Load data ──
    t.start("load_corpus")
    corpus = load_corpus()
    t.stop("load_corpus")

    t.start("load_queries")
    queries = load_queries(query_set)
    qrels = load_qrels(query_set)
    t.stop("load_queries")

    if max_queries:
        query_ids = list(queries.keys())[:max_queries]
        queries = {qid: queries[qid] for qid in query_ids}

    logger.info(
        "Pipeline: %s retriever, %s queries, top_k=%d, reranker=%s",
        retriever_name,
        query_set,
        top_k,
        "OFF" if skip_reranker else "MiniLM",
    )

    # ── Initialize retriever ──
    t.start("init_retriever")
    retriever = _get_retriever(retriever_name, retriever_config)
    t.stop("init_retriever")

    # ── Initialize reranker ──
    reranker = None
    if not skip_reranker:
        reranker_cfg = base_config.get("reranker", {})
        reranker = Reranker(RerankerConfig(**reranker_cfg))

    # ── Run pipeline ──
    score_log = ScoreLogger()

    for i, (qid, query_text) in enumerate(queries.items()):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info("Processing query %d/%d: %s", i + 1, len(queries), qid)

        # Stage 1: Retrieve
        t.start("retrieval")
        stage1_results = retriever.search(query_text, k=top_k)
        t.stop("retrieval")

        # Build candidate list
        candidates = []
        for rank, (doc_id, score) in enumerate(stage1_results, start=1):
            candidates.append({
                "doc_id": doc_id,
                "stage_1_rank": rank,
                "stage_1_score": score,
            })

        if skip_reranker:
            # No reranking — Stage 1 scores are final
            for cand in candidates:
                score_log.add(
                    query_id=qid,
                    doc_id=cand["doc_id"],
                    stage_1_rank=cand["stage_1_rank"],
                    stage_1_score=cand["stage_1_score"],
                    in_candidate_set=True,
                    stage_2_rank=cand["stage_1_rank"],
                    stage_2_score=cand["stage_1_score"],
                    final_rank=cand["stage_1_rank"],
                )
        else:
            # Stage 2: Rerank
            t.start("reranking")
            rerank_pairs = [
                (cand["doc_id"], corpus.get(cand["doc_id"], ""))
                for cand in candidates
            ]
            reranked = reranker.rerank(query_text, rerank_pairs)
            t.stop("reranking")

            # Build doc_id → reranker result lookup
            rerank_lookup = {doc.doc_id: doc for doc in reranked}

            for cand in candidates:
                did = cand["doc_id"]
                rdoc = rerank_lookup.get(did)

                score_log.add(
                    query_id=qid,
                    doc_id=did,
                    stage_1_rank=cand["stage_1_rank"],
                    stage_1_score=cand["stage_1_score"],
                    in_candidate_set=True,
                    stage_2_rank=rdoc.rank if rdoc else None,
                    stage_2_score=rdoc.score if rdoc else None,
                    final_rank=rdoc.rank if rdoc else cand["stage_1_rank"],
                )

    # ── Attach relevance labels ──
    df = score_log.to_dataframe(strict=False)
    df = attach_relevance_labels(df, qrels)

    # ── Save ──
    if output_path is None:
        reranker_tag = "none" if skip_reranker else "minilm"
        output_path = f"results/scores/{retriever_name}_{reranker_tag}.parquet"

    out = save_parquet(df, output_path, "score_matrix")
    logger.info("Saved score matrix to %s (%d rows)", out, len(df))

    # ── Verify baselines ──
    ndcg_scores = compute_ndcg_from_score_matrix(df, k=10)
    ndcg_ci = bootstrap_ci(ndcg_scores)
    logger.info(
        "nDCG@10 = %.4f [%.4f, %.4f]  (n=%d queries)",
        ndcg_ci["mean"],
        ndcg_ci["ci_low"],
        ndcg_ci["ci_high"],
        ndcg_ci["n_queries"],
    )

    # Check against expected range if available
    if retriever_config:
        expected = retriever_config.get("expected_ndcg10_range")
        if expected:
            lo, hi = expected
            actual = ndcg_ci["mean"]
            if actual < lo - 0.02 or actual > hi + 0.02:
                logger.warning(
                    "⚠  nDCG@10 = %.4f is outside expected range [%.2f, %.2f]! "
                    "Debug the retriever before running attribution experiments.",
                    actual, lo, hi,
                )
            else:
                logger.info("✓  nDCG@10 within expected range.")

    t.report()
    return out


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run Stage 1 → Stage 2 retrieval pipeline"
    )
    parser.add_argument(
        "--retriever", required=True,
        choices=["bm25", "dpr", "colbert", "splade"],
        help="Which retriever to use for Stage 1",
    )
    parser.add_argument(
        "--query-set", default="dl19",
        choices=["dl19", "dl20", "dev_small"],
        help="Which query set to evaluate on",
    )
    parser.add_argument(
        "--config", default="config/base.yaml",
        help="Path to base config",
    )
    parser.add_argument(
        "--retriever-config", default=None,
        help="Path to retriever-specific config (e.g., config/bm25.yaml)",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K_DEFAULT,
        help="Stage 1 candidate set size (default: 1000)",
    )
    parser.add_argument(
        "--no-reranker", action="store_true",
        help="Skip Stage 2 reranking (produce Stage 1 only baseline)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override output Parquet path",
    )
    parser.add_argument(
        "--max-queries", type=int, default=None,
        help="Process only N queries (for debugging)",
    )

    args = parser.parse_args()

    run_pipeline(
        retriever_name=args.retriever,
        query_set=args.query_set,
        config_path=args.config,
        retriever_config_path=args.retriever_config,
        top_k=args.top_k,
        skip_reranker=args.no_reranker,
        output_path=args.output,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()
