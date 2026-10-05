"""
run_intervention.py — Apply an intervention to a pipeline's score matrix.

Reads a full-pipeline score matrix (from run_pipeline.py), applies one
intervention from interventions/, and writes the intervention results.

Usage:
    # No model or GPU needed
    python -m experiments.run_intervention --intervention remove_stage2 \\
        --score-matrix results/scores/dpr_minilm_dl19.parquet

    # Needs the retriever (top-5000) and the reranker
    python -m experiments.run_intervention --intervention remove_gating \\
        --score-matrix results/scores/dpr_minilm_dl19.parquet \\
        --retriever dpr --retriever-config config/dpr.yaml --query-set dl19

Default output: results/interventions/<score-matrix name>_<intervention>.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.schema import load_parquet, save_parquet
from shared.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

INTERVENTIONS = ("remove_stage2", "remove_gating")


def summarize(df, top_k: int = 10) -> None:
    """Log how documents of the final top-k moved under the intervention."""
    top = df[df["rank_before"] <= top_k]
    if top.empty:
        return
    moved = top["displacement"].abs()
    logger.info(
        "Final top-%d documents (n=%d): mean |displacement| %.1f, median %.0f, "
        "%.1f%% still in top-%d after the intervention",
        top_k, len(top), moved.mean(), moved.median(),
        100 * (top["rank_after"] <= top_k).mean(), top_k,
    )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Apply an intervention to a score matrix")
    parser.add_argument("--intervention", required=True, choices=INTERVENTIONS)
    parser.add_argument("--score-matrix", required=True,
                        help="Full-pipeline score matrix (Parquet)")
    parser.add_argument("--output", default=None)
    # remove_gating only
    parser.add_argument("--retriever", choices=["bm25", "dpr", "colbert", "splade"])
    parser.add_argument("--retriever-config", default=None)
    parser.add_argument("--query-set", choices=["dl19", "dl20", "dev_small"])
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--expanded-k", type=int, default=5000,
                        help="remove_gating: Stage 1 depth fed to the reranker")
    args = parser.parse_args()

    score_matrix = load_parquet(args.score_matrix, "score_matrix")

    if args.intervention == "remove_stage2":
        from interventions import remove_stage2
        result = remove_stage2.run(score_matrix)
    else:
        if not (args.retriever and args.query_set):
            parser.error("remove_gating needs --retriever and --query-set")
        from experiments.run_pipeline import _get_retriever
        from interventions import remove_gating
        from shared.data import load_passages, load_queries
        from shared.reranker import Reranker, RerankerConfig

        retriever_config = load_config(args.retriever_config) if args.retriever_config else {}
        base_config = load_config(args.config)
        result = remove_gating.run(
            score_matrix,
            retriever=_get_retriever(args.retriever, retriever_config),
            reranker=Reranker(RerankerConfig(**base_config.get("reranker", {}))),
            queries=load_queries(args.query_set),
            load_passages=load_passages,
            expanded_k=args.expanded_k,
        )

    output = args.output or (
        f"results/interventions/{Path(args.score_matrix).stem}_{args.intervention}.parquet"
    )
    out = save_parquet(result, output, "intervention")
    logger.info("Saved %d %s rows to %s", len(result), args.intervention, out)
    summarize(result)


if __name__ == "__main__":
    main()
