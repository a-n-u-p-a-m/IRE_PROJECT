"""
run_attribution.py — Run an attribution method on a pipeline's score matrix.

Any module attribution/<method>.py exposing
    compute_attribution(score_matrix, top_k=10) -> DataFrame
can be run here (roadmap 1.7 API).

Usage:
    python -m experiments.run_attribution --method stage_shapley \\
        --score-matrix results/scores/dpr_minilm_dl19.parquet

    # Stage-Shapley sensitivity check: Stage 2 alone = remove_gating ranks
    python -m experiments.run_attribution --method stage_shapley \\
        --score-matrix results/scores/dpr_minilm_dl19.parquet \\
        --stage2-alone results/interventions/dpr_minilm_dl19_remove_gating.parquet

Default output: results/attributions/<score-matrix name>_<method>[_<variant>].parquet
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.schema import load_parquet, save_parquet
from shared.utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run an attribution method")
    parser.add_argument("--method", required=True,
                        help="Module name in attribution/, e.g. stage_shapley")
    parser.add_argument("--score-matrix", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default=None)
    # stage_shapley options
    parser.add_argument("--value", default=None, choices=["rr", "dcg", "topk"],
                        help="stage_shapley: per-document utility (default rr)")
    parser.add_argument("--stage2-alone", default=None,
                        help="stage_shapley: remove_gating intervention file used as "
                             "'Stage 2 alone' (sensitivity check)")
    args = parser.parse_args()

    method = importlib.import_module(f"attribution.{args.method}")
    score_matrix = load_parquet(args.score_matrix, "score_matrix")

    kwargs, variant = {}, []
    if args.value or args.stage2_alone:
        if args.method != "stage_shapley":
            parser.error("--value and --stage2-alone apply to stage_shapley only")
        if args.value:
            kwargs["value"] = args.value
            variant.append(args.value)
        if args.stage2_alone:
            gating = load_parquet(args.stage2_alone, "intervention")
            kwargs["stage2_alone"] = method.stage2_alone_from_intervention(gating)
            variant.append("s2alone")

    attr = method.compute_attribution(score_matrix, top_k=args.top_k, **kwargs)

    suffix = "".join(f"_{v}" for v in variant)
    output = args.output or (
        f"results/attributions/{Path(args.score_matrix).stem}_{args.method}{suffix}.parquet"
    )
    out = save_parquet(attr, output, "attribution")
    s1 = attr["stage_1_attribution"]
    logger.info("Saved %d rows to %s | stage_1_attribution mean %.3f, median %.3f, "
                "%.0f%% Stage-1-dominant (> 0.5)", len(attr), out, s1.mean(), s1.median(),
                100 * (s1 > 0.5).mean())


if __name__ == "__main__":
    main()
