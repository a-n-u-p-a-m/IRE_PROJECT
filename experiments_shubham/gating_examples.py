"""
gating_examples.py - The qualitative figure: real gating failures, with text.

A gating failure is a passage Stage 1 discarded that the no-gate world would
have ranked usefully (`shared.effects.gating_failures`). The selection rule is
fixed here, before the full-run effects exist, so the examples cannot be
picked for how well they tell the story:

    system and setting    the given system at the headline K_c, uncondensed
    order                 largest loss first (most negative E_gate), then the
                          better no-gate rank, then query_id and doc_id
    one per query         the worst failure of each query, so three examples
                          are three different queries
    n                     3

Each example carries the query, the excluded passage with its grade, its
Stage-1 rank and its rank without the gate, and for contrast the passage the
actual cascade ranked first.

Passage text comes in separately (the repository has none), as a TSV of
doc_id<TAB>text for exactly the ids `select` writes:

    python -m experiments_shubham.gating_examples select --effects-dir effects/real --system bm25
    # fetch the passages for analysis/gating_examples/doc_ids.txt, then:
    python -m experiments_shubham.gating_examples render --passages passages.tsv
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from experiments_shubham.analysis import load_stage
from experiments_shubham.power_h7 import headline_k_c
from shared.config import load_config
from shared.effects import gating_failures
from shared.schema import E_RERANK, load

logger = logging.getLogger(__name__)

N_EXAMPLES = 3


def select(stage: pd.DataFrame, n: int = N_EXAMPLES) -> pd.DataFrame:
    """The frozen rule: worst failure per query, largest losses first."""
    fails = gating_failures(stage)
    fails = fails[~fails["condensed"].fillna(False).astype(bool)]
    fails = fails.sort_values(["effect", "rank_control", "query_id", "doc_id"])
    worst = fails.drop_duplicates("query_id").head(n)
    top = stage[(stage["effect_type"] == E_RERANK) & (stage["rank_treatment"] == 1)]
    top = top.set_index("query_id")["doc_id"]
    return pd.DataFrame({
        "query_id": worst["query_id"].astype(str).to_numpy(),
        "doc_id": worst["doc_id"].astype(str).to_numpy(),
        "rank_without_gate": worst["rank_control"].astype(int).to_numpy(),
        "e_gate": worst["effect"].to_numpy(float),
        "actual_top_doc_id": [str(top.get(q, "")) for q in worst["query_id"]],
    })


def annotate(examples: pd.DataFrame, system: str, k_c: int,
             queries: Mapping[str, str], qrels: pd.DataFrame,
             stage1: pd.DataFrame, depth_m: int) -> pd.DataFrame:
    """
    Query text, grades and the Stage-1 rank that put each passage below K_c.

    A Stage-1 rank beyond `depth_m` is the floor policy's rank: the retriever
    did not return the passage at all, which is a different failure from
    ranking it just below the cutoff, so it is flagged.
    """
    grade = {(str(q), str(d)): int(g) for q, d, g, j in
             zip(qrels["query_id"], qrels["doc_id"], qrels["relevance"], qrels["judged"])
             if bool(j)}
    s1 = stage1[stage1["system"] == system]
    s1_rank = {(str(q), str(d)): int(r)
               for q, d, r in zip(s1["query_id"], s1["doc_id"], s1["rank"])}
    out = examples.copy()
    out.insert(0, "system", system)
    out.insert(1, "k_c", k_c)
    out["query"] = [queries[q] for q in out["query_id"]]
    out["grade"] = [grade.get((q, d)) for q, d in zip(out["query_id"], out["doc_id"])]
    out["stage1_rank"] = [s1_rank[(q, d)] for q, d in zip(out["query_id"], out["doc_id"])]
    out["stage1_retrieved"] = out["stage1_rank"] <= depth_m
    out["depth_m"] = depth_m
    out["actual_top_grade"] = [grade.get((q, d)) for q, d
                               in zip(out["query_id"], out["actual_top_doc_id"])]
    return out


def read_passages(path: str | Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {row[0]: row[1] for row in csv.reader(fh, delimiter="\t") if len(row) >= 2}


def _grade(g) -> str:
    return "unjudged" if g is None or pd.isna(g) else f"grade {int(g)}"


def _stage1(ex) -> str:
    if ex.stage1_retrieved:
        return f"Stage-1 rank {ex.stage1_rank:,}"
    return f"not in {ex.system.upper()}'s top {ex.depth_m:,}"


def render_markdown(examples: pd.DataFrame, passages: Mapping[str, str]) -> str:
    missing = sorted({*examples["doc_id"], *examples["actual_top_doc_id"]} - set(passages))
    if missing:
        raise KeyError(f"no passage text for {missing}")
    parts = []
    for i, ex in enumerate(examples.itertuples(), start=1):
        parts.append(
            f"**Example {i}.** Query {ex.query_id}: *{ex.query}*\n\n"
            f"- **Excluded by {ex.system.upper()}** ({_stage1(ex)}, "
            f"K_c = {ex.k_c}; rank {ex.rank_without_gate} without the gate; "
            f"{_grade(ex.grade)}; E_gate = {ex.e_gate:.2f}), "
            f"passage {ex.doc_id}: {passages[ex.doc_id]}\n"
            f"- **Ranked first by the cascade** ({_grade(ex.actual_top_grade)}), "
            f"passage {ex.actual_top_doc_id}: {passages[ex.actual_top_doc_id]}\n")
    return "\n".join(parts)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Real gating failures with passage text")
    sub = parser.add_subparsers(dest="command", required=True)
    sel = sub.add_parser("select", help="apply the frozen rule and list the doc ids to fetch")
    sel.add_argument("--effects-dir", default="effects/real")
    sel.add_argument("--system", default="bm25")
    sel.add_argument("--k-c", type=int, default=None)
    sel.add_argument("--n", type=int, default=N_EXAMPLES)
    ren = sub.add_parser("render", help="write the examples with their passage text")
    ren.add_argument("--passages", required=True, help="TSV: doc_id<TAB>text")
    for p in (sel, ren):
        p.add_argument("--config", default="config/base.yaml")
        p.add_argument("--out-dir", default="analysis/gating_examples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out_dir)
    if args.command == "select":
        k_c = args.k_c or headline_k_c(cfg)[0]
        stage = load_stage(args.effects_dir, args.system, k_c)
        with open(cfg.queries_file, encoding="utf-8") as fh:
            queries = dict(line.rstrip("\n").split("\t", 1) for line in fh if "\t" in line)
        stage1 = load(Path(cfg.paths.get("data", "data"))
                      / f"stage1_over_universe_{args.system}.parquet", "run")
        examples = annotate(select(stage, args.n), args.system, k_c, queries,
                            load(cfg.qrels_file, "qrels"), stage1, cfg.universe_depth_m)
        out.mkdir(parents=True, exist_ok=True)
        examples.to_csv(out / "examples.csv", index=False)
        ids = sorted({*examples["doc_id"], *examples["actual_top_doc_id"]})
        (out / "doc_ids.txt").write_text("\n".join(ids) + "\n")
        logger.info("%d examples, %d passages to fetch -> %s", len(examples), len(ids),
                    out / "doc_ids.txt")
    else:
        examples = pd.read_csv(out / "examples.csv", dtype={"query_id": str, "doc_id": str,
                                                             "actual_top_doc_id": str})
        text = render_markdown(examples, read_passages(args.passages))
        (out / "examples.md").write_text(text, encoding="utf-8")
        logger.info("written %s", out / "examples.md")


if __name__ == "__main__":
    main()
