"""
stage1_over_universe.py - Stage-1 scores for every document in U_q.

    python -m experiments_anupam.stage1_over_universe

A Stage-1 run reaches depth M. The worlds need Stage-1 over the WHOLE universe,
because `select_outsiders` works on the discarded set D_q = U_q \\ C_q:

    boundary      needs s1 over D_q, to take the documents immediately below
                  the cutoff
    alternative   needs the OTHER retriever's rank over D_q, to ask a second
                  opinion about what the primary retriever threw away

Neither is answerable from a top-M run alone, so this module extends each
Stage-1 run to U_q under the declared floor policy and records what that cost.

**The floor is an assertion about depth, not about relevance.** A universe
document the retriever did not return gets a score strictly below everything it
did return - which is exactly what the retriever said: *not in my top-M*. It
never lets an unretrieved document outrank a retrieved one, and all floored
documents tie, so the frozen `doc_id` tie-break orders them. `missing="error"`
exists for paths where a gap is a bug; here a gap is a fact about depth.

The output is a `run` table whose depth is |U_q| rather than M, so it validates
with the same machinery and replays through `PrecomputedRun`. The floored flag
cannot ride along - `schema._coerce` keeps only the declared run columns - so
the floored fraction is a separate report, which is the right shape anyway: it
belongs beside every cross-paradigm result. A retriever that floors most of
U_q is doing far less work than the comparison assumes.

Four checks run before anything is written. The first is the one that matters:

  * **top-K_c agreement.** The top-K_c of this table must equal the top-K_c of
    the source run at every configured K_c. If it does not, C_q as reconstructed
    from U_q is not the candidate set the cascade actually used, and every
    effect computed against it describes a cascade that never ran.
  * the table covers exactly U_q, no more and no less;
  * every floored document scores strictly below every retrieved one;
  * the discarded set can actually supply each outsider policy at B = K_c.

Tables land in `data/` and the report in `analysis/`, both gitignored: all of
it is reproducible from the committed run tables plus the frozen universe, so
only the manifest is tracked. Same rule as the universe parquet.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiments_anupam.retrievers import floor_counts, scores_from_run
from shared.config import load_config
from shared.ordering import order_by
from shared.provenance import run_manifest, write_manifest
from shared.schema import load, save
from shared.universe import load_frozen, universe_sets

logger = logging.getLogger(__name__)


def build_table(
    run: pd.DataFrame,
    system: str,
    universe: dict[str, list[str]],
    checkpoint: str,
    tie_break: str,
) -> pd.DataFrame:
    """
    One `run` table over U_q: ranks 1..|U_q|, floored documents last.

    Ranks are contiguous because `validate_run` requires it, and because a
    gap would make "rank within U_q" ambiguous exactly where the outsider
    policies read it.
    """
    s1 = scores_from_run(run, system, universe, missing="floor")
    rows = []
    for qid in sorted(universe):
        scores = s1[qid]
        ordered = order_by(universe[qid], scores, tie_break)
        rows.extend(
            {
                "query_id": qid,
                "doc_id": doc,
                "system": system,
                "rank": rank,
                "score": float(scores[doc]),
                "checkpoint": checkpoint,
            }
            for rank, doc in enumerate(ordered, start=1)
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────────────

def check_top_k_agreement(
    table: pd.DataFrame, run: pd.DataFrame, system: str, candidate_depths
) -> list[str]:
    """
    C_q reconstructed from U_q must equal C_q from the source run.

    This is the invariant the whole reconstruction rests on. It can fail for a
    reason that is easy to miss: the floor makes unretrieved documents tie, and
    if the tie-break here differed from the one that produced the run, the
    boundary of C_q would shift silently and every gate effect would be
    computed against a candidate set the cascade never used.
    """
    problems = []
    src = run[run["system"] == system]
    by_q_src = {
        str(q): list(g.sort_values("rank")["doc_id"].astype(str))
        for q, g in src.groupby("query_id", observed=True)
    }
    by_q_new = {
        str(q): list(g.sort_values("rank")["doc_id"].astype(str))
        for q, g in table.groupby("query_id", observed=True)
    }
    for k_c in candidate_depths:
        bad = [
            qid for qid, docs in by_q_src.items()
            if by_q_new.get(qid, [])[:k_c] != docs[:k_c]
        ]
        if bad:
            problems.append(
                f"{system} K_c={k_c}: top-{k_c} over U_q disagrees with the "
                f"source run for {len(bad)} queries, e.g. {sorted(bad)[:3]}. "
                "C_q reconstructed from U_q is not the candidate set the "
                "cascade used."
            )
    return problems


def check_covers_universe(
    table: pd.DataFrame, universe: dict[str, list[str]], system: str
) -> list[str]:
    problems = []
    got = {
        str(q): set(g["doc_id"].astype(str))
        for q, g in table.groupby("query_id", observed=True)
    }
    if set(got) != set(universe):
        problems.append(
            f"{system}: covers {len(got)} queries against {len(universe)} in U_q"
        )
        return problems
    for qid, docs in universe.items():
        if got[qid] != set(docs):
            problems.append(
                f"{system} query {qid}: covers {len(got[qid])} documents "
                f"against |U_q|={len(docs)}"
            )
            break
    return problems


def check_floor_ordering(
    table: pd.DataFrame, run: pd.DataFrame, system: str
) -> list[str]:
    """No floored document may outrank a retrieved one."""
    src = run[run["system"] == system]
    retrieved = {
        str(q): set(g["doc_id"].astype(str))
        for q, g in src.groupby("query_id", observed=True)
    }
    for qid, grp in table.groupby("query_id", observed=True):
        ret = retrieved.get(str(qid), set())
        is_ret = grp["doc_id"].astype(str).isin(ret)
        if not is_ret.any() or is_ret.all():
            continue
        if float(grp.loc[~is_ret, "score"].max()) >= float(grp.loc[is_ret, "score"].min()):
            return [
                f"{system} query {qid}: a floored document scores at or above "
                "a retrieved one, so the floor is not below the observed minimum"
            ]
    return []


def check_policy_supply(
    table: pd.DataFrame,
    run: pd.DataFrame,
    system: str,
    other: pd.DataFrame | None,
    other_name: str | None,
    candidate_depths,
) -> tuple[list[str], dict]:
    """
    Can D_q actually supply each outsider policy at B = K_c?

    `boundary` needs genuinely RANKED documents below the cutoff - floored ones
    are tied, so taking the "n immediately below" from a tied block is an
    arbitrary choice dressed as a policy. `alternative` needs documents the
    other retriever ranked. Cheap to check now; expensive to discover after a
    cache fill.
    """
    problems: list[str] = []
    stats: dict = {}

    depth = int(run[run["system"] == system]
                .groupby("query_id", observed=True)["rank"].max().median())
    other_ranked = None
    if other is not None and other_name is not None:
        o = other[other["system"] == other_name]
        other_ranked = {
            str(q): set(g["doc_id"].astype(str))
            for q, g in o.groupby("query_id", observed=True)
        }

    for k_c in candidate_depths:
        budget = k_c                      # b_equals_kc
        boundary_supply, alt_supply = [], []
        for qid, grp in table.groupby("query_id", observed=True):
            g = grp.sort_values("rank")
            discarded = list(g["doc_id"].astype(str))[k_c:]
            # Genuinely ranked within the discarded set: positions k_c+1..depth
            boundary_supply.append(max(depth - k_c, 0))
            if other_ranked is not None:
                alt_supply.append(
                    len(set(discarded) & other_ranked.get(str(qid), set()))
                )

        b_min = int(np.min(boundary_supply)) if boundary_supply else 0
        stats[f"k_c={k_c}"] = {
            "budget_b": budget,
            "boundary_ranked_supply_min": b_min,
            "alternative_supply_min": int(np.min(alt_supply)) if alt_supply else None,
            "alternative_supply_median": (
                float(np.median(alt_supply)) if alt_supply else None
            ),
        }
        if b_min < budget:
            problems.append(
                f"{system} K_c={k_c}: only {b_min} genuinely ranked documents "
                f"below the cutoff against a budget of {budget}. The boundary "
                "policy would have to draw from the floored block, where every "
                "score is tied and the selection is the tie-break rather than "
                "the policy."
            )
        if alt_supply and int(np.min(alt_supply)) < budget:
            problems.append(
                f"{system} K_c={k_c}: the weakest query offers only "
                f"{int(np.min(alt_supply))} documents ranked by {other_name} "
                f"within D_q, against a budget of {budget}. The alternative "
                "policy falls back to the doc_id tie-break for the remainder."
            )
    return problems, stats


# ──────────────────────────────────────────────────────────────────────

def run(
    config_path: str,
    runs_dir: str,
    universe_path: str,
    out_dir: str,
) -> tuple[dict, list[str]]:
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    universe = universe_sets(
        load_frozen(universe_path, fingerprint=cfg.fingerprint())
    )
    logger.info(
        "universe: %d queries, |U_q| %d-%d",
        len(universe),
        min(len(v) for v in universe.values()),
        max(len(v) for v in universe.values()),
    )

    systems = list(cfg.stage1_systems)
    runs = {s: load(Path(runs_dir) / f"{s}.parquet", "run") for s in systems}

    problems: list[str] = []
    extra: dict = {"systems": {}, "floor_policy": "below the observed minimum"}

    for system in systems:
        src = runs[system]
        cps = sorted(set(src[src["system"] == system]["checkpoint"]
                         .dropna().astype(str)))
        if len(cps) != 1:
            raise SystemExit(
                f"{system}: run table carries {len(cps)} checkpoints {cps}. "
                "A table that cannot say which checkpoint produced it must not "
                "be extended over the universe."
            )

        table = build_table(src, system, universe, cps[0], cfg.tie_break)

        problems += check_top_k_agreement(table, src, system, cfg.candidate_depths)
        problems += check_covers_universe(table, universe, system)
        problems += check_floor_ordering(table, src, system)

        others = [s for s in systems if s != system]
        other_name = others[0] if others else None
        supply_problems, supply = check_policy_supply(
            table, src, system,
            runs[other_name] if other_name else None, other_name,
            cfg.candidate_depths,
        )
        problems += supply_problems

        table_path = save(table, out / f"stage1_over_universe_{system}.parquet", "run")

        report = floor_counts(src, system, universe)
        report_path = Path("analysis") / f"stage1_floor_{system}.parquet"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_parquet(report_path, index=False)

        extra["systems"][system] = {
            "checkpoint": cps[0],
            "artifact": str(table_path),
            "floor_report": str(report_path),
            "n_rows": int(len(table)),
            "floored_fraction_median": round(
                float(report["floored_fraction"].median()), 4),
            "floored_fraction_min": round(
                float(report["floored_fraction"].min()), 4),
            "floored_fraction_max": round(
                float(report["floored_fraction"].max()), 4),
            "n_retrieved_median": int(report["n_retrieved"].median()),
            "outsider_policy_supply": supply,
        }
        logger.info(
            "%s: %d rows, floored fraction %.3f (median), %d retrieved per query",
            system, len(table),
            float(report["floored_fraction"].median()),
            int(report["n_retrieved"].median()),
        )

    extra["n_queries"] = len(universe)
    extra["problems"] = problems
    write_manifest(
        out / "stage1_over_universe_manifest.json",
        run_manifest(cfg.fingerprint(), owner="anupam", extra=extra),
    )
    return extra, problems


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Extend each Stage-1 run to the whole universe"
    )
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--universe", default="data/universe.parquet")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any check failed")
    args = p.parse_args()

    extra, problems = run(args.config, args.runs_dir, args.universe, args.out_dir)

    print()
    for system, s in extra["systems"].items():
        print(f"{system}")
        print(f"  floored fraction of U_q   {s['floored_fraction_median']} "
              f"(median), {s['floored_fraction_min']}–{s['floored_fraction_max']}")
        print(f"  retrieved per query       {s['n_retrieved_median']}")
        for k, v in s["outsider_policy_supply"].items():
            print(f"  {k}: B={v['budget_b']}  boundary supply ≥ "
                  f"{v['boundary_ranked_supply_min']}  alternative supply ≥ "
                  f"{v['alternative_supply_min']} (median "
                  f"{v['alternative_supply_median']})")
        print()

    if problems:
        print(f"{len(problems)} problem(s):")
        for pr in problems:
            print(f"  - {pr}")
        return 1 if args.strict else 0
    print("Stage-1 extended to U_q: top-K_c agrees with the source runs at "
          "every depth, coverage exact, floor ordering holds, and every "
          "outsider policy has supply at B = K_c.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
