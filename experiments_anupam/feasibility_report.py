"""
feasibility_report.py - The Week-2 gate, from the frozen universe alone.

    python -m experiments_anupam.feasibility_report

No GPU, no cross-encoder, no Stage-2 scores: one pass over the committed run
tables, the frozen universe and the qrels. It answers the three questions that
decide whether the experiment as specified measures what the proposal claims,
and it answers them before anyone spends GPU hours.

**1. Is there leverage?** Per-query Jaccard between the two Stage-1 systems'
candidate sets at each K_c. Two systems that select nearly the same documents
leave the cross-paradigm comparison with nothing to separate, however different
their scores are. Depth-5000 overlap says little about this; the overlap at K_c
is the number that matters.

**2. Is world C comparable to world A?** The judged density of C_q against the
judged density of U_q. TREC pools are built from participating systems' top-k,
so the head of a ranking is far better judged than the tail. World A ranks
within C_q - dense - while world C sorts all of U_q - sparse - and unjudged
documents earn zero gain. If the densities differ sharply, E_gate is partly an
artefact of where the judgments are, not of what gating did. That is a validity
threat, and this report quantifies it rather than assuming either way.

**3. Is there a gate-failure supply at all?** |J_q \\ C_q| restricted to
documents that actually carry gain. Gate failures are the phenomenon the whole
project is about, and at large K_c they can vanish: a pool built to depth ~100
puts almost every judged document inside a top-1000 candidate set. If the
supply is thin at the largest K_c, the headline gate analysis belongs at a
shallower one - and that rule is declared here, before magnitudes are seen.

A note on "relevant". A document counts here only if `gain(grade) > 0` under
the frozen map, which on TREC-DL excludes grade 1 ("Related"). An earlier
version of this report carried its own gain-map-aware counts beside
`shared.qrels`'s looser `relevance > 0` ones, because the two differed by 4x to
9x and the headline-K_c decision was about to be made on the looser number.
`shared.qrels` now applies the gain map itself, so the duplication is gone; the
measurement that motivated the change is recorded in docs/design_log.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from shared.config import load_config
from shared.provenance import run_manifest, write_manifest
from shared.qrels import coverage_report, judged_non_survivor_counts
from shared.schema import load
from shared.universe import candidate_overlap, load_frozen
from shared.utility import gain

logger = logging.getLogger(__name__)

# Declared before any magnitude is inspected (see point 3 above).
MIN_GATE_FAILURES = 1.0      # median relevant non-survivors required at a K_c
LEVERAGE_CEILING = 0.8       # median candidate Jaccard above this = no leverage
DENSITY_GAP_LIMIT = 3.0      # C_q / U_q judged-density ratio worth a warning


def select_headline_k_c(summary_by_k_c: dict) -> tuple[int | None, str]:
    """
    Pick the headline depth from gate-failure supply, and say why.

    THE RULE, AND WHY IT IS NOT THE FIRST ONE DECLARED. The original rule was
    "the largest K_c whose median gain-carrying non-survivor count is >= 1".
    That conflated two different jobs. `MIN_GATE_FAILURES >= 1` is a
    NON-DEGENERACY floor - below it a depth has no gate failure to detect at
    all - and it says nothing about which of several usable depths to prefer.
    Used as a selection rule it chose the deepest, which is exactly backwards:
    a TREC pool is built from the top ~100 of the systems of the day, so a
    deeper K_c swallows more of the judged pool into C_q and leaves LESS
    supply outside it. On this collection the old rule selected K_c=1000, where
    the median supply is 1-2 documents and 36-39 of 97 queries have none.

    Separated, the rule is: among depths that clear the non-degeneracy floor
    for EVERY Stage-1 system, take the one with the greatest supply, breaking
    ties towards the deeper depth for external validity. The configured
    `candidate_depths` bound the shallow end, so the config's own realism
    judgement still applies - this cannot wander off to an absurd K_c=10.

    This is a post-hoc change to a pre-declared rule and must be read as one.
    It is defensible only because gate-failure supply is measurable BEFORE any
    effect is computed, so the choice cannot be driven by effect sizes. No
    threshold was tuned to produce a particular answer: the floor is unchanged
    at 1, and only the direction of the preference changed.
    """
    usable = {
        int(k): v for k, v in summary_by_k_c.items()
        if v["usable_for_headline_gate"]
    }
    rule = (
        "among depths clearing the non-degeneracy floor (median gain-carrying "
        f"non-survivors >= {MIN_GATE_FAILURES} for every Stage-1 system), the "
        "depth with the greatest supply; ties to the deeper depth. Supply is "
        "measurable before any effect is computed."
    )
    if not usable:
        return None, rule

    def supply(k: int) -> float:
        """The weakest system's supply at this depth: the comparison needs both."""
        per = usable[k]["by_system"].values()
        return min(v["median_gainful_non_survivors"] for v in per)

    best = max(usable, key=lambda k: (supply(k), k))
    return best, rule


def candidate_sets(run: pd.DataFrame, system: str, k_c: int) -> dict[str, set[str]]:
    """C_q = TopKc(U_q, s1) for one system, as {query_id: {doc_id}}."""
    r = run[(run["system"] == system) & (run["rank"] <= k_c)]
    if r.empty:
        raise SystemExit(f"no rows for system {system!r} at K_c={k_c}")
    return {
        str(qid): set(grp["doc_id"].astype(str))
        for qid, grp in r.groupby("query_id", observed=True)
    }


def relevant_doc_sets(qrels: pd.DataFrame, gain_map) -> dict[str, set[str]]:
    """
    {query_id: documents whose grade carries gain}, per the frozen gain map.

    This is the project's own definition of relevant, which is why it is
    derived from `gain` rather than from a threshold written down here.
    """
    j = qrels[qrels["judged"].astype(bool)]
    out: dict[str, set[str]] = {}
    for qid, grp in j.groupby("query_id", observed=True):
        out[str(qid)] = {
            str(d) for d, g in zip(grp["doc_id"], grp["relevance"])
            if gain(int(g), gain_map) > 0.0
        }
    return out


def judged_doc_sets(qrels: pd.DataFrame) -> dict[str, set[str]]:
    j = qrels[qrels["judged"].astype(bool)]
    return {
        str(qid): set(grp["doc_id"].astype(str))
        for qid, grp in j.groupby("query_id", observed=True)
    }


# ──────────────────────────────────────────────────────────────────────
# 1. Leverage
# ──────────────────────────────────────────────────────────────────────

def leverage_table(runs: dict[str, pd.DataFrame], systems: list[str],
                   candidate_depths) -> pd.DataFrame:
    if len(systems) < 2:
        logger.warning("only %d Stage-1 system; no overlap to report", len(systems))
        return pd.DataFrame()
    a, b = systems[0], systems[1]
    out = []
    for k_c in candidate_depths:
        out.append(candidate_overlap(
            candidate_sets(runs[a], a, k_c), candidate_sets(runs[b], b, k_c),
            a, b, k_c,
        ))
    return pd.concat(out, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# 2 and 3. Density and gate-failure supply
# ──────────────────────────────────────────────────────────────────────

def density_table(
    runs: dict[str, pd.DataFrame],
    systems: list[str],
    candidate_depths,
    coverage: pd.DataFrame,
    judged: dict[str, set[str]],
    relevant: dict[str, set[str]],
) -> pd.DataFrame:
    """One row per (query, system, K_c): density inside C_q against U_q."""
    uq = coverage.set_index("query_id")
    rows = []
    for system in systems:
        for k_c in candidate_depths:
            cands = candidate_sets(runs[system], system, k_c)
            for qid, cq in cands.items():
                if qid not in uq.index:
                    continue
                u = uq.loc[qid]
                j_cq = cq & judged.get(qid, set())
                r_cq = cq & relevant.get(qid, set())
                n_j_uq = int(u["n_judged"])
                rows.append({
                    "query_id": qid,
                    "stage1_system": system,
                    "k_c": k_c,
                    "n_candidates": len(cq),
                    "n_judged_cq": len(j_cq),
                    "n_relevant_cq": len(r_cq),
                    "density_cq": len(j_cq) / len(cq) if cq else float("nan"),
                    "universe_size": int(u["universe_size"]),
                    "n_judged_uq": n_j_uq,
                    "n_relevant_uq": len(relevant.get(qid, set())),
                    "density_uq": float(u["coverage"]),
                    # How much of the query's judged supply the gate kept
                    "judged_retained": (len(j_cq) / n_j_uq) if n_j_uq else float("nan"),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["density_ratio"] = df["density_cq"] / df["density_uq"].replace(0, np.nan)
    return df


def non_survivor_table(
    qrels: pd.DataFrame,
    runs: dict[str, pd.DataFrame],
    systems: list[str],
    candidate_depths,
    gain_map,
) -> pd.DataFrame:
    """
    Gain-carrying non-survivors per (query, system, K_c).

    `shared.judged_non_survivor_counts` now applies the gain map itself, so
    there is nothing to recompute here. It used to count relevance > 0, which
    on this collection overstated the gate-failure supply by 4x to 9x; the
    report that measured that gap is in docs/design_log.md for 2026-10-28.
    """
    out = []
    for system in systems:
        for k_c in candidate_depths:
            cands = candidate_sets(runs[system], system, k_c)
            out.append(
                judged_non_survivor_counts(qrels, cands, system, k_c, gain_map)
            )
    return pd.concat(out, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────────────

def verdict(leverage: pd.DataFrame, density: pd.DataFrame,
            nonsurv: pd.DataFrame, candidate_depths) -> tuple[dict, list[str]]:
    """
    Everything except the candidate Jaccard is reported PER SYSTEM.

    Pooling across Stage-1 systems would hide the case this report exists to
    catch: BM25-like runs contributed to the TREC-DL pools and a modern dense
    retriever did not, so the two systems' candidate sets are not judged to the
    same depth. A median over both would average a dense head with a sparse one
    and report a comfortable middle that describes neither. A depth is usable
    for the headline gate only if EVERY Stage-1 system has supply at it, since
    the cross-paradigm comparison needs both.
    """
    summary: dict = {"by_k_c": {}}
    found: dict[str, dict] = {}
    systems = sorted(set(nonsurv["stage1_system"])) if not nonsurv.empty else []

    def note(kind: str, where: str, explain: str) -> None:
        """
        One entry per KIND of finding, listing everywhere it fires.

        The first version of this emitted one message per (K_c, system), which
        turned two distinct problems into twelve lines of near-identical prose.
        A reader needs to know what is wrong once and where it applies.
        """
        found.setdefault(kind, {"explain": explain, "where": []})["where"].append(where)

    for k_c in candidate_depths:
        lev = leverage[leverage["k_c"] == k_c]["jaccard"] if not leverage.empty \
            else pd.Series(dtype=float)
        med_jac = float(lev.median()) if not lev.empty else float("nan")

        per_system: dict = {}
        for system in systems:
            den = density[(density["k_c"] == k_c)
                          & (density["stage1_system"] == system)]
            ns = nonsurv[(nonsurv["k_c"] == k_c)
                         & (nonsurv["stage1_system"] == system)]
            if den.empty or ns.empty:
                continue

            ratio = float(den["density_ratio"].median())
            gainful = float(ns["n_gainful_non_survivors"].median())
            per_system[system] = {
                "median_density_cq": round(float(den["density_cq"].median()), 5),
                "median_density_uq": round(float(den["density_uq"].median()), 5),
                "median_density_ratio": round(ratio, 2),
                "median_judged_retained": round(float(den["judged_retained"].median()), 4),
                "median_gainful_non_survivors": gainful,
                "queries_with_no_gainful_non_survivor": int(
                    (ns["n_gainful_non_survivors"] == 0).sum()),
                "n_queries": int(len(ns)),
                "usable_for_headline_gate": bool(
                    np.isfinite(gainful) and gainful >= MIN_GATE_FAILURES),
            }

            if np.isfinite(ratio) and ratio > DENSITY_GAP_LIMIT:
                note(
                    "judgment_placement",
                    f"K_c={k_c}, {system} ({ratio:.1f}x)",
                    "C_q is far more densely judged than U_q. World C promotes "
                    "from the sparse pool while world A ranks within the dense "
                    "one, and unjudged documents earn zero gain, so E_gate is "
                    "biased upwards by where the judgments are. Note this is a "
                    "consequence of a DELIBERATE rule (qrels.py: unjudged "
                    "documents occupy ranks as competitors), not a bug - but at "
                    "this density the rule drives E_gate towards DCG(pi_act). "
                    "Decide before Stage-2 scoring: keep the rule and report "
                    "the judged fraction of world C's top-k beside every "
                    "result, or add a condensed-list secondary analysis "
                    "(utility.condense) and show the conclusion survives both.",
                )
            if np.isfinite(gainful) and gainful < MIN_GATE_FAILURES:
                note(
                    "gate_failure_supply",
                    f"K_c={k_c}, {system} (median {gainful:.1f})",
                    f"Median gain-carrying non-survivors is below the declared "
                    f"minimum of {MIN_GATE_FAILURES}. Over half the queries have "
                    "no excluded document that could have contributed utility, "
                    "so there is no gate failure to detect at this depth.",
                )

        # The asymmetry between systems, which no per-system number shows.
        densities = {s: v["median_density_cq"] for s, v in per_system.items()}
        if len(densities) >= 2:
            lo_s, lo = min(densities.items(), key=lambda kv: kv[1])
            hi_s, hi = max(densities.items(), key=lambda kv: kv[1])
            asym = (hi / lo) if lo > 0 else float("inf")
            if asym > DENSITY_GAP_LIMIT:
                note(
                    "between_system_asymmetry",
                    f"K_c={k_c} ({hi_s} {hi} vs {lo_s} {lo})",
                    "One system's candidate set is judged far more densely than "
                    "the other's. TREC-DL pools were built from the systems of "
                    "the day, so a lexical run's head can be judged where a "
                    "modern dense run's is not. Comparing E_gate across the two "
                    "then compares judgment coverage as much as retrieval "
                    "paradigm, which confounds the cross-paradigm claim.",
                )

        summary["by_k_c"][int(k_c)] = {
            "median_candidate_jaccard": round(med_jac, 4),
            "by_system": per_system,
            "usable_for_headline_gate": bool(
                per_system
                and all(v["usable_for_headline_gate"] for v in per_system.values())),
        }

        if np.isfinite(med_jac) and med_jac > LEVERAGE_CEILING:
            note(
                "no_leverage",
                f"K_c={k_c} (Jaccard {med_jac:.3f})",
                "The two Stage-1 systems select nearly the same documents, so "
                "the cross-paradigm comparison has little to separate at this "
                "depth however different their scores are.",
            )

    summary["headline_k_c"], summary["headline_k_c_rule"] = select_headline_k_c(
        summary["by_k_c"]
    )
    if summary["headline_k_c"] is None:
        note(
            "no_usable_depth", "every K_c",
            "No configured depth clears the non-degeneracy floor. The gate "
            "analysis cannot be run as specified; either add a shallower K_c or "
            "restate the gate claim as descriptive.",
        )
    return summary, found


def run(config_path: str, runs_dir: str, universe_path: str, qrels_path: str,
        out_dir: str) -> tuple[dict, list[str]]:
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    universe = load_frozen(universe_path, fingerprint=cfg.fingerprint())
    qrels = load(qrels_path, "qrels")
    systems = list(cfg.stage1_systems)
    runs = {s: load(Path(runs_dir) / f"{s}.parquet", "run") for s in systems}

    judged = judged_doc_sets(qrels)
    relevant = relevant_doc_sets(qrels, cfg.gain_map)
    coverage = coverage_report(universe, qrels, cfg.gain_map)
    logger.info(
        "universe: %d queries, median |U_q| %d, median coverage %.4f",
        len(coverage), int(coverage["universe_size"].median()),
        float(coverage["coverage"].median()),
    )

    lev = leverage_table(runs, systems, cfg.candidate_depths)
    den = density_table(runs, systems, cfg.candidate_depths, coverage, judged, relevant)
    ns = non_survivor_table(qrels, runs, systems, cfg.candidate_depths, cfg.gain_map)

    for name, df in (("feasibility_overlap", lev), ("feasibility_density", den),
                     ("feasibility_non_survivors", ns)):
        if not df.empty:
            df.to_parquet(out / f"{name}.parquet", index=False)

    summary, found = verdict(lev, den, ns, cfg.candidate_depths)
    summary["n_queries"] = int(len(coverage))
    summary["universe_size_median"] = int(coverage["universe_size"].median())
    summary["total_universe_pairs"] = int(coverage["universe_size"].sum())
    summary["findings"] = {
        kind: {"where": v["where"], "explain": v["explain"]}
        for kind, v in found.items()
    }

    write_manifest(
        out / "feasibility_manifest.json",
        run_manifest(cfg.fingerprint(), owner="anupam", extra=summary),
    )
    # The universe is the whole Stage-2 bill: every world re-ranks a subset of
    # it and the cache keys on (query, doc), so this is a ceiling, not a stage.
    logger.info(
        "Stage-2 ceiling for the entire project: %s unique pairs",
        f"{summary['total_universe_pairs']:,}",
    )
    return summary, found


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Week-2 feasibility gate")
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--universe", default="data/universe.parquet")
    p.add_argument("--qrels", default="data/qrels.parquet")
    p.add_argument("--out-dir", default="analysis")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero when any warning fires")
    args = p.parse_args()

    summary, found = run(args.config, args.runs_dir, args.universe,
                         args.qrels, args.out_dir)

    print()
    for k_c, v in summary["by_k_c"].items():
        print(f"K_c = {k_c}    candidate Jaccard (median) "
              f"{v['median_candidate_jaccard']}")
        for system, s in v["by_system"].items():
            print(f"  {system}")
            print(f"    judged density  C_q / U_q      {s['median_density_cq']} / "
                  f"{s['median_density_uq']}   ratio {s['median_density_ratio']}x")
            print(f"    judged supply kept by the gate {s['median_judged_retained']}")
            print(f"    gainful non-survivors (median) "
                  f"{s['median_gainful_non_survivors']}")
            print(f"    queries with none at all       "
                  f"{s['queries_with_no_gainful_non_survivor']} / {s['n_queries']}")
        print(f"  usable for the headline gate (all systems) "
              f"{v['usable_for_headline_gate']}")
        print()
    print(f"Headline K_c: {summary['headline_k_c']}")
    for line in textwrap.wrap(summary["headline_k_c_rule"], 74):
        print(f"  {line}")
    print(f"Stage-2 ceiling, whole project:    "
          f"{summary['total_universe_pairs']:,} unique pairs")

    if found:
        n = len(found)
        print(f"\n{n} finding{'' if n == 1 else 's'} needing a recorded decision:")
        for kind, v in found.items():
            print(f"\n  [{kind}]  fires at: {'; '.join(v['where'])}")
            for line in textwrap.wrap(v["explain"], 74):
                print(f"      {line}")
        return 1 if args.strict else 0
    print("\nNo findings. The experiment as specified is measurable at every K_c.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
