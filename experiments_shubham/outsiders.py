"""
outsiders.py - What each outsider policy actually promotes, what it misses,
and how much the random control moves with its seed.

Three Week-3 checks, all from the effect tables at the headline K_c, uncondensed:

    promotion       per (system, policy, t): how many outsiders world D promoted,
                    the share judged, the share relevant (gain > 0 under the
                    frozen map), the share that reached the final top-k, and how
                    many of them were real gating failures
    false negatives per (system, policy, t): of the exact gating failures - non-
                    survivors the no-gate world puts in the top-k - the share
                    the policy did NOT promote at that budget. For `boundary`
                    this is the boundary-probe false-negative rate the proposal
                    says to measure rather than assume away
    seed variation  per (system, t) for `random`: mean Delta DCG and both H7
                    statistics under each budget seed separately, and their
                    spread. A control whose verdict changes with its seed is
                    not a control

Budgets are the t-experiment's own, (1 - t) * B outsiders per query, so these
are statements about the allocations the reported Delta_t came from.

    python -m experiments_shubham.outsiders      # -> analysis/outsiders/
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments_shubham.analysis import diagnose, h7, load_stage
from experiments_shubham.engine import table_name
from experiments_shubham.power_h7 import headline_k_c
from shared.config import FrozenConfig, load_config
from shared.provenance import run_manifest, write_manifest
from shared.schema import E_GATE, NON_SURVIVOR, SELECTED_OUTSIDER, load
from shared.utility import gain

logger = logging.getLogger(__name__)

EPS = 1e-12


def _budget(path: Path, columns: list[str], filters=None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns, filters=filters)
    for c in ("query_id", "doc_id"):
        if c in df:
            df[c] = df[c].astype(str)
    return df


def gating_failure_set(stage: pd.DataFrame) -> pd.DataFrame:
    """(query_id, doc_id) of every exact gating failure."""
    g = stage[(stage["effect_type"] == E_GATE) & (stage["stratum"] == NON_SURVIVOR)
              & (stage["utility_control"] > EPS)]
    return g[["query_id", "doc_id"]].astype(str).drop_duplicates()


def promotion(promoted: pd.DataFrame, failures: pd.DataFrame, relevant: set,
              n_queries: int) -> pd.DataFrame:
    """Per (policy, t): what the promoted outsiders were, averaged over seeds."""
    p = promoted.copy()
    p["relevant"] = [(q, d) in relevant for q, d in zip(p["query_id"], p["doc_id"])]
    p["reached_top_k"] = p["utility_treatment"] > EPS
    fail = set(zip(failures["query_id"], failures["doc_id"]))
    p["gating_failure"] = [(q, d) in fail for q, d in zip(p["query_id"], p["doc_id"])]
    per_seed = p.groupby(["policy", "t", "seed"], dropna=False).agg(
        n_promoted=("doc_id", "size"), share_judged=("judged", "mean"),
        share_relevant=("relevant", "mean"), share_reached_top_k=("reached_top_k", "mean"),
        gating_failures_promoted=("gating_failure", "sum")).reset_index()
    per_seed["promoted_per_query"] = per_seed["n_promoted"] / n_queries
    out = per_seed.drop(columns="seed").groupby(["policy", "t"]).mean().reset_index()
    out["n_seeds"] = per_seed.groupby(["policy", "t"]).size().to_numpy()
    return out


def false_negatives(promoted: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """Per (policy, t): the share of exact gating failures the policy left out."""
    fail = failures.assign(is_failure=True)
    rows = []
    for (policy, t, seed), blk in promoted.groupby(["policy", "t", "seed"], dropna=False):
        hit = fail.merge(blk[["query_id", "doc_id"]].drop_duplicates(),
                         on=["query_id", "doc_id"], how="left", indicator=True)
        found = hit["_merge"] == "both"
        per_q = found.groupby(hit["query_id"]).any()
        rows.append({"policy": policy, "t": t, "seed": seed,
                     "gating_failures": int(len(hit)), "found": int(found.sum()),
                     "false_negative_rate": float(1 - found.mean()) if len(hit) else np.nan,
                     "queries_with_failures": int(per_q.size),
                     "queries_all_missed": int((~per_q).sum())})
    out = pd.DataFrame(rows)
    return (out.drop(columns="seed").groupby(["policy", "t"]).mean().reset_index()
            .assign(n_seeds=out.groupby(["policy", "t"]).size().to_numpy()))


def seed_variation(budget: pd.DataFrame, diag: pd.DataFrame, n_permutations: int,
                   seed: int) -> pd.DataFrame:
    """Each random seed on its own: mean Delta DCG and H7, then the spread."""
    per = (budget.groupby(["query_id", "t", "seed"])["effect"].sum()
           .rename("delta_dcg").reset_index())
    rows = []
    for s, blk in per.groupby("seed"):
        outcomes = blk.drop(columns="seed").assign(policy="random")
        stats = h7(diag, outcomes, n_permutations, seed).set_index("t")
        for t, tb in blk.groupby("t"):
            r = stats.loc[float(t)]
            rows.append({"seed": int(s), "t": float(t), "mean_delta": tb["delta_dcg"].mean(),
                         "group_difference": r["group_difference"], "group_p": r["group_p"],
                         "spearman_rho": r["spearman_rho"],
                         "spearman_p_one_sided": r["spearman_p_one_sided"]})
    return pd.DataFrame(rows)


def run_system(system: str, cfg: FrozenConfig, k_c: int, effects_dir: Path,
               relevant: set, n_permutations: int) -> dict[str, pd.DataFrame]:
    stage = load_stage(effects_dir, system, k_c)
    stage["query_id"] = stage["query_id"].astype(str)
    stage["doc_id"] = stage["doc_id"].astype(str)
    failures = gating_failure_set(stage)
    diag = diagnose(stage, cfg.tau_query)
    n_q = stage["query_id"].nunique()
    cols = ["query_id", "doc_id", "t", "seed", "judged", "utility_treatment", "policy"]
    promoted = pd.concat(
        [_budget(effects_dir / f"{table_name(system, k_c, p, False)}.parquet", cols,
                 [("stratum", "==", SELECTED_OUTSIDER)]).assign(policy=p)
         for p in cfg.outsider_policies], ignore_index=True)
    promoted["judged"] = promoted["judged"].fillna(False).astype(bool)
    rnd = _budget(effects_dir / f"{table_name(system, k_c, 'random', False)}.parquet",
                  ["query_id", "doc_id", "t", "seed", "effect"], [("judged", "==", True)])
    out = {"promotion": promotion(promoted, failures, relevant, n_q),
           "false_negatives": false_negatives(promoted, failures),
           "seed_variation": seed_variation(rnd, diag, n_permutations, cfg.seed)}
    for df in out.values():
        df.insert(0, "system", system)
    out["n_gating_failures"] = len(failures)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="What the outsider policies promote and miss")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--effects-dir", default="effects/real")
    parser.add_argument("--k-c", type=int, default=None)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--out-dir", default="analysis/outsiders")
    args = parser.parse_args()

    cfg = load_config(args.config)
    k_c = args.k_c or headline_k_c(cfg)[0]
    qrels = load(cfg.qrels_file, "qrels")
    qrels = qrels[qrels["judged"].fillna(False).astype(bool)]
    relevant = {(str(q), str(d)) for q, d, g in
                zip(qrels["query_id"], qrels["doc_id"], qrels["relevance"])
                if gain(int(g), cfg.gain_map) > 0}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    parts: dict[str, list[pd.DataFrame]] = {}
    failures = {}
    for system in cfg.stage1_systems:
        res = run_system(system, cfg, k_c, Path(args.effects_dir), relevant,
                         args.n_permutations)
        failures[system] = res.pop("n_gating_failures")
        for name, df in res.items():
            parts.setdefault(name, []).append(df)
        logger.info("%s done", system)
    tables = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)

    def records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
        t = df[cols].round(4)
        return t.astype(object).where(t.notna(), None).to_dict("records")

    sv = tables["seed_variation"]
    spread = (sv.groupby(["system", "t"])
              .agg(mean_delta_min=("mean_delta", "min"), mean_delta_max=("mean_delta", "max"),
                   rho_min=("spearman_rho", "min"), rho_max=("spearman_rho", "max"),
                   group_p_min=("group_p", "min"), group_p_max=("group_p", "max"))
              .reset_index())
    write_manifest(out / "outsiders_manifest.json", run_manifest(
        cfg.fingerprint(), owner="shubham", extra={
            "effects_dir": args.effects_dir, "k_c": k_c, "n_permutations": args.n_permutations,
            "gating_failures": failures,
            "promotion": records(tables["promotion"],
                                 ["system", "policy", "t", "promoted_per_query", "share_judged",
                                  "share_relevant", "share_reached_top_k",
                                  "gating_failures_promoted"]),
            "false_negatives": records(tables["false_negatives"],
                                       ["system", "policy", "t", "gating_failures", "found",
                                        "false_negative_rate", "queries_all_missed"]),
            "random_seed_spread": records(spread, list(spread.columns)),
            "files": sorted(p.name for p in out.iterdir()),
            "seconds": round(time.perf_counter() - start, 1),
        }))
    logger.info("written to %s", out)


if __name__ == "__main__":
    main()
