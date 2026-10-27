"""
power_h7.py - How many reference queries H7 needs (hand-off H2).

H7 runs only on the reference subset, the one place the full no-gate world
exists, so the subset's size bounds its power. This sets that size on
synthetic data before any GPU budget is committed (config `power`).

1. A pool of synthetic queries. Each draws its own stage quality - a1, a2 in
   [0.45, 0.85] and sigma1, sigma2 in [0.15, 0.60], the envelope the six
   regimes span - so the pool mixes gate-limited, reranker-limited and stable
   queries instead of assuming one regime. For each query and each K_c:
   the diagnosis label from worlds A/B/C at the frozen tau_query, and
   Delta DCG@k at every t < 1 and outsider policy (random: mean over seeds).
2. Power. `shared.diagnosis.required_reference_size` draws reference subsets
   of each candidate size from the pool and runs the H7 permutation test.

The recommendation, declared before the run: the smallest candidate size that
reaches the target power for EVERY policy at the smallest t, at the headline
K_c (the largest candidate depth, provisionally). If none does, it says so.

Conditional on the synthetic mechanism: real effect sizes and label shares
can differ, and the Week-3 reference effects are the check on this number.

    python -m experiments_shubham.power_h7                  # ~10 min
    python -m experiments_shubham.power_h7 --pool-size 200  # quick
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from shared.allocation import RANDOM
from shared.config import FrozenConfig, load_config
from shared.diagnosis import (
    GATE_SENSITIVE,
    QUERY_LABELS,
    RERANK_SENSITIVE,
    query_delta_dcg,
    query_diagnosis,
    required_reference_size,
    smallest_sufficient_size,
)
from shared.provenance import run_manifest, write_manifest
from shared.synthetic import SyntheticParams, alternative_ranks, generate, to_lookups
from shared.worlds import build_t_grid, build_worlds

logger = logging.getLogger(__name__)

ALIGNMENT_RANGE = (0.45, 0.85)
NOISE_RANGE = (0.15, 0.60)


def simulate_pool(cfg: FrozenConfig, pool_size: int, n_docs: int | None = None,
                  seed: int | None = None) -> pd.DataFrame:
    """
    One row per (query, K_c, policy, t < 1): the query's diagnosis and its
    budget outcome.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    n_docs = n_docs or cfg.synthetic["n_docs_per_query"]
    k, gain_map = cfg.final_cutoff_k, cfg.gain_map
    rows = []
    for i in range(pool_size):
        a1, a2 = rng.uniform(*ALIGNMENT_RANGE, size=2)
        sigma1, sigma2 = rng.uniform(*NOISE_RANGE, size=2)
        query_seed = int(rng.integers(1 << 31))
        df = generate(SyntheticParams(
            float(a1), float(a2), float(sigma1), float(sigma2), n_queries=1,
            n_docs=n_docs, relevance_quantiles=tuple(cfg.synthetic["relevance_quantiles"]),
            regime="h7_pool", seed=query_seed,
        ))
        universe, s1, s2, rel = (next(iter(x.values())) for x in to_lookups(df))
        alt = next(iter(alternative_ranks(df, seed=query_seed).values()))

        for k_c in cfg.candidate_depths:
            w = build_worlds(f"p{i:05d}", universe, s1, s2, k_c)
            diag = query_diagnosis(w, rel, k, gain_map, cfg.tau_query)
            for policy in cfg.outsider_policies:
                seeds = cfg.budget_seeds if policy == RANDOM else cfg.budget_seeds[:1]
                deltas = [
                    query_delta_dcg(build_t_grid(w, s1, s2, cfg.budget_for(k_c), cfg.t_grid,
                                                 policy, alternative_rank=alt, seed=s),
                                    rel, k, gain_map)
                    for s in seeds
                ]
                for t in cfg.t_grid:
                    if t == 1.0:
                        continue
                    rows.append({
                        "query": i, "a1": a1, "a2": a2, "sigma1": sigma1, "sigma2": sigma2,
                        "k_c": k_c, "policy": policy, "t": float(t),
                        "label": diag["label"], "g_q": diag["g_q"], "r_q": diag["r_q"],
                        "reranker_harms": diag["reranker_harms"],
                        "delta_dcg": float(np.mean([d[t] for d in deltas])),
                    })
        if (i + 1) % 100 == 0:
            logger.info("pool: %d/%d queries", i + 1, pool_size)
    return pd.DataFrame(rows)


def power_tables(pool: pd.DataFrame, cfg: FrozenConfig) -> pd.DataFrame:
    """Estimated power per (K_c, policy, t) at every candidate size."""
    pw = cfg.power
    out = []
    for (k_c, policy, t), blk in pool.groupby(["k_c", "policy", "t"]):
        labels = blk["label"].to_numpy()
        deltas = blk["delta_dcg"].to_numpy(float)

        def simulate(rng, labels=labels, deltas=deltas):
            j = int(rng.integers(len(labels)))
            return labels[j], deltas[j]

        table = required_reference_size(
            simulate, pw["candidate_sizes"], n_simulations=pw["n_simulations"],
            alpha=pw["alpha"], target_power=pw["target_power"], seed=cfg.seed,
        )
        gate = deltas[labels == GATE_SENSITIVE]
        rer = deltas[labels == RERANK_SENSITIVE]
        out.append(table.assign(
            k_c=k_c, policy=policy, t=t,
            share_gate=float(np.mean(labels == GATE_SENSITIVE)),
            share_rerank=float(np.mean(labels == RERANK_SENSITIVE)),
            pool_difference=float(gate.mean() - rer.mean()) if len(gate) and len(rer)
            else float("nan"),
        ))
    return pd.concat(out, ignore_index=True)


def recommend(power: pd.DataFrame, cfg: FrozenConfig) -> dict:
    """The declared rule: every policy at the smallest t, at the headline K_c."""
    k_c, t_min = max(cfg.candidate_depths), min(cfg.t_grid)
    per_policy = {}
    for policy in cfg.outsider_policies:
        blk = power[(power["k_c"] == k_c) & (power["policy"] == policy)
                    & (power["t"] == t_min)]
        per_policy[policy] = smallest_sufficient_size(blk)
    sizes = list(per_policy.values())
    size = None if any(s is None for s in sizes) else max(sizes)
    return {"k_c": k_c, "t": t_min, "per_policy": per_policy, "reference_size": size,
            "largest_tested": max(cfg.power["candidate_sizes"])}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="H7 power simulation (H2)")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--n-docs", type=int, default=None)
    parser.add_argument("--out-dir", default=None, help="default: analysis/power")
    args = parser.parse_args()

    cfg = load_config(args.config)
    analysis_root = Path(cfg.paths.get("analysis", "analysis"))
    out_dir = Path(args.out_dir) if args.out_dir else analysis_root / "power"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    pool = simulate_pool(cfg, args.pool_size, args.n_docs)
    pool.to_csv(out_dir / "h7_pool.csv", index=False)
    power = power_tables(pool, cfg)
    power.to_csv(out_dir / "h7_power.csv", index=False)
    rec = recommend(power, cfg)

    labels = (pool.drop_duplicates(["query", "k_c"]).groupby("k_c")["label"]
              .value_counts(normalize=True).unstack().reindex(columns=list(QUERY_LABELS))
              .fillna(0.0))
    manifest = run_manifest(cfg.fingerprint(), owner="shubham", extra={
        "experiment": "H7 power simulation (H2)", "pool_size": args.pool_size,
        "n_docs": args.n_docs or cfg.synthetic["n_docs_per_query"],
        "alignment_range": ALIGNMENT_RANGE, "noise_range": NOISE_RANGE,
        "tau_query": cfg.tau_query, "power": cfg.power,
        "label_shares": {int(k): v for k, v in labels.to_dict("index").items()},
        "recommendation": rec, "seconds": round(time.perf_counter() - start, 1),
    })
    manifest_path = (analysis_root if args.out_dir is None else out_dir) / "h7_power_manifest.json"
    write_manifest(manifest_path, manifest)

    logger.info("label shares by K_c:\n%s", labels.round(3))
    head = power[(power["k_c"] == rec["k_c"]) & (power["t"] == rec["t"])]
    logger.info("power at K_c=%d, t=%.2f:\n%s", rec["k_c"], rec["t"],
                head.pivot_table(index="reference_size", columns="policy", values="power"))
    if rec["reference_size"] is None:
        logger.warning("no tested size up to %d reaches power %.2f for every policy: %s",
                       rec["largest_tested"], cfg.power["target_power"], rec["per_policy"])
    else:
        logger.info("recommended reference subset: %d queries (%s)",
                    rec["reference_size"], rec["per_policy"])


if __name__ == "__main__":
    main()
