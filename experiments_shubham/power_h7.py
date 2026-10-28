"""
power_h7.py - How many reference queries H7 needs (hand-off H2).

H7 runs only on the reference subset, the one place the full no-gate world
exists, so the subset's size bounds its power. This sets that size on
synthetic data before any GPU budget is committed (config `power`).

1. A pool of synthetic queries. Each draws its own stage quality - a1, a2 in
   [0.45, 0.85] and sigma1, sigma2 in [0.15, 0.60], the envelope the six
   regimes span - so the pool mixes gate-limited, reranker-limited and stable
   queries instead of assuming one regime. For each query and each K_c:
   the diagnosis from worlds A/B/C at the frozen tau_query, and Delta DCG@k at
   every t < 1 and outsider policy (random: mean over seeds). Pools are drawn
   at two universe sizes: the synthetic config's, and 8000, a realistic
   |U_q| for the union of two top-5000 runs.
2. Power of two tests, by drawing reference subsets from the pool:
     group       the proposal's test: Delta DCG of gate-sensitive vs
                 reranking-sensitive queries (shared.diagnosis.h7_test). A
                 subset with fewer than two queries in a group cannot be
                 tested, which is a failed H7, so it counts as a miss.
                 `shared.diagnosis` counts it that way itself: its `power` is
                 over every simulation drawn, and `power_given_testable`
                 conditions on the testable ones and overstates what a real
                 subset would give. `unconditional_power` here is the former,
                 kept as a name the continuous test can share.
     continuous  the same hypothesis on every reference query: one-sided
                 Spearman correlation between G_q and Delta DCG. Proposed to
                 the team, not frozen.

Recommendations at the headline K_c (the largest candidate depth) and the
smallest t, required at every universe size:
     group       every policy reaches the target unconditional power
     continuous  boundary and alternative reach it (random is the control)

Conditional on the synthetic mechanism, whose relevance density is far above
real TREC-DL: real effect sizes and label shares will differ, and the Week-3
reference effects are the check on this number.

    python -m experiments_shubham.power_h7                   # ~30 min
    python -m experiments_shubham.power_h7 --pool-size 100 --n-docs 2000   # quick
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shared.allocation import RANDOM
from shared.config import FrozenConfig, load_config
from shared.diagnosis import (
    GATE_SENSITIVE,
    QUERY_LABELS,
    RERANK_SENSITIVE,
    query_delta_dcg,
    query_diagnosis,
    required_reference_size,
)
from shared.provenance import run_manifest, write_manifest
from shared.synthetic import SyntheticParams, alternative_ranks, generate, to_lookups
from shared.worlds import build_t_grid, build_worlds

logger = logging.getLogger(__name__)

ALIGNMENT_RANGE = (0.45, 0.85)
NOISE_RANGE = (0.15, 0.60)
REALISTIC_UNIVERSE = 8000
ALL_QUERIES = 97            # TREC-DL 2019 + 2020 judged queries


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
                        "n_docs": n_docs, "query": i, "a1": a1, "a2": a2,
                        "sigma1": sigma1, "sigma2": sigma2,
                        "k_c": k_c, "policy": policy, "t": float(t),
                        "label": diag["label"], "g_q": diag["g_q"], "r_q": diag["r_q"],
                        "reranker_harms": diag["reranker_harms"],
                        "delta_dcg": float(np.mean([d[t] for d in deltas])),
                    })
        if (i + 1) % 100 == 0:
            logger.info("pool |U|=%d: %d/%d queries", n_docs, i + 1, pool_size)
    return pd.DataFrame(rows)


GROUPS = ["n_docs", "k_c", "policy", "t"]


def group_power(pool: pd.DataFrame, cfg: FrozenConfig, sizes: list[int]) -> pd.DataFrame:
    """The proposal's group test, per (n_docs, K_c, policy, t) and size."""
    pw = cfg.power
    out = []
    for key, blk in pool.groupby(GROUPS):
        labels = blk["label"].to_numpy()
        deltas = blk["delta_dcg"].to_numpy(float)

        def simulate(rng, labels=labels, deltas=deltas):
            j = int(rng.integers(len(labels)))
            return labels[j], deltas[j]

        table = required_reference_size(
            simulate, sizes, n_simulations=pw["n_simulations"],
            alpha=pw["alpha"], target_power=pw["target_power"], seed=cfg.seed,
        )
        # `shared.diagnosis` now divides detections by every simulation drawn,
        # so its `power` is already the unconditional quantity and the local
        # rescaling that used to live here would double-correct. The conditional
        # figure, if ever wanted, is `power_given_testable`.
        table["unconditional_power"] = table["power"]
        gate = deltas[labels == GATE_SENSITIVE]
        rer = deltas[labels == RERANK_SENSITIVE]
        out.append(table.assign(
            **dict(zip(GROUPS, key)), test="group",
            share_gate=float(np.mean(labels == GATE_SENSITIVE)),
            share_rerank=float(np.mean(labels == RERANK_SENSITIVE)),
            pool_effect=float(gate.mean() - rer.mean()) if len(gate) and len(rer)
            else float("nan"),
        ))
    return pd.concat(out, ignore_index=True)


def correlation_power(pool: pd.DataFrame, cfg: FrozenConfig,
                      sizes: list[int]) -> pd.DataFrame:
    """One-sided Spearman(G_q, Delta DCG) on every reference query."""
    pw = cfg.power
    rows = []
    for key, blk in pool.groupby(GROUPS):
        g = blk["g_q"].to_numpy(float)
        d = blk["delta_dcg"].to_numpy(float)
        rng = np.random.default_rng(cfg.seed)
        pool_rho = float(spearmanr(g, d).statistic)
        for n in sizes:
            hits = 0
            for _ in range(pw["n_simulations"]):
                j = rng.integers(len(g), size=n)
                p = spearmanr(g[j], d[j], alternative="greater").pvalue
                hits += bool(np.isfinite(p) and p < pw["alpha"])
            power = hits / pw["n_simulations"]
            rows.append({**dict(zip(GROUPS, key)), "test": "continuous",
                         "reference_size": n, "n_simulations": pw["n_simulations"],
                         "unconditional_power": power, "pool_effect": pool_rho})
    return pd.DataFrame(rows)


def _smallest(blk: pd.DataFrame, target: float) -> int | None:
    ok = blk[blk["unconditional_power"] >= target]
    return int(ok["reference_size"].min()) if len(ok) else None


def recommend(power: pd.DataFrame, cfg: FrozenConfig) -> dict:
    """The declared rules, at the headline K_c and the smallest t."""
    k_c, t_min = max(cfg.candidate_depths), min(cfg.t_grid)
    target = cfg.power["target_power"]
    head = power[(power["k_c"] == k_c) & (power["t"] == t_min)]
    rules = {
        "group": list(cfg.outsider_policies),
        "continuous": [p for p in cfg.outsider_policies if p != RANDOM],
    }
    out = {"k_c": k_c, "t": t_min, "target_power": target}
    for test, policies in rules.items():
        per = {
            f"{policy}@{n_docs}": _smallest(
                head[(head["test"] == test) & (head["policy"] == policy)
                     & (head["n_docs"] == n_docs)], target)
            for policy in policies for n_docs in sorted(head["n_docs"].unique())
        }
        sizes = list(per.values())
        out[test] = {"policies": policies, "smallest_per_setting": per,
                     "reference_size": None if None in sizes else max(sizes)}
    return out


def headline_table(power: pd.DataFrame, cfg: FrozenConfig) -> dict:
    """Unconditional power at the headline K_c and smallest t, for the manifest."""
    k_c, t_min = max(cfg.candidate_depths), min(cfg.t_grid)
    head = power[(power["k_c"] == k_c) & (power["t"] == t_min)]
    return {
        f"{test}|{policy}|U={n_docs}": {int(r.reference_size): round(float(r.unconditional_power), 3)
                                        for r in blk.itertuples()}
        for (test, policy, n_docs), blk in head.groupby(["test", "policy", "n_docs"])
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    # one line per generated pool query otherwise
    logging.getLogger("shared.synthetic").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="H7 power simulation (H2)")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--n-docs", type=int, nargs="+", default=None,
                        help=f"universe sizes (default: the config's and {REALISTIC_UNIVERSE})")
    parser.add_argument("--out-dir", default=None, help="default: analysis/power")
    args = parser.parse_args()

    cfg = load_config(args.config)
    analysis_root = Path(cfg.paths.get("analysis", "analysis"))
    out_dir = Path(args.out_dir) if args.out_dir else analysis_root / "power"
    out_dir.mkdir(parents=True, exist_ok=True)
    universes = args.n_docs or [cfg.synthetic["n_docs_per_query"], REALISTIC_UNIVERSE]
    sizes = sorted(set(cfg.power["candidate_sizes"]) | {ALL_QUERIES})

    start = time.perf_counter()
    pool = pd.concat([simulate_pool(cfg, args.pool_size, n) for n in universes],
                     ignore_index=True)
    pool.to_csv(out_dir / "h7_pool.csv", index=False)
    power = pd.concat([group_power(pool, cfg, sizes), correlation_power(pool, cfg, sizes)],
                      ignore_index=True)
    power.to_csv(out_dir / "h7_power.csv", index=False)
    rec = recommend(power, cfg)

    labels = (pool.drop_duplicates(["n_docs", "query", "k_c"])
              .groupby(["n_docs", "k_c"])["label"].value_counts(normalize=True)
              .unstack().reindex(columns=list(QUERY_LABELS)).fillna(0.0).round(3))
    manifest = run_manifest(cfg.fingerprint(), owner="shubham", extra={
        "experiment": "H7 power simulation (H2)", "pool_size": args.pool_size,
        "universe_sizes": universes, "reference_sizes": sizes,
        "alignment_range": ALIGNMENT_RANGE, "noise_range": NOISE_RANGE,
        "tau_query": cfg.tau_query, "power": cfg.power,
        "label_shares": {f"U={n}|K_c={k}": row for (n, k), row
                         in labels.to_dict("index").items()},
        "headline_power": headline_table(power, cfg),
        "recommendation": rec, "seconds": round(time.perf_counter() - start, 1),
    })
    manifest_path = ((analysis_root if args.out_dir is None else out_dir)
                     / "h7_power_manifest.json")
    write_manifest(manifest_path, manifest)

    logger.info("label shares:\n%s", labels)
    for test in ("group", "continuous"):
        r = rec[test]
        logger.info("%s test: reference size %s (%s)", test,
                    r["reference_size"] if r["reference_size"] else "none tested reaches "
                    f"power {rec['target_power']}", r["smallest_per_setting"])


if __name__ == "__main__":
    main()
