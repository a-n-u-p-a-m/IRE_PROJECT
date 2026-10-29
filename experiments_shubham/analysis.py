"""
analysis.py - Effect distributions, document labels, the query diagnosis and
H7, computed from effect tables alone.

Everything here reads the effect tables `engine.run_grid` writes
(`{system}_stage_kc{K}.parquet`, `{system}_budget_kc{K}_{policy}.parquet`) and
nothing else, so the same code runs on synthetic tables now and on the real
ones in Week 3:

    distributions   per (effect type, K_c, t, policy, stratum): size, queries,
                    mean, spread and quantiles - all documents, and judged only
    additivity      E_rerank + E_gate against U_d(pi_act): the two effects are
                    not a decomposition, and the figure shows by how much
    labels          shared.effects.label_documents at the frozen tau_document,
                    and the gating failures behind the qualitative examples
    diagnosis       G_q = DCG(pi_no1) - DCG(pi_act) = -sum_d E_gate,
                    R_q = |DCG(pi_act) - DCG(pi_no2)| = |sum_d E_rerank|,
                    labelled by the frozen rule in shared.diagnosis
    H7              per (policy, t): the group test (shared.diagnosis.h7_test)
                    and the continuous one-sided Spearman(G_q, Delta DCG), with
                    group sizes beside every p-value

    python -m experiments_shubham.engine --regime weak_retriever
    python -m experiments_shubham.analysis --effects-dir effects/synthetic/weak_retriever \\
        --system synthetic
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments_shubham.power_h7 import headline_k_c
from shared.config import FrozenConfig, load_config
from shared.diagnosis import (
    GATE_SENSITIVE,
    MIXED_STABLE,
    RERANK_SENSITIVE,
    h7_table,
    h7_test,
)
from shared.effects import check_additivity, gating_failures, label_documents
from shared.provenance import run_manifest, write_manifest
from shared.schema import E_GATE, E_RERANK, load

logger = logging.getLogger(__name__)

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def load_stage(effects_dir: str | Path, system: str, k_c: int) -> pd.DataFrame:
    return load(Path(effects_dir) / f"{system}_stage_kc{k_c}.parquet", "effect")


def load_budget(effects_dir: str | Path, system: str, k_c: int,
                policies) -> dict[str, pd.DataFrame]:
    return {p: load(Path(effects_dir) / f"{system}_budget_kc{k_c}_{p}.parquet", "effect")
            for p in policies}


# ──────────────────────────────────────────────────────────────────────
# Distributions
# ──────────────────────────────────────────────────────────────────────

def distributions(effects: pd.DataFrame, judged_only: bool = False) -> pd.DataFrame:
    """Per (effect type, K_c, t, policy, stratum): always with its size."""
    eff = effects[effects["judged"].fillna(False)] if judged_only else effects
    keys = ["effect_type", "k_c", "t", "policy", "stratum"]
    rows = []
    for key, blk in eff.groupby(keys, dropna=False, observed=True):
        x = blk["effect"].to_numpy(float)
        rows.append({
            **dict(zip(keys, key)), "judged_only": judged_only,
            "n": len(x), "n_queries": blk["query_id"].nunique(),
            "mean": x.mean(), "sd": x.std(ddof=1) if len(x) > 1 else float("nan"),
            "share_nonzero": float(np.mean(np.abs(x) > 1e-9)),
            **{f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in QUANTILES},
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Query diagnosis and H7
# ──────────────────────────────────────────────────────────────────────

def diagnose(stage: pd.DataFrame, tau_query: float) -> pd.DataFrame:
    """
    The frozen labelling rule of shared.diagnosis.query_diagnosis, from the
    effect table: summing a per-document effect over the universe gives the
    system-level DCG difference, because DCG@k is a sum of document utilities.
    """
    act = (stage[stage["effect_type"] == E_RERANK]
           .groupby("query_id", observed=True)["utility_treatment"].sum())
    total = stage.groupby(["query_id", "effect_type"], observed=True)["effect"].sum()
    total = total.unstack()
    out = pd.DataFrame({
        "query_id": total.index.astype(str),
        "dcg_actual": act.reindex(total.index).to_numpy(float),
        "g_q": -total[E_GATE].to_numpy(float),
        "rerank_signed": total[E_RERANK].to_numpy(float),
    })
    out["r_q"] = out["rerank_signed"].abs()
    out["reranker_harms"] = out["rerank_signed"] < 0
    g, r = out["g_q"], out["r_q"]
    out["label"] = np.where(
        (g > tau_query) & (g > r), GATE_SENSITIVE,
        np.where((r > tau_query) & (r >= np.maximum(g, 0.0)), RERANK_SENSITIVE,
                 MIXED_STABLE))
    return out


def budget_outcomes(budgets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """DCG@k at each t minus DCG@k at t=1, per (query, policy, t); random: mean over seeds."""
    frames = []
    for policy, budget in budgets.items():
        per_run = (budget.assign(seed=budget["seed"].fillna(-1))
                   .groupby(["query_id", "t", "seed"], observed=True)["effect"].sum()
                   .groupby(["query_id", "t"]).mean().rename("delta_dcg").reset_index())
        frames.append(per_run.assign(policy=policy))
    out = pd.concat(frames, ignore_index=True)
    out["query_id"] = out["query_id"].astype(str)
    return out


def h7(diagnosis: pd.DataFrame, outcomes: pd.DataFrame, n_permutations: int = 10000,
       seed: int = 42) -> pd.DataFrame:
    """Both H7 tests per (policy, t), with group sizes beside every p-value."""
    rows = []
    g_by_q = dict(zip(diagnosis["query_id"], diagnosis["g_q"]))
    for policy, blk in outcomes.groupby("policy"):
        by_query = {q: dict(zip(b["t"], b["delta_dcg"]))
                    for q, b in blk.groupby("query_id")}
        group = h7_test(h7_table(diagnosis, by_query), n_permutations, seed)
        for t, tb in blk.groupby("t"):
            g = np.array([g_by_q[q] for q in tb["query_id"]])
            d = tb["delta_dcg"].to_numpy(float)
            res = spearmanr(g, d, alternative="greater") if len(g) > 2 else None
            grp = group[group["t"] == t].iloc[0].to_dict() if (group["t"] == t).any() else {}
            rows.append({
                "policy": policy, "t": float(t), "n_queries": len(g),
                "n_gate_sensitive": grp.get("n_gate_sensitive", 0),
                "n_rerank_sensitive": grp.get("n_rerank_sensitive", 0),
                "group_difference": grp.get("difference", float("nan")),
                "group_p": grp.get("p_value", float("nan")),
                "group_note": grp.get("note", ""),
                "spearman_rho": float(res.statistic) if res else float("nan"),
                "spearman_p_one_sided": float(res.pvalue) if res else float("nan"),
            })
    return pd.DataFrame(rows).sort_values(["policy", "t"], ascending=[True, False])


# ──────────────────────────────────────────────────────────────────────
# Labels, gating failures, additivity
# ──────────────────────────────────────────────────────────────────────

def label_summary(stage: pd.DataFrame, tau_document: float) -> pd.DataFrame:
    labels = label_documents(stage, tau_document)
    return (labels.groupby(["stratum", "label"], observed=True).size()
            .rename("n").reset_index())


def gating_failure_examples(stage: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The largest gating failures: what each excluded document would have earned."""
    fails = gating_failures(stage)
    cols = ["query_id", "doc_id", "judged", "rank_control", "utility_control", "effect"]
    return (fails[cols].rename(columns={"rank_control": "rank_without_gate",
                                        "utility_control": "utility_without_gate",
                                        "effect": "e_gate"})
            .head(n).reset_index(drop=True))


def additivity_figure(stage: pd.DataFrame, path: str | Path, title: str = "") -> dict:
    """
    E_rerank + E_gate against U_d(pi_act), per document with any nonzero term.

    On the diagonal the two effects would add up to the document's actual
    utility; they do not, because each is measured against a different
    reference world.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wide = check_additivity(stage)
    s = wide[f"effect_{E_RERANK}"] + wide[f"effect_{E_GATE}"]
    u = wide[f"utility_treatment_{E_RERANK}"]
    keep = (s.abs() > 1e-9) | (u.abs() > 1e-9)
    s, u, res = s[keep], u[keep], wide.loc[keep, "additivity_residual"]

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    ax.scatter(u, s, s=7, alpha=0.35, color="#1d4e89", linewidths=0)
    lo, hi = float(min(u.min(), s.min())), float(max(u.max(), s.max()))
    ax.plot([lo, hi], [lo, hi], color="#9aa5b1", lw=1, ls="--", label="additive (y = x)")
    ax.set_xlabel("U_d(π_act): the document's actual utility")
    ax.set_ylabel("E_rerank + E_gate")
    ax.set_title(title or "The two stage effects are not a decomposition", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return {"n_documents": int(keep.sum()),
            "share_off_diagonal": float(np.mean(np.abs(res) > 1e-9)),
            "max_abs_residual": float(np.abs(res).max()) if len(res) else 0.0}


# ──────────────────────────────────────────────────────────────────────
# The run
# ──────────────────────────────────────────────────────────────────────

def run(effects_dir: str | Path, system: str, k_c: int, cfg: FrozenConfig,
        out_dir: str | Path, n_permutations: int = 10000) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = load_stage(effects_dir, system, k_c)
    budgets = load_budget(effects_dir, system, k_c, cfg.outsider_policies)

    dist = pd.concat([distributions(e, judged) for e in [stage, *budgets.values()]
                      for judged in (False, True)], ignore_index=True)
    diag = diagnose(stage, cfg.tau_query)
    outcomes = budget_outcomes(budgets)
    tests = h7(diag, outcomes, n_permutations, cfg.seed)
    labels = label_summary(stage, cfg.tau_document)
    examples = gating_failure_examples(stage)
    additivity = additivity_figure(stage, out_dir / "additivity.png",
                                   f"{system}, K_c = {k_c}: E_rerank + E_gate vs U_d(π_act)")

    tables = {"distributions": dist, "diagnosis": diag, "budget_outcomes": outcomes,
              "h7": tests, "labels": labels, "gating_failures": examples}
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return {"tables": tables, "additivity": additivity,
            "label_counts": diag["label"].value_counts().to_dict()}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Effect analysis and H7 from effect tables")
    parser.add_argument("--effects-dir", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--k-c", type=int, default=None,
                        help="default: the feasibility report's headline K_c")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--out-dir", default=None,
                        help="default: analysis/<effects-dir name>_<system>_kc<K>")
    parser.add_argument("--n-permutations", type=int, default=10000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    k_c = args.k_c or headline_k_c(cfg)[0]
    name = f"{Path(args.effects_dir).name}_{args.system}_kc{k_c}"
    out_dir = Path(args.out_dir or Path(cfg.paths.get("analysis", "analysis")) / name)
    start = time.perf_counter()
    result = run(args.effects_dir, args.system, k_c, cfg, out_dir, args.n_permutations)

    write_manifest(out_dir / "analysis_manifest.json", run_manifest(
        cfg.fingerprint(), owner="shubham", extra={
            "effects_dir": str(args.effects_dir), "system": args.system, "k_c": k_c,
            "tau_query": cfg.tau_query, "tau_document": cfg.tau_document,
            "label_counts": result["label_counts"], "additivity": result["additivity"],
            "files": sorted(p.name for p in out_dir.iterdir()),
            "seconds": round(time.perf_counter() - start, 1),
        }))
    logger.info("query labels: %s", result["label_counts"])
    logger.info("additivity: %s", result["additivity"])
    logger.info("H7:\n%s", result["tables"]["h7"].to_string(index=False))
    logger.info("written to %s", out_dir)


if __name__ == "__main__":
    main()
