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
    unjudged_at_k   per (query, world A/B/C): the unjudged share of the top-k,
                    which decides how far the uncondensed E_gate can be trusted
    contrast        with --contrast-with: is a label (gate_sensitive by default)
                    more frequent under one Stage-1 system than the other on the
                    same queries - exact McNemar, the queries being paired
    intervals       percentile bootstrap over QUERIES (shared.metrics.bootstrap_ci,
                    predictors.bootstrap_iterations / bootstrap_alpha) for the
                    headline quantities: mean DCG, G_q and R_q, the label shares,
                    and per (policy, t) the mean Delta DCG and both H7 statistics;
                    with --contrast-with, the same for system a minus system b,
                    resampling query PAIRS

Each of these runs once per ranking mode, reading that mode's own files, and
every output row carries `condensed`. False is the primary estimand; True is
the robustness check reported beside it (freeze v2). The modes are never
pooled: `diagnose` and `budget_outcomes` refuse a frame that mixes them.

    python -m experiments_shubham.engine --regime weak_retriever
    python -m experiments_shubham.analysis --effects-dir effects/synthetic/weak_retriever \\
        --system synthetic
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from experiments_shubham.engine import MODES, table_name
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
from shared.metrics import bootstrap_ci
from shared.provenance import run_manifest, write_manifest
from shared.schema import E_GATE, E_RERANK, load, single_mode

logger = logging.getLogger(__name__)

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def load_stage(effects_dir: str | Path, system: str, k_c: int,
               condensed: bool = False) -> pd.DataFrame:
    name = table_name(system, k_c, None, condensed)
    return load(Path(effects_dir) / f"{name}.parquet", "effect")


def load_budget(effects_dir: str | Path, system: str, k_c: int, policies,
                condensed: bool = False) -> dict[str, pd.DataFrame]:
    return {p: load(Path(effects_dir) / f"{table_name(system, k_c, p, condensed)}.parquet",
                    "effect")
            for p in policies}


# ──────────────────────────────────────────────────────────────────────
# Distributions
# ──────────────────────────────────────────────────────────────────────

def distributions(effects: pd.DataFrame, judged_only: bool = False) -> pd.DataFrame:
    """Per (effect type, K_c, t, policy, stratum): always with its size."""
    eff = effects[effects["judged"].fillna(False)] if judged_only else effects
    keys = ["condensed", "effect_type", "k_c", "t", "policy", "stratum"]
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


#: (world, effect type, rank column): where each world's ranking lives in a
#: stage table. World C is E_gate's control; A and B are E_rerank's two sides.
WORLD_RANKS = (("A", E_RERANK, "rank_treatment"), ("B", E_RERANK, "rank_control"),
               ("C", E_GATE, "rank_control"))


def unjudged_at_k(stage: pd.DataFrame) -> pd.DataFrame:
    """
    Per (query, world): how much of the top-k is unjudged.

    This is the measurement that decides how much the uncondensed E_gate is
    worth. World C ranks all of U_q, which is ~1.65% judged, so unjudged
    documents can displace judged survivors out of its top-k and drag E_gate
    towards DCG(pi_act) as an artefact of pooling. A small share makes the
    concern a footnote; a large one makes the condensed figure the headline.
    The condensed table gives 0 here by construction.
    """
    single_mode(stage, "unjudged_at_k")
    rows = []
    for world, effect_type, col in WORLD_RANKS:
        blk = stage[(stage["effect_type"] == effect_type)
                    & stage[col].notna() & (stage[col] <= stage["k"])]
        judged = blk["judged"].fillna(False).astype(bool)
        per = (blk.assign(unjudged=~judged)
               .groupby(["query_id", "k_c", "k"], observed=True)["unjudged"]
               .agg(n_top="size", n_unjudged="sum").reset_index())
        per.insert(1, "world", world)
        rows.append(per)
    out = pd.concat(rows, ignore_index=True)
    out["share_unjudged"] = out["n_unjudged"] / out["n_top"]
    return out


# ──────────────────────────────────────────────────────────────────────
# Query diagnosis and H7
# ──────────────────────────────────────────────────────────────────────

def diagnose(stage: pd.DataFrame, tau_query: float) -> pd.DataFrame:
    """
    The frozen labelling rule of shared.diagnosis.query_diagnosis, from the
    effect table: summing a per-document effect over the universe gives the
    system-level DCG difference, because DCG@k is a sum of document utilities.
    """
    single_mode(stage, "diagnose")
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
        single_mode(budget, f"budget_outcomes[{policy}]")
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


def label_contrast(diag_a: pd.DataFrame, diag_b: pd.DataFrame,
                   label: str = GATE_SENSITIVE) -> dict:
    """
    Is `label` more frequent under system a than under b, on the same queries?

    Exact McNemar. Both systems are diagnosed on the same queries, so the unit
    is the query pair and only the discordant queries carry information.
    Fisher's test on the two counts would treat them as independent samples.
    """
    single_mode(diag_a, "label_contrast")
    single_mode(diag_b, "label_contrast")
    a = diag_a.set_index("query_id")["label"].eq(label)
    b = diag_b.set_index("query_id")["label"].eq(label)
    if set(a.index) != set(b.index):
        raise ValueError("a paired contrast needs both systems on the same queries")
    b = b.reindex(a.index)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    return {
        "label": label, "n_queries": len(a), "n_a": int(a.sum()), "n_b": int(b.sum()),
        "both": int((a & b).sum()), "only_a": only_a, "only_b": only_b,
        "p_one_sided": binomtest(only_a, n, 0.5, alternative="greater").pvalue if n
        else float("nan"),
        "p_two_sided": binomtest(only_a, n, 0.5).pvalue if n else float("nan"),
    }


def contrast(effects_dir: str | Path, system: str, other: str, k_c: int,
             cfg: FrozenConfig, modes: Sequence[bool] = MODES) -> pd.DataFrame:
    """`label_contrast` of `system` against `other` at one K_c, per ranking mode."""
    rows = []
    for condensed in modes:
        diags = [diagnose(load_stage(effects_dir, s, k_c, condensed), cfg.tau_query)
                 for s in (system, other)]
        rows.append({"condensed": condensed, "system_a": system, "system_b": other,
                     "k_c": k_c, **label_contrast(*diags)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Bootstrap intervals
# ──────────────────────────────────────────────────────────────────────

def _delta_column(policy: str, t: float) -> str:
    return f"delta_{policy}_t{t:g}"


def query_table(diagnosis: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """One row per query: its diagnosis, and Delta DCG per (policy, t) as columns."""
    single_mode(diagnosis, "query_table")
    wide = outcomes.pivot_table(index="query_id", columns=["policy", "t"],
                                values="delta_dcg", aggfunc="first")
    wide.columns = [_delta_column(p, t) for p, t in wide.columns]
    return diagnosis.merge(wide.reset_index(), on="query_id", how="left",
                           validate="one_to_one")


def _mean(col: str):
    return lambda df: float(df[col].mean())


def _share(col: str, value):
    return lambda df: float((df[col] == value).mean())


def _group_difference(col: str):
    """h7_test's statistic: mean Delta of gate- minus rerank-sensitive queries."""
    def stat(df: pd.DataFrame) -> float:
        gate = df.loc[df["label"] == GATE_SENSITIVE, col]
        rer = df.loc[df["label"] == RERANK_SENSITIVE, col]
        if len(gate) < 2 or len(rer) < 2:     # h7_test's own testability guard
            return float("nan")
        return float(gate.mean() - rer.mean())
    return stat


def _spearman(col: str):
    def stat(df: pd.DataFrame) -> float:
        if df["g_q"].nunique() < 2 or df[col].nunique() < 2:
            return float("nan")
        return float(spearmanr(df["g_q"], df[col]).statistic)
    return stat


def _intervals(table: pd.DataFrame, specs, n_iter: int, alpha: float,
               seed: int) -> pd.DataFrame:
    """
    `bootstrap_ci` for each (quantity, policy, t, statistic) in `specs`.

    Every statistic is drawn with the same seed, so all intervals in one call
    come from the same resamples of queries. A statistic that is undefined on
    the full data (one query in an H7 group, say) gets no interval, even though
    some resamples could define it. `n_draws` below `n_iter` means resamples
    that left a group untestable were dropped, and the interval is conditional
    on the group being testable: read it with `n_draws`.
    """
    rows = []
    for quantity, policy, t, stat in specs:
        ci = bootstrap_ci(table, stat, n_iter=n_iter, alpha=alpha, seed=seed)
        defined = np.isfinite(ci["point"])
        rows.append({
            "quantity": quantity, "policy": policy, "t": t, "point": ci["point"],
            "lo": ci["lo"] if defined else float("nan"),
            "hi": ci["hi"] if defined else float("nan"),
            "level": 1 - alpha, "n_queries": ci["n"], "n_draws": ci.get("n_draws", 0),
        })
    return pd.DataFrame(rows)


def intervals(diagnosis: pd.DataFrame, outcomes: pd.DataFrame, n_iter: int,
              alpha: float, seed: int) -> pd.DataFrame:
    """
    Percentile bootstrap intervals, resampling queries, for the headline numbers.

    Per system: mean DCG@k of the cascade, mean G_q, R_q and signed R_q, the
    shares of gate-sensitive queries and of queries the reranker harms; per
    (policy, t): mean Delta DCG, the H7 group difference and Spearman rho.
    Each point estimate is the number the rest of the analysis reports.
    """
    table = query_table(diagnosis, outcomes)
    specs = [("dcg_actual_mean", None, None, _mean("dcg_actual")),
             ("g_q_mean", None, None, _mean("g_q")),
             ("r_q_mean", None, None, _mean("r_q")),
             ("rerank_signed_mean", None, None, _mean("rerank_signed")),
             ("share_gate_sensitive", None, None, _share("label", GATE_SENSITIVE)),
             ("share_reranker_harms", None, None, _share("reranker_harms", True))]
    for (policy, t), _ in outcomes.groupby(["policy", "t"]):
        if float(t) == 1.0:                   # the baseline: Delta is 0 by definition
            continue
        col = _delta_column(policy, t)
        specs += [("delta_mean", policy, float(t), _mean(col)),
                  ("h7_group_difference", policy, float(t), _group_difference(col)),
                  ("h7_spearman_rho", policy, float(t), _spearman(col))]
    return _intervals(table, specs, n_iter, alpha, seed)


def paired_intervals(diag_a: pd.DataFrame, diag_b: pd.DataFrame, n_iter: int,
                     alpha: float, seed: int) -> pd.DataFrame:
    """
    System a minus system b on the same queries, resampling query PAIRS.

    Resampling each system's queries separately would ignore the pairing, the
    same mistake Fisher's test makes against McNemar's.
    """
    single_mode(diag_a, "paired_intervals")
    single_mode(diag_b, "paired_intervals")
    a, b = diag_a.set_index("query_id"), diag_b.set_index("query_id")
    if set(a.index) != set(b.index):
        raise ValueError("a paired interval needs both systems on the same queries")
    b = b.reindex(a.index)
    pair = pd.DataFrame({"query_id": a.index.astype(str)})
    for col in ("dcg_actual", "g_q", "r_q"):
        pair[f"{col}_diff"] = a[col].to_numpy(float) - b[col].to_numpy(float)
    pair["gate_sensitive_diff"] = (a["label"].eq(GATE_SENSITIVE).to_numpy(int)
                                   - b["label"].eq(GATE_SENSITIVE).to_numpy(int))
    pair["reranker_harms_diff"] = (a["reranker_harms"].to_numpy(int)
                                   - b["reranker_harms"].to_numpy(int))
    specs = [("dcg_actual_mean_diff", None, None, _mean("dcg_actual_diff")),
             ("g_q_mean_diff", None, None, _mean("g_q_diff")),
             ("r_q_mean_diff", None, None, _mean("r_q_diff")),
             ("share_gate_sensitive_diff", None, None, _mean("gate_sensitive_diff")),
             ("share_reranker_harms_diff", None, None, _mean("reranker_harms_diff"))]
    return _intervals(pair, specs, n_iter, alpha, seed)


def contrast_intervals(effects_dir: str | Path, system: str, other: str, k_c: int,
                       cfg: FrozenConfig, n_iter: int | None = None,
                       modes: Sequence[bool] = MODES) -> pd.DataFrame:
    """`paired_intervals` of `system` minus `other` at one K_c, per ranking mode."""
    n_iter = n_iter or cfg.bootstrap_iterations
    frames = []
    for condensed in modes:
        diags = [diagnose(load_stage(effects_dir, s, k_c, condensed), cfg.tau_query)
                 for s in (system, other)]
        out = paired_intervals(*diags, n_iter, cfg.bootstrap_alpha, cfg.seed)
        out.insert(0, "condensed", condensed)
        out.insert(1, "system_a", system)
        out.insert(2, "system_b", other)
        out.insert(3, "k_c", k_c)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


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

def _mode_name(condensed: bool) -> str:
    return "condensed" if condensed else "uncondensed"


def run(effects_dir: str | Path, system: str, k_c: int, cfg: FrozenConfig,
        out_dir: str | Path, n_permutations: int = 10000,
        modes: Sequence[bool] = MODES, n_bootstrap: int | None = None) -> dict:
    n_bootstrap = n_bootstrap or cfg.bootstrap_iterations
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: dict[str, list[pd.DataFrame]] = {}
    additivity, label_counts = {}, {}
    for condensed in modes:
        stage = load_stage(effects_dir, system, k_c, condensed)
        budgets = load_budget(effects_dir, system, k_c, cfg.outsider_policies, condensed)

        diag = diagnose(stage, cfg.tau_query)
        tables = {
            "distributions": pd.concat(
                [distributions(e, judged) for e in [stage, *budgets.values()]
                 for judged in (False, True)], ignore_index=True),
            "diagnosis": diag,
            "budget_outcomes": budget_outcomes(budgets),
            "labels": label_summary(stage, cfg.tau_document),
            "gating_failures": gating_failure_examples(stage),
            "unjudged_at_k": unjudged_at_k(stage),
        }
        tables["h7"] = h7(diag, tables["budget_outcomes"], n_permutations, cfg.seed)
        tables["intervals"] = intervals(diag, tables["budget_outcomes"], n_bootstrap,
                                        cfg.bootstrap_alpha, cfg.seed)
        for name, df in tables.items():
            df = df.drop(columns="condensed", errors="ignore")
            df.insert(0, "condensed", condensed)
            parts.setdefault(name, []).append(df)

        mode = _mode_name(condensed)
        suffix = "_condensed" if condensed else ""
        additivity[mode] = additivity_figure(
            stage, out_dir / f"additivity{suffix}.png",
            f"{system}, K_c = {k_c}, {mode}: E_rerank + E_gate vs U_d(π_act)")
        label_counts[mode] = diag["label"].value_counts().to_dict()

    tables = {name: pd.concat(dfs, ignore_index=True) for name, dfs in parts.items()}
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return {"tables": tables, "additivity": additivity, "label_counts": label_counts,
            "n_bootstrap": n_bootstrap}


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
    parser.add_argument("--n-bootstrap", type=int, default=None,
                        help="default: predictors.bootstrap_iterations in the config")
    parser.add_argument("--contrast-with", default=None, metavar="SYSTEM",
                        help="also test whether gate-sensitive queries are more frequent "
                             "under --system than under SYSTEM (paired, exact McNemar), "
                             "with paired bootstrap intervals for the differences")
    args = parser.parse_args()

    cfg = load_config(args.config)
    k_c = args.k_c or headline_k_c(cfg)[0]
    name = f"{Path(args.effects_dir).name}_{args.system}_kc{k_c}"
    out_dir = Path(args.out_dir or Path(cfg.paths.get("analysis", "analysis")) / name)
    start = time.perf_counter()
    result = run(args.effects_dir, args.system, k_c, cfg, out_dir, args.n_permutations,
                 n_bootstrap=args.n_bootstrap)
    if args.contrast_with:
        paired = contrast(args.effects_dir, args.system, args.contrast_with, k_c, cfg)
        paired.to_csv(out_dir / f"contrast_vs_{args.contrast_with}.csv", index=False)
        logger.info("contrast:\n%s", paired.to_string(index=False))
        diffs = contrast_intervals(args.effects_dir, args.system, args.contrast_with, k_c,
                                   cfg, result["n_bootstrap"])
        diffs.to_csv(out_dir / f"contrast_intervals_vs_{args.contrast_with}.csv", index=False)
        logger.info("paired intervals:\n%s", diffs.to_string(index=False))

    write_manifest(out_dir / "analysis_manifest.json", run_manifest(
        cfg.fingerprint(), owner="shubham", extra={
            "effects_dir": str(args.effects_dir), "system": args.system, "k_c": k_c,
            "tau_query": cfg.tau_query, "tau_document": cfg.tau_document,
            "condensed_modes": list(MODES),
            "bootstrap": {"iterations": result["n_bootstrap"],
                          "alpha": cfg.bootstrap_alpha, "seed": cfg.seed,
                          "unit": "query", "method": "percentile"},
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
