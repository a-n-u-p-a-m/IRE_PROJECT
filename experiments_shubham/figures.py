"""
figures.py - The report and video figures, one command each.

    python -m experiments_shubham.figures                       # all five
    python -m experiments_shubham.figures --only cost_fidelity  # one

| name | what it shows | reads |
|---|---|---|
| effect_distributions | nonzero effects by stratum, with stratum sizes, bm25 vs dpr | effects/real (judged rows) |
| utility_vs_t | mean Delta DCG@10 against t per policy, 95% bootstrap intervals | analysis/real_<s>_kc100/intervals.csv |
| h7 | Delta for gate- vs rerank-sensitive queries per policy and t, group sizes and p | analysis/real_<s>_kc100/{diagnosis,budget_outcomes,h7}.csv |
| cost_fidelity | gate loss recovered and G_q Spearman against extra Stage-2 calls per query | analysis/prediction/query_level.csv |
| gating_examples | the three pre-selected gating failures with passage text | analysis/gating_examples/{examples.csv,passages.tsv} |

Every figure is written to `figures/<name>.png` with the exact numbers it plots
in `figures/data/<name>.csv`, so a figure can be checked, or restyled, without
rerunning anything. Headline setting throughout: K_c = 100, DCG@10, uncondensed.

Colour does one job: identity. Systems are blue (bm25) and orange (dpr);
policies are blue (alternative), orange (boundary), aqua (random), each series
also labelled directly, since aqua is below 3:1 contrast on white.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shared.config import load_config  # noqa: E402
from shared.metrics import bootstrap_ci  # noqa: E402
from shared.provenance import run_manifest, write_manifest  # noqa: E402

logger = logging.getLogger(__name__)

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#ffffff"
SLOT = ("#2a78d6", "#eb6834", "#1baf7a")
SYSTEM_COLOR = {"bm25": SLOT[0], "dpr": SLOT[1]}
POLICY_COLOR = {"alternative": SLOT[0], "boundary": SLOT[1], "random": SLOT[2]}
SYSTEMS = ("bm25", "dpr")
POLICIES = ("alternative", "boundary", "random")
NAMES = ("effect_distributions", "utility_vs_t", "h7", "cost_fidelity", "gating_examples")


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "lines.linewidth": 2,
        "legend.frameon": False, "legend.fontsize": 8.5,
    })


def _direct_labels(ax, items, min_gap_pt: float = 10.0, dx_pt: float = 5.0) -> None:
    """
    Label line ends directly, nudging labels apart where lines converge.

    `items` is [(x, y, text)]. Call after the layout is final: the gap is
    measured in display space.
    """
    fig = ax.figure
    fig.canvas.draw()
    pts = sorted(((x, y, t, ax.transData.transform((x, y))[1]) for x, y, t in items),
                 key=lambda p: p[3])
    gap = min_gap_pt * fig.dpi / 72
    placed: list[float] = []
    for p in pts:
        placed.append(p[3] if not placed else max(p[3], placed[-1] + gap))
    shift = np.mean([p[3] for p in pts]) - np.mean(placed)
    for (x, y, text, py), ty in zip(pts, placed):
        ax.annotate(text, (x, y), xytext=(dx_pt, (ty + shift - py) * 72 / fig.dpi),
                    textcoords="offset points", va="center", fontsize=8, color=INK)


def _figure_legend(fig, handles, labels, ncol: int) -> None:
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=ncol,
               handlelength=1.6, columnspacing=1.4)


def _save(fig, out: Path, name: str, data: pd.DataFrame) -> dict:
    (out / "data").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    data.to_csv(out / "data" / f"{name}.csv", index=False)
    logger.info("figure %s", out / f"{name}.png")
    return {"png": f"{name}.png", "data": f"data/{name}.csv", "rows": int(len(data))}


# ──────────────────────────────────────────────────────────────────────
# 1. Effect distributions by stratum
# ──────────────────────────────────────────────────────────────────────

PANELS = (("e_rerank", None, "survivor", "E_rerank · survivors"),
          ("e_gate", None, "survivor", "E_gate · survivors"),
          ("e_gate", None, "non_survivor", "E_gate · non-survivors"),
          ("delta_t", "alternative", "retained", "Δ_t · retained"),
          ("delta_t", "alternative", "removed", "Δ_t · removed"),
          ("delta_t", "alternative", "selected_outsider", "Δ_t · selected outsider"))


def _effect_rows(effects_dir: Path, system: str, k_c: int) -> dict[tuple, pd.DataFrame]:
    cols = ["query_id", "effect_type", "stratum", "t", "effect", "judged"]
    stage = pd.read_parquet(effects_dir / f"{system}_stage_kc{k_c}.parquet", columns=cols)
    alt = pd.read_parquet(effects_dir / f"{system}_budget_kc{k_c}_alternative.parquet",
                          columns=cols, filters=[("t", "==", 0.5)])
    out = {}
    for effect_type, policy, stratum, _ in PANELS:
        src = alt if policy else stage
        out[(effect_type, policy, stratum)] = src[(src["effect_type"] == effect_type)
                                                  & (src["stratum"] == stratum)]
    return out


def effect_distributions(cfg, k_c: int, effects_dir: Path, analysis_dir: Path):
    rows, nonzero = [], {}
    for system in SYSTEMS:
        for key, blk in _effect_rows(effects_dir, system, k_c).items():
            judged = blk[blk["judged"].fillna(False).astype(bool)]
            nz = judged.loc[judged["effect"].abs() > 1e-9, "effect"].to_numpy(float)
            nonzero[(system, *key)] = nz
            q = np.quantile(nz, [0.05, 0.25, 0.5, 0.75, 0.95]) if len(nz) else [np.nan] * 5
            rows.append({"system": system, "effect_type": key[0], "policy": key[1],
                         "stratum": key[2], "n_documents": len(blk), "n_judged": len(judged),
                         "n_nonzero": len(nz),
                         "share_nonzero_of_judged": len(nz) / len(judged) if len(judged) else np.nan,
                         "q05": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q95": q[4]})
    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4))
    for ax, (effect_type, policy, stratum, title) in zip(axes.flat, PANELS):
        vals = [nonzero[(s, effect_type, policy, stratum)] for s in SYSTEMS]
        bp = ax.boxplot(vals, vert=False, widths=0.55, patch_artist=True, showfliers=True,
                        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.35,
                                    "markeredgewidth": 0},
                        medianprops={"color": INK, "linewidth": 1.2})
        for patch, flier, s in zip(bp["boxes"], bp["fliers"], SYSTEMS):
            patch.set_facecolor(SYSTEM_COLOR[s])
            patch.set_alpha(0.75)
            patch.set_edgecolor(SURFACE)
            flier.set_markerfacecolor(SYSTEM_COLOR[s])
        sub = data[(data["effect_type"] == effect_type) & (data["stratum"] == stratum)
                   & (data["policy"].fillna("") == (policy or ""))].set_index("system")
        ax.set_yticks([1, 2], [f"{s.upper()}\n{int(sub.loc[s, 'n_nonzero']):,} of "
                               f"{int(sub.loc[s, 'n_judged']):,}" for s in SYSTEMS],
                      fontsize=7.5)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    fig.supxlabel("effect on DCG@10 of each nonzero judged document "
                  "(tick labels: nonzero of judged documents in the stratum)", fontsize=8.5)
    fig.suptitle(f"Effects sit in a few documents (K_c = {k_c}, uncondensed; "
                 "Δ_t: alternative policy, t = 0.5)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    return fig, data


# ──────────────────────────────────────────────────────────────────────
# 2. Utility against t
# ──────────────────────────────────────────────────────────────────────

def utility_vs_t(cfg, k_c: int, effects_dir: Path, analysis_dir: Path):
    rows = []
    for system in SYSTEMS:
        ci = pd.read_csv(analysis_dir / f"real_{system}_kc{k_c}" / "intervals.csv")
        ci = ci[(~ci["condensed"]) & (ci["quantity"] == "delta_mean")]
        for policy in POLICIES:
            rows.append({"system": system, "policy": policy, "t": 1.0, "delta": 0.0,
                         "lo": 0.0, "hi": 0.0})
            for _, r in ci[ci["policy"] == policy].iterrows():
                rows.append({"system": system, "policy": policy, "t": r["t"],
                             "delta": r["point"], "lo": r["lo"], "hi": r["hi"]})
    data = pd.DataFrame(rows).sort_values(["system", "policy", "t"], ascending=[True, True, False])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    handles = []
    for ax, system in zip(axes, SYSTEMS):
        for policy in POLICIES:
            d = data[(data["system"] == system) & (data["policy"] == policy)]
            c = POLICY_COLOR[policy]
            ax.fill_between(d["t"], d["lo"], d["hi"], color=c, alpha=0.14, linewidth=0)
            (h,) = ax.plot(d["t"], d["delta"], color=c, marker="o", markersize=4.5,
                           markeredgecolor=SURFACE, markeredgewidth=1, label=policy)
            if system == SYSTEMS[0]:
                handles.append(h)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_xlim(1.02, 0.36)
        ax.set_xticks([1.0, 0.9, 0.75, 0.5])
        ax.set_xlabel("t (share of the Stage-2 budget kept for C_q)")
        ax.set_title(system.upper(), loc="left")
    axes[0].set_ylabel("Δ DCG@10 vs t = 1")
    fig.suptitle("Moving budget to the other retriever's outsiders helps BM25 only",
                 x=0.01, ha="left", fontsize=10.5)
    _figure_legend(fig, handles, [f"{p} (95% CI band)" for p in POLICIES], ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.5)
    for ax, system in zip(axes, SYSTEMS):
        ends = data[(data["system"] == system) & (data["t"] == 0.5)]
        _direct_labels(ax, [(0.5, r["delta"], r["policy"]) for _, r in ends.iterrows()])
    return fig, data


# ──────────────────────────────────────────────────────────────────────
# 3. H7
# ──────────────────────────────────────────────────────────────────────

GROUPS = (("gate_sensitive", "gate-sensitive", SLOT[0]),
          ("rerank_sensitive", "rerank-sensitive", SLOT[1]))


def h7(cfg, k_c: int, effects_dir: Path, analysis_dir: Path):
    rows = []
    for system in SYSTEMS:
        base = analysis_dir / f"real_{system}_kc{k_c}"
        diag = pd.read_csv(base / "diagnosis.csv", dtype={"query_id": str})
        diag = diag[~diag["condensed"]]
        out = pd.read_csv(base / "budget_outcomes.csv", dtype={"query_id": str})
        out = out[~out["condensed"]].merge(diag[["query_id", "label"]], on="query_id")
        tests = pd.read_csv(base / "h7.csv")
        tests = tests[~tests["condensed"]].set_index(["policy", "t"])
        for (policy, t), blk in out.groupby(["policy", "t"]):
            for label, _, _ in GROUPS:
                g = blk[blk["label"] == label]
                ci = bootstrap_ci(g, lambda x: float(x["delta_dcg"].mean()),
                                  cfg.bootstrap_iterations, cfg.bootstrap_alpha, cfg.seed)
                rows.append({"system": system, "policy": policy, "t": t, "group": label,
                             "n_queries": len(g), "mean_delta": ci["point"], "lo": ci["lo"],
                             "hi": ci["hi"], "group_p": tests.loc[(policy, t), "group_p"],
                             "spearman_rho": tests.loc[(policy, t), "spearman_rho"]})
    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0), sharex=True)
    handles = []
    for i, system in enumerate(SYSTEMS):
        for j, policy in enumerate(POLICIES):
            ax = axes[i, j]
            d = data[(data["system"] == system) & (data["policy"] == policy)]
            for k, (label, _, color) in enumerate(GROUPS):
                g = d[d["group"] == label].sort_values("t")
                x = g["t"].to_numpy() + (0.012 if k else -0.012)
                h = ax.errorbar(x, g["mean_delta"], yerr=[g["mean_delta"] - g["lo"],
                                                          g["hi"] - g["mean_delta"]],
                                color=color, marker="o", markersize=4.5, linewidth=1.6,
                                capsize=0, elinewidth=1.2, markeredgecolor=SURFACE)
                if i == 0 and j == 0:
                    handles.append(h)
            for t, r in d.drop_duplicates("t").set_index("t").iterrows():
                p = r["group_p"]
                ptxt = "p<0.001" if p < 0.001 else f"p={p:.2f}" if p >= 0.01 else f"p={p:.3f}"
                ax.annotate(ptxt, (t, 1.0), xycoords=("data", "axes fraction"), ha="center",
                            va="bottom", xytext=(0, 2), textcoords="offset points",
                            fontsize=6.8, color=INK_2)
            ax.axhline(0, color=INK_2, linewidth=0.8)
            ax.set_xlim(0.97, 0.43)
            ax.set_xticks([0.9, 0.75, 0.5])
            ax.set_title(f"{system.upper()} · {policy}", loc="left", pad=13)
            if j == 0:
                n = d.drop_duplicates("group").set_index("group")["n_queries"]
                ax.set_ylabel(f"mean Δ DCG@10\n(n = {n['gate_sensitive']} vs "
                              f"{n['rerank_sensitive']})")
            if i == 1:
                ax.set_xlabel("t")
    fig.suptitle("H7: gate-sensitive queries gain more from outsiders than rerank-sensitive ones",
                 x=0.01, ha="left", fontsize=10.5)
    _figure_legend(fig, handles, ["gate-sensitive queries", "rerank-sensitive queries",
                                  ], ncol=2)
    fig.text(0.99, 0.955, "95% CIs; group-test p above each t", ha="right", va="top",
             fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, data


# ──────────────────────────────────────────────────────────────────────
# 4. Cost-fidelity
# ──────────────────────────────────────────────────────────────────────

def cost_fidelity(cfg, k_c: int, effects_dir: Path, analysis_dir: Path):
    q = pd.read_csv(analysis_dir / "prediction" / "query_level.csv")
    probes = q[(q["scope"] == "all") & q["approximation"].isin(POLICIES)]
    data = (probes.groupby(["system", "approximation", "budget"])
            [["extra_calls_per_query", "gq_spearman", "loss_recovered_share", "gs_f1"]]
            .mean().reset_index().rename(columns={"approximation": "policy"}))
    pred = q[(q["scope"] == "test") & q["approximation"].isin(["ridge", "gbm"])]
    best = pred.groupby("system")["gq_spearman"].max()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex=True)
    rows = (("loss_recovered_share", "gate loss recovered"),
            ("gq_spearman", "ρ, probed vs exact G_q"))
    handles, ends = [], {}
    for j, system in enumerate(SYSTEMS):
        for i, (col, ylabel) in enumerate(rows):
            ax = axes[i, j]
            for policy in POLICIES:
                d = data[(data["system"] == system) & (data["policy"] == policy)].sort_values("budget")
                d = d[d["budget"] > 0] if col == "gq_spearman" else d
                (h,) = ax.plot(d["budget"], d[col], color=POLICY_COLOR[policy], marker="o",
                               markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1)
                if i == 0 and j == 0:
                    handles.append(h)
                ends.setdefault((i, j), []).append((d["budget"].iloc[-1], d[col].iloc[-1], policy))
            if col == "gq_spearman" and system in best:
                ax.axhline(best[system], color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
                ax.annotate(f"best predictor, 0 calls (test queries): {best[system]:.2f}",
                            (0.02, best[system]), xycoords=("axes fraction", "data"),
                            xytext=(0, -4), textcoords="offset points", ha="left", va="top",
                            fontsize=7, color=INK_2)
            ax.set_xscale("symlog", linthresh=20)
            ax.set_xticks([0, 20, 50, 100, 500], ["0", "20", "50", "100", "500"])
            ax.set_xlim(-2, 1400)
            ax.set_ylim(-0.05 if col == "loss_recovered_share" else -0.25, 1.05)
            if i == 0:
                ax.set_title(f"{system.upper()} · all 97 queries", loc="left")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xlabel("extra Stage-2 calls per query (exact: ~8,900)")
    fig.suptitle("Twenty calls on the other retriever's outsiders recover most of BM25's gate loss",
                 x=0.01, ha="left", fontsize=10.5)
    _figure_legend(fig, handles, [f"{p} probe" for p in POLICIES], ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3.0)
    for (i, j), items in ends.items():
        _direct_labels(axes[i, j], items)
    return fig, data


# ──────────────────────────────────────────────────────────────────────
# 5. Gating-failure examples
# ──────────────────────────────────────────────────────────────────────

def _passages(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r[0]: r[1] for r in csv.reader(fh, delimiter="\t") if len(r) >= 2}


def _clip(text: str, n: int = 230) -> str:
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + " …"


def gating_examples(cfg, k_c: int, effects_dir: Path, analysis_dir: Path):
    base = analysis_dir / "gating_examples"
    ex = pd.read_csv(base / "examples.csv", dtype={"query_id": str, "doc_id": str,
                                                   "actual_top_doc_id": str})
    text = _passages(base / "passages.tsv")
    data = ex.assign(excluded_text=[_clip(text[d]) for d in ex["doc_id"]],
                     top_text=[_clip(text[d]) for d in ex["actual_top_doc_id"]])
    fig, axes = plt.subplots(len(data), 1, figsize=(7.2, 1.75 * len(data) + 0.5))
    for ax, (_, r) in zip(np.atleast_1d(axes), data.iterrows()):
        ax.axis("off")
        where = (f"BM25 rank {int(r['stage1_rank']):,}" if r["stage1_retrieved"]
                 else f"not in BM25's top {int(r['depth_m']):,}")
        ax.text(0, 1, f"“{r['query']}”", fontsize=10, fontweight="bold", va="top",
                transform=ax.transAxes)
        ax.text(0, 0.84, f"Excluded by the gate · {where} · grade {int(r['grade'])} · "
                         f"would rank #{int(r['rank_without_gate'])} without the gate",
                color=SLOT[1], fontsize=8.2, va="top", transform=ax.transAxes)
        ax.text(0, 0.70, textwrap.fill(r["excluded_text"], 118), fontsize=7.8, va="top",
                transform=ax.transAxes, color=INK)
        ax.text(0, 0.40, f"Ranked first by the cascade · grade {int(r['actual_top_grade'])}",
                color=SLOT[0], fontsize=8.2, va="top", transform=ax.transAxes)
        ax.text(0, 0.26, textwrap.fill(r["top_text"], 118), fontsize=7.8, va="top",
                transform=ax.transAxes, color=INK)
    fig.suptitle(f"Three gating failures, chosen by a rule fixed before the run "
                 f"(BM25, K_c = {k_c})", x=0.01, ha="left", fontsize=10.5)
    fig.text(0.01, 0.005, "Grades: TREC-DL qrels (3 = perfectly relevant). Passages: MS MARCO "
             "v1, mojibake repaired with ftfy for display.", fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    # The figure shows short excerpts; the committed data file keeps ids, ranks
    # and grades only, so the repository still carries no passage text.
    return fig, data.drop(columns=["excluded_text", "top_text"])


FIGURES = {"effect_distributions": effect_distributions, "utility_vs_t": utility_vs_t,
           "h7": h7, "cost_fidelity": cost_fidelity, "gating_examples": gating_examples}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Report and video figures")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--only", choices=NAMES, nargs="+", default=None)
    parser.add_argument("--k-c", type=int, default=None)
    parser.add_argument("--effects-dir", default="effects/real")
    parser.add_argument("--analysis-dir", default="analysis")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    from experiments_shubham.power_h7 import headline_k_c

    cfg = load_config(args.config)
    k_c = args.k_c or headline_k_c(cfg)[0]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _style()
    manifest_path = out / "figures_manifest.json"
    done = json.loads(manifest_path.read_text()).get("figures", {}) if manifest_path.exists() else {}
    for name in args.only or NAMES:
        fig, data = FIGURES[name](cfg, k_c, Path(args.effects_dir), Path(args.analysis_dir))
        done[name] = {**_save(fig, out, name, data),
                      "command": f"python -m experiments_shubham.figures --only {name}"}
    write_manifest(manifest_path, run_manifest(cfg.fingerprint(), owner="shubham", extra={
        "k_c": k_c, "effects_dir": args.effects_dir, "analysis_dir": args.analysis_dir,
        "figures": dict(sorted(done.items()))}))


if __name__ == "__main__":
    main()
