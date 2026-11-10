"""
compare.py - BM25 against DPR, one axis changed at a time.

The Week-4 comparison. Every summary is computed per setting - (system, K_c,
utility, k, mode) - from the effect tables alone, then laid out so that each
block changes exactly one axis away from the headline and shows the two
retrievers side by side:

    headline     both systems at the headline K_c, the frozen utility and
                 cutoff, uncondensed
    k_c          the other frozen candidate depths
    utility      the sensitivity kinds (top-k, RBP) at the headline cutoff
    k            the frozen utility at the sensitivity cutoffs
    condensed    the robustness mode

Holding everything else fixed is what lets a difference between the columns be
read as a difference between the retrievers rather than between settings.

Query labels use the frozen tau_query, which was set on the DCG@10 scale, so
under another utility or cutoff they are descriptive only. H7 is not
recomputed here: it is pre-registered at the headline setting and lives in
analysis.py.

    python -m experiments_shubham.compare                 # effects/real*
    python -m experiments_shubham.compare --preview       # effects/real_preview*
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from experiments_shubham.analysis import budget_outcomes, diagnose
from experiments_shubham.engine import MODES, table_name
from experiments_shubham.power_h7 import headline_k_c
from experiments_shubham.real import effects_dir_name
from shared.config import FrozenConfig, load_config
from shared.diagnosis import GATE_SENSITIVE, MIXED_STABLE, RERANK_SENSITIVE
from shared.effects import gating_failures
from shared.provenance import run_manifest, write_manifest

logger = logging.getLogger(__name__)

AXES = ("k_c", "utility", "k", "condensed")
_STAGE_COLUMNS = ["query_id", "doc_id", "effect_type", "stratum", "condensed",
                  "utility_treatment", "utility_control", "effect"]
_BUDGET_COLUMNS = ["query_id", "t", "seed", "condensed", "effect"]


def frozen_settings(cfg: FrozenConfig) -> list[tuple[str, int]]:
    """(utility, k): the headline, then each utility kind and each cutoff alone."""
    head_u, head_k = cfg.utility_kind, cfg.final_cutoff_k
    return ([(head_u, head_k)] + [(u, head_k) for u in cfg.sensitivity_kinds]
            + [(head_u, k) for k in cfg.cutoff_sensitivity])


def summarise(stage: pd.DataFrame, budgets: Mapping[str, pd.DataFrame],
              tau_query: float) -> dict[str, float]:
    """The comparison's metrics for one (system, K_c, utility, k, mode)."""
    diag = diagnose(stage, tau_query)
    fails = gating_failures(stage)
    out = {
        "n_queries": len(diag),
        "utility_actual_mean": diag["dcg_actual"].mean(),
        "g_q_mean": diag["g_q"].mean(),
        "g_q_median": diag["g_q"].median(),
        "r_q_mean": diag["r_q"].mean(),
        "rerank_signed_mean": diag["rerank_signed"].mean(),
        "n_gate_sensitive": int((diag["label"] == GATE_SENSITIVE).sum()),
        "n_rerank_sensitive": int((diag["label"] == RERANK_SENSITIVE).sum()),
        "n_mixed_stable": int((diag["label"] == MIXED_STABLE).sum()),
        "n_reranker_harms": int(diag["reranker_harms"].sum()),
        "n_gating_failures": len(fails),
        "n_queries_with_gating_failure": fails["query_id"].nunique(),
    }
    if budgets:
        outcomes = budget_outcomes(budgets)
        for (policy, t), blk in outcomes.groupby(["policy", "t"]):
            out[f"delta_{policy}_t{t:g}"] = blk["delta_dcg"].mean()
    return out


def _read(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns))


def collect(
    dirs: Mapping[tuple[str, int], Path],
    systems: Sequence[str],
    cfg: FrozenConfig,
    modes: Sequence[bool] = MODES,
) -> pd.DataFrame:
    """
    One long row per (system, K_c, utility, k, mode, metric), for every table
    that exists. A sensitivity setting run at the headline K_c only simply
    contributes fewer rows.
    """
    rows = []
    for (utility, k), root in dirs.items():
        for system in systems:
            for k_c in cfg.candidate_depths:
                for condensed in modes:
                    path = root / f"{table_name(system, k_c, None, condensed)}.parquet"
                    if not path.exists():
                        continue
                    budgets = {}
                    for policy in cfg.outsider_policies:
                        bp = root / f"{table_name(system, k_c, policy, condensed)}.parquet"
                        if bp.exists():
                            budgets[policy] = _read(bp, _BUDGET_COLUMNS)
                    metrics = summarise(_read(path, _STAGE_COLUMNS), budgets, cfg.tau_query)
                    key = {"system": system, "k_c": k_c, "utility": utility, "k": k,
                           "condensed": condensed}
                    rows += [{**key, "metric": m, "value": float(v)}
                             for m, v in metrics.items()]
                    logger.info("%s %s@%d K_c=%d condensed=%s", system, utility, k,
                                k_c, condensed)
    return pd.DataFrame(rows)


def one_axis(long: pd.DataFrame, headline: Mapping[str, object],
             systems: Sequence[str]) -> pd.DataFrame:
    """
    The rows at most one axis away from `headline`, systems side by side.

    `varied` names the axis that changed ("headline" for none) and `setting`
    its value, so each block reads as one controlled comparison. The last
    column is the first system minus the second.
    """
    off = pd.DataFrame({a: long[a] != headline[a] for a in AXES})
    near = long[off.sum(axis=1) <= 1].copy()
    off = off.loc[near.index]
    near["varied"] = off.idxmax(axis=1).where(off.any(axis=1), "headline")
    near["setting"] = [str(row[v]) if v != "headline" else ""
                       for v, row in zip(near["varied"], near.to_dict("records"))]
    order = {"headline": 0, **{a: i + 1 for i, a in enumerate(AXES)}}
    wide = (near.pivot_table(index=["varied", "setting", *AXES, "metric"],
                             columns="system", values="value", aggfunc="first")
            .reindex(columns=list(systems)).reset_index())
    a, b = systems[:2]
    wide[f"{a}_minus_{b}"] = wide[a] - wide[b]
    wide = wide.sort_values(["varied", "setting", "metric"],
                            key=lambda s: s.map(order) if s.name == "varied" else s.astype(str))
    wide.columns.name = None
    return wide.reset_index(drop=True)


def run(effects_root: str | Path, cfg: FrozenConfig, out_dir: str | Path,
        preview: bool = False, k_c: int | None = None,
        systems: Iterable[str] | None = None) -> dict:
    systems = list(systems or cfg.stage1_systems)
    root = Path(effects_root)
    dirs = {s: root / effects_dir_name(cfg, *s, subset=preview) for s in frozen_settings(cfg)}
    found = {s: d for s, d in dirs.items() if d.exists()}
    missing = sorted(str(d) for s, d in dirs.items() if s not in found)
    if missing:
        logger.warning("settings not run yet, left out: %s", missing)
    long = collect(found, systems, cfg)
    if long.empty:
        raise SystemExit(f"no effect tables under {root} for {list(map(str, dirs.values()))}")
    headline = {"k_c": k_c or headline_k_c(cfg)[0], "utility": cfg.utility_kind,
                "k": cfg.final_cutoff_k, "condensed": False}
    wide = one_axis(long, headline, systems)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "compare_long.csv", index=False)
    wide.to_csv(out / "compare_one_axis.csv", index=False)
    write_manifest(out / "compare_manifest.json", run_manifest(
        cfg.fingerprint(), owner="shubham", extra={
            "effects_dirs": {f"{u}@{k}": str(d) for (u, k), d in found.items()},
            "missing_settings": missing, "systems": systems, "headline": headline,
            "tau_query": cfg.tau_query,
        }))
    return {"long": long, "one_axis": wide, "missing": missing}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="BM25 vs DPR, one axis changed at a time")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--effects-root", default=None, help="default: paths.effects")
    parser.add_argument("--preview", action="store_true",
                        help="read effects/real_preview* (one shard) instead of effects/real*")
    parser.add_argument("--k-c", type=int, default=None,
                        help="headline K_c; default: the feasibility report's")
    parser.add_argument("--out-dir", default=None,
                        help="default: <paths.analysis>/compare, or compare_preview")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.paths
    name = "compare_preview" if args.preview else "compare"
    out_dir = args.out_dir or Path(paths.get("analysis", "analysis")) / name
    result = run(args.effects_root or paths.get("effects", "effects"), cfg, out_dir,
                 args.preview, args.k_c)
    head = result["one_axis"]
    logger.info("headline:\n%s", head[head["varied"] == "headline"].to_string(index=False))
    logger.info("written to %s", out_dir)


if __name__ == "__main__":
    main()
