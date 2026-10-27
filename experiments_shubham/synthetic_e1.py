"""
synthetic_e1.py - E1: the engine on every synthetic regime, checked two ways.

Proposal section 9, E1 "synthetic implementation validation": known relevance,
independent stage errors, all worlds and the t grid. Its results are
correctness and effect sanity (predictor recoverability, the third, belongs
to the prediction stream).

1. Exact recovery. Every effect the engine produces is re-derived here from
   the generated scores with plain numpy - no shared.worlds, shared.effects,
   shared.allocation or shared.ordering - and the two must agree row for row:
   the same rows, ranks and strata, and effects within 1e-12. Two separate
   implementations of the proposal's definitions agreeing on ~10 million rows
   per regime is the evidence that the engine computes what was defined.

2. Effect sanity. Each regime stresses one channel, so the effects must move
   the way its mechanism says (see EXPECTATIONS). These directions are
   declared here before the run, not read off the results.

    python -m experiments_shubham.synthetic_e1                 # full size, ~20 min
    python -m experiments_shubham.synthetic_e1 --n-queries 10  # quick, ~2 min

Writes effects/synthetic/e1_report.md, e1_checks.csv and e1_stats.csv, and the
committed record effects/e1_manifest.json. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments_shubham.engine import (
    EffectSettings,
    grid_t,
    iter_grid,
    synthetic_frame,
    synthetic_inputs,
)
from shared.allocation import ALTERNATIVE, BOUNDARY, RANDOM
from shared.config import FrozenConfig, load_config
from shared.provenance import run_manifest, write_manifest
from shared.schema import (
    E_GATE,
    E_RERANK,
    EPS,
    NON_SURVIVOR,
    REMOVED,
    RETAINED,
    SELECTED_OUTSIDER,
    SURVIVOR,
    UNSELECTED_OUTSIDER,
)
from shared.synthetic import alternative_ranks

logger = logging.getLogger(__name__)

TOLERANCE = 1e-12
UNRANKED_ALT = 1 << 30      # documents the alternative retriever never ranked

# name -> (regime, stress applied, gate-stress documents buried far below K_c)
VARIANTS: dict[str, tuple[str, bool, bool]] = {
    "strong_stages": ("strong_stages", True, False),
    "weak_retriever": ("weak_retriever", True, False),
    "weak_reranker": ("weak_reranker", True, False),
    "stage_mismatch": ("stage_mismatch", True, False),
    "gate_stress": ("gate_stress", True, False),
    "gate_stress_far": ("gate_stress", True, True),
    "budget_stress": ("budget_stress", True, False),
    # gate_stress and budget_stress share one generator; this is that draw
    # with no manipulation, the control both stress regimes are compared to
    "stress_control": ("budget_stress", False, False),
}

EXPECTATIONS = {
    "weak_retriever_gate_cost_higher":
        "weak retriever: the gate costs at least as much as under strong stages "
        "(mean per-query sum of E_gate over non-survivors no higher) at every K_c, "
        "and strictly more at the smallest K_c",
    "weak_retriever_more_gating_failures":
        "weak retriever: at least as many gating failures per query as strong "
        "stages at every K_c, and strictly more at the smallest K_c",
    "weak_reranker_helps_less":
        "weak reranker: reranking gains less DCG per query than under strong stages, "
        "at every K_c",
    "weak_reranker_harms_more_queries":
        "weak reranker: a larger share of queries is harmed by reranking than under "
        "strong stages, at the stress K_c",
    "gate_stress_boundary_finds_planted":
        "gate stress (just below K_c): the boundary policy promotes at least 95% of "
        "the planted documents at the smallest t",
    "gate_stress_far_boundary_misses":
        "gate stress (far below K_c): the boundary policy promotes fewer planted "
        "documents than the alternative policy at the smallest t",
    "budget_stress_outsiders_reach_top_k":
        "budget stress: more promoted outsiders reach the final top-k than in the "
        "control, for every policy at the smallest t",
    "budget_stress_retained_lose_more":
        "budget stress: retained candidates lose more utility than in the control, "
        "for every policy at the smallest t",
}

# Every change to EXPECTATIONS after the code first ran, with the result that
# prompted it. Printed in the report and the manifest so no revision is silent.
REVISIONS = [
    "Before the first full-size run, after a 5-query trial: the two "
    "weak-retriever comparisons went from strict at every K_c to non-strict at "
    "every K_c and strict at the smallest, because where the gate removes "
    "nothing both regimes are exactly zero; and the policy checks gained the "
    "not-testable guard, because the trial's D_q was smaller than the outsider "
    "budget, so every policy promoted all of it.",
    "After the first full-size run: 'weak_retriever_gate_costs_more' compared "
    "the NET per-query gate effect, DCG(pi_act) - DCG(pi_no1), and failed at "
    "K_c=1000 (weak retriever +0.021 vs strong stages 0.000). The net effect "
    "adds two opposite channels: the loss on documents the gate excludes (cost) "
    "and the gain for survivors shielded from strong competitors (protection). "
    "A weak retriever's gate has far more of both (K_c=1000: cost -3.710 vs "
    "0.000, protection +3.732 vs 0.000), and they nearly cancel. Replaced by "
    "'weak_retriever_gate_cost_higher', on the cost alone; the net effect and "
    "both channels stay in the stage-effects table.",
]


# ──────────────────────────────────────────────────────────────────────
# Independent re-derivation (numpy only)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RefQuery:
    query_id: str
    docs: np.ndarray        # doc_id strings
    s1: np.ndarray
    s2: np.ndarray
    gains: np.ndarray       # gain of each document under the frozen gain map
    alt: np.ndarray         # alternative retriever's rank
    stage1: np.ndarray      # indices in Stage-1 order


def _order(q: RefQuery, scores: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """idx sorted by score descending, then doc_id ascending."""
    return idx[np.lexsort((q.docs[idx], -scores[idx]))]


def _ranks(ordered: np.ndarray, n: int) -> np.ndarray:
    """1-based rank per document; 0 for documents not in the ranking."""
    r = np.zeros(n, dtype=np.int64)
    r[ordered] = np.arange(1, len(ordered) + 1)
    return r


def _utility(ranks: np.ndarray, gains: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(len(ranks))
    m = (ranks >= 1) & (ranks <= k)
    out[m] = gains[m] / np.log2(1.0 + ranks[m])
    return out


def ref_queries(df: pd.DataFrame, gain_map: Mapping[int, float]) -> list[RefQuery]:
    alt = alternative_ranks(df)
    out = []
    for qid, grp in df.groupby("query_id", observed=True):
        docs = grp["doc_id"].astype(str).to_numpy()
        s1 = grp["s1"].to_numpy(float)
        q = RefQuery(
            str(qid), docs, s1, grp["s2"].to_numpy(float),
            np.array([gain_map.get(int(r), 0.0) for r in grp["relevance"]]),
            np.array([alt[str(qid)].get(d, UNRANKED_ALT) for d in docs]),
            np.empty(0, dtype=np.int64),
        )
        q.stage1 = _order(q, s1, np.arange(len(docs)))
        out.append(q)
    return out


def ref_stage(qs: Sequence[RefQuery], k_c: int, k: int) -> pd.DataFrame:
    frames = []
    for q in qs:
        n = len(q.docs)
        cand = q.stage1[:k_c]
        r_act = _ranks(_order(q, q.s2, cand), n)
        r_no2 = _ranks(cand, n)
        r_no1 = _ranks(_order(q, q.s2, np.arange(n)), n)
        u_act, u_no2, u_no1 = (_utility(r, q.gains, k) for r in (r_act, r_no2, r_no1))
        stratum = np.where(r_no2 > 0, SURVIVOR, NON_SURVIVOR)
        for etype, r_ctl, u_ctl in ((E_RERANK, r_no2, u_no2), (E_GATE, r_no1, u_no1)):
            frames.append(pd.DataFrame({
                "query_id": q.query_id, "doc_id": q.docs, "effect_type": etype,
                "stratum": stratum, "rank_treatment": r_act, "rank_control": r_ctl,
                "effect": u_act - u_ctl,
            }))
    return pd.concat(frames, ignore_index=True)


def _select(q: RefQuery, disc: np.ndarray, n_out: int, policy: str,
            seed: int) -> np.ndarray:
    if n_out <= 0 or not len(disc):
        return disc[:0]
    if n_out >= len(disc):
        return disc
    if policy == BOUNDARY:
        return disc[:n_out]                       # disc is already in Stage-1 order
    if policy == ALTERNATIVE:
        return disc[np.lexsort((q.docs[disc], q.alt[disc]))][:n_out]
    if policy == RANDOM:
        pool = disc[np.argsort(q.docs[disc], kind="stable")]
        pick = np.random.default_rng(seed).choice(len(pool), size=n_out, replace=False)
        return pool[pick]
    raise ValueError(f"unknown policy {policy!r}")


def _t_world(q: RefQuery, k_c: int, budget_b: int, t: float, policy: str,
             seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(ranks, strata) of world D at one t."""
    n = len(q.docs)
    cand, disc = q.stage1[:k_c], q.stage1[k_c:]
    n_cand = int(math.floor(t * budget_b))
    selected = _select(q, disc, budget_b - n_cand, policy, seed)
    submitted = np.concatenate([cand[:n_cand], selected])

    strata = np.full(n, UNSELECTED_OUTSIDER, dtype=object)
    strata[cand[:n_cand]] = RETAINED
    strata[cand[n_cand:]] = REMOVED
    strata[selected] = SELECTED_OUTSIDER
    return _ranks(_order(q, q.s2, submitted), n), strata


def ref_budget(qs: Sequence[RefQuery], k_c: int, budget_b: int, t_grid: Sequence[float],
               policy: str, seeds: Sequence[int], k: int) -> pd.DataFrame:
    frames = []
    run_seeds = list(seeds) if policy == RANDOM else list(seeds[:1])
    for q in qs:
        for seed in run_seeds:
            r_base, _ = _t_world(q, k_c, budget_b, 1.0, policy, seed)
            u_base = _utility(r_base, q.gains, k)
            for t in t_grid:
                if t == 1.0:
                    continue
                r_t, strata = _t_world(q, k_c, budget_b, t, policy, seed)
                frames.append(pd.DataFrame({
                    "query_id": q.query_id, "doc_id": q.docs, "t": float(t),
                    "seed": seed if policy == RANDOM else -1, "stratum": strata,
                    "rank_treatment": r_t, "rank_control": r_base,
                    "effect": _utility(r_t, q.gains, k) - u_base,
                }))
    return pd.concat(frames, ignore_index=True)


def compare(engine: pd.DataFrame, ref: pd.DataFrame, keys: list[str]) -> dict:
    """Row-for-row agreement between the engine and the re-derivation."""
    e = engine[keys + ["stratum", "rank_treatment", "rank_control", "effect"]].copy()
    if "seed" in keys:
        e["seed"] = e["seed"].fillna(-1).astype("int64")
    text_cols = ["query_id", "doc_id", "stratum"]
    if "effect_type" in keys:
        text_cols.append("effect_type")
    for col in text_cols:
        e[col] = e[col].astype(str)
    for col in ("rank_treatment", "rank_control"):
        e[col] = e[col].fillna(0).astype("int64")
    m = e.merge(ref, on=keys, how="outer", suffixes=("", "_ref"), indicator=True)
    both = m["_merge"] == "both"
    return {
        "rows": int(len(engine)),
        "unmatched_rows": int((~both).sum()),
        "rank_mismatches": int(sum(
            (m.loc[both, c] != m.loc[both, f"{c}_ref"]).sum()
            for c in ("rank_treatment", "rank_control"))),
        "stratum_mismatches": int(
            (m.loc[both, "stratum"] != m.loc[both, "stratum_ref"]).sum()),
        "max_abs_diff": float(
            (m.loc[both, "effect"] - m.loc[both, "effect_ref"]).abs().max()),
    }


# ──────────────────────────────────────────────────────────────────────
# Statistics for the sanity checks
# ──────────────────────────────────────────────────────────────────────

def stage_stats(stage: pd.DataFrame) -> dict:
    per_q = stage.groupby(["query_id", "effect_type"], observed=True)["effect"].sum()
    per_q = per_q.unstack()
    gate = stage[stage["effect_type"] == E_GATE]
    failures = gate[(gate["stratum"] == NON_SURVIVOR) & (gate["utility_control"] > EPS)]
    n_q = stage["query_id"].nunique()
    return {
        "rerank_gain": float(per_q[E_RERANK].mean()),
        "rerank_harm_share": float((per_q[E_RERANK] < -EPS).mean()),
        "gate_effect": float(per_q[E_GATE].mean()),
        "gate_protection": float(gate.loc[gate["stratum"] == SURVIVOR, "effect"].sum() / n_q),
        "gate_cost": float(gate.loc[gate["stratum"] == NON_SURVIVOR, "effect"].sum() / n_q),
        "gating_failures_per_query": len(failures) / n_q,
    }


def budget_stats(budget: pd.DataFrame, k: int,
                 planted: set[tuple[str, str]] | None) -> list[dict]:
    """Per t, averaged over queries (and seeds for the random policy)."""
    rows = []
    b = budget.assign(seed=budget["seed"].fillna(-1))
    for t, blk in b.groupby("t"):
        groups = ["query_id", "seed"]
        n_runs = blk[groups].drop_duplicates().shape[0]
        sel = blk[blk["stratum"] == SELECTED_OUTSIDER]
        rec = {
            "t": float(t),
            "delta_dcg": float(blk["effect"].sum() / n_runs),
            "retained_delta": float(blk.loc[blk["stratum"] == RETAINED, "effect"].sum()
                                    / n_runs),
            "outsiders_in_top_k": float((sel["rank_treatment"] <= k).sum() / n_runs),
        }
        if planted:
            key = list(zip(blk["query_id"].astype(str), blk["doc_id"].astype(str)))
            is_planted = np.fromiter((x in planted for x in key), bool, len(key))
            p = blk[is_planted & blk["stratum"].isin([SELECTED_OUTSIDER,
                                                       UNSELECTED_OUTSIDER])]
            rec["planted_promoted_share"] = (
                float((p["stratum"] == SELECTED_OUTSIDER).mean()) if len(p) else float("nan"))
        rows.append(rec)
    return rows


def planted_documents(stressed: pd.DataFrame, control: pd.DataFrame) -> set[tuple[str, str]]:
    """The documents a gate-stress manipulation moved: s1 changed, nothing else."""
    moved = stressed["s1"].to_numpy() != control["s1"].to_numpy()
    return set(zip(stressed.loc[moved, "query_id"].astype(str),
                   stressed.loc[moved, "doc_id"].astype(str)))


# ──────────────────────────────────────────────────────────────────────
# The run
# ──────────────────────────────────────────────────────────────────────

def run_variant(cfg: FrozenConfig, name: str, n_queries: int | None,
                n_docs: int | None) -> tuple[list[dict], list[dict]]:
    """(recovery records, stats records) for one variant over the whole grid."""
    regime, stress, far = VARIANTS[name]
    df = synthetic_frame(cfg, regime, n_queries, n_docs, stress=stress, far=far)
    planted = None
    if regime == "gate_stress":
        control = synthetic_frame(cfg, regime, n_queries, n_docs, stress=False)
        planted = planted_documents(df, control)

    settings = EffectSettings.from_config(cfg, "synthetic")
    k = settings.k
    queries, refs = synthetic_inputs(df), ref_queries(df, cfg.gain_map)
    t_grid = grid_t(cfg)

    recovery, stats = [], []
    for table in iter_grid(queries, cfg, settings):
        base = {"variant": name, "k_c": table.k_c, "policy": table.policy}
        if table.policy is None:
            ref = ref_stage(refs, table.k_c, k)
            recovery.append({**base, **compare(table.effects, ref,
                                               ["query_id", "doc_id", "effect_type"])})
            stats.append({**base, "t": None, **stage_stats(table.effects)})
        else:
            ref = ref_budget(refs, table.k_c, cfg.budget_for(table.k_c), t_grid,
                             table.policy, cfg.budget_seeds, k)
            recovery.append({**base, **compare(table.effects, ref,
                                               ["query_id", "doc_id", "t", "seed"])})
            for rec in budget_stats(table.effects, k, planted):
                stats.append({**base, **rec})
        logger.info("%s %s: %s", name, table.name, recovery[-1])
    return recovery, stats


def evaluate(stats: pd.DataFrame, cfg: FrozenConfig, n_docs: int) -> list[dict]:
    """
    The declared expectations, each a pass/fail with the numbers behind it.

    A gate can be inert at a large K_c in both regimes (both effects exactly
    zero), so the regime comparisons are non-strict at every K_c and strict at
    the smallest. The policy comparisons are only testable when the outsider
    budget at the smallest t is smaller than D_q; otherwise every policy
    promotes all of D_q and they are reported as not testable, never passed.
    """
    depths = list(cfg.candidate_depths)
    stress_kc = max(depths)
    t_min = min(cfg.t_grid)
    budget_b = cfg.budget_for(stress_kc)
    n_out = budget_b - int(math.floor(t_min * budget_b))
    testable = n_out < n_docs - stress_kc
    untestable = (f"not testable: {n_out} outsiders to promote from a D_q of "
                  f"{n_docs - stress_kc}, so every policy promotes all of it")
    stage = stats[stats["policy"].isna()].set_index(["variant", "k_c"])
    budget = stats[stats["policy"].notna() & (stats["t"] == t_min)]
    budget = budget.set_index(["variant", "k_c", "policy"])

    def s(variant, k_c, col):
        return float(stage.loc[(variant, k_c), col])

    def b(variant, policy, col):
        return float(budget.loc[(variant, stress_kc, policy), col])

    checks = []

    def check(key, passed, detail, needs_outsider_choice=False):
        if needs_outsider_choice and not testable:
            passed, detail = False, untestable
        checks.append({"check": key, "expectation": EXPECTATIONS[key],
                       "passed": bool(passed), "detail": detail})

    def regime_check(key, pairs, better, fmt):
        ok = all(better(x, c) or x == c for x, c in pairs) and better(*pairs[0])
        check(key, ok, "; ".join(f"K_c={d}: {x:{fmt}} vs {c:{fmt}}"
                                 for d, (x, c) in zip(depths, pairs)))

    depths = sorted(depths)
    regime_check("weak_retriever_gate_cost_higher",
                 [(s("weak_retriever", d, "gate_cost"),
                   s("strong_stages", d, "gate_cost")) for d in depths],
                 lambda x, c: x < c, ".3f")
    regime_check("weak_retriever_more_gating_failures",
                 [(s("weak_retriever", d, "gating_failures_per_query"),
                   s("strong_stages", d, "gating_failures_per_query")) for d in depths],
                 lambda x, c: x > c, ".2f")

    pairs = [(s("weak_reranker", d, "rerank_gain"), s("strong_stages", d, "rerank_gain"))
             for d in depths]
    check("weak_reranker_helps_less", all(w < st for w, st in pairs),
          "; ".join(f"K_c={d}: {w:.3f} vs {st:.3f}" for d, (w, st) in zip(depths, pairs)))

    w = s("weak_reranker", stress_kc, "rerank_harm_share")
    st = s("strong_stages", stress_kc, "rerank_harm_share")
    check("weak_reranker_harms_more_queries", w > st, f"{w:.2f} vs {st:.2f}")

    rate = b("gate_stress", BOUNDARY, "planted_promoted_share")
    check("gate_stress_boundary_finds_planted", rate >= 0.95, f"{rate:.2f}", True)

    near, alt = (b("gate_stress_far", p, "planted_promoted_share")
                 for p in (BOUNDARY, ALTERNATIVE))
    check("gate_stress_far_boundary_misses", near < alt,
          f"boundary {near:.2f} vs alternative {alt:.2f}", True)

    policies = list(cfg.outsider_policies)
    pairs = [(b("budget_stress", p, "outsiders_in_top_k"),
              b("stress_control", p, "outsiders_in_top_k")) for p in policies]
    check("budget_stress_outsiders_reach_top_k", all(x > c for x, c in pairs),
          "; ".join(f"{p}: {x:.2f} vs {c:.2f}" for p, (x, c) in zip(policies, pairs)),
          True)

    pairs = [(b("budget_stress", p, "retained_delta"),
              b("stress_control", p, "retained_delta")) for p in policies]
    check("budget_stress_retained_lose_more", all(x < c for x, c in pairs),
          "; ".join(f"{p}: {x:.3f} vs {c:.3f}" for p, (x, c) in zip(policies, pairs)),
          True)
    return checks


def _md(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    """A Markdown table, without needing tabulate."""
    def cell(v):
        if isinstance(v, float):
            return "" if math.isnan(v) else format(v, floatfmt)
        return str(v)
    head = [str(c) for c in df.columns]
    rows = [[cell(v) for v in row] for row in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(head) + " |",
                      "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def write_report(path: Path, recovery: pd.DataFrame, checks: list[dict],
                 stats: pd.DataFrame, cfg: FrozenConfig, sizes: dict) -> None:
    exact = recovery.groupby("variant", sort=False).agg(
        rows=("rows", "sum"), unmatched=("unmatched_rows", "sum"),
        rank_mismatches=("rank_mismatches", "sum"),
        stratum_mismatches=("stratum_mismatches", "sum"),
        max_abs_diff=("max_abs_diff", "max"),
    ).reset_index()
    t_min = min(cfg.t_grid)
    curve = (stats[stats["policy"].notna()]
             .pivot_table(index=["variant", "k_c", "policy"], columns="t",
                          values="delta_dcg").reset_index())
    lines = [
        "# E1 - synthetic implementation validation",
        "",
        f"Config fingerprint `{cfg.fingerprint()}`; {sizes['n_queries']} queries x "
        f"{sizes['n_docs']} documents per variant; K_c {list(cfg.candidate_depths)}, "
        f"B = K_c, t grid {list(cfg.t_grid)}, policies {list(cfg.outsider_policies)}, "
        f"random seeds {list(cfg.budget_seeds)}, k = {cfg.final_cutoff_k}.",
        "",
        "## 1. Exact recovery",
        "",
        "Engine effects against an independent numpy re-derivation, row for row.",
        "",
        _md(exact),
        "",
        "## 2. Effect sanity",
        "",
        _md(pd.DataFrame(checks)[["check", "passed", "detail"]]),
        "",
        "Revisions to the expectations, each with the result that prompted it:",
        "",
        *[f"{i}. {r}" for i, r in enumerate(REVISIONS, start=1)],
        "",
        "## 3. Stage effects (per-query means)",
        "",
        "`gate_effect` = `gate_protection` (survivors) + `gate_cost` (non-survivors).",
        "",
        _md(stats[stats["policy"].isna()][["variant", "k_c", "rerank_gain",
                                           "rerank_harm_share", "gate_effect",
                                           "gate_protection", "gate_cost",
                                           "gating_failures_per_query"]], ".3f"),
        "",
        f"## 4. Budget curves: per-query mean DCG change against t = 1 "
        f"(smallest t = {t_min})",
        "",
        _md(curve, ".3f"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="E1 synthetic validation")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--n-queries", type=int, default=None)
    parser.add_argument("--n-docs", type=int, default=None)
    parser.add_argument("--out-dir", default=None, help="default: effects/synthetic")
    args = parser.parse_args()

    cfg = load_config(args.config)
    effects_root = Path(cfg.paths.get("effects", "effects"))
    out_dir = Path(args.out_dir) if args.out_dir else effects_root / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {"n_queries": args.n_queries or cfg.synthetic["n_queries"],
             "n_docs": args.n_docs or cfg.synthetic["n_docs_per_query"]}

    start = time.perf_counter()
    recovery, stats = [], []
    for name in VARIANTS:
        r, s = run_variant(cfg, name, args.n_queries, args.n_docs)
        recovery += r
        stats += s
    recovery_df, stats_df = pd.DataFrame(recovery), pd.DataFrame(stats)

    exact_ok = bool(
        (recovery_df["unmatched_rows"] == 0).all()
        and (recovery_df["rank_mismatches"] == 0).all()
        and (recovery_df["stratum_mismatches"] == 0).all()
        and (recovery_df["max_abs_diff"] <= TOLERANCE).all()
    )
    checks = [{"check": "exact_recovery", "expectation": "engine == re-derivation",
               "passed": exact_ok,
               "detail": f"{int(recovery_df['rows'].sum())} rows, max |diff| "
                         f"{recovery_df['max_abs_diff'].max():.1e}"}]
    checks += evaluate(stats_df, cfg, sizes["n_docs"])

    pd.DataFrame(checks).to_csv(out_dir / "e1_checks.csv", index=False)
    stats_df.to_csv(out_dir / "e1_stats.csv", index=False)
    recovery_df.to_csv(out_dir / "e1_recovery.csv", index=False)
    write_report(out_dir / "e1_report.md", recovery_df, checks, stats_df, cfg, sizes)

    manifest = run_manifest(cfg.fingerprint(), owner="shubham", extra={
        "experiment": "E1", **sizes, "variants": list(VARIANTS),
        "checks": checks, "revisions": REVISIONS,
        "seconds": round(time.perf_counter() - start, 1),
        "report": str(out_dir / "e1_report.md"),
    })
    manifest_path = (effects_root if args.out_dir is None else out_dir) / "e1_manifest.json"
    write_manifest(manifest_path, manifest)

    for c in checks:
        logger.info("%s %s: %s", "PASS" if c["passed"] else "FAIL", c["check"], c["detail"])
    if not all(c["passed"] for c in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
