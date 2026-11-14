"""
predict.py - Can cheap signals stand in for the reference effects? (RQ3; H1, H5)

Two ways to approximate an effect without paying for it, put on one cost axis,
the extra Stage-2 calls per query:

    predictors  features from the factual run only (proposal §6.1, the F1
                conventional set), fitted on the training queries of the frozen
                split and scored on its held-out test queries. One target at a
                time (E_rerank, E_gate, and Delta_t per outsider policy, never
                mixed, §6.3), one model per stratum, against the three shared
                baselines (zero, membership, stratum mean) and, for H1, rank
                displacement alone. Cost: 0 extra Stage-2 calls.
    probes      spend b extra Stage-2 calls per query on the outsiders a policy
                picks (boundary / alternative / random, `shared.allocation`) and
                recompute the no-gate world over C_q plus the probed set (H5).
                An unprobed document gets 0. b sweeps the config's outsider and
                boundary budgets; b = |D_q| is the exact reference, equal to it
                by construction, and checked rather than assumed.

THE FEATURE BOUNDARY. Stage-2 scores enter only for C_q, the documents the
factual run actually scored. Any other Stage-2 score exists only because world
C was computed, and for a non-survivor it determines E_gate exactly - using it
would predict the target from the target. So models are fitted PER STRATUM:
survivors see Stage-1, alternative-retriever and Stage-2 features; non-survivors
and selected outsiders see Stage-1 and alternative-retriever features only. A
probe's Stage-2 scores are paid for, and counted.

THE EVALUATION POPULATION. Judged documents, via
`shared.metrics.evaluate_within_strata`: an unjudged document has zero gain, so
its effect is zero under every world and every predictor, and including it
would bury the comparison under free zeros. Models are trained on judged
training rows for the same reason. Query-level numbers sum over judged
documents, which is the same as summing over all of U_q.

    python -m experiments_shubham.predict                  # both systems, headline K_c
    python -m experiments_shubham.predict --systems bm25 --n-bootstrap 200
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import tracemalloc
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments_shubham.analysis import diagnose
from experiments_shubham.engine import QueryInputs, table_name
from experiments_shubham.power_h7 import headline_k_c
from shared.allocation import select_outsiders
from shared.baselines import attach_all
from shared.config import FrozenConfig, load_config
from shared.diagnosis import GATE_SENSITIVE
from shared.metrics import (
    bootstrap_ci,
    evaluate_within_strata,
    macro_fidelity,
    mae,
    spearman,
    top_m_recovery,
)
from shared.ordering import order_by
from shared.provenance import run_manifest, write_manifest
from shared.schema import (
    DELTA_T,
    E_GATE,
    E_RERANK,
    NON_SURVIVOR,
    REMOVED,
    RETAINED,
    SELECTED_OUTSIDER,
    SURVIVOR,
)
from shared.utility import ranking_utilities

logger = logging.getLogger(__name__)

#: Stage-1 and alternative-retriever features: free for every document in U_q.
SHARED_FEATURES = ("s1_log_rank", "s1_margin", "s1_floored",
                   "alt_log_rank", "alt_top_kc", "alt_floored")
#: Plus the factual run's Stage-2 view, which exists only inside C_q.
SURVIVOR_FEATURES = SHARED_FEATURES + ("s2_score", "s2_rank", "s2_margin_k", "displacement")
#: H1's predictor: within-candidate rank displacement and nothing else.
DISPLACEMENT_FEATURES = ("displacement",)

MODELS = ("ridge", "gbm")
BASELINES = ("pred_zero", "pred_membership", "pred_stratum_mean")
MIN_TRAIN_ROWS = 20

#: Per target, the features each stratum's model may see. A stratum missing
#: here is structurally zero for that target and is predicted as 0.
STRATUM_FEATURES = {
    E_RERANK: {SURVIVOR: SURVIVOR_FEATURES},
    E_GATE: {SURVIVOR: SURVIVOR_FEATURES, NON_SURVIVOR: SHARED_FEATURES},
    DELTA_T: {RETAINED: SURVIVOR_FEATURES + ("t",), REMOVED: SURVIVOR_FEATURES + ("t",),
              SELECTED_OUTSIDER: SHARED_FEATURES + ("t",)},
}


# ──────────────────────────────────────────────────────────────────────
# Features
# ──────────────────────────────────────────────────────────────────────

def query_features(q: QueryInputs, k_c: int, k: int, depth_m: int) -> pd.DataFrame:
    """
    F1 features for every document of one query, from the factual run alone.

    `q.s2` is read for the K_c candidates only: `factual` is the one place a
    Stage-2 score enters, and the leakage test perturbs every other score to
    prove nothing else reads it.
    """
    order = order_by(q.universe, q.s1)
    n = len(order)
    cands = order[:k_c]
    factual = {d: float(q.s2[d]) for d in cands}
    act = order_by(cands, factual)
    s2_rank = {d: i for i, d in enumerate(act, start=1)}
    s2_k = factual[act[min(k, len(act)) - 1]]

    s1 = np.fromiter((q.s1[d] for d in order), float, n)
    rank = np.arange(1, n + 1)
    alt_map = q.alternative_rank or {}
    alt = np.fromiter((alt_map.get(d, n + 1) for d in order), float, n)
    survivor = rank <= k_c
    out = pd.DataFrame({
        "query_id": str(q.query_id), "doc_id": [str(d) for d in order],
        "survivor": survivor,
        "s1_log_rank": np.log(rank), "s1_margin": s1 - s1[k_c - 1],
        "s1_floored": (rank > depth_m).astype(float),
        "alt_log_rank": np.log(alt), "alt_top_kc": (alt <= k_c).astype(float),
        "alt_floored": (alt > depth_m).astype(float),
    })
    s2 = np.full(n, np.nan)
    s2r = np.full(n, np.nan)
    s2[:k_c] = [factual[d] for d in cands]
    s2r[:k_c] = [s2_rank[d] for d in cands]
    out["s2_score"] = s2
    out["s2_rank"] = s2r
    out["s2_margin_k"] = s2 - s2_k
    out["displacement"] = np.where(survivor, rank - s2r, np.nan)
    return out


def feature_table(queries: Iterable[QueryInputs], k_c: int, k: int,
                  depth_m: int) -> pd.DataFrame:
    return pd.concat([query_features(q, k_c, k, depth_m) for q in queries],
                     ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# Predictors
# ──────────────────────────────────────────────────────────────────────

def _model(kind: str, seed: int):
    """
    Ridge, and gradient-boosted trees for the nonlinear case.

    Not HistGradientBoostingRegressor: its OpenMP runtime segfaults in any
    process that has already imported torch (two libomp copies on macOS), which
    the test suite does via the DPR tests. The training sets here are a few
    thousand rows, so the classic, single-threaded implementation costs seconds.
    """
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "gbm":
        return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=20, random_state=seed)
    raise ValueError(f"unknown model {kind!r}")


def fit_predict(frame: pd.DataFrame, strata_features: Mapping[str, Sequence[str]],
                train_queries: set[str], kind: str, seed: int) -> tuple[pd.Series, dict]:
    """
    One model per stratum, fitted on judged rows of training queries.

    Returns predictions for every row of `frame` (0 in a stratum with no
    model, which is how structurally zero strata are predicted) and, per
    stratum, the training size and what was fitted. A stratum with fewer than
    MIN_TRAIN_ROWS training rows, or a constant target, gets the training mean
    and says so - never a silent NaN.
    """
    pred = pd.Series(0.0, index=frame.index)
    info = {}
    is_train = frame["query_id"].isin(train_queries) & frame["judged"].astype(bool)
    for stratum, cols in strata_features.items():
        rows = frame["stratum"] == stratum
        if not rows.any():
            continue
        cols = list(cols)
        tr = rows & is_train
        y = frame.loc[tr, "effect"].to_numpy(float)
        if len(y) < MIN_TRAIN_ROWS or np.allclose(y, y[0] if len(y) else 0.0):
            pred[rows] = float(y.mean()) if len(y) else 0.0
            info[stratum] = {"n_train": int(len(y)), "fitted": "train_mean"}
            continue
        m = _model(kind, seed).fit(frame.loc[tr, cols].to_numpy(float), y)
        pred[rows] = m.predict(frame.loc[rows, cols].to_numpy(float))
        info[stratum] = {"n_train": int(len(y)), "fitted": kind, "features": cols}
    return pred, info


def _join(effects: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = effects.merge(features, on=["query_id", "doc_id"], how="left",
                        validate="many_to_one")
    if out["s1_log_rank"].isna().any():
        raise ValueError("effect rows without features: the feature table must "
                         "cover U_q for every query")
    return out


def predictor_block(effects: pd.DataFrame, features: pd.DataFrame, effect_type: str,
                    split: Mapping[str, set[str]], seed: int) -> tuple[pd.DataFrame, dict]:
    """
    Every predictor for one target on the held-out test queries, as columns.

    Baselines are fitted on judged training rows only (`shared.baselines`), so
    the comparison with the learned models is like for like.
    """
    frame = _join(effects, features)
    info = {}
    for kind in MODELS:
        frame[f"pred_{kind}"], info[kind] = fit_predict(
            frame, STRATUM_FEATURES[effect_type], split["train"], kind, seed)
    if effect_type in (E_RERANK, E_GATE):
        disp = {SURVIVOR: DISPLACEMENT_FEATURES}
        frame["pred_displacement"], info["displacement"] = fit_predict(
            frame, disp, split["train"], "ridge", seed)
        frame.loc[frame["stratum"] != SURVIVOR, "pred_displacement"] = np.nan
    train = frame[frame["query_id"].isin(split["train"]) & frame["judged"].astype(bool)]
    test = frame[frame["query_id"].isin(split["test"])]
    return attach_all(test, effect_type, train=train), info


def predictor_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("pred_")]


def fidelity(test: pd.DataFrame, effect_type: str, cfg: FrozenConfig,
             n_iter: int) -> pd.DataFrame:
    rows = [evaluate_within_strata(test, col, effect_type, top_m=cfg.top_m, n_iter=n_iter,
                                   alpha=cfg.bootstrap_alpha, seed=cfg.seed)
            for col in predictor_columns(test) if test[col].notna().any()]
    return pd.concat(rows, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# Probes
# ──────────────────────────────────────────────────────────────────────

def probe_effects(q: QueryInputs, k_c: int, k: int, gain_map, kind: str, policy: str,
                  budget: int, seed: int, persistence: float = 0.8) -> tuple[dict, int]:
    """
    Approximate E_gate from b extra Stage-2 calls, and the calls actually spent.

    The probed no-gate world ranks C_q plus the b outsiders the policy picks,
    by Stage-2 score. Every document outside it keeps E_gate = 0. Returns
    {doc_id: approximate E_gate} for the documents whose effect is nonzero,
    with the engine's sign: E_gate = U_d(actual) - U_d(no gate), so >= 0 for
    survivors and <= 0 for outsiders, and G_q = -sum E_gate.
    """
    order = order_by(q.universe, q.s1)
    cands, discarded = list(order[:k_c]), list(order[k_c:])
    probed = select_outsiders(discarded, budget, policy, s1=q.s1,
                              alternative_rank=q.alternative_rank, seed=seed)
    act = order_by(cands, q.s2)
    world = order_by(cands + list(probed), q.s2)
    u_act = ranking_utilities(act, q.relevance, k, gain_map, kind, persistence)
    u_new = ranking_utilities(world, q.relevance, k, gain_map, kind, persistence)
    effects = {d: u_act.get(d, 0.0) - u_new.get(d, 0.0) for d in {*u_act, *u_new}}
    return {d: e for d, e in effects.items() if abs(e) > 1e-12}, len(probed)


def _budgets(cfg: FrozenConfig) -> list[int]:
    return sorted({0, *cfg.outsider_budgets, *cfg.boundary_budgets})


def _label(g: np.ndarray, r: np.ndarray, tau: float) -> np.ndarray:
    """The frozen gate-sensitive rule of `analysis.diagnose`, on arrays."""
    return (g > tau) & (g > r)


def _label_agreement(pred: np.ndarray, true: np.ndarray) -> dict:
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"gs_true": int(true.sum()), "gs_predicted": int(pred.sum()), "gs_tp": tp,
            "gs_precision": prec, "gs_recall": rec, "gs_f1": f1}


def query_level(name: str, budget: int | None, calls: float, g_hat: pd.Series,
                diag: pd.DataFrame, cfg: FrozenConfig, n_iter: int) -> dict:
    """Approximate G_q against the reference, and the gate-sensitive label it implies."""
    d = diag.set_index("query_id").loc[g_hat.index]
    g, r = d["g_q"].to_numpy(float), d["r_q"].to_numpy(float)
    gh = g_hat.to_numpy(float)
    frame = pd.DataFrame({"query_id": g_hat.index, "g": g, "gh": gh})
    ci = bootstrap_ci(frame, lambda x: spearman(x["gh"].to_numpy(float),
                                                x["g"].to_numpy(float)),
                      n_iter=n_iter, alpha=cfg.bootstrap_alpha, seed=cfg.seed)
    return {"approximation": name, "budget": budget, "extra_calls_per_query": calls,
            "n_queries": len(gh), "gq_spearman": ci["point"], "gq_spearman_lo": ci["lo"],
            "gq_spearman_hi": ci["hi"], "gq_mae": mae(gh, g),
            **_label_agreement(_label(gh, r, cfg.tau_query),
                               (d["label"] == GATE_SENSITIVE).to_numpy())}


def probe_tables(queries: Sequence[QueryInputs], stage: pd.DataFrame, diag: pd.DataFrame,
                 cfg: FrozenConfig, k_c: int, scopes: Mapping[str, set[str]],
                 n_iter: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Document- and query-level fidelity of every (policy, budget, seed), per scope.

    The full budget is run once per query as the exact-reference check: it
    must reproduce the engine's E_gate row for row, or the run stops.
    """
    gate = stage[(stage["effect_type"] == E_GATE) & stage["judged"].astype(bool)]
    gate = gate[["query_id", "doc_id", "stratum", "effect", "utility_control"]].copy()
    key = list(zip(gate["query_id"], gate["doc_id"]))
    failures = gate[(gate["stratum"] == NON_SURVIVOR) & (gate["utility_control"] > 1e-12)]

    configs = [(p, s) for p in cfg.outsider_policies
               for s in (cfg.budget_seeds if p == "random" else (cfg.seed,))]
    doc_rows, query_rows = [], []
    max_diff = 0.0
    full_q = {}
    for q in queries:
        full, _ = probe_effects(q, k_c, cfg.final_cutoff_k, cfg.gain_map, cfg.utility_kind,
                                "boundary", len(q.universe), cfg.seed, cfg.rbp_persistence)
        full_q[q.query_id] = full
    approx_full = np.array([full_q[qid].get(did, 0.0) for qid, did in key])
    max_diff = float(np.abs(approx_full - gate["effect"].to_numpy(float)).max())
    if max_diff > 1e-9:
        raise AssertionError(f"probing every outsider does not reproduce E_gate "
                             f"(max |diff| {max_diff:.3g}): the probe code is wrong")

    for policy, seed in configs:
        for budget in _budgets(cfg):
            per_q, calls = {}, {}
            for q in queries:
                per_q[q.query_id], calls[q.query_id] = probe_effects(
                    q, k_c, cfg.final_cutoff_k, cfg.gain_map, cfg.utility_kind,
                    policy, budget, seed, cfg.rbp_persistence)
            approx = np.array([per_q[qid].get(did, 0.0) for qid, did in key])
            g_hat = pd.Series({qid: -sum(e.values()) for qid, e in per_q.items()})
            for scope, qset in scopes.items():
                in_scope = gate["query_id"].isin(qset).to_numpy()
                for stratum in (SURVIVOR, NON_SURVIVOR):
                    sel = in_scope & (gate["stratum"] == stratum).to_numpy()
                    ref = gate["effect"].to_numpy(float)[sel]
                    doc_rows.append({
                        "policy": policy, "seed": seed, "budget": budget, "scope": scope,
                        "stratum": stratum, "n_documents": int(sel.sum()),
                        "mae": mae(approx[sel], ref), "spearman": spearman(approx[sel], ref),
                        "top_m_recovery": top_m_recovery(approx[sel], ref, cfg.top_m)})
                f_sel = failures["query_id"].isin(qset)
                found = [per_q[qid].get(did, 0.0) < -1e-12 for qid, did
                         in zip(failures.loc[f_sel, "query_id"], failures.loc[f_sel, "doc_id"])]
                lost = failures.loc[f_sel, "effect"].to_numpy(float)
                found = np.asarray(found, bool)
                ids = sorted(qset)
                rec = query_level(policy, budget,
                                  float(np.mean([calls[i] for i in ids])),
                                  g_hat.loc[ids], diag, cfg, n_iter)
                rec.update({"seed": seed, "scope": scope,
                            "failures": int(len(found)),
                            "failures_found": int(found.sum()),
                            "loss_recovered_share": float(lost[found].sum() / lost.sum())
                            if lost.sum() else float("nan")})
                query_rows.append(rec)
    return (pd.DataFrame(doc_rows), pd.DataFrame(query_rows),
            {"exact_at_full_budget_max_abs_diff": max_diff})


# ──────────────────────────────────────────────────────────────────────
# The run
# ──────────────────────────────────────────────────────────────────────

def read_split(path: str | Path) -> dict[str, set[str]]:
    raw = json.loads(Path(path).read_text())
    splits = raw.get("splits", raw)
    return {k: {str(x) for x in v} for k, v in splits.items() if isinstance(v, list)}


def _read_effects(path: Path, keep_unselected: bool = False) -> pd.DataFrame:
    cols = ["query_id", "doc_id", "effect_type", "stratum", "judged", "condensed",
            "t", "seed", "effect", "utility_control"]
    filters = [("judged", "==", True)]
    if not keep_unselected:
        filters.append(("stratum", "!=", "unselected_outsider"))
    df = pd.read_parquet(path, columns=cols, filters=filters)
    df["query_id"] = df["query_id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    return df


def run_system(system: str, cfg: FrozenConfig, k_c: int, queries: Sequence[QueryInputs],
               effects_dir: Path, split: Mapping[str, set[str]],
               n_iter: int) -> dict[str, pd.DataFrame | dict]:
    from experiments_shubham.analysis import load_stage

    cost: dict = {"system": system}
    t0 = time.perf_counter()
    tracemalloc.start()
    feats = feature_table(queries, k_c, cfg.final_cutoff_k, cfg.universe_depth_m)
    cost["feature_seconds"] = round(time.perf_counter() - t0, 2)
    cost["feature_peak_mb"] = round(tracemalloc.get_traced_memory()[1] / 2**20, 1)
    tracemalloc.stop()
    cost["feature_rows"] = int(len(feats))

    stage_path = effects_dir / f"{table_name(system, k_c, None, False)}.parquet"
    stage = _read_effects(stage_path)
    fid, macro, info = [], [], {}
    t0 = time.perf_counter()
    blocks = [(E_RERANK, None, stage[stage["effect_type"] == E_RERANK]),
              (E_GATE, None, stage[stage["effect_type"] == E_GATE])]
    for policy in cfg.outsider_policies:
        path = effects_dir / f"{table_name(system, k_c, policy, False)}.parquet"
        blocks.append((DELTA_T, policy, _read_effects(path)))
    tests = {}
    for effect_type, policy, eff in blocks:
        test, info[f"{effect_type}|{policy or ''}"] = predictor_block(
            eff, feats, effect_type, split, cfg.seed)
        tests[(effect_type, policy)] = test
        f = fidelity(test, effect_type, cfg, n_iter)
        f.insert(0, "policy", policy)
        fid.append(f)
        m = macro_fidelity(f)
        m.insert(0, "policy", policy)
        macro.append(m)
        logger.info("%s %s%s: %d test rows", system, effect_type,
                    f" ({policy})" if policy else "", len(test))
    cost["predictor_seconds"] = round(time.perf_counter() - t0, 2)

    # The predictors' own point on the cost axis: G_q from predicted E_gate.
    full_stage = load_stage(effects_dir, system, k_c)
    diag = diagnose(full_stage, cfg.tau_query)
    diag["query_id"] = diag["query_id"].astype(str)
    gate_test = tests[(E_GATE, None)]
    q_rows = []
    for col in predictor_columns(gate_test):
        if gate_test[col].isna().any():
            continue
        g_hat = (-gate_test.groupby("query_id")[col].sum()).reindex(sorted(split["test"]),
                                                                    fill_value=0.0)
        rec = query_level(col.removeprefix("pred_"), 0, 0.0, g_hat, diag, cfg, n_iter)
        rec.update({"seed": None, "scope": "test"})
        q_rows.append(rec)

    t0 = time.perf_counter()
    scopes = {"test": split["test"], "all": {q.query_id for q in queries}}
    probe_doc, probe_q, check = probe_tables(queries, _read_effects(stage_path),
                                             diag, cfg, k_c, scopes, n_iter)
    cost["probe_seconds"] = round(time.perf_counter() - t0, 2)
    cost["exact_reference_calls"] = int(sum(len(q.universe) - k_c for q in queries))
    cost.update(check)
    query = pd.concat([pd.DataFrame(q_rows), probe_q], ignore_index=True)
    for df in (*fid, *macro, query, probe_doc):
        df.insert(0, "system", system)
    return {"fidelity": pd.concat(fid, ignore_index=True),
            "macro": pd.concat(macro, ignore_index=True),
            "query_level": query, "probe_doc": probe_doc, "cost": cost, "fits": info}


def summary(result: Mapping[str, pd.DataFrame]) -> dict:
    """The numbers a reader needs, per system, for the manifest."""
    fid, q = result["fidelity"], result["query_level"]
    out: dict = {}
    for system, f in fid.groupby("system"):
        rec: dict = {"fidelity": {}}
        for (effect_type, policy, stratum), blk in f.groupby(
                ["effect_type", "policy", "stratum"], dropna=False):
            if blk["structural_zero"].any():
                continue
            row = blk.set_index("predictor")
            cell = {p.removeprefix("pred_"): {"mae": round(float(row.loc[p, "mae"]), 4),
                                              "spearman": None if pd.isna(row.loc[p, "spearman"])
                                              else round(float(row.loc[p, "spearman"]), 3)}
                    for p in row.index if pd.notna(row.loc[p, "mae"])}
            name = f"{effect_type}|{'' if pd.isna(policy) else policy}|{stratum}"
            rec["fidelity"][name] = cell
        # Predictors are scored on test only; probes need no training, so they
        # are also given on all queries, which is their headline scope.
        for scope in ("test", "all"):
            qs = q[(q["system"] == system) & (q["scope"] == scope)]
            cols = [c for c in ("extra_calls_per_query", "gq_spearman", "gs_f1",
                                "loss_recovered_share") if c in qs]
            table = (qs.groupby(["approximation", "budget"], dropna=False)[cols]
                     .mean().round(3).reset_index())
            rec[f"query_level_{scope}"] = (table.astype(object)
                                           .where(table.notna(), None).to_dict("records"))
        out[system] = rec
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Predictors and probes against the reference effects")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument("--k-c", type=int, default=None)
    parser.add_argument("--effects-dir", default="effects/real")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--query-split", default="data/query_split.json")
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--out-dir", default="analysis/prediction")
    args = parser.parse_args()

    from experiments_shubham.real import load_queries
    from shared.cache import Stage2Cache

    cfg = load_config(args.config)
    k_c = args.k_c or headline_k_c(cfg)[0]
    n_iter = args.n_bootstrap or cfg.bootstrap_iterations
    split = read_split(args.query_split)
    cache = Stage2Cache(args.cache or cfg.paths.get("cache", "cache/stage2.sqlite"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    parts: dict[str, list] = {}
    costs, fits = [], {}
    for system in args.systems or cfg.stage1_systems:
        t0 = time.perf_counter()
        queries = load_queries(cfg, system, cache)
        load_s = round(time.perf_counter() - t0, 2)
        res = run_system(system, cfg, k_c, queries, Path(args.effects_dir), split, n_iter)
        res["cost"]["load_seconds"] = load_s
        costs.append(res["cost"])
        fits[system] = res["fits"]
        for name in ("fidelity", "macro", "query_level", "probe_doc"):
            parts.setdefault(name, []).append(res[name])
    tables = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    tables["cost"] = pd.DataFrame(costs)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)

    write_manifest(out / "prediction_manifest.json", run_manifest(
        cfg.fingerprint(), owner="shubham", extra={
            "effects_dir": args.effects_dir, "k_c": k_c, "query_split": args.query_split,
            "split_sizes": {k: len(v) for k, v in split.items()},
            "trained_on": "train", "evaluated_on": "test (predictors); test and all (probes)",
            "models": list(MODELS), "features": {"shared": list(SHARED_FEATURES),
                                                 "survivor": list(SURVIVOR_FEATURES)},
            "probe_budgets": _budgets(cfg), "bootstrap_iterations": n_iter,
            "evaluation_population": "judged documents",
            "cost": costs, "fits": fits, "summary": summary(tables),
            "files": sorted(p.name for p in out.iterdir()),
            "seconds": round(time.perf_counter() - start, 1),
        }))
    logger.info("written to %s", out)


if __name__ == "__main__":
    main()
