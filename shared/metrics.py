"""
metrics.py - Predictor fidelity, computed within strata.

The rule this module enforces mechanically: for E_gate and Delta_t the sign of
the effect is fixed by membership, so a POOLED statistic rewards a predictor
for detecting a partition it was handed for free. Every evaluator here returns
one row per stratum plus a macro average, and the pooled variant is available
only behind an explicit flag that labels its own output as secondary.

Bootstrap resampling is over QUERIES, never documents. Documents are nested
within queries, so resampling documents treats correlated observations as
independent and produces intervals far too narrow.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from .strata import INFORMATIVE_STRATA, STRUCTURAL_ZERO

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Point metrics
# ──────────────────────────────────────────────────────────────────────

def _clean(predicted, reference) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(predicted, dtype=float)
    r = np.asarray(reference, dtype=float)
    mask = np.isfinite(p) & np.isfinite(r)
    return p[mask], r[mask]


def mae(predicted, reference) -> float:
    p, r = _clean(predicted, reference)
    return float(np.mean(np.abs(p - r))) if len(p) else float("nan")


def rmse(predicted, reference) -> float:
    p, r = _clean(predicted, reference)
    return float(np.sqrt(np.mean((p - r) ** 2))) if len(p) else float("nan")


def signed_bias(predicted, reference) -> float:
    """Mean signed error: separates systematic over/under-prediction from noise."""
    p, r = _clean(predicted, reference)
    return float(np.mean(p - r)) if len(p) else float("nan")


def spearman(predicted, reference) -> float:
    """Spearman rho. NaN for constant input rather than an exception."""
    p, r = _clean(predicted, reference)
    if len(p) < 3 or np.allclose(p, p[0]) or np.allclose(r, r[0]):
        return float("nan")
    return float(sps.spearmanr(p, r).statistic)


def pearson(predicted, reference) -> float:
    p, r = _clean(predicted, reference)
    if len(p) < 3 or np.allclose(p, p[0]) or np.allclose(r, r[0]):
        return float("nan")
    return float(sps.pearsonr(p, r).statistic)


def top_m_recovery(predicted, reference, m: int, use_abs: bool = True) -> float:
    """
    Fraction of the m largest reference effects the predictor also ranks in its
    own top m. The operationally relevant question is whether a cheap signal
    LOCALISES the documents a stage really moved, not whether it tracks the
    whole distribution.
    """
    p, r = _clean(predicted, reference)
    if len(r) == 0:
        return float("nan")
    m = min(m, len(r))
    key_p, key_r = (np.abs(p), np.abs(r)) if use_abs else (p, r)
    top_r = set(np.argsort(-key_r)[:m].tolist())
    top_p = set(np.argsort(-key_p)[:m].tolist())
    return len(top_r & top_p) / m


def calibration_curve(predicted, reference, n_bins: int = 10) -> pd.DataFrame:
    """
    Mean predicted against mean reference, binned by reference magnitude.
    Answers whether errors are systematic (and correctable) or just noise.
    """
    p, r = _clean(predicted, reference)
    if len(r) == 0:
        return pd.DataFrame(columns=["bin", "n", "mean_reference", "mean_predicted"])
    edges = np.unique(np.quantile(r, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return pd.DataFrame(
            [{"bin": 0, "n": len(r), "mean_reference": float(r.mean()),
              "mean_predicted": float(p.mean())}]
        )
    idx = np.clip(np.digitize(r, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            rows.append(
                {"bin": b, "n": int(sel.sum()),
                 "mean_reference": float(r[sel].mean()),
                 "mean_predicted": float(p[sel].mean())}
            )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Bootstrap over queries
# ──────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """
    Percentile bootstrap resampling QUERIES.

    Resampling documents would treat observations nested within a query as
    independent and give intervals that are far too narrow.
    """
    rng = np.random.default_rng(seed)
    keys = df["query_id"].astype(str)
    queries = keys.unique()
    if len(queries) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}

    groups = {q: g for q, g in df.groupby(keys, observed=True)}
    point = statistic(df)

    draws = []
    for _ in range(n_iter):
        pick = rng.choice(queries, size=len(queries), replace=True)
        sample = pd.concat([groups[q] for q in pick], ignore_index=True)
        val = statistic(sample)
        if np.isfinite(val):
            draws.append(val)

    if not draws:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"),
                "n": int(len(queries)), "n_draws": 0}
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "point": float(point), "lo": float(lo), "hi": float(hi),
        "n": int(len(queries)), "n_draws": len(draws),
    }


# ──────────────────────────────────────────────────────────────────────
# Stratified evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_within_strata(
    frame: pd.DataFrame,
    predicted_column: str,
    effect_type: str,
    target_column: str = "effect",
    stratum_column: str = "stratum",
    judged_only: bool = True,
    top_m: int = 20,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    min_documents: int = 30,
) -> pd.DataFrame:
    """
    Fidelity of one predictor against one effect type, within stratum.

    Returns one row per stratum. There is intentionally no pooled row: see the
    module docstring. Strata that are structurally zero for this effect type
    get counts only, with correlations suppressed, because a correlation
    against a constant is undefined and reporting NaN invites someone to fill
    the cell in later.
    """
    df = frame[frame["effect_type"] == effect_type]
    if judged_only:
        df = df[df["judged"].fillna(False)]
    if predicted_column not in df.columns:
        raise KeyError(
            f"predictor column {predicted_column!r} is not present; join the "
            "feature table to the effect table before evaluating"
        )

    zero_strata = set(STRUCTURAL_ZERO.get(effect_type, ()))
    rows = []
    for stratum, grp in df.groupby(stratum_column, observed=True):
        pred = grp[predicted_column].to_numpy(dtype=float)
        ref = grp[target_column].to_numpy(dtype=float)
        structural = str(stratum) in zero_strata

        rec: dict[str, object] = {
            "effect_type": effect_type,
            "predictor": predicted_column,
            "stratum": str(stratum),
            "structural_zero": structural,
            "n_documents": int(len(grp)),
            "n_queries": int(grp["query_id"].nunique()),
            "underpowered": bool(len(grp) < min_documents),
        }

        if structural:
            rec.update(
                {"mae": mae(pred, ref), "rmse": float("nan"),
                 "signed_bias": signed_bias(pred, ref),
                 "spearman": float("nan"), "pearson": float("nan"),
                 "top_m_recovery": float("nan"),
                 "mae_lo": float("nan"), "mae_hi": float("nan"),
                 "spearman_lo": float("nan"), "spearman_hi": float("nan"),
                 "note": "effect is zero by construction; counts only"}
            )
            rows.append(rec)
            continue

        def _mae(d: pd.DataFrame) -> float:
            return mae(d[predicted_column].to_numpy(float), d[target_column].to_numpy(float))

        def _rho(d: pd.DataFrame) -> float:
            return spearman(d[predicted_column].to_numpy(float), d[target_column].to_numpy(float))

        mae_ci = bootstrap_ci(grp, _mae, n_iter, alpha, seed)
        rho_ci = bootstrap_ci(grp, _rho, n_iter, alpha, seed)

        rec.update(
            {
                "mae": mae(pred, ref),
                "rmse": rmse(pred, ref),
                "signed_bias": signed_bias(pred, ref),
                "spearman": spearman(pred, ref),
                "pearson": pearson(pred, ref),
                "top_m_recovery": top_m_recovery(pred, ref, top_m),
                "mae_lo": mae_ci["lo"], "mae_hi": mae_ci["hi"],
                "spearman_lo": rho_ci["lo"], "spearman_hi": rho_ci["hi"],
                "note": "",
            }
        )
        rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out) and out["underpowered"].any():
        thin = out.loc[out["underpowered"], "stratum"].tolist()
        logger.warning(
            "strata %s have fewer than %d documents; their correlations are "
            "noise wearing a number", thin, min_documents,
        )
    return out


def macro_fidelity(per_stratum: pd.DataFrame) -> pd.DataFrame:
    """
    Macro-average the informative strata into one headline row per predictor.

    Unweighted on purpose: weighting by stratum size would let the largest
    stratum dominate, and for Delta_t the largest is usually the structurally
    zero one.
    """
    rows = []
    for (effect_type, predictor), block in per_stratum.groupby(
        ["effect_type", "predictor"], observed=True
    ):
        informative = set(INFORMATIVE_STRATA.get(str(effect_type), ()))
        sel = block[block["stratum"].isin(informative)] if informative else block
        sel = sel[~sel["structural_zero"]]
        if sel.empty:
            continue
        rec = {"effect_type": effect_type, "predictor": predictor,
               "n_strata": int(len(sel)),
               "strata": ",".join(sorted(sel["stratum"].astype(str)))}
        for col in ("mae", "rmse", "spearman", "pearson", "top_m_recovery",
                    "signed_bias"):
            vals = pd.to_numeric(sel[col], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            rec[f"macro_{col}"] = float(vals.mean()) if len(vals) else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def pooled_fidelity(
    frame: pd.DataFrame,
    predicted_column: str,
    effect_type: str,
    target_column: str = "effect",
    judged_only: bool = True,
) -> pd.DataFrame:
    """
    The pooled statistic, for completeness only.

    Behind an explicit call and labelled in its own output, because for E_gate
    and Delta_t a pooled correlation mostly measures membership detection. Any
    pooled number reported must sit beside the corresponding shortcut baseline
    from `shared.baselines`, and must never be the headline.
    """
    df = frame[frame["effect_type"] == effect_type]
    if judged_only:
        df = df[df["judged"].fillna(False)]
    pred = df[predicted_column].to_numpy(float)
    ref = df[target_column].to_numpy(float)
    from .strata import describe_shortcut

    return pd.DataFrame(
        [
            {
                "effect_type": effect_type,
                "predictor": predicted_column,
                "scope": "pooled_secondary",
                "n_documents": int(len(df)),
                "mae": mae(pred, ref),
                "spearman": spearman(pred, ref),
                "pearson": pearson(pred, ref),
                "warning": describe_shortcut(effect_type),
            }
        ]
    )


def cost_fidelity_curve(
    results: Sequence[tuple[int, pd.DataFrame]],
    stratum: str,
    metric: str = "spearman",
) -> pd.DataFrame:
    """
    Assemble a budget sweep into extra Stage-2 calls against fidelity.

    `results` is [(extra_calls, per_stratum_frame), ...]. The curve is the most
    directly useful artefact the project can hand a practitioner, and it must
    converge on the exact reference as the budget reaches the full universe.
    """
    rows = []
    for budget, frame in results:
        sel = frame[frame["stratum"] == stratum]
        if sel.empty:
            continue
        r = sel.iloc[0]
        rows.append(
            {
                "extra_stage2_calls": int(budget),
                "stratum": stratum,
                "predictor": r.get("predictor"),
                metric: r.get(metric),
                "mae": r.get("mae"),
                "top_m_recovery": r.get("top_m_recovery"),
                "n_documents": r.get("n_documents"),
            }
        )
    return pd.DataFrame(rows).sort_values("extra_stage2_calls")


__all__ = [
    "mae", "rmse", "signed_bias", "spearman", "pearson", "top_m_recovery",
    "calibration_curve", "bootstrap_ci", "evaluate_within_strata",
    "macro_fidelity", "pooled_fidelity", "cost_fidelity_curve",
]
