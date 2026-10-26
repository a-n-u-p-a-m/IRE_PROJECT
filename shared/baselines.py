"""
baselines.py - The bars every real predictor has to clear.

Three of them, each the operationalisation of a way to look good while knowing
nothing.

zero
    Predicts no effect anywhere. For survivors under E_gate, and for retained
    candidates under Delta_t, this is a genuinely hard baseline: most documents
    are barely moved, so "nothing happened" is usually right. A predictor that
    cannot beat it on MAE has not earned its compute.

membership_only
    Predicts +1 for survivors and -1 for non-survivors. Carries no magnitude
    information at all, yet scores a large POOLED correlation against E_gate
    because sign(E_gate) is fixed by survivorship. Report it beside any pooled
    number so the reader can see how much of that number was free.

stratum_mean
    Predicts the training-set mean effect of a document's stratum, and exactly
    zero for the structurally zero stratum. This is the bar inside a stratum:
    beating it means a predictor knows something about WHICH documents moved,
    not merely which group they fell in.

All three are fitted on training queries only and applied to held-out ones,
like any other predictor - otherwise the comparison is not like for like.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .schema import DELTA_T, NON_SURVIVOR, SURVIVOR
from .strata import STRUCTURAL_ZERO, describe_shortcut


def zero_baseline(frame: pd.DataFrame, column: str = "pred_zero") -> pd.DataFrame:
    """Attach a column of zeros."""
    out = frame.copy()
    out[column] = 0.0
    return out


def membership_baseline(
    frame: pd.DataFrame, column: str = "pred_membership"
) -> pd.DataFrame:
    """
    Attach +1 for survivors, -1 for non-survivors.

    Defined only where survivorship strata apply (E_gate, E_rerank); rows with
    allocation strata get NaN so they cannot be silently averaged in.
    """
    out = frame.copy()
    out[column] = np.where(
        out["stratum"] == SURVIVOR, 1.0,
        np.where(out["stratum"] == NON_SURVIVOR, -1.0, np.nan),
    )
    return out


def fit_stratum_means(
    train: pd.DataFrame,
    effect_type: str,
    target_column: str = "effect",
) -> dict[str, float]:
    """
    Mean effect per stratum, learned on training queries.

    Structurally zero strata are pinned to exactly 0.0 rather than to their
    sample mean, which is 0.0 anyway but should not depend on the sample.
    """
    block = train[train["effect_type"] == effect_type]
    means = (
        block.groupby("stratum", observed=True)[target_column].mean().to_dict()
    )
    for stratum in STRUCTURAL_ZERO.get(effect_type, ()):
        means[stratum] = 0.0
    return {str(k): float(v) for k, v in means.items()}


def apply_stratum_means(
    frame: pd.DataFrame,
    means: Mapping[str, float],
    column: str = "pred_stratum_mean",
) -> pd.DataFrame:
    """Attach the fitted stratum means as a predictor column."""
    out = frame.copy()
    out[column] = out["stratum"].astype(str).map(dict(means)).astype(float)
    # A stratum unseen in training gets the overall training mean, not NaN,
    # so the baseline stays defined on held-out data.
    if out[column].isna().any():
        fallback = float(np.mean(list(means.values()))) if means else 0.0
        out[column] = out[column].fillna(fallback)
    return out


def attach_all(
    frame: pd.DataFrame,
    effect_type: str,
    train: pd.DataFrame | None = None,
    target_column: str = "effect",
) -> pd.DataFrame:
    """
    Attach all three baselines in one call.

    `train` supplies the stratum means; when omitted the frame itself is used,
    which is fine for a synthetic sanity check and NOT fine for a reported
    result, where the means must come from training queries only.
    """
    out = zero_baseline(frame)
    out = membership_baseline(out)
    source = train if train is not None else frame
    out = apply_stratum_means(out, fit_stratum_means(source, effect_type, target_column))
    return out


def baseline_report(
    frame: pd.DataFrame,
    effect_type: str,
    target_column: str = "effect",
) -> pd.DataFrame:
    """
    Pooled correlation of the membership baseline, with the warning attached.

    Print this next to every pooled statistic. On a typical cascade the
    membership baseline alone reaches a pooled Spearman well clear of zero
    against E_gate, which is exactly how much of a pooled number comes free.
    """
    from .metrics import spearman

    block = frame[frame["effect_type"] == effect_type]
    if "pred_membership" not in block.columns:
        block = membership_baseline(block)
    rho = spearman(
        block["pred_membership"].to_numpy(float),
        block[target_column].to_numpy(float),
    )
    return pd.DataFrame(
        [
            {
                "effect_type": effect_type,
                "baseline": "membership_only",
                "pooled_spearman": rho,
                "n_documents": int(len(block)),
                "warning": describe_shortcut(effect_type),
            }
        ]
    )


def delta_t_stratum_report(
    frame: pd.DataFrame, target_column: str = "effect"
) -> pd.DataFrame:
    """
    The Delta_t analogue: how much of the signal is pure allocation membership.

    Three of the four allocation strata have fixed signs, so this is the number
    a stratum-blind predictor gets for free on the budget experiment.
    """
    from .metrics import spearman

    block = frame[frame["effect_type"] == DELTA_T].copy()
    if block.empty:
        return pd.DataFrame()
    means = fit_stratum_means(block, DELTA_T, target_column)
    block = apply_stratum_means(block, means)
    rho = spearman(
        block["pred_stratum_mean"].to_numpy(float),
        block[target_column].to_numpy(float),
    )
    return pd.DataFrame(
        [
            {
                "effect_type": DELTA_T,
                "baseline": "stratum_mean",
                "pooled_spearman": rho,
                "n_documents": int(len(block)),
                "stratum_means": {k: round(v, 6) for k, v in means.items()},
                "warning": describe_shortcut(DELTA_T),
            }
        ]
    )


__all__ = [
    "zero_baseline", "membership_baseline", "fit_stratum_means",
    "apply_stratum_means", "attach_all", "baseline_report",
    "delta_t_stratum_report",
]
