"""
strata.py - Membership strata and why every statistic lives inside one.

Two separate stratifications, for two separate reasons, both of which exist to
stop a predictor earning credit for knowing nothing but where a document sits.

E_gate - survivorship (proposal section 3)
    Under a pointwise reranker and a monotone rank utility, E_gate >= 0 for
    survivors and <= 0 for non-survivors. So sign(E_gate) is a deterministic
    function of membership in C_q. A predictor that does nothing but detect
    the Stage-1 cutoff therefore scores a large POOLED correlation against
    E_gate while predicting nothing whatever about magnitude.

Delta_t - allocation membership (proposal section 4.4)
    Three of the four groups have signs fixed by membership: removed
    candidates can only lose, promoted outsiders can only gain, and unselected
    outsiders are structurally zero. The same shortcut, one layer along.

The remedy in both cases is the same. Report within stratum, macro-average
across strata, and compare against a baseline that knows ONLY the stratum. If
a real predictor cannot beat that baseline inside a stratum, it has learned
the partition and nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .schema import (
    DELTA_T,
    E_GATE,
    E_RERANK,
    NON_SURVIVOR,
    REMOVED,
    RETAINED,
    SELECTED_OUTSIDER,
    SURVIVOR,
    T_STRATA,
    UNSELECTED_OUTSIDER,
    single_mode,
)

# Which strata carry usable magnitude information for each effect type. The
# others are either structurally zero or sign-determined with little spread,
# and reporting a correlation on them is misleading.
INFORMATIVE_STRATA: dict[str, tuple[str, ...]] = {
    E_RERANK: (SURVIVOR,),
    E_GATE: (SURVIVOR, NON_SURVIVOR),
    DELTA_T: (RETAINED, REMOVED, SELECTED_OUTSIDER),
}

# Strata whose effect is zero by construction. Report counts, never correlation.
STRUCTURAL_ZERO: dict[str, tuple[str, ...]] = {
    E_RERANK: (NON_SURVIVOR,),
    E_GATE: (),
    DELTA_T: (UNSELECTED_OUTSIDER,),
}


def survivorship_stratum(doc_id: str, candidate_set: set[str]) -> str:
    return SURVIVOR if doc_id in candidate_set else NON_SURVIVOR


def survivorship_map(
    universe: Sequence[str], candidate_set: set[str]
) -> dict[str, str]:
    return {d: survivorship_stratum(d, candidate_set) for d in universe}


def strata_for(effect_type: str) -> tuple[str, ...]:
    """Every stratum label valid for an effect type."""
    if effect_type in (E_RERANK, E_GATE):
        return (SURVIVOR, NON_SURVIVOR)
    if effect_type == DELTA_T:
        return T_STRATA
    raise ValueError(f"unknown effect type {effect_type!r}")


def check_strata_valid(effects: pd.DataFrame) -> None:
    """Every row's stratum must belong to its effect type."""
    for effect_type, block in effects.groupby("effect_type", observed=True):
        allowed = set(strata_for(str(effect_type)))
        found = set(block["stratum"].dropna().unique())
        bad = found - allowed
        if bad:
            raise ValueError(
                f"effect_type {effect_type!r} has strata {sorted(bad)}, which "
                f"belong to a different effect type; allowed: {sorted(allowed)}"
            )


def stratum_sizes(effects: pd.DataFrame) -> pd.DataFrame:
    """Row and query counts per (effect_type, stratum). Report this always."""
    single_mode(effects, "stratum_sizes")
    return (
        effects.groupby(["effect_type", "stratum"], observed=True)
        .agg(
            n_documents=("doc_id", "size"),
            n_queries=("query_id", "nunique"),
            n_judged=("judged", lambda s: int(s.fillna(False).sum())),
            mean_effect=("effect", "mean"),
            mean_abs_effect=("effect", lambda s: float(s.abs().mean())),
        )
        .reset_index()
    )


def thin_strata(effects: pd.DataFrame, min_documents: int = 30) -> pd.DataFrame:
    """
    Strata too small to support a correlation.

    Call before reporting. A Spearman computed on eight documents is noise
    wearing a number, and a stratified table invites exactly that mistake by
    guaranteeing every cell gets filled in.
    """
    sizes = stratum_sizes(effects)
    return sizes[sizes["n_documents"] < min_documents]


def macro_average(
    per_stratum: pd.DataFrame,
    value_columns: Sequence[str],
    effect_type_column: str = "effect_type",
    weight_column: str | None = None,
) -> pd.DataFrame:
    """
    Average a per-stratum table across strata.

    Unweighted by default, deliberately. Weighting by stratum size would let
    the largest stratum - usually unselected outsiders, which are structurally
    zero - dominate the headline number and reintroduce the shortcut the
    stratification exists to remove.
    """
    rows = []
    for effect_type, block in per_stratum.groupby(effect_type_column, observed=True):
        informative = set(INFORMATIVE_STRATA.get(str(effect_type), ()))
        sel = block[block["stratum"].isin(informative)] if informative else block
        if sel.empty:
            continue
        rec: dict[str, object] = {
            effect_type_column: effect_type,
            "n_strata": int(len(sel)),
            "strata": ",".join(sorted(sel["stratum"].astype(str))),
        }
        for col in value_columns:
            if col not in sel.columns:
                continue
            vals = pd.to_numeric(sel[col], errors="coerce").to_numpy(dtype=float)
            if weight_column and weight_column in sel.columns:
                w = pd.to_numeric(sel[weight_column], errors="coerce").to_numpy(float)
                mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
                rec[f"macro_{col}"] = (
                    float(np.average(vals[mask], weights=w[mask])) if mask.any()
                    else float("nan")
                )
            else:
                mask = np.isfinite(vals)
                rec[f"macro_{col}"] = float(vals[mask].mean()) if mask.any() else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def describe_shortcut(effect_type: str) -> str:
    """One line explaining which shortcut stratification removes. For reports."""
    if effect_type == E_GATE:
        return (
            "sign(E_gate) is fixed by survivorship, so a pooled correlation "
            "rewards detecting the Stage-1 cutoff rather than predicting magnitude"
        )
    if effect_type == DELTA_T:
        return (
            "three of four allocation strata have signs fixed by membership, so "
            "a pooled correlation rewards detecting the allocation rather than "
            "predicting magnitude"
        )
    if effect_type == E_RERANK:
        return (
            "E_rerank is exactly zero off the candidate set, so including "
            "non-survivors pads the sample with a structural constant"
        )
    return ""


__all__ = [
    "INFORMATIVE_STRATA", "STRUCTURAL_ZERO",
    "survivorship_stratum", "survivorship_map", "strata_for",
    "check_strata_valid", "stratum_sizes", "thin_strata", "macro_average",
    "describe_shortcut",
]
