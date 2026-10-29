"""
effects.py - The three per-document estimands (proposal section 3).

    E_rerank(q,d) = U_d(pi_act) - U_d(pi_no2)
    E_gate(q,d)   = U_d(pi_act) - U_d(pi_no1)
    Delta_t(q,d)  = U_d(pi_t)   - U_d(pi_{t=1})

All three are signed, bounded and NOT an additive decomposition. Each is a
comparison against a different reference world, so in general

    E_gate + E_rerank != U_d(pi_act)

and any sum-to-one presentation would be a category error. `check_additivity`
exists only to demonstrate that, not to enforce it.

Rows are produced in long format with an `effect_type` column, because the
three estimands carry different invariants and different strata and must never
share one unconditional validator. `shared.schema.validate_effect` dispatches
the sign checks on that column.

Rows are emitted for EVERY document in the universe, judged or not. Restricting
to the judged subset happens at analysis time, so the unjudged competitors stay
visible in the ranking that produced the numbers - they really did occupy those
positions.

That is the primary mode. `condensed=True` gives the robustness check decided
at freeze v2: every world's ranking is condensed (`utility.condense`) before
ranks are assigned, so a judged document's rank counts only judged documents
above it. Worlds, candidate sets and allocations are unchanged - only the
ranks are recounted - so it needs no Stage-2 call. Unjudged documents have no
rank in a condensed world, and rows are emitted for judged documents only.
The sign invariants survive condensing: dropping the same documents from both
rankings cannot move a survivor above a judged competitor it was below.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import pandas as pd

from .schema import (
    DELTA_T,
    E_GATE,
    E_RERANK,
    EPS,
    NON_SURVIVOR,
    SURVIVOR,
    empty,
    validate_effect,
)
from .utility import (
    GainMap,
    UtilityKind,
    condense,
    document_utility,
    is_monotone,
    ranks_of,
)
from .worlds import TWorld, Worlds

logger = logging.getLogger(__name__)

# Presentation labels (section 3). The threshold is pre-registered as
# diagnosis.tau_document; the continuous signed effects remain the scientific
# object and a label distribution never substitutes for an effect distribution.
GATE_SENSITIVE = "gate_sensitive"
RERANK_SENSITIVE = "rerank_sensitive"
JOINTLY_SENSITIVE = "jointly_sensitive"
STABLE = "stable"
GATING_FAILURE = "gating_failure"


def _row(
    *,
    query_id: str,
    doc_id: str,
    stage1_system: str,
    effect_type: str,
    k_c: int,
    k: int,
    utility: str,
    stratum: str,
    judged: bool,
    rank_treatment: int | None,
    rank_control: int | None,
    u_treatment: float,
    u_control: float,
    condensed: bool = False,
    budget_b: int | None = None,
    t: float | None = None,
    policy: str | None = None,
    seed: int | None = None,
) -> dict:
    return {
        "query_id": query_id,
        "doc_id": doc_id,
        "stage1_system": stage1_system,
        "effect_type": effect_type,
        "k_c": k_c,
        "k": k,
        "utility": utility,
        "condensed": condensed,
        "budget_b": budget_b,
        "t": t,
        "policy": policy,
        "seed": seed,
        "stratum": stratum,
        "judged": judged,
        "rank_treatment": rank_treatment,
        "rank_control": rank_control,
        "utility_treatment": u_treatment,
        "utility_control": u_control,
        "effect": u_treatment - u_control,
    }


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty("effect")
    for col in ("query_id", "doc_id", "stage1_system", "effect_type", "utility",
                "policy", "stratum"):
        df[col] = df[col].astype("string")
    for col in ("k_c", "k", "budget_b", "seed", "rank_treatment", "rank_control"):
        df[col] = df[col].astype("Int64")
    df["t"] = df["t"].astype("float64")
    df["judged"] = df["judged"].astype("boolean")
    df["condensed"] = df["condensed"].astype("boolean")
    return df


def _judged_only(
    universe: Sequence[str],
    rankings: Sequence[Sequence[str]],
    judged: Mapping[str, bool],
) -> tuple[list[str], list[list[str]]]:
    """The judged documents, and every ranking condensed to them."""
    keep = {d for d in universe if judged.get(d, False)}
    return [d for d in universe if d in keep], [condense(r, keep) for r in rankings]


# ──────────────────────────────────────────────────────────────────────
# E_rerank and E_gate
# ──────────────────────────────────────────────────────────────────────

def stage_effects_for_query(
    worlds: Worlds,
    relevances: Mapping[str, int],
    judged: Mapping[str, bool],
    stage1_system: str,
    k: int,
    gain_map: GainMap,
    utility: UtilityKind = "dcg",
    persistence: float = 0.8,
    *,
    condensed: bool = False,
) -> pd.DataFrame:
    """E_rerank and E_gate for every document in the universe, for one query."""
    docs = worlds.universe
    rankings = (worlds.pi_act, worlds.pi_no2, worlds.pi_no1)
    if condensed:
        docs, rankings = _judged_only(docs, rankings, judged)
    r_act, r_no2, r_no1 = (ranks_of(r) for r in rankings)
    cand = worlds.candidate_set

    def u(rank: int | None, doc: str) -> float:
        return document_utility(
            rank, relevances.get(doc), k, gain_map, utility, persistence
        )

    rows = []
    for doc in docs:
        stratum = SURVIVOR if doc in cand else NON_SURVIVOR
        is_judged = bool(judged.get(doc, False))
        ra, r2, r1 = r_act.get(doc), r_no2.get(doc), r_no1.get(doc)
        ua, u2, u1 = u(ra, doc), u(r2, doc), u(r1, doc)

        common = dict(
            query_id=worlds.query_id, doc_id=doc, stage1_system=stage1_system,
            k_c=worlds.k_c, k=k, utility=utility, stratum=stratum, judged=is_judged,
            condensed=condensed,
        )
        rows.append(_row(
            effect_type=E_RERANK, rank_treatment=ra, rank_control=r2,
            u_treatment=ua, u_control=u2, **common,
        ))
        rows.append(_row(
            effect_type=E_GATE, rank_treatment=ra, rank_control=r1,
            u_treatment=ua, u_control=u1, **common,
        ))

    return _typed(pd.DataFrame(rows))


# ──────────────────────────────────────────────────────────────────────
# Delta_t
# ──────────────────────────────────────────────────────────────────────

def budget_effects_for_query(
    worlds: Worlds,
    t_worlds: Mapping[float, TWorld],
    relevances: Mapping[str, int],
    judged: Mapping[str, bool],
    stage1_system: str,
    k: int,
    gain_map: GainMap,
    utility: UtilityKind = "dcg",
    persistence: float = 0.8,
    *,
    condensed: bool = False,
) -> pd.DataFrame:
    """
    Delta_t for every t in the grid, measured against the t=1 world.

    The baseline is computed from the t=1 world explicitly rather than reused
    from pi_act, because the two coincide only when B = K_c. Every row carries
    budget_b and t so a budget effect can never be read without the budget it
    was measured at.
    """
    if 1.0 not in t_worlds:
        raise ValueError(
            "budget effects need the t=1 world as a baseline; build it even "
            "when it is not otherwise of interest"
        )
    docs = worlds.universe
    rank_t = {t: tw.pi_t for t, tw in t_worlds.items()}
    if condensed:
        docs, ranked = _judged_only(docs, list(rank_t.values()), judged)
        rank_t = dict(zip(rank_t, ranked))
    r_base = ranks_of(rank_t[1.0])

    def u(rank: int | None, doc: str) -> float:
        return document_utility(
            rank, relevances.get(doc), k, gain_map, utility, persistence
        )

    rows = []
    for t, tw in sorted(t_worlds.items()):
        if t == 1.0:
            continue                      # the baseline has nothing to compare to
        r_t = ranks_of(rank_t[t])
        for doc in docs:
            rows.append(_row(
                query_id=worlds.query_id, doc_id=doc, stage1_system=stage1_system,
                effect_type=DELTA_T, k_c=worlds.k_c, k=k, utility=utility,
                stratum=tw.strata.get(doc, ""), judged=bool(judged.get(doc, False)),
                rank_treatment=r_t.get(doc), rank_control=r_base.get(doc),
                u_treatment=u(r_t.get(doc), doc), u_control=u(r_base.get(doc), doc),
                condensed=condensed, budget_b=tw.budget_b, t=tw.t, policy=tw.policy,
                seed=tw.seed,
            ))

    return _typed(pd.DataFrame(rows))


# ──────────────────────────────────────────────────────────────────────
# Assembly and checks
# ──────────────────────────────────────────────────────────────────────

def concat_effects(frames: Sequence[pd.DataFrame], strict: bool = True) -> pd.DataFrame:
    """Concatenate per-query effect frames and validate the result."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return empty("effect")
    return validate_effect(pd.concat(frames, ignore_index=True), strict=strict)


def check_invariants(
    effects: pd.DataFrame,
    utility: UtilityKind = "dcg",
    pointwise: bool = True,
) -> None:
    """
    Verify the structural properties of the effect table.

    Delegates to the schema validator, which dispatches sign checks on
    effect_type, and adds the precondition for the survivor monotonicity
    result: it holds for a POINTWISE Stage 2 under a MONOTONE rank utility.
    A listwise reranker re-scores survivors when the candidate set changes, so
    pass pointwise=False there and say in the write-up that the estimand
    changed.
    """
    if pointwise and not is_monotone(utility):
        raise ValueError(
            f"utility {utility!r} is not monotone in rank, so the survivor "
            "invariant E_gate >= 0 does not follow. Either use a monotone "
            "utility or pass pointwise=False and document the change."
        )
    validate_effect(effects, strict=pointwise)


def check_additivity(effects: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrate that the two stage effects do NOT sum to the actual utility.

    Returns the residual per document. This is evidence for the write-up, not
    a failure: the two interventions are comparisons against different
    reference worlds, so a sum-to-one decomposition would be a category error.
    """
    wide = (
        effects[effects["effect_type"].isin([E_RERANK, E_GATE])]
        .pivot_table(
            index=["query_id", "doc_id", "stage1_system", "k_c", "k", "utility",
                   "condensed"],
            columns="effect_type",
            values=["effect", "utility_treatment"],
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns = [
        "_".join(str(p) for p in col if p) if isinstance(col, tuple) else col
        for col in wide.columns
    ]
    e_r = wide.get(f"effect_{E_RERANK}")
    e_g = wide.get(f"effect_{E_GATE}")
    u_a = wide.get(f"utility_treatment_{E_RERANK}")
    if e_r is None or e_g is None or u_a is None:
        return pd.DataFrame()
    wide["additivity_residual"] = e_r + e_g - u_a
    return wide


def label_documents(effects: pd.DataFrame, tau_document: float) -> pd.DataFrame:
    """
    Categorical presentation labels per (query, doc).

    `tau_document` must be pre-registered. These labels are for plots and
    error analysis only; a label distribution never substitutes for the effect
    distribution.
    """
    stage = effects[effects["effect_type"].isin([E_RERANK, E_GATE])]
    wide = stage.pivot_table(
        index=["query_id", "doc_id", "stage1_system", "k_c", "k", "utility",
               "condensed", "stratum"],
        columns="effect_type",
        values=["effect", "utility_control"],
        aggfunc="first",
    ).reset_index()
    wide.columns = [
        "_".join(str(p) for p in col if p) if isinstance(col, tuple) else col
        for col in wide.columns
    ]

    e_g = wide[f"effect_{E_GATE}"].abs()
    e_r = wide[f"effect_{E_RERANK}"].abs()
    # utility_control for e_gate is the no-gate utility: what the document
    # would have contributed had the gate not excluded it.
    u_no1 = wide[f"utility_control_{E_GATE}"]

    label = pd.Series(STABLE, index=wide.index, dtype="object")
    label[(e_g > tau_document) & (e_r <= tau_document)] = GATE_SENSITIVE
    label[(e_r > tau_document) & (e_g <= tau_document)] = RERANK_SENSITIVE
    label[(e_g > tau_document) & (e_r > tau_document)] = JOINTLY_SENSITIVE
    label[(wide["stratum"] == NON_SURVIVOR) & (u_no1 > EPS)] = GATING_FAILURE
    wide["label"] = label.astype("string")
    return wide


def gating_failures(effects: pd.DataFrame) -> pd.DataFrame:
    """
    Documents the gate excluded that the no-gate world would have ranked
    usefully. The clearest Stage-1 failure case, and the source of the
    qualitative examples.
    """
    gate = effects[effects["effect_type"] == E_GATE]
    mask = (gate["stratum"] == NON_SURVIVOR) & (gate["utility_control"] > EPS)
    return gate[mask].sort_values("effect")


def recovered_outsiders(effects: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """
    Promoted outsiders that actually entered the final top-k under a t-world.

    This is the operational payoff of the budget experiment: not whether
    outsiders were scored, but whether any of them displaced a candidate.
    """
    d = effects[effects["effect_type"] == DELTA_T]
    from .schema import SELECTED_OUTSIDER

    sel = d[d["stratum"] == SELECTED_OUTSIDER].copy()
    cutoff = k if k is not None else sel["k"].max()
    sel["entered_top_k"] = sel["rank_treatment"].notna() & (
        sel["rank_treatment"] <= cutoff
    )
    return sel


def summarise(effects: pd.DataFrame) -> pd.DataFrame:
    """Compact per-(effect_type, stratum) summary, reported separately always."""
    group = ["effect_type", "stage1_system", "k_c", "utility", "condensed", "stratum"]
    if effects["t"].notna().any():
        group = ["effect_type", "stage1_system", "k_c", "utility", "condensed", "t",
                 "policy", "stratum"]
    return (
        effects.groupby(group, observed=True, dropna=False)
        .agg(
            n=("doc_id", "size"),
            n_judged=("judged", lambda s: int(s.fillna(False).sum())),
            mean_effect=("effect", "mean"),
            mean_abs_effect=("effect", lambda s: float(s.abs().mean())),
            max_abs_effect=("effect", lambda s: float(s.abs().max())),
        )
        .reset_index()
    )


__all__ = [
    "GATE_SENSITIVE", "RERANK_SENSITIVE", "JOINTLY_SENSITIVE", "STABLE",
    "GATING_FAILURE",
    "stage_effects_for_query", "budget_effects_for_query", "concat_effects",
    "check_invariants", "check_additivity", "label_documents", "gating_failures",
    "recovered_outsiders", "summarise",
]
