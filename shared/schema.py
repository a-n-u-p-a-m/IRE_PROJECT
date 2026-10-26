"""
schema.py - The data contract (proposal section 14).

Six tables, defined once, validated everywhere. The three members integrate
through files, not handoffs, so these schemas ARE the interface.

    run             (query_id, doc_id, system, rank, score, checkpoint)
    universe        (query_id, doc_id, sources)
    qrels           (query_id, doc_id, relevance, judged)
    stage2_cache    (query_id, doc_id, model, score)
    effect          per-document outcome under one world comparison
    feature         predictor inputs, kept strictly apart from targets

The effect table is the primary scientific object and its validator is the
most important code in this package. Critically, the three effect types do NOT
share invariants:

    e_rerank    d not in C_q  =>  exactly 0
    e_gate      d in C_q      =>  >= 0        d not in C_q  =>  <= 0
    delta_t     four membership strata, three of which have fixed signs

So sign checking is conditional on effect_type. A single unconditional
validator would either reject correct delta_t rows or wave through incorrect
e_gate rows; both have happened in earlier drafts of this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

TableKind = Literal["run", "universe", "qrels", "stage2_cache", "effect", "feature"]

# Effect types. Each selects a different set of sign invariants.
E_RERANK = "e_rerank"
E_GATE = "e_gate"
DELTA_T = "delta_t"
EFFECT_TYPES = frozenset({E_RERANK, E_GATE, DELTA_T})

# Survivorship strata (e_gate, e_rerank).
SURVIVOR = "survivor"
NON_SURVIVOR = "non_survivor"

# Membership strata for delta_t (proposal section 4.4).
RETAINED = "retained"                       # in the top floor(tB) of C_q
REMOVED = "removed"                         # in C_q but below the retained cut
SELECTED_OUTSIDER = "selected_outsider"     # promoted from D_q
UNSELECTED_OUTSIDER = "unselected_outsider" # in D_q, never scored
T_STRATA = (RETAINED, REMOVED, SELECTED_OUTSIDER, UNSELECTED_OUTSIDER)

VALID_UTILITIES = frozenset({"dcg", "topk", "rbp"})

# Effects are differences of utilities and inherit float64 noise from the gain
# and discount terms.
EPS = 1e-9


class SchemaError(ValueError):
    """Raised when a table violates the data contract."""


# ──────────────────────────────────────────────────────────────────────
# Column definitions
# ──────────────────────────────────────────────────────────────────────

RUN_COLUMNS: dict[str, str] = {
    "query_id": "string",
    "doc_id": "string",
    "system": "string",
    "rank": "Int64",            # 1-indexed within (query_id, system)
    "score": "float64",
    "checkpoint": "string",     # exact model/index version that produced it
}

UNIVERSE_COLUMNS: dict[str, str] = {
    "query_id": "string",
    "doc_id": "string",
    "sources": "string",        # comma-joined contributing systems, sorted
}

QRELS_COLUMNS: dict[str, str] = {
    "query_id": "string",
    "doc_id": "string",
    "relevance": "Int64",       # graded; NA when unjudged
    "judged": "boolean",
}

STAGE2_CACHE_COLUMNS: dict[str, str] = {
    "query_id": "string",
    "doc_id": "string",
    "model": "string",
    "score": "float64",
}

EFFECT_COLUMNS: dict[str, str] = {
    "query_id": "string",
    "doc_id": "string",
    "stage1_system": "string",
    "effect_type": "string",    # e_rerank | e_gate | delta_t
    "k_c": "Int64",
    "k": "Int64",
    "utility": "string",
    "budget_b": "Int64",        # NA unless effect_type == delta_t
    "t": "float64",             # NA unless effect_type == delta_t
    "policy": "string",         # outsider policy; NA unless delta_t
    "seed": "Int64",            # NA unless the policy is randomised
    "stratum": "string",        # survivor/non_survivor, or one of T_STRATA
    "judged": "boolean",
    "rank_treatment": "Int64",  # rank in the world being measured; NA if absent
    "rank_control": "Int64",    # rank in the comparison world; NA if absent
    "utility_treatment": "float64",
    "utility_control": "float64",
    "effect": "float64",        # utility_treatment - utility_control
}

FEATURE_COLUMNS: dict[str, str] = {
    "query_id": "string",
    "doc_id": "string",
    "feature_version": "string",
    "feature_name": "string",
    "value": "float64",
    "generation_seconds": "float64",
}

_SCHEMAS: dict[str, dict[str, str]] = {
    "run": RUN_COLUMNS,
    "universe": UNIVERSE_COLUMNS,
    "qrels": QRELS_COLUMNS,
    "stage2_cache": STAGE2_CACHE_COLUMNS,
    "effect": EFFECT_COLUMNS,
    "feature": FEATURE_COLUMNS,
}

_KEYS: dict[str, list[str]] = {
    "run": ["query_id", "doc_id", "system"],
    "universe": ["query_id", "doc_id"],
    "qrels": ["query_id", "doc_id"],
    "stage2_cache": ["query_id", "doc_id", "model"],
    "effect": ["query_id", "doc_id", "stage1_system", "effect_type",
               "k_c", "k", "utility", "budget_b", "t", "policy", "seed"],
    "feature": ["query_id", "doc_id", "feature_name", "feature_version"],
}


# ──────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────

def empty(kind: TableKind) -> pd.DataFrame:
    """An empty, correctly typed frame of the given kind."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _SCHEMAS[kind].items()})


def _coerce(df: pd.DataFrame, kind: TableKind) -> pd.DataFrame:
    """Copy, check columns, cast dtypes. Never mutates the caller's frame."""
    spec = _SCHEMAS[kind]
    missing = set(spec) - set(df.columns)
    if missing:
        raise SchemaError(f"{kind}: missing required columns {sorted(missing)}")

    out = df.copy()
    for col, dtype in spec.items():
        try:
            out[col] = out[col].astype(dtype)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{kind}.{col}: cannot cast to {dtype} ({exc})") from exc
    return out[list(spec)]


def _check_keys(df: pd.DataFrame, kind: TableKind, strict: bool) -> None:
    keys = _KEYS[kind]
    # Nullable key columns are legitimate on the effect table: budget_b, t,
    # policy and seed are NA for e_gate and e_rerank rows by design.
    nullable = {"budget_b", "t", "policy", "seed"} if kind == "effect" else set()
    for col in keys:
        if col in nullable:
            continue
        if df[col].isna().any():
            raise SchemaError(f"{kind}.{col}: null values in key column")
    if strict and len(df):
        dup = df.duplicated(subset=keys, keep=False)
        if dup.any():
            raise SchemaError(
                f"{kind}: duplicate keys on {keys}; first offenders:\n"
                f"{df.loc[dup, keys].head()}"
            )


# ──────────────────────────────────────────────────────────────────────
# Per-table validators
# ──────────────────────────────────────────────────────────────────────

def validate_run(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    out = _coerce(df, "run")
    _check_keys(out, "run", strict)
    if len(out) and (out["rank"] < 1).any():
        raise SchemaError("run.rank: values < 1 (ranks are 1-indexed)")
    if strict and len(out):
        for (qid, sysname), grp in out.groupby(["query_id", "system"], observed=True):
            ranks = sorted(grp["rank"].dropna().tolist())
            if ranks != list(range(1, len(ranks) + 1)):
                raise SchemaError(
                    f"run: ranks for ({qid}, {sysname}) are not a contiguous "
                    f"1..{len(ranks)} sequence"
                )
    return out


def validate_universe(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    out = _coerce(df, "universe")
    _check_keys(out, "universe", strict)
    if len(out) and out["sources"].str.len().eq(0).any():
        raise SchemaError("universe.sources: empty source list")
    return out


def validate_qrels(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    out = _coerce(df, "qrels")
    _check_keys(out, "qrels", strict)
    if len(out):
        judged = out["judged"].fillna(False)
        if out.loc[judged, "relevance"].isna().any():
            raise SchemaError("qrels: judged=True rows must carry a relevance value")
        if out.loc[~judged, "relevance"].notna().any():
            raise SchemaError(
                "qrels: judged=False rows must have relevance=NA. Never silently "
                "convert unjudged to non-relevant: the no-gate universe is far "
                "larger than the judgment pool, so that substitution fabricates "
                "exactly the evidence E_gate depends on."
            )
    return out


def validate_stage2_cache(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    out = _coerce(df, "stage2_cache")
    _check_keys(out, "stage2_cache", strict)
    if len(out) and out["score"].isna().any():
        raise SchemaError("stage2_cache.score: null scores")
    return out


def validate_feature(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    out = _coerce(df, "feature")
    _check_keys(out, "feature", strict)
    return out


# ──────────────────────────────────────────────────────────────────────
# The effect validator
# ──────────────────────────────────────────────────────────────────────

def _check_definition(out: pd.DataFrame) -> None:
    """effect == utility_treatment - utility_control, for every row."""
    residual = (out["effect"] - (out["utility_treatment"] - out["utility_control"])).abs()
    if (residual > EPS).any():
        bad = int((residual > EPS).sum())
        raise SchemaError(
            f"effect: definition violated on {bad} rows. "
            "effect must equal utility_treatment - utility_control exactly."
        )


def _check_e_rerank(block: pd.DataFrame) -> None:
    """
    Non-survivors have no Stage-2 rank inside C_q in either world, so their
    reranking effect is exactly zero. A nonzero value means the no-reranker
    world leaked outside the candidate set, which would confound the gate and
    reranker interventions.
    """
    outside = block["stratum"] == NON_SURVIVOR
    if (block.loc[outside, "effect"].abs() > EPS).any():
        raise SchemaError(
            "effect/e_rerank: a non-survivor has a nonzero effect. Both the "
            "actual and no-reranker worlds rank exactly C_q, so any document "
            "outside C_q must score exactly 0."
        )


def _check_e_gate(block: pd.DataFrame, strict: bool) -> None:
    """
    Under a pointwise reranker and a monotone rank utility, removing the gate
    can only add competitors: a survivor can never improve, and a non-survivor
    starts from zero actual utility so it can never lose.
    """
    if not strict:
        return
    survivor = block["stratum"] == SURVIVOR
    if (block.loc[survivor, "effect"] < -EPS).any():
        n = int((block.loc[survivor, "effect"] < -EPS).sum())
        raise SchemaError(
            f"effect/e_gate: {n} survivors have effect < 0. Removing the gate "
            "only adds competitors, so a survivor cannot improve. Check that "
            "the no-gate world reuses the same Stage-2 scores and that "
            "tie-breaking is deterministic. Pass strict=False only for a "
            "listwise Stage 2, which is a different estimand."
        )
    if (block.loc[~survivor, "effect"] > EPS).any():
        raise SchemaError(
            "effect/e_gate: a non-survivor has effect > 0. Its actual utility "
            "is zero, so the effect cannot be positive."
        )


def _check_delta_t(block: pd.DataFrame) -> None:
    """
    Proposal section 4.4. Three of the four membership strata have signs fixed
    by allocation membership alone, which is why pooled correlation against
    delta_t is meaningless and why `metrics` reports within-stratum.
    """
    for stratum, rule in (
        (REMOVED, "le"),               # dropped from the submitted set
        (SELECTED_OUTSIDER, "ge"),     # promoted into it
        (UNSELECTED_OUTSIDER, "zero"),  # absent from both
    ):
        sel = block["stratum"] == stratum
        if not sel.any():
            continue
        vals = block.loc[sel, "effect"]
        if rule == "le" and (vals > EPS).any():
            raise SchemaError(
                f"effect/delta_t: stratum '{REMOVED}' has effect > 0. A document "
                "dropped from the submitted set scores zero at this t, so its "
                "effect cannot be positive."
            )
        if rule == "ge" and (vals < -EPS).any():
            raise SchemaError(
                f"effect/delta_t: stratum '{SELECTED_OUTSIDER}' has effect < 0. "
                "It was absent from the t=1 baseline, so its effect cannot be "
                "negative."
            )
        if rule == "zero" and (vals.abs() > EPS).any():
            raise SchemaError(
                f"effect/delta_t: stratum '{UNSELECTED_OUTSIDER}' has a nonzero "
                "effect. It is absent from both worlds and is structurally zero."
            )

    bad = set(block["stratum"].dropna().unique()) - set(T_STRATA)
    if bad:
        raise SchemaError(f"effect/delta_t: unknown strata {sorted(bad)}")

    missing = {"budget_b", "t"} - set(
        c for c in ("budget_b", "t") if block[c].notna().all()
    )
    if missing:
        raise SchemaError(
            f"effect/delta_t: rows must carry {sorted(missing)}; a budget effect "
            "is meaningless without the budget and allocation it was measured at."
        )


def validate_effect(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    Validate the effect table, applying sign invariants conditionally on
    effect_type.

    strict=False relaxes only the e_gate sign constraints, which are the ones
    that depend on the reranker being pointwise.
    """
    out = _coerce(df, "effect")
    _check_keys(out, "effect", strict)
    if not len(out):
        return out

    bad_util = set(out["utility"].dropna()) - VALID_UTILITIES
    if bad_util:
        raise SchemaError(f"effect.utility: unknown {sorted(bad_util)}")
    bad_type = set(out["effect_type"].dropna()) - EFFECT_TYPES
    if bad_type:
        raise SchemaError(f"effect.effect_type: unknown {sorted(bad_type)}")

    _check_definition(out)

    for effect_type, block in out.groupby("effect_type", observed=True):
        if effect_type == E_RERANK:
            _check_e_rerank(block)
        elif effect_type == E_GATE:
            _check_e_gate(block, strict)
        elif effect_type == DELTA_T:
            _check_delta_t(block)

    return out


_VALIDATORS = {
    "run": validate_run,
    "universe": validate_universe,
    "qrels": validate_qrels,
    "stage2_cache": validate_stage2_cache,
    "effect": validate_effect,
    "feature": validate_feature,
}


def validate(df: pd.DataFrame, kind: TableKind, strict: bool = True) -> pd.DataFrame:
    if kind not in _VALIDATORS:
        raise SchemaError(f"unknown table kind {kind!r}")
    return _VALIDATORS[kind](df, strict=strict)


# ──────────────────────────────────────────────────────────────────────
# Parquet I/O
# ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, path: str | Path, kind: TableKind,
         strict: bool = True) -> Path:
    """Validate, then write Parquet. Parent directories are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate(df, kind, strict=strict).to_parquet(path, index=False)
    return path


def load(path: str | Path, kind: TableKind, strict: bool = True) -> pd.DataFrame:
    """Read Parquet and validate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{kind} table not found: {path}")
    return validate(pd.read_parquet(path), kind, strict=strict)


__all__ = [
    "SchemaError", "TableKind", "EPS",
    "E_RERANK", "E_GATE", "DELTA_T", "EFFECT_TYPES",
    "SURVIVOR", "NON_SURVIVOR",
    "RETAINED", "REMOVED", "SELECTED_OUTSIDER", "UNSELECTED_OUTSIDER", "T_STRATA",
    "VALID_UTILITIES",
    "RUN_COLUMNS", "UNIVERSE_COLUMNS", "QRELS_COLUMNS",
    "STAGE2_CACHE_COLUMNS", "EFFECT_COLUMNS", "FEATURE_COLUMNS",
    "empty", "validate", "validate_run", "validate_universe", "validate_qrels",
    "validate_stage2_cache", "validate_effect", "validate_feature",
    "save", "load",
]
