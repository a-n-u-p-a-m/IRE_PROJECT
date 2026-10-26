"""
universe.py - Build, freeze and sample the fixed evaluation universe U_q.

    U_q = union over sources of TopM(q, source)

Built once, written to disk, never regenerated during an experiment series.
Every counterfactual claim is relative to it, so a silent change invalidates
every effect computed earlier. The fingerprint sidecar turns that from a
convention into a mechanical refusal.

This module also owns the reference-subset sampler. Which queries receive full
no-gate scoring is a design decision, not a convenience: selecting queries with
small |U_q| would bias towards queries where the retrievers agree, which is
exactly where gate effects are smallest. The rule is declared, seeded and
recorded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import load, save, validate_run, validate_universe

logger = logging.getLogger(__name__)


class UniverseError(ValueError):
    """Raised when a universe is inconsistent with the runs built against it."""


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────

def build_universe(
    runs: dict[str, pd.DataFrame],
    depth_m: int,
    max_size: int | None = None,
) -> pd.DataFrame:
    """
    Union the top-M of each source run into a frozen universe.

    Source runs are produced sequentially and only the top-M document
    identifiers are needed here, so the sparse, dense and impact indexes never
    have to be resident at the same time.

    `max_size` caps |U_q| after the union, dropping from the deepest source
    ranks first so that every source keeps its strongest contributions.
    """
    if not runs:
        raise UniverseError("no source runs supplied")

    frames = []
    for system, run in runs.items():
        r = validate_run(run)
        top = r[r["rank"] <= depth_m][["query_id", "doc_id", "rank"]].copy()
        top["system"] = system
        frames.append(top)

    stacked = pd.concat(frames, ignore_index=True)

    best = (
        stacked.groupby(["query_id", "doc_id"], observed=True)["rank"]
        .min().rename("best_rank").reset_index()
    )
    srcs = (
        stacked.groupby(["query_id", "doc_id"], observed=True)["system"]
        .apply(lambda s: ",".join(sorted(set(s)))).rename("sources").reset_index()
    )
    uni = best.merge(srcs, on=["query_id", "doc_id"], how="inner")

    if max_size is not None:
        uni = (
            uni.sort_values(["query_id", "best_rank", "doc_id"])
            .groupby("query_id", observed=True, group_keys=False)
            .head(max_size)
        )

    uni = uni.drop(columns=["best_rank"]).sort_values(["query_id", "doc_id"])
    out = validate_universe(uni.reset_index(drop=True))

    sizes = out.groupby("query_id", observed=True).size()
    logger.info(
        "universe: %d queries, |U_q| min=%d median=%d max=%d, sources=%s",
        len(sizes), int(sizes.min()), int(sizes.median()), int(sizes.max()),
        sorted(runs),
    )
    return out


def freeze(universe: pd.DataFrame, path: str | Path, fingerprint: str) -> Path:
    """Write the universe plus a sidecar recording the config that produced it."""
    path = Path(path)
    save(universe, path, "universe")
    path.with_suffix(".frozen").write_text(fingerprint + "\n", encoding="utf-8")
    logger.info("universe frozen at %s (config %s)", path, fingerprint)
    return path


def load_frozen(path: str | Path, fingerprint: str | None = None) -> pd.DataFrame:
    """Load a frozen universe, refusing a design-freeze mismatch."""
    path = Path(path)
    uni = load(path, "universe")
    sidecar = path.with_suffix(".frozen")
    if fingerprint is not None and sidecar.exists():
        stored = sidecar.read_text(encoding="utf-8").strip()
        if stored != fingerprint:
            raise UniverseError(
                f"universe at {path} was frozen under config {stored} but the "
                f"current config is {fingerprint}. Effects computed against "
                "different universes are not comparable. Restore the old config "
                "or regenerate the universe and recompute every effect."
            )
    return uni


def universe_sets(universe: pd.DataFrame) -> dict[str, list[str]]:
    """{query_id: [doc_id, ...]} for fast lookup."""
    return {
        str(q): g["doc_id"].astype(str).tolist()
        for q, g in universe.groupby("query_id", observed=True)
    }


def universe_sizes(universe: pd.DataFrame) -> pd.Series:
    return universe.groupby("query_id", observed=True).size()


# ──────────────────────────────────────────────────────────────────────
# Consistency checks
# ──────────────────────────────────────────────────────────────────────

def check_containment(
    run: pd.DataFrame, universe: pd.DataFrame, k_c: int, system: str
) -> None:
    """
    Verify C_q = TopKc(U_q, s1) is contained in U_q for this system.

    Fails loudly when a Stage-1 retriever was not a universe source, which is
    the easiest way to silently break the whole counterfactual frame.
    """
    r = validate_run(run)
    r = r[r["system"] == system]
    if r.empty:
        raise UniverseError(f"run contains no rows for system {system!r}")

    uni_keys = set(
        map(tuple, universe[["query_id", "doc_id"]].drop_duplicates().astype(str).to_numpy())
    )
    cand_keys = set(
        map(tuple, r[r["rank"] <= k_c][["query_id", "doc_id"]].astype(str).to_numpy())
    )
    escaped = cand_keys - uni_keys
    if escaped:
        raise UniverseError(
            f"{len(escaped)} candidates of system {system!r} at K_c={k_c} lie "
            f"outside the frozen universe, e.g. {sorted(escaped)[:5]}. Every "
            "Stage-1 system must contribute its top-M before it is used."
        )


def candidate_overlap(
    candidates_a: dict[str, set[str]],
    candidates_b: dict[str, set[str]],
    name_a: str,
    name_b: str,
    k_c: int,
) -> pd.DataFrame:
    """
    Per-query Jaccard between two systems' candidate sets.

    If two Stage-1 systems select nearly the same documents, the cross-paradigm
    experiment has no leverage however different their scores are. This is a
    pre-run gate, not a result.
    """
    rows = []
    for qid in sorted(set(candidates_a) | set(candidates_b)):
        ca, cb = candidates_a.get(qid, set()), candidates_b.get(qid, set())
        union = ca | cb
        rows.append(
            {
                "query_id": qid,
                "system_a": name_a,
                "system_b": name_b,
                "k_c": k_c,
                "jaccard": (len(ca & cb) / len(union)) if union else float("nan"),
                "n_a_only": len(ca - cb),
                "n_b_only": len(cb - ca),
            }
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Reference subset
# ──────────────────────────────────────────────────────────────────────

def sample_reference_subset(
    universe: pd.DataFrame,
    size: int,
    strategy: str = "stratified_universe_size",
    seed: int = 42,
    n_strata: int = 5,
    jaccard: pd.DataFrame | None = None,
) -> list[str]:
    """
    Choose the queries that receive full no-gate scoring.

    Strategies
    ----------
    random
        Uniform sample of queries.
    stratified_universe_size
        Split queries into `n_strata` equal-count bins by |U_q| and draw
        proportionally. Guards against over-sampling small universes, where
        the retrievers agree and gate effects are smallest.
    stratified_jaccard
        Same, binned by BM25/DPR candidate-set Jaccard. Needs `jaccard`, a
        frame with columns (query_id, jaccard). Use this when the headline
        claim is about cross-paradigm differences, since it balances the
        subset across agreement levels.

    The chosen list is frozen in the config record and written to the run
    manifest. Never pick this subset by convenience or by inspecting effects.
    """
    sizes = universe_sizes(universe)
    queries = sorted(sizes.index.astype(str))
    if size >= len(queries):
        logger.info("reference subset requested (%d) >= all queries (%d); using all",
                    size, len(queries))
        return queries

    rng = np.random.default_rng(seed)

    if strategy == "random":
        idx = rng.choice(len(queries), size=size, replace=False)
        return sorted(queries[int(i)] for i in idx)

    if strategy == "stratified_universe_size":
        key = pd.Series(sizes.values, index=sizes.index.astype(str)).astype(float)
    elif strategy == "stratified_jaccard":
        if jaccard is None:
            raise ValueError(
                "strategy 'stratified_jaccard' needs the jaccard frame from "
                "universe.candidate_overlap"
            )
        key = (
            jaccard.groupby(jaccard["query_id"].astype(str), observed=True)["jaccard"]
            .mean().astype(float)
        )
        key = key.reindex(queries).fillna(key.mean())
    else:
        raise ValueError(f"unknown reference-subset strategy {strategy!r}")

    key = key.reindex(queries)
    ranks = key.rank(method="first")
    bins = np.minimum(
        ((ranks - 1) / (len(queries) / n_strata)).astype(int), n_strata - 1
    )

    chosen: list[str] = []
    per_stratum = size // n_strata
    remainder = size - per_stratum * n_strata
    for b in range(n_strata):
        members = [q for q, bb in zip(queries, bins) if bb == b]
        take = per_stratum + (1 if b < remainder else 0)
        take = min(take, len(members))
        if take:
            idx = rng.choice(len(members), size=take, replace=False)
            chosen.extend(members[int(i)] for i in idx)

    # Top up if a stratum was short.
    if len(chosen) < size:
        pool = [q for q in queries if q not in set(chosen)]
        extra = rng.choice(len(pool), size=min(size - len(chosen), len(pool)),
                           replace=False)
        chosen.extend(pool[int(i)] for i in extra)

    return sorted(chosen)


def write_reference_subset(path: str | Path, query_ids: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(query_ids) + "\n", encoding="utf-8")
    return path


def read_reference_subset(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = [
    "UniverseError", "build_universe", "freeze", "load_frozen", "universe_sets",
    "universe_sizes", "check_containment", "candidate_overlap",
    "sample_reference_subset", "write_reference_subset", "read_reference_subset",
]
