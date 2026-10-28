"""
qrels.py - Relevance judgments and the unjudged policy.

The no-gate world ranks the whole universe, which is far larger than the
judgment pool. Two rules follow, enforced here rather than left to each
member's analysis code:

  * unjudged is UNKNOWN, never non-relevant. Converting unjudged to grade 0
    would fabricate exactly the evidence E_gate depends on.
  * per-document effect analysis is restricted to the judged subset, but
    unjudged documents still OCCUPY RANKS as competitors, because the policy
    being simulated really would have ranked them.

Those two rules pull in opposite directions and both must hold at once. The
lookup returned by `relevance_lookup` therefore omits unjudged documents
entirely, which forces callers to treat them as unknown rather than letting a
default slip in.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .schema import empty, validate_qrels
from .utility import gain

logger = logging.getLogger(__name__)


def read_trec_qrels(path: str | Path) -> pd.DataFrame:
    """
    Read a TREC qrels file: `query_id iteration doc_id relevance`.

    Every row present is judged by definition. Documents absent from the file
    are simply not represented here; `attach_judgments` marks them unjudged
    once a universe is supplied.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"qrels not found: {path}")

    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(
                    f"{path}:{lineno}: expected 4 whitespace-separated fields "
                    f"(query_id iteration doc_id relevance), got {len(parts)}"
                )
            qid, _, did, rel = parts
            rows.append((qid, did, int(rel)))

    df = pd.DataFrame(rows, columns=["query_id", "doc_id", "relevance"])
    df["judged"] = True
    logger.info("read %d judgments from %s", len(df), path)
    return validate_qrels(df)


def attach_judgments(universe: pd.DataFrame, qrels: pd.DataFrame) -> pd.DataFrame:
    """
    Expand qrels to cover every (query, doc) pair in the universe.

    Universe documents with no judgment get judged=False and relevance=NA.
    Judgments for documents outside U_q are dropped: they cannot participate in
    any world, so carrying them would only inflate apparent coverage.
    """
    uni = universe[["query_id", "doc_id"]].astype("string").drop_duplicates()
    qr = qrels[["query_id", "doc_id", "relevance", "judged"]].copy()
    qr[["query_id", "doc_id"]] = qr[["query_id", "doc_id"]].astype("string")

    merged = uni.merge(qr, on=["query_id", "doc_id"], how="left")
    merged["judged"] = merged["judged"].fillna(False).astype("boolean")
    merged.loc[~merged["judged"].fillna(False), "relevance"] = pd.NA
    merged["relevance"] = merged["relevance"].astype("Int64")

    kept = int(merged["judged"].fillna(False).sum())
    dropped = len(qr) - kept
    if dropped > 0:
        logger.info(
            "%d of %d judgments fall outside the universe and were dropped",
            dropped, len(qr),
        )
    return validate_qrels(merged)


def judged_subset(qrels: pd.DataFrame) -> pd.DataFrame:
    """J_q: the judged documents, which carry the primary effect analysis."""
    return qrels[qrels["judged"].fillna(False)].copy()


def relevance_lookup(qrels: pd.DataFrame) -> dict[tuple[str, str], int]:
    """
    {(query_id, doc_id): relevance} over JUDGED documents only.

    Unjudged documents are absent by design. Callers must treat a missing key
    as unknown; `utility.document_utility` takes relevance=None and returns a
    zero contribution, which is the correct behaviour for a competitor whose
    relevance cannot be asserted.
    """
    j = judged_subset(qrels)
    return {
        (str(q), str(d)): int(r)
        for q, d, r in zip(j["query_id"], j["doc_id"], j["relevance"])
        if pd.notna(r)
    }


def judged_flags(qrels: pd.DataFrame) -> dict[tuple[str, str], bool]:
    """{(query_id, doc_id): judged} over the whole universe."""
    return {
        (str(q), str(d)): bool(j)
        for q, d, j in zip(qrels["query_id"], qrels["doc_id"], qrels["judged"].fillna(False))
    }


def coverage_report(
    universe: pd.DataFrame, qrels: pd.DataFrame, gain_map: Mapping[int, float]
) -> pd.DataFrame:
    """
    Per-query judgment coverage of the universe.

    Returns (query_id, universe_size, n_judged, n_gainful, coverage). Coverage
    is reported, never assumed: strong gate-failure claims are not available
    when outsider judgments are thin.

    `n_gainful` counts documents whose grade carries gain under the frozen map,
    NOT documents with relevance > 0. On TREC-DL those differ by a factor of
    four to nine, because grade 1 ("Related") maps to 0.0: a grade-1 document
    is judged, is not relevant for this project's utility, and contributes
    nothing wherever it is ranked. The gain map is a required argument rather
    than a default so the definition always travels with the configuration that
    set it.
    """
    uni = universe[["query_id", "doc_id"]].astype(str)
    qr = qrels[["query_id", "doc_id", "relevance", "judged"]].copy()
    qr[["query_id", "doc_id"]] = qr[["query_id", "doc_id"]].astype(str)

    merged = uni.merge(qr, on=["query_id", "doc_id"], how="left")
    merged["judged"] = merged["judged"].fillna(False).astype(bool)
    merged["is_gainful"] = merged["judged"] & merged["relevance"].map(
        lambda r: pd.notna(r) and gain(int(r), gain_map) > 0.0
    ).fillna(False)

    out = (
        merged.groupby("query_id", observed=True)
        .agg(
            universe_size=("doc_id", "size"),
            n_judged=("judged", "sum"),
            n_gainful=("is_gainful", "sum"),
        )
        .reset_index()
    )
    out["coverage"] = out["n_judged"] / out["universe_size"]
    return out


def judged_non_survivor_counts(
    qrels: pd.DataFrame,
    candidates: dict[str, set[str]],
    system: str,
    k_c: int,
    gain_map: Mapping[int, float],
) -> pd.DataFrame:
    """
    |J_q \\ C_q| per query: the judged documents the gate excluded.

    Gate failures live here, and they can be scarce at large K_c because TREC
    pools are built from participating systems' top-k. If this is thin at a
    given depth, the headline gate analysis belongs at a shallower one - and
    that decision rule must be frozen before magnitudes are inspected.

    `n_gainful_non_survivors` is the count that matters, and it is NOT the
    count of excluded documents with relevance > 0. Excluding a grade-1
    document costs nothing under a gain map that scores it 0.0, so it is not a
    gate failure. Measured on TREC-DL 2019+2020 the two differ by 4x to 9x -
    at K_c=1000 the DPR candidate set has a median of 9 excluded documents with
    relevance > 0 and exactly 1 that carries gain - so a headline K_c chosen
    on the looser count would be chosen on a number nine times too large.
    """
    j = judged_subset(qrels)[["query_id", "doc_id", "relevance"]].astype(
        {"query_id": "string", "doc_id": "string"}
    )

    recs = []
    for qid, grp in j.groupby("query_id", observed=True):
        cq = candidates.get(str(qid), set())
        docs = set(grp["doc_id"].astype(str))
        outside = docs - cq
        rel_by_doc = dict(zip(grp["doc_id"].astype(str), grp["relevance"]))
        gainful_outside = {
            d for d in outside
            if pd.notna(rel_by_doc.get(d))
            and gain(int(rel_by_doc[d]), gain_map) > 0.0
        }
        recs.append(
            {
                "query_id": str(qid),
                "stage1_system": system,
                "k_c": k_c,
                "n_judged": len(docs),
                "n_judged_non_survivors": len(outside),
                "n_gainful_non_survivors": len(gainful_outside),
            }
        )
    return pd.DataFrame(recs)


def empty_qrels() -> pd.DataFrame:
    return empty("qrels")


__all__ = [
    "read_trec_qrels", "attach_judgments", "judged_subset", "relevance_lookup",
    "judged_flags", "coverage_report", "judged_non_survivor_counts", "empty_qrels",
]
