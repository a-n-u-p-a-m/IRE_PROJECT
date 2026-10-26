"""
cache.py - Persistent Stage-2 score store.

The engineering keystone. Because the core reranker is pointwise,

    s2(q, d | C_q) == s2(q, d | U_q) == s2(q, d)

so one inference pass over U_q serves every downstream experiment: every K_c,
every t, every outsider policy, both the actual and no-gate worlds, and every
probe budget. All of that becomes sorting and filtering over cached scores
rather than new GPU work. Only this module's filler needs a GPU.

SQLite rather than Parquet because scoring is long-running and interruptible:
writes land incrementally, so a killed job resumes instead of restarting, and
three members can fill disjoint query shards and merge.

Cost note: one row is one INFERENCE, not one document. On a passage collection
those coincide. On a document collection with segmentation they do not, and the
passage-level count is what must be budgeted and reported.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stage2 (
    query_id TEXT NOT NULL,
    doc_id   TEXT NOT NULL,
    model    TEXT NOT NULL,
    score    REAL NOT NULL,
    PRIMARY KEY (query_id, doc_id, model)
);
CREATE INDEX IF NOT EXISTS idx_stage2_query ON stage2 (query_id, model);
"""


class Stage2Cache:
    """
    Keyed store of pointwise Stage-2 scores.

    The model name is part of the key so MiniLM, a larger cross-encoder and a
    listwise model cannot silently overwrite one another - which matters for
    experiment E7, where two rerankers are compared on the same universe.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            yield con
            con.commit()
        finally:
            con.close()

    # ── writing ──────────────────────────────────────────────────────

    def put_many(self, model: str, rows: Iterable[tuple[str, str, float]]) -> int:
        """
        Insert or replace (query_id, doc_id, score) triples for one model.

        Call per batch during scoring so an interrupted run keeps its progress.
        """
        payload = [(str(q), str(d), model, float(s)) for q, d, s in rows]
        if not payload:
            return 0
        with self._conn() as con:
            con.executemany(
                "INSERT OR REPLACE INTO stage2 (query_id, doc_id, model, score) "
                "VALUES (?, ?, ?, ?)",
                payload,
            )
        return len(payload)

    # ── reading ──────────────────────────────────────────────────────

    def get_query(self, query_id: str, model: str) -> dict[str, float]:
        """{doc_id: score} for one query."""
        with self._conn() as con:
            cur = con.execute(
                "SELECT doc_id, score FROM stage2 WHERE query_id=? AND model=?",
                (str(query_id), model),
            )
            return {d: float(s) for d, s in cur.fetchall()}

    def missing(
        self, query_id: str, doc_ids: Sequence[str], model: str
    ) -> list[str]:
        """Which of these documents still need scoring. Drives resumable fills."""
        have = set(self.get_query(query_id, model))
        return [d for d in doc_ids if d not in have]

    def count(self, model: str | None = None) -> int:
        with self._conn() as con:
            if model is None:
                return int(con.execute("SELECT COUNT(*) FROM stage2").fetchone()[0])
            return int(
                con.execute(
                    "SELECT COUNT(*) FROM stage2 WHERE model=?", (model,)
                ).fetchone()[0]
            )

    def models(self) -> list[str]:
        with self._conn() as con:
            return sorted(r[0] for r in con.execute("SELECT DISTINCT model FROM stage2"))

    def queries(self, model: str) -> list[str]:
        with self._conn() as con:
            return sorted(
                r[0] for r in con.execute(
                    "SELECT DISTINCT query_id FROM stage2 WHERE model=?", (model,)
                )
            )

    # ── export and merge ─────────────────────────────────────────────

    def to_frame(self, model: str | None = None) -> pd.DataFrame:
        """Export as a `stage2_cache` table for archiving alongside results."""
        with self._conn() as con:
            if model is None:
                df = pd.read_sql_query(
                    "SELECT query_id, doc_id, model, score FROM stage2", con
                )
            else:
                df = pd.read_sql_query(
                    "SELECT query_id, doc_id, model, score FROM stage2 WHERE model=?",
                    con, params=(model,),
                )
        for col in ("query_id", "doc_id", "model"):
            df[col] = df[col].astype("string")
        return df

    def merge_from(self, other: Stage2Cache | str | Path, model: str) -> int:
        """
        Fold another cache's rows for one model into this one. Idempotent.

        This is how three shard fills become one cache. Re-merging changes
        nothing, so it is safe to run after every shard finishes.
        """
        src = other if isinstance(other, Stage2Cache) else Stage2Cache(other)
        rows = src.to_frame(model)
        n = self.put_many(
            model,
            ((q, d, s) for q, d, s in zip(rows["query_id"], rows["doc_id"], rows["score"])),
        )
        logger.info("merged %d rows for %s from %s", n, model, src.path)
        return n

    # ── completeness ─────────────────────────────────────────────────

    def coverage_report(
        self, universe: pd.DataFrame, model: str, query_ids: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """
        Per-query count of universe documents still unscored.

        The no-gate world needs s2 for EVERY document in U_q. A partial cache
        produces a silently truncated no-gate ranking, so check this before
        computing any gate effect. Pass `query_ids` to check only the reference
        subset, which is the usual case.
        """
        recs = []
        wanted = None if query_ids is None else {str(q) for q in query_ids}
        for qid, grp in universe.groupby("query_id", observed=True):
            qid = str(qid)
            if wanted is not None and qid not in wanted:
                continue
            docs = grp["doc_id"].astype(str).tolist()
            n_missing = len(self.missing(qid, docs, model))
            recs.append(
                {
                    "query_id": qid,
                    "universe_size": len(docs),
                    "n_scored": len(docs) - n_missing,
                    "n_missing": n_missing,
                }
            )
        out = pd.DataFrame(recs)
        if len(out) and int(out["n_missing"].sum()):
            logger.warning(
                "stage2 cache incomplete for model %s: %d/%d documents unscored",
                model, int(out["n_missing"].sum()), int(out["universe_size"].sum()),
            )
        return out

    def require_complete(
        self, universe: pd.DataFrame, model: str, query_ids: Sequence[str] | None = None
    ) -> None:
        """Raise unless every required document has a cached score."""
        rep = self.coverage_report(universe, model, query_ids)
        if rep.empty:
            raise RuntimeError(
                "cache completeness check matched no queries; check that the "
                "requested query ids exist in the frozen universe"
            )
        missing = int(rep["n_missing"].sum())
        if missing:
            worst = rep.sort_values("n_missing", ascending=False).head(3)
            raise RuntimeError(
                f"Stage-2 cache is missing {missing} scores for model {model!r}. "
                "The no-gate world requires s2 over the entire universe for "
                "every reference query; a partial cache would truncate it and "
                f"produce plausible-looking wrong gate effects.\nWorst queries:\n"
                f"{worst.to_string(index=False)}"
            )


__all__ = ["Stage2Cache"]
