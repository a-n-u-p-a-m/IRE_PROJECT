"""
real.py - The engine on the frozen universe and the real Stage-2 cache.

    python -m experiments_shubham.real --system bm25
    python -m experiments_shubham.real --system dpr

`engine.main` feeds the runners generated `QueryInputs`. This module reads them
from the four artefacts that define the real cascade:

    data/universe.parquet                    U_q, frozen under the config fingerprint
    data/stage1_over_universe_<s>.parquet    s1 over all of U_q (floor policy)
    data/qrels.parquet                       relevance and the judged flag
    cache/stage2.sqlite                      s2 over all of U_q, one reranker

Before a single effect is computed:

  * the universe must carry the current config fingerprint;
  * both Stage-1 tables must cover exactly U_q, the system's own and the
    alternative retriever's. A gap would silently move a document to the
    bottom of world B or out of the alternative policy's reach;
  * the cache must hold a score for every (q, d) in U_q under the configured
    reranker. World C ranks all of U_q, so a partial cache truncates it and
    yields plausible-looking wrong gate effects.

After the grid is written, world D is checked against the published
allocations (`experiments_anupam.allocations`): the selected outsiders in the
effect tables must equal `analysis/allocation_promoted.parquet` at every
(K_c, t, policy, seed, query). The two come from different code paths, and
this is the check that they describe the same world D. A mismatch exits
non-zero and is recorded in the manifest.

The alternative retriever is the other Stage-1 system: DPR for BM25 and BM25
for DPR, as in the published allocations. With two systems that is the only
reading under which `alternative` differs from `boundary` for both of them.

The manifest records the reranker, the number of cached rows and a digest of
their content (sorted, so independent of the order the shards were merged
in), so every effect table names the exact scores it was computed from.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments_shubham.engine import (
    EffectSettings,
    QueryInputs,
    grid_t,
    run_grid,
    table_name,
)
from shared.allocation import RANDOM
from shared.cache import Stage2Cache
from shared.config import FrozenConfig, load_config
from shared.provenance import read_manifest, write_manifest
from shared.schema import DELTA_T, SELECTED_OUTSIDER, load
from shared.universe import load_frozen, universe_sets

logger = logging.getLogger(__name__)


class RealInputError(ValueError):
    """The real artefacts disagree with each other or with the config."""


def alternative_system(cfg: FrozenConfig, system: str) -> str:
    """The retriever whose ranks drive the `alternative` policy for `system`."""
    others = [s for s in cfg.stage1_systems if s != system]
    if system not in cfg.stage1_systems or len(others) != 1:
        raise RealInputError(
            f"expected {system!r} to be one of exactly two Stage-1 systems, "
            f"got {list(cfg.stage1_systems)}")
    return others[0]


def stage1_path(tables_dir: str | Path, system: str) -> Path:
    return Path(tables_dir) / f"stage1_over_universe_{system}.parquet"


def _per_query(df: pd.DataFrame, column: str, cast) -> dict[str, dict[str, object]]:
    return {
        str(q): dict(zip(g["doc_id"].astype(str), g[column].map(cast)))
        for q, g in df.groupby("query_id", observed=True)
    }


def _stage1(tables_dir: str | Path, system: str,
            universe: dict[str, list[str]]) -> pd.DataFrame:
    path = stage1_path(tables_dir, system)
    if not path.exists():
        raise RealInputError(
            f"missing {path}. Run `python -m experiments_anupam.stage1_over_universe`.")
    table = load(path, "run")
    table = table[table["system"] == system]
    have = {str(q): set(g["doc_id"].astype(str))
            for q, g in table.groupby("query_id", observed=True)}
    bad = [q for q, docs in universe.items() if have.get(q) != set(docs)]
    if bad:
        raise RealInputError(
            f"{path} does not cover exactly U_q for {len(bad)} queries "
            f"(first: {bad[:3]}). It must have been built from a different "
            "universe; rebuild it from the frozen one.")
    return table


def load_queries(
    cfg: FrozenConfig,
    system: str,
    cache: Stage2Cache,
    model: str | None = None,
    universe_path: str | Path = "data/universe.parquet",
    tables_dir: str | Path = "data",
    qrels_path: str | Path = "data/qrels.parquet",
    query_ids: Sequence[str] | None = None,
) -> list[QueryInputs]:
    """Engine inputs for every query, or for `query_ids`, from the real artefacts."""
    model = model or cfg.reranker_model
    universe = load_frozen(universe_path, fingerprint=cfg.fingerprint())
    if query_ids is not None:
        wanted = {str(q) for q in query_ids}
        universe = universe[universe["query_id"].isin(wanted)]
    sets = universe_sets(universe)
    if not sets:
        raise RealInputError("no queries selected from the frozen universe")

    s1 = _per_query(_stage1(tables_dir, system, sets), "score", float)
    alt = _per_query(_stage1(tables_dir, alternative_system(cfg, system), sets),
                     "rank", int)

    qrels = load(qrels_path, "qrels")
    qrels = qrels[qrels["judged"].fillna(False).astype(bool)]
    relevance = _per_query(qrels, "relevance", int)

    cache.require_complete(universe, model, list(sets))
    queries = []
    for q in sorted(sets):
        rel = relevance.get(q, {})
        queries.append(QueryInputs(
            q, tuple(sets[q]), s1[q], cache.get_query(q, model), rel,
            {d: True for d in rel}, alt[q]))
    logger.info("%s: %d queries, %d pairs, Stage-2 model %s", system,
                len(queries), sum(len(q.universe) for q in queries), model)
    return queries


def cache_digest(cache: Stage2Cache, model: str) -> str:
    """Content hash of one model's cached scores, independent of insertion order."""
    rows = cache.to_frame(model).sort_values(["query_id", "doc_id"])
    h = hashlib.sha256()
    for q, d, s in zip(rows["query_id"], rows["doc_id"], rows["score"]):
        h.update(f"{q}\t{d}\t{float(s)!r}\n".encode())
    return h.hexdigest()[:16]


def _selected_sets(frame: pd.DataFrame, seed_free: bool) -> dict[tuple, frozenset]:
    keys = ["query_id", "t"] if seed_free else ["query_id", "t", "seed"]
    return {tuple(k): frozenset(g["doc_id"].astype(str))
            for k, g in frame.groupby(keys, observed=True)}


def check_allocations(
    effects_dir: str | Path,
    system: str,
    cfg: FrozenConfig,
    published_path: str | Path,
) -> dict:
    """
    Compare world D's selected outsiders with the published allocations.

    Reads the uncondensed budget tables only: condensing drops unjudged
    documents, so a condensed table does not carry the whole selected set.
    Boundary and alternative are deterministic and carry seed=NA in the effect
    tables, so they are compared without the seed.
    """
    published = pd.read_parquet(published_path)
    published = published[(published["stage1_system"] == system)
                          & (published["t"] < 1.0)]
    published = published.astype({"query_id": str, "doc_id": str})
    n_checked, mismatched = 0, []
    for k_c in cfg.candidate_depths:
        for policy in cfg.outsider_policies:
            seed_free = policy != RANDOM
            pub = published[(published["k_c"] == k_c) & (published["policy"] == policy)]
            path = Path(effects_dir) / f"{table_name(system, k_c, policy)}.parquet"
            ours = pd.read_parquet(
                path, columns=["query_id", "doc_id", "effect_type", "stratum", "t", "seed"],
                filters=[("stratum", "==", SELECTED_OUTSIDER)])
            ours = ours[(ours["effect_type"] == DELTA_T)
                        & ours["t"].isin(set(pub["t"]))]
            want = _selected_sets(pub, seed_free)
            have = _selected_sets(ours.astype({"query_id": str}), seed_free)
            n_checked += len(want)
            for key in sorted(set(want) | set(have), key=str):
                if want.get(key) != have.get(key):
                    mismatched.append({"k_c": k_c, "policy": policy, "key": list(key)})
    return {"published": str(published_path), "n_allocations": n_checked,
            "n_mismatched": len(mismatched), "examples": mismatched[:5],
            "ok": n_checked > 0 and not mismatched}


def _published_fingerprint(published_path: Path) -> str | None:
    manifest = published_path.with_name("allocation_manifest.json")
    return read_manifest(manifest).get("config_fingerprint") if manifest.exists() else None


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="World and budget runners on the frozen universe and the real Stage-2 cache")
    parser.add_argument("--system", required=True)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--cache", default=None, help="default: paths.cache in the config")
    parser.add_argument("--model", default=None, help="default: stage2.model in the config")
    parser.add_argument("--universe", default=None, help="default: paths.universe")
    parser.add_argument("--tables-dir", default=None,
                        help="where stage1_over_universe_<s>.parquet live; default: paths.data")
    parser.add_argument("--qrels", default=None, help="default: dataset.qrels_file")
    parser.add_argument("--published-allocations", default=None,
                        help="default: <paths.analysis>/allocation_promoted.parquet")
    parser.add_argument("--out-dir", default=None, help="default: <paths.effects>/real")
    parser.add_argument("--include-optional-t", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = cfg.paths
    if args.system not in cfg.stage1_systems:
        parser.error(f"--system must be one of {list(cfg.stage1_systems)}")
    model = args.model or cfg.reranker_model
    cache_path = Path(args.cache or paths.get("cache", "cache/stage2.sqlite"))
    if not cache_path.exists():
        # Stage2Cache would create an empty file here; refuse instead.
        parser.error(f"no Stage-2 cache at {cache_path}")
    published = Path(args.published_allocations
                     or Path(paths.get("analysis", "analysis")) / "allocation_promoted.parquet")
    if not published.exists():
        parser.error(f"no published allocations at {published}. Run "
                     "`python -m experiments_anupam.allocations` first.")
    stored = _published_fingerprint(published)
    if stored is not None and stored != cfg.fingerprint():
        parser.error(f"{published} was built under config {stored}, "
                     f"not {cfg.fingerprint()}")

    cache = Stage2Cache(cache_path)
    universe_path = args.universe or paths.get("universe", "data/universe.parquet")
    queries = load_queries(
        cfg, args.system, cache, model, universe_path,
        args.tables_dir or paths.get("data", "data"), args.qrels or cfg.qrels_file)
    out_dir = Path(args.out_dir or Path(paths.get("effects", "effects")) / "real")
    written = run_grid(
        queries, cfg, EffectSettings.from_config(cfg, args.system), out_dir,
        include_optional_t=args.include_optional_t,
        extra_manifest={
            "source": "real", "universe": str(universe_path),
            "alternative_system": alternative_system(cfg, args.system),
            "stage2_model": model, "stage2_cache": str(cache_path),
            "stage2_rows": cache.count(model), "stage2_digest": cache_digest(cache, model),
            "t_grid_run": list(grid_t(cfg, args.include_optional_t)),
        },
    )

    check = check_allocations(out_dir, args.system, cfg, published)
    manifest = read_manifest(written["manifest"])
    manifest["allocation_check"] = check
    write_manifest(written["manifest"], manifest)
    if not check["ok"]:
        logger.error("world D disagrees with the published allocations: %s", check)
        return 1
    logger.info("world D matches the published allocations at all %d allocations",
                check["n_allocations"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
