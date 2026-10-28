"""
smoke_pipeline.py - Drive the whole Week-1 chain end to end with no retriever.

    python -m experiments_anupam.smoke_pipeline

    queries + qrels  ->  split
                     ->  stage-1 runs  ->  frozen universe  ->  containment

Why this exists. Every Week-1 module has unit tests, but unit tests do not
prove the modules compose: that `run_stage1`'s output is what `build_universe`
expects to read, that the manifests land beside the artefacts the next step
looks for, that the config fingerprint written by one step is the one the next
step demands. This script runs the real entry points - `data_prep.run`,
`run_stage1.run`, `build_universe.run` - in order, in a throwaway directory,
and asserts the invariants the frame depends on.

It fakes exactly one thing: the retriever. Both runs come from `PrecomputedRun`
over synthetic rankings, via the production `--from-run` path. So when the real
BM25 run lands, the retriever is the only new variable and everything
downstream of it has already been exercised. That is the point: no Pyserini, no
JDK, no index, no network, a few seconds.

It is not a test of correctness - the unit suites do that - and the numbers it
prints are meaningless. It answers one question: does the chain hold together.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from shared.config import load_config
from shared.schema import load, save
from shared.universe import check_containment, load_frozen, universe_sizes

logger = logging.getLogger(__name__)

N_QUERIES = 24

# The document pool is derived from the retrieval depth, not fixed: the real
# config's largest K_c is 1000, so M runs into the thousands and a small pool
# could not supply a ranking that deep. Two runs of M drawn from 2M overlap by
# roughly half, which is what makes the union broader than either source while
# keeping it comfortably above max(K_c).
POOL_FACTOR = 2


def synthetic_queries_and_qrels(root: Path, grades: dict, pool: int) -> tuple[Path, Path]:
    """
    Query and qrels files in the real TREC formats.

    Qrels use only grades the config's gain map covers, because an unmapped
    grade is a separate failure mode with its own test; here it would just mask
    the composition failures this script is looking for.
    """
    rng = np.random.default_rng(0)
    qids = [str(1000 + i) for i in range(N_QUERIES)]

    queries = root / "queries.tsv"
    queries.write_text(
        "".join(f"{q}\tsynthetic query {q} about topic {i % 7}\n"
                for i, q in enumerate(qids)),
        encoding="utf-8",
    )

    usable = sorted(grades)
    lines = []
    for q in qids:
        for d in rng.choice(pool, size=30, replace=False):
            lines.append(f"{q} 0 d{d:04d} {usable[int(rng.integers(len(usable)))]}\n")
    qrels = root / "qrels.txt"
    qrels.write_text("".join(lines), encoding="utf-8")
    return queries, qrels


def synthetic_run(system: str, depth: int, pool: int, seed: int,
                  checkpoint: str) -> pd.DataFrame:
    """
    A run table over a shared document pool, so the two systems partly agree.

    Partial agreement is deliberate: if the sources ranked disjoint documents
    the union would be trivially large and the |U_q| > max(K_c) check would
    never be near its boundary.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_QUERIES):
        qid = str(1000 + i)
        docs = rng.choice(pool, size=depth, replace=False)
        for rank, d in enumerate(docs, start=1):
            rows.append({"query_id": qid, "doc_id": f"d{int(d):04d}",
                         "system": system, "rank": rank,
                         "score": float(depth - rank), "checkpoint": checkpoint})
    return pd.DataFrame(rows)


def shrink_config(src: str, root: Path, depth_m: int) -> Path:
    """
    The frozen config with the universe shrunk to smoke-test size.

    Only depth and cap change. The candidate depths, cutoff and gain map stay
    exactly as committed, so the |U_q| > max(K_c) relation under test is the
    real one rather than one invented for the fixture.
    """
    with open(src, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["universe"]["depth_m"] = depth_m
    raw["universe"]["max_size"] = depth_m * 2
    out = root / "smoke_config.yaml"
    out.write_text(yaml.safe_dump(raw))
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="End-to-end dry run of the Week-1 chain, no retriever"
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--keep", action="store_true",
                        help="leave the scratch directory in place for inspection")
    args = parser.parse_args()

    from experiments_anupam import build_universe, data_prep, run_stage1

    root = Path(tempfile.mkdtemp(prefix="week1_smoke_"))
    logger.info("scratch directory %s", root)
    failures: list[str] = []

    try:
        base = load_config(args.config)
        # Deep enough to clear the real largest K_c with room for the union to
        # be interesting, shallow enough to run in seconds.
        depth_m = max(base.candidate_depths) + max(base.candidate_depths) // 2
        pool = depth_m * POOL_FACTOR
        cfg_path = shrink_config(args.config, root, depth_m)
        cfg = load_config(cfg_path)
        logger.info(
            "config %s | M=%d | pool=%d | K_c=%s | k=%d",
            cfg.fingerprint(), cfg.universe_depth_m, pool,
            list(cfg.candidate_depths), cfg.final_cutoff_k,
        )

        # ── step 1: queries, qrels, split ────────────────────────────
        queries, qrels_txt = synthetic_queries_and_qrels(root, cfg.gain_map, pool)
        split = data_prep.run(
            queries_path=str(queries),
            qrels_txt=str(qrels_txt),
            qrels_out=str(root / "qrels.parquet"),
            split_out=str(root / "query_split.json"),
            config_path=str(cfg_path),
        )
        assigned = sum(len(v) for v in split.values())
        if assigned != N_QUERIES:
            failures.append(f"split covers {assigned} of {N_QUERIES} queries")
        for name in ("qrels.parquet", "query_split.json", "data_prep_manifest.json"):
            if not (root / name).exists():
                failures.append(f"step 1 did not write {name}")

        # ── step 2: two Stage-1 runs via the production --from-run path ──
        runs_dir = root / "runs"
        runs_dir.mkdir()
        seeds = {"bm25": 1, "dpr": 2}
        checkpoints = {
            "bm25": "synthetic|bm25-k1=0.9-b=0.4",
            "dpr": "synthetic|dpr",
        }
        for system in cfg.universe_sources:
            if system not in seeds:
                failures.append(f"no synthetic source defined for {system!r}")
                continue
            seed_path = root / f"seed_{system}.parquet"
            save(synthetic_run(system, depth_m, pool, seeds[system],
                               checkpoints[system]), seed_path, "run")
            run_stage1.run(
                system=system,
                queries_path=str(queries),
                out_path=str(runs_dir / f"{system}.parquet"),
                config_path=str(cfg_path),
                depth=depth_m,
                args=SimpleNamespace(from_run=str(seed_path), limit_queries=None),
            )
            manifest = runs_dir / f"{system}_manifest.json"
            if not manifest.exists():
                failures.append(f"step 2 did not write {manifest.name}")
            written = load(runs_dir / f"{system}.parquet", "run")
            if set(written["checkpoint"]) != {checkpoints[system]}:
                failures.append(f"{system} lost its checkpoint through run_stage1")

        # ── step 3: freeze the universe ──────────────────────────────
        out = build_universe.run(
            config_path=str(cfg_path),
            runs_dir=str(runs_dir),
            out_path=str(root / "data" / "universe.parquet"),
            qrels_path=str(root / "qrels.parquet"),
            force=False,
        )
        frozen = load_frozen(out, fingerprint=cfg.fingerprint())
        sizes = universe_sizes(frozen)
        logger.info(
            "universe: %d queries, |U_q| %d-%d (median %d)",
            sizes.size, int(sizes.min()), int(sizes.max()), int(sizes.median()),
        )
        if int(sizes.min()) <= max(cfg.candidate_depths):
            failures.append("frozen universe does not exceed the largest K_c")
        if sizes.size != N_QUERIES:
            failures.append(f"universe covers {sizes.size} of {N_QUERIES} queries")
        for name in ("universe_manifest.json", "universe_coverage.parquet"):
            if not (out.parent / name).exists():
                failures.append(f"step 3 did not write {name}")

        # ── step 4: the invariants the frame rests on ────────────────
        for system in cfg.stage1_systems:
            run_df = load(runs_dir / f"{system}.parquet", "run")
            for k_c in cfg.candidate_depths:
                try:
                    check_containment(run_df, frozen, k_c, system)
                except Exception as exc:                        # noqa: BLE001
                    failures.append(f"containment failed for {system} at K_c={k_c}: {exc}")

        # A rebuild must be refused, or the frozen universe is not frozen.
        try:
            build_universe.run(str(cfg_path), str(runs_dir),
                               str(out), str(root / "qrels.parquet"), force=False)
            failures.append("a frozen universe was silently overwritten")
        except SystemExit:
            pass

        # A stale fingerprint must be refused, or freezes mean nothing.
        try:
            load_frozen(out, fingerprint="0" * 12)
            failures.append("a fingerprint mismatch was accepted")
        except Exception:                                       # noqa: BLE001
            pass

    finally:
        if args.keep:
            logger.info("kept %s", root)
        else:
            shutil.rmtree(root, ignore_errors=True)

    if failures:
        print("\nWeek-1 chain FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nWeek-1 chain holds: split, two Stage-1 runs, frozen universe, "
          "containment at every K_c, rebuild and fingerprint both refused.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
