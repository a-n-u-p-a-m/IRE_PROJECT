"""
engine.py - The counterfactual engine: the world runner and the budget runner.

    world runner    worlds A, B, C per (query, K_c)        -> E_rerank, E_gate
    budget runner   world D per (query, K_c, t, policy, seed) -> Delta_t

Both are thin drivers over `shared.worlds` and `shared.effects`; the estimands
themselves are defined there and nowhere else. What the runners add is the
loop over the frozen grid and the checks that must pass before a single effect
is trusted:

  * every (query, K_c) passes `verify_isolation` (world B ranks exactly C_q,
    world C only adds competitors, no duplicates);
  * every t-grid passes `verify_t_baseline` (with B = K_c the t=1 world IS the
    actual cascade);
  * the assembled table passes the schema's sign invariants in strict mode,
    and every row's stratum belongs to its effect type.

Strict is the default. `strict=False` relaxes only the E_gate sign checks and
is for a listwise Stage 2, which is a different estimand.

Inputs are per-query `QueryInputs`, so the same runners take the synthetic
generator now and the frozen universe plus Stage-2 cache in Week 3:

    python -m experiments_shubham.engine --regime strong_stages --n-queries 20
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from shared.allocation import RANDOM
from shared.config import FrozenConfig, load_config
from shared.effects import (
    budget_effects_for_query,
    check_invariants,
    concat_effects,
    stage_effects_for_query,
    summarise,
)
from shared.provenance import run_manifest, write_manifest
from shared.schema import save
from shared.strata import check_strata_valid
from shared.synthetic import (
    alternative_ranks,
    apply_budget_stress,
    apply_gate_stress,
    generate,
    params_for_regime,
    to_lookups,
)
from shared.worlds import build_t_grid, build_worlds, verify_isolation, verify_t_baseline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryInputs:
    """Everything the engine needs for one query."""

    query_id: str
    universe: tuple[str, ...]
    s1: Mapping[str, float]
    s2: Mapping[str, float]
    relevance: Mapping[str, int]                # judged documents only
    judged: Mapping[str, bool]
    alternative_rank: Mapping[str, int] | None = None


@dataclass(frozen=True)
class EffectSettings:
    """The frozen settings every effect row depends on."""

    stage1_system: str
    k: int
    gain_map: Mapping[int, float]
    utility: str = "dcg"
    persistence: float = 0.8

    @classmethod
    def from_config(cls, cfg: FrozenConfig, stage1_system: str) -> EffectSettings:
        return cls(stage1_system, cfg.final_cutoff_k, cfg.gain_map,
                   cfg.utility_kind, cfg.rbp_persistence)


def _finish(frames: list[pd.DataFrame], settings: EffectSettings,
            strict: bool) -> pd.DataFrame:
    effects = concat_effects(frames, strict=strict)
    check_invariants(effects, utility=settings.utility, pointwise=strict)
    check_strata_valid(effects)
    return effects


# ──────────────────────────────────────────────────────────────────────
# World runner: A, B, C
# ──────────────────────────────────────────────────────────────────────

def run_worlds(
    queries: Iterable[QueryInputs],
    k_c: int,
    settings: EffectSettings,
    strict: bool = True,
) -> pd.DataFrame:
    """E_rerank and E_gate for every document of every query at one K_c."""
    frames = []
    for q in queries:
        w = build_worlds(q.query_id, q.universe, q.s1, q.s2, k_c)
        verify_isolation(w)
        frames.append(stage_effects_for_query(
            w, q.relevance, q.judged, settings.stage1_system, settings.k,
            settings.gain_map, settings.utility, settings.persistence,
        ))
    return _finish(frames, settings, strict)


# ──────────────────────────────────────────────────────────────────────
# Budget runner: D
# ──────────────────────────────────────────────────────────────────────

def run_budget(
    queries: Iterable[QueryInputs],
    k_c: int,
    budget_b: int,
    t_grid: Sequence[float],
    policy: str,
    seeds: Sequence[int],
    settings: EffectSettings,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Delta_t for every document of every query at one K_c and one policy.

    Only the random policy is repeated over seeds. Boundary and alternative
    are deterministic, so they run once and carry seed=NA, as the effect
    schema specifies; repeating them would only duplicate identical rows.
    """
    if 1.0 not in t_grid:
        raise ValueError("t_grid must contain 1.0, the baseline of every Delta_t")
    randomised = policy == RANDOM
    run_seeds = list(seeds) if randomised else list(seeds[:1])

    frames = []
    for q in queries:
        w = build_worlds(q.query_id, q.universe, q.s1, q.s2, k_c)
        verify_isolation(w)
        for seed in run_seeds:
            tw = build_t_grid(w, q.s1, q.s2, budget_b, t_grid, policy,
                              alternative_rank=q.alternative_rank, seed=seed)
            verify_t_baseline(tw, w)
            frame = budget_effects_for_query(
                w, tw, q.relevance, q.judged, settings.stage1_system, settings.k,
                settings.gain_map, settings.utility, settings.persistence,
            )
            if not randomised:
                frame["seed"] = pd.NA
            frames.append(frame)
    return _finish(frames, settings, strict)


# ──────────────────────────────────────────────────────────────────────
# The whole frozen grid
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GridTable:
    """One block of the grid: stage effects at a K_c, or budget effects at a
    (K_c, policy)."""

    name: str
    k_c: int
    policy: str | None
    effects: pd.DataFrame


def grid_t(cfg: FrozenConfig, include_optional_t: bool = False) -> tuple[float, ...]:
    t_grid = tuple(cfg.t_grid)
    if include_optional_t:
        t_grid += tuple(cfg.t_grid_optional)
    return t_grid


def iter_grid(
    queries: Sequence[QueryInputs],
    cfg: FrozenConfig,
    settings: EffectSettings,
    budget_for: Callable[[int], int] | None = None,
    include_optional_t: bool = False,
    strict: bool = True,
) -> Iterator[GridTable]:
    """
    Every K_c, and for world D every policy and seed, one table at a time.

    One table per K_c for the stage effects and per (K_c, policy) for the
    budget effects keeps each one small enough to validate in memory.
    """
    budget_for = budget_for or cfg.budget_for
    t_grid = grid_t(cfg, include_optional_t)
    system = settings.stage1_system
    for k_c in cfg.candidate_depths:
        stage = run_worlds(queries, k_c, settings, strict)
        yield GridTable(f"{system}_stage_kc{k_c}", k_c, None, stage)
        for policy in cfg.outsider_policies:
            budget = run_budget(queries, k_c, budget_for(k_c), t_grid, policy,
                                cfg.budget_seeds, settings, strict)
            yield GridTable(f"{system}_budget_kc{k_c}_{policy}", k_c, policy, budget)


def run_grid(
    queries: Sequence[QueryInputs],
    cfg: FrozenConfig,
    settings: EffectSettings,
    out_dir: str | Path,
    budget_for: Callable[[int], int] | None = None,
    include_optional_t: bool = False,
    strict: bool = True,
    extra_manifest: Mapping | None = None,
) -> dict[str, Path]:
    """The whole grid, written one validated Parquet file per table."""
    out_dir = Path(out_dir)
    budget_for = budget_for or cfg.budget_for
    t_grid = grid_t(cfg, include_optional_t)
    system = settings.stage1_system

    written: dict[str, Path] = {}
    summaries = []
    start = time.perf_counter()
    for table in iter_grid(queries, cfg, settings, budget_for, include_optional_t,
                           strict):
        path = out_dir / f"{table.name}.parquet"
        written[table.name] = save(table.effects, path, "effect", strict)
        summaries.append(summarise(table.effects))
        logger.info("%s: %d rows", table.name, len(table.effects))

    summary_path = out_dir / f"{system}_summary.csv"
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    written["summary"] = summary_path

    manifest = run_manifest(cfg.fingerprint(), owner="shubham", extra={
        "stage1_system": system, "n_queries": len(queries),
        "candidate_depths": list(cfg.candidate_depths),
        "budgets": {k: budget_for(k) for k in cfg.candidate_depths},
        "t_grid": list(t_grid), "policies": list(cfg.outsider_policies),
        "seeds": list(cfg.budget_seeds), "k": settings.k, "utility": settings.utility,
        "strict": strict, "files": sorted(p.name for p in written.values()),
        "seconds": round(time.perf_counter() - start, 1),
        **dict(extra_manifest or {}),
    })
    written["manifest"] = write_manifest(out_dir / f"{system}_manifest.json", manifest)
    return written


# ──────────────────────────────────────────────────────────────────────
# Synthetic inputs
# ──────────────────────────────────────────────────────────────────────

def synthetic_frame(cfg: FrozenConfig, regime: str, n_queries: int | None = None,
                    n_docs: int | None = None, stress_kc: int | None = None,
                    seed: int | None = None, stress: bool = True,
                    far: bool = False) -> pd.DataFrame:
    """
    One generated regime, with its stress manipulation applied at `stress_kc`
    (default: the largest candidate depth) for gate_stress and budget_stress.

    `far` buries the gate-stress documents far below K_c instead of just under
    it; `stress=False` gives the same draw with no manipulation, as a control.
    """
    syn = cfg.synthetic
    params = params_for_regime(
        regime, n_queries or syn["n_queries"], n_docs or syn["n_docs_per_query"],
        syn["relevance_quantiles"], seed=cfg.seed if seed is None else seed,
    )
    df = generate(params)
    if not stress:
        return df
    stress_kc = stress_kc or max(cfg.candidate_depths)
    if regime == "gate_stress":
        df = apply_gate_stress(df, stress_kc, syn["gate_stress_planted"],
                               seed=params.seed, far=far)
    elif regime == "budget_stress":
        df = apply_budget_stress(df, stress_kc, syn["budget_stress_outsiders"],
                                 seed=params.seed)
    return df


def synthetic_inputs(df: pd.DataFrame) -> list[QueryInputs]:
    """Per-query engine inputs from a synthetic frame. Every label is known."""
    universe, s1, s2, rel = to_lookups(df)
    alt = alternative_ranks(df)
    return [
        QueryInputs(q, tuple(universe[q]), s1[q], s2[q], rel[q],
                    {d: True for d in universe[q]}, alt[q])
        for q in universe
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="World and budget runners on synthetic data")
    parser.add_argument("--regime", required=True)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--n-queries", type=int, default=None)
    parser.add_argument("--n-docs", type=int, default=None)
    parser.add_argument("--out-dir", default=None,
                        help="default: effects/synthetic/<regime>")
    parser.add_argument("--include-optional-t", action="store_true")
    parser.add_argument("--non-strict", action="store_true",
                        help="relax E_gate signs; only for a listwise Stage 2")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = synthetic_frame(cfg, args.regime, args.n_queries, args.n_docs)
    queries = synthetic_inputs(df)
    effects_root = Path(cfg.paths.get("effects", "effects"))
    out_dir = args.out_dir or effects_root / "synthetic" / args.regime
    written = run_grid(
        queries, cfg, EffectSettings.from_config(cfg, "synthetic"), out_dir,
        include_optional_t=args.include_optional_t, strict=not args.non_strict,
        extra_manifest={"regime": args.regime,
                        "n_docs": int(df.groupby("query_id").size().max())},
    )
    for name, path in written.items():
        logger.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
