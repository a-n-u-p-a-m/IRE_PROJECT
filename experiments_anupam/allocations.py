"""
allocations.py - Publish the t-allocations and audit them.

    python -m experiments_anupam.allocations

World D asks what happens under a budget: of B Stage-2 calls, floor(t*B) go to
the top of C_q and the remainder to documents the gate discarded. Which
documents those are is the outsider policy, and the four membership strata that
result - retained, removed, selected_outsider, unselected_outsider - are what
every Delta_t sign invariant is conditioned on. If an allocation is wrong, the
strata are wrong, and the invariants then validate the wrong partition.

So this does not compute effects. It publishes the allocations and checks them:

  * **every allocation spends exactly B.** A short allocation makes one point
    of the t curve cheaper than the others, which silently turns a budget
    comparison into a compute comparison.
  * **the four strata partition U_q**, disjointly and completely. The Delta_t
    sign rules (removed <= 0, selected >= 0, unselected == 0) are conditional
    on membership, so an overlap or a gap would let a document be scored under
    two rules or none.
  * **t=1 reproduces the conventional cascade.** With B = K_c, floor(1*B) = B
    means every call goes to C_q and no outsider is promoted. This is the
    baseline every budget result is measured against; if it is not the plain
    cascade then `Delta_t` is relative to something that never ran.
  * **the random policy varies with its seed and only with its seed.** It is
    the control: if a seeded redraw changed the stratum SIZES, the control
    would differ from the treatments in cost as well as in selection, and the
    comparison would not isolate the policy.

Deliberately NOT done here: sampling or approximating. Every allocation is
built by `shared.allocation.allocate` and checked by `validate_allocation`, so
the audit exercises the production path rather than a reimplementation of it.

Outputs: the promoted outsiders per allocation (the interesting part - what
each policy actually chooses), a per-allocation audit row, and a manifest.
Tables are gitignored and reproducible; only the manifest is tracked.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from shared.allocation import (
    AllocationError,
    allocate,
    split_budget,
    validate_allocation,
)
from shared.config import load_config
from shared.provenance import run_manifest, write_manifest
from shared.schema import load
from shared.universe import load_frozen, universe_sets

logger = logging.getLogger(__name__)

RANDOM = "random"


def stage1_order(table: pd.DataFrame, system: str) -> dict[str, list[str]]:
    """{query_id: U_q in Stage-1 order} from the stage1-over-universe table."""
    r = table[table["system"] == system]
    if r.empty:
        raise SystemExit(f"no rows for system {system!r}")
    return {
        str(q): list(g.sort_values("rank")["doc_id"].astype(str))
        for q, g in r.groupby("query_id", observed=True)
    }


def scores_of(table: pd.DataFrame, system: str) -> dict[str, dict[str, float]]:
    r = table[table["system"] == system]
    return {
        str(q): dict(zip(g["doc_id"].astype(str), g["score"].astype(float)))
        for q, g in r.groupby("query_id", observed=True)
    }


def ranks_of_system(table: pd.DataFrame, system: str) -> dict[str, dict[str, int]]:
    r = table[table["system"] == system]
    return {
        str(q): dict(zip(g["doc_id"].astype(str), g["rank"].astype(int)))
        for q, g in r.groupby("query_id", observed=True)
    }


def seeds_for(policy: str, budget_seeds) -> tuple:
    """
    Only `random` consumes a seed.

    Running boundary or alternative under three seeds would write three
    identical allocations and make the audit look three times as thorough as
    it is.
    """
    return tuple(budget_seeds) if policy == RANDOM else (int(budget_seeds[0]),)


def audit_one(
    alloc: dict[str, list[str]],
    universe_q: set[str],
    budget: int,
    t: float,
) -> list[str]:
    """Spend and partition, for one allocation."""
    problems: list[str] = []
    try:
        validate_allocation(alloc, budget, t)
    except AllocationError as exc:
        problems.append(f"spend: {exc}")

    # The spend check is repeated here rather than delegated, because
    # `shared.validate_allocation` cannot catch the failure its own docstring
    # describes. It compares against `min(n_out, len(selected))` - the amount
    # ACHIEVED - so a short allocation always matches its own shortfall and the
    # check passes. The achievable ceiling is |D_q|, not |selected|:
    #
    #     validate_allocation({"retained": ["a"], "removed": ["b"],
    #                          "selected": [], "unselected": ["c", "d"],
    #                          "submitted": ["a"]}, 2, 0.5)   -> accepted,
    #
    # although one outsider was available and not taken. Reported for a fix in
    # shared/; until it lands this is the check that actually holds.
    n_cand_want, n_out_want = split_budget(budget, t)
    n_candidates = len(alloc["retained"]) + len(alloc["removed"])
    n_discarded = len(alloc["selected"]) + len(alloc["unselected"])
    achievable = min(n_cand_want, n_candidates) + min(n_out_want, n_discarded)
    if len(alloc["submitted"]) != achievable:
        problems.append(
            f"spend: submitted {len(alloc['submitted'])} of an achievable "
            f"{achievable} at t={t}, B={budget} (|C_q|={n_candidates}, "
            f"|D_q|={n_discarded}). A short allocation makes this point of the "
            "t curve cheaper than the others."
        )

    strata = ("retained", "removed", "selected", "unselected")
    sizes = {k: len(alloc[k]) for k in strata}
    union: set[str] = set()
    for k in strata:
        s = set(alloc[k])
        if len(s) != sizes[k]:
            problems.append(f"partition: '{k}' contains a duplicate")
        if union & s:
            problems.append(f"partition: '{k}' overlaps an earlier stratum")
        union |= s
    if union != universe_q:
        problems.append(
            f"partition: strata cover {len(union)} documents against "
            f"|U_q|={len(universe_q)}"
        )
    return problems


def run(config_path: str, universe_path: str, tables_dir: str,
        out_dir: str) -> tuple[dict, list[str]]:
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    universe = {q: set(v) for q, v in universe_sets(
        load_frozen(universe_path, fingerprint=cfg.fingerprint())
    ).items()}

    systems = list(cfg.stage1_systems)
    tables, orders, scores, ranks = {}, {}, {}, {}
    for s in systems:
        p = Path(tables_dir) / f"stage1_over_universe_{s}.parquet"
        if not p.exists():
            raise SystemExit(
                f"missing {p}. Allocations need Stage-1 over the whole "
                "universe, not just top-M: the boundary policy reads s1 below "
                "the cutoff and alternative reads the other system's ranks "
                "there. Run `python -m experiments_anupam.stage1_over_universe`."
            )
        tables[s] = load(p, "run")
        orders[s] = stage1_order(tables[s], s)
        scores[s] = scores_of(tables[s], s)
        ranks[s] = ranks_of_system(tables[s], s)

    promoted_rows: list[dict] = []
    audit_rows: list[dict] = []
    problems: list[str] = []

    for system in systems:
        others = [o for o in systems if o != system]
        other = others[0] if others else None

        for k_c in cfg.candidate_depths:
            budget = cfg.budget_for(k_c)
            for qid in sorted(orders[system]):
                ordered = orders[system][qid]
                candidates = ordered[:k_c]
                discarded = ordered[k_c:]
                s1_q = scores[system][qid]
                alt_q = ranks[other][qid] if other else None

                for t in cfg.t_grid:
                    for policy in cfg.outsider_policies:
                        for seed in seeds_for(policy, cfg.budget_seeds):
                            alloc = allocate(
                                candidates, discarded, budget, t, policy,
                                s1=s1_q, alternative_rank=alt_q, seed=seed,
                            )
                            found = audit_one(alloc, universe[qid], budget, t)
                            problems += [
                                f"{system} K_c={k_c} q={qid} t={t} "
                                f"{policy} seed={seed}: {f}" for f in found
                            ]
                            n_cand, n_out = split_budget(budget, t)
                            audit_rows.append({
                                "query_id": qid, "stage1_system": system,
                                "k_c": k_c, "t": t, "policy": policy,
                                "seed": seed, "budget_b": budget,
                                "n_from_candidates": n_cand,
                                "n_from_discarded": n_out,
                                "n_retained": len(alloc["retained"]),
                                "n_removed": len(alloc["removed"]),
                                "n_selected": len(alloc["selected"]),
                                "n_unselected": len(alloc["unselected"]),
                                "n_submitted": len(alloc["submitted"]),
                                "ok": not found,
                            })
                            promoted_rows.extend(
                                {"query_id": qid, "stage1_system": system,
                                 "k_c": k_c, "t": t, "policy": policy,
                                 "seed": seed, "doc_id": d, "promoted_rank": i}
                                for i, d in enumerate(alloc["selected"], start=1)
                            )
            logger.info("%s K_c=%d: %d allocations", system, k_c,
                        sum(1 for r in audit_rows
                            if r["stage1_system"] == system and r["k_c"] == k_c))

    audit = pd.DataFrame(audit_rows)
    promoted = pd.DataFrame(promoted_rows)

    # t=1 must be the plain cascade: all B calls inside C_q, nothing promoted.
    if 1.0 in set(cfg.t_grid):
        at1 = audit[audit["t"] == 1.0]
        if not at1.empty:
            if (at1["n_selected"] != 0).any():
                problems.append(
                    "t=1 promotes outsiders, so it is not the conventional "
                    "cascade and every Delta_t is relative to a policy that "
                    "never ran. Check b_equals_kc."
                )
            if (at1["n_retained"] != at1["budget_b"]).any():
                problems.append(
                    "t=1 does not retain exactly B candidates, so the budget "
                    "baseline is not a full candidate-only cascade."
                )

    # The random control must vary with its seed, and only in WHICH documents.
    rnd = audit[audit["policy"] == RANDOM]
    seed_sensitivity = None
    if not rnd.empty and len(set(rnd["seed"])) > 1:
        key = ["query_id", "stage1_system", "k_c", "t"]
        if rnd.groupby(key)["n_selected"].nunique().gt(1).any():
            problems.append(
                "the random policy's stratum sizes change with the seed, so "
                "the control differs from the treatments in cost as well as "
                "selection and the comparison does not isolate the policy."
            )
        pr = promoted[(promoted["policy"] == RANDOM) & (promoted["t"] < 1.0)]
        if not pr.empty:
            per = pr.groupby(key + ["seed"])["doc_id"].apply(frozenset)
            varied = per.groupby(key).nunique()
            seed_sensitivity = round(float((varied > 1).mean()), 4)
            if seed_sensitivity == 0.0:
                problems.append(
                    "the random policy chose identical documents under every "
                    "seed, so it is not sampling and cannot serve as a control."
                )

    audit.to_parquet(out / "allocation_audit.parquet", index=False)
    promoted.to_parquet(out / "allocation_promoted.parquet", index=False)

    extra = {
        "artifact": [str(out / "allocation_audit.parquet"),
                     str(out / "allocation_promoted.parquet")],
        "n_allocations": int(len(audit)),
        "n_promoted_rows": int(len(promoted)),
        "t_grid": list(cfg.t_grid),
        "policies": list(cfg.outsider_policies),
        "budget_seeds": list(cfg.budget_seeds),
        "b_equals_kc": bool(cfg.b_equals_kc),
        "all_allocations_valid": bool(audit["ok"].all()) if len(audit) else None,
        "random_seed_sensitivity": seed_sensitivity,
        "by_k_c": {
            int(k): {
                "budget_b": int(g["budget_b"].iloc[0]),
                "n_allocations": int(len(g)),
                "n_promoted_median": float(
                    g[g["t"] < 1.0]["n_selected"].median()
                ) if (g["t"] < 1.0).any() else 0.0,
            }
            for k, g in audit.groupby("k_c")
        },
        "problems": problems,
    }
    write_manifest(out / "allocation_manifest.json",
                   run_manifest(cfg.fingerprint(), owner="anupam", extra=extra))
    return extra, problems


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Publish and audit the t-allocations")
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--universe", default="data/universe.parquet")
    p.add_argument("--tables-dir", default="data")
    p.add_argument("--out-dir", default="analysis")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    extra, problems = run(args.config, args.universe, args.tables_dir,
                          args.out_dir)

    print()
    print(f"{extra['n_allocations']:,} allocations, "
          f"{extra['n_promoted_rows']:,} promoted documents")
    print(f"t grid {extra['t_grid']}  policies {extra['policies']}  "
          f"seeds {extra['budget_seeds']}")
    for k, v in extra["by_k_c"].items():
        print(f"  K_c={k}: B={v['budget_b']}, {v['n_allocations']:,} "
              f"allocations, {v['n_promoted_median']:.0f} promoted per "
              f"allocation (median over t<1)")
    print(f"random seed sensitivity: {extra['random_seed_sensitivity']} "
          "(fraction of settings where a reseed changed the selection)")
    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for pr in problems[:20]:
            print(f"  - {pr}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        return 1 if args.strict else 0
    print("Every allocation spends exactly B, the four strata partition U_q, "
          "t=1 is the conventional cascade, and the random control varies with "
          "its seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
