"""
reference_subset.py - Publish which queries receive full no-gate scoring.

    python -m experiments_anupam.reference_subset

The reference subset is the set of queries for which world C is computed over
the whole universe rather than approximated. It is a CONTRACT: Shubham's
analysis reads it to decide which queries carry a measured E_gate, so "all 97"
living in a chat message is not the same as a committed file. `.gitignore`
names `data/reference_subset.txt` as deliberately tracked for that reason.

The size is not chosen here. It comes from the H7 power simulation, and this
module's job is to read that result, check it was produced against THIS freeze
and THIS universe, and refuse rather than guess if either disagrees.

What the power simulation returned, at the real median |U_q| of 9,097 and the
headline K_c of 100:

    group test        alternative 97    boundary -    random -
    continuous test   alternative 97    boundary -

Only the `alternative` outsider policy reaches the 0.80 target, and only with
every query. `boundary` clears at the smaller synthetic universe (U=2,000) and
not at the real one, so H7 is CONFIRMATORY for alternative and DESCRIPTIVE for
boundary. `random` is the control and is not expected to reach power - a
control that did would mean the effect is not specific to the policy.

That split is declared here, before any real effect is computed, and recorded
in the manifest so the Week-4 report cannot quietly promote a descriptive
result to a confirmatory one.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from shared.config import load_config
from shared.provenance import run_manifest, write_manifest
from shared.universe import (
    load_frozen,
    read_reference_subset,
    universe_sizes,
    write_reference_subset,
)

logger = logging.getLogger(__name__)

TARGET_POWER_KEY = "target_power"


def read_power_result(path: str | Path, fingerprint: str, universe_median: int) -> dict:
    """
    Load the H7 power manifest, refusing anything not comparable to this freeze.

    Two refusals, both about comparability rather than correctness. A power
    result from a different config fingerprint answers a question about a
    different experiment; one computed against a different universe size
    answers a question about a different universe. Either would let a
    reference size chosen elsewhere look authoritative here.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"no power result at {p}. The reference-subset size comes from the "
            "H7 power simulation (H2, Shubham); it is not chosen here. Run "
            "`python -m experiments_shubham.power_h7` first."
        )
    d = json.loads(p.read_text(encoding="utf-8"))

    got_fp = d.get("config_fingerprint")
    if got_fp != fingerprint:
        raise SystemExit(
            f"{p} was produced under freeze {got_fp!r} but the config is "
            f"{fingerprint!r}. A reference size from a different freeze is not "
            "comparable; re-run the power simulation under this one."
        )

    ru = (d.get("realistic_universe") or {}).get("value")
    if ru is not None and int(ru) != int(universe_median):
        raise SystemExit(
            f"{p} assumed a median |U_q| of {ru}, but the frozen universe has "
            f"{universe_median}. Power depends on the universe size, so this "
            "result does not describe the universe we froze."
        )
    return d


def policy_verdicts(power: dict, universe_median: int) -> dict:
    """
    Which policies reach target power at the real universe size, per test.

    `smallest_per_setting` keys look like "alternative@9097". A null value means
    no tested reference size reached the target, which is a finding to declare,
    not a gap to fill.
    """
    rec = power.get("recommendation", {})
    target = rec.get(TARGET_POWER_KEY)
    suffix = f"@{universe_median}"
    out: dict = {"target_power": target, "headline_k_c": rec.get("k_c"),
                 "t": rec.get("t"), "by_test": {}}

    for test in ("group", "continuous"):
        block = rec.get(test) or {}
        smallest = block.get("smallest_per_setting") or {}
        here = {
            k.split("@")[0]: v for k, v in smallest.items() if k.endswith(suffix)
        }
        out["by_test"][test] = {
            "confirmatory": sorted(p for p, v in here.items() if v is not None),
            "descriptive": sorted(p for p, v in here.items() if v is None),
            "smallest_size": {p: v for p, v in sorted(here.items())},
        }
    return out


def required_size(verdicts: dict, n_queries: int) -> tuple[int, str]:
    """
    The subset size: the largest size any confirmatory setting needs.

    Taking the largest is the only safe reduction. A smaller size would leave
    at least one policy under-powered while the report still presented it as
    confirmatory, which is exactly the failure the pre-declaration exists to
    prevent.
    """
    needed = [
        v for test in verdicts["by_test"].values()
        for v in test["smallest_size"].values() if v is not None
    ]
    if not needed:
        return n_queries, (
            "no policy reached target power at any tested size, so H7 is "
            "descriptive throughout; the subset is every query so the "
            "descriptive result is at least computed over the full query set"
        )
    size = max(needed)
    if size > n_queries:
        raise SystemExit(
            f"the power simulation needs {size} reference queries but the "
            f"universe has only {n_queries}. H7 cannot be confirmatory on this "
            "collection at any budget; say so rather than reporting it."
        )
    return size, (
        f"the largest size any confirmatory setting requires ({size}); taking "
        "the maximum keeps every policy declared confirmatory actually powered"
    )


def run(config_path: str, universe_path: str, power_path: str,
        out_path: str) -> dict:
    cfg = load_config(config_path)
    uni = load_frozen(universe_path, fingerprint=cfg.fingerprint())
    sizes = universe_sizes(uni)
    query_ids = sorted(str(q) for q in uni["query_id"].unique())
    median = int(sizes.median())

    power = read_power_result(power_path, cfg.fingerprint(), median)
    verdicts = policy_verdicts(power, median)
    size, why = required_size(verdicts, len(query_ids))

    if size == len(query_ids):
        chosen = query_ids
        selection = "every query; no smaller size reached target power"
    else:
        # Sampling would need a declared, seeded rule. We never reach here on
        # this collection, and a silent head-of-list truncation would bias
        # towards whatever order the ids happen to be in.
        raise SystemExit(
            f"the power result allows a subset of {size} of {len(query_ids)} "
            "queries. Sampling needs a declared seeded strategy "
            "(`universe.sample_reference_subset`); it is not done implicitly."
        )

    path = write_reference_subset(out_path, chosen)
    assert read_reference_subset(path) == chosen, "round-trip failed"

    extra = {
        "artifact": str(path),
        "n_reference_queries": len(chosen),
        "n_queries_total": len(query_ids),
        "selection": selection,
        "size_rationale": why,
        "universe_size_median": median,
        "power_result": str(power_path),
        "power_verdicts": verdicts,
        "world_c_pairs": int(sizes.sum()),
    }
    write_manifest(
        Path(out_path).with_name("reference_subset_manifest.json"),
        run_manifest(cfg.fingerprint(), owner="anupam", extra=extra),
    )
    return extra


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Publish the reference subset")
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--universe", default="data/universe.parquet")
    p.add_argument("--power", default="analysis/h7_power_manifest.json")
    p.add_argument("--out", default="data/reference_subset.txt")
    args = p.parse_args()

    extra = run(args.config, args.universe, args.power, args.out)

    print()
    print(f"reference subset: {extra['n_reference_queries']} of "
          f"{extra['n_queries_total']} queries -> {extra['artifact']}")
    print(f"  {extra['selection']}")
    print(f"  world C over the subset: {extra['world_c_pairs']:,} pairs")
    print()
    v = extra["power_verdicts"]
    print(f"H7 at K_c={v['headline_k_c']}, t={v['t']}, "
          f"target power {v['target_power']}:")
    for test, b in v["by_test"].items():
        print(f"  {test:11} confirmatory: {', '.join(b['confirmatory']) or 'none'}")
        print(f"  {'':11} descriptive : {', '.join(b['descriptive']) or 'none'}")
    print()
    print("Declared before any real effect is computed. A descriptive policy "
          "must not be reported as confirmatory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
