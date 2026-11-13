# experiments_shubham

**Owner: Shubham.** DPR retrieval; the counterfactual engine (worlds A-D,
`E_rerank`, `E_gate`, `Delta_t`); the synthetic benchmark; the power
simulation; H7.

Expected here: the DPR run script, the world runner, the budget runner, the
synthetic regime driver, the exact-recovery report (E1), the `simulate_query`
closure and the power simulation, the diagnosis and H7 analysis.

Import from `shared/`. Never import another member's module — go through the
file schemas in `shared/schema.py`.

## DPR Stage-1 run

| file | what it does |
|---|---|
| `fetch_trec_dl.py` | writes `data/queries.tsv` (97 judged TREC-DL 2019+2020 queries) and `data/qrels.txt` |
| `dpr.py` | cosDPR-distil encoder, exact search over the 89 float16 shards (local or rclone remote), index build |
| `run_dpr.py` | top-M run for every query → `runs/dpr.parquet` + `runs/dpr_manifest.json` |
| `dpr_run_colab.ipynb` | the run on Colab: index download via the Drive API, encoder check, run, sanity check |
| `tests/` | `python -m pytest experiments_shubham` (fake encoder; no model or index needed) |

The index lives in the shared Drive folder `IRE_PROJECT/dpr_index_cosdpr_distil`.

## Counterfactual engine

| file | what it does |
|---|---|
| `engine.py` | world runner (A/B/C → `E_rerank`, `E_gate`) and budget runner (D → `Δ_t`) over the frozen grid, strict by default |
| `tests/test_engine.py` | toy-fixture numbers, seed handling, refusal cases, every synthetic regime in strict mode, condensed effects recounted from the uncondensed table |

Every (query, K_c) passes `verify_isolation`, every t-grid passes
`verify_t_baseline`, and every table passes the schema's sign invariants before
it is written. Only the random outsider policy repeats over seeds; boundary and
alternative run once with `seed=NA`.

    python -m experiments_shubham.engine --regime weak_retriever            # ~2 min per mode, ~2 GB
    python -m experiments_shubham.engine --regime strong_stages --n-queries 10

Writes one effect table per K_c (stage effects) and per (K_c, policy) (budget
effects) in each ranking mode, a summary CSV and a manifest to
`effects/synthetic/<regime>/`. `<table>.parquet` is the primary, uncondensed
estimand; `<table>_condensed.parquet` is the freeze-v2 robustness check, where
unjudged documents are dropped before ranks are assigned and rows cover judged
documents only. Pick one mode per computation (`shared.schema.select_mode`);
the shared readers refuse a frame holding both.

## Real effects (Week 3)

| file | what it does |
|---|---|
| `real.py` | the engine on the frozen universe: `QueryInputs` from `data/universe.parquet`, `data/stage1_over_universe_<s>.parquet`, `data/qrels.parquet` and the Stage-2 cache, then the whole grid, then world D checked against the published allocations |
| `cache_fill_colab.ipynb` | fills one member's cache shard on a Colab T4 (set `OWNER`): rebuilds the universe, shuffle spot check, resumable fill with Drive copies, completeness check, shard manifest with digest |
| `tests/test_real.py` | a synthetic cascade written out as the real artefacts reproduces the synthetic path exactly; each refusal; a corrupted published allocation is caught for every policy |

    # inputs, all reproducible from the committed runs (no GPU):
    python -m experiments_anupam.data_prep --queries data/queries.tsv --qrels-txt data/qrels.txt
    python -m experiments_anupam.build_universe --qrels data/qrels.parquet
    python -m experiments_anupam.stage1_over_universe
    python -m experiments_anupam.allocations

    python -m experiments_shubham.real --system bm25      # ~10 min, ~3.5 GB peak
    python -m experiments_shubham.real --system dpr
    python -m experiments_shubham.analysis --effects-dir effects/real --system bm25 --contrast-with dpr

    # one shard, before the others land (~3 min per system for 32 queries):
    python -m experiments_shubham.real --system bm25 \
        --cache cache/stage2_shard_shubham.sqlite \
        --queries-from cache/stage2_shard_shubham_manifest.json   # -> effects/real_preview

It refuses to start unless the universe carries the current fingerprint, both
Stage-1 tables cover exactly U_q, and the cache holds every (q, d) in U_q under
`stage2.model`. The alternative retriever is the other Stage-1 system, as in the
published allocations. The manifest records the cache's row count and an
order-independent digest of its scores, and `allocation_check`: every selected
outsider in the budget tables must equal `analysis/allocation_promoted.parquet`,
or the run exits non-zero.

Dry run on the real universe with stand-in Stage-2 scores (2026-11-05, M1, 16 GB):
both systems, all 12 tables per mode, invariants pass, world D equal to the
published allocations at all 4,365 allocations per system; 9.5 min per system,
~400 MB of Parquet in all.

## E1 - synthetic implementation validation

| file | what it does |
|---|---|
| `synthetic_e1.py` | the engine on all six regimes plus far-buried gate stress and an unstressed control, checked two ways |
| `tests/test_synthetic_e1.py` | re-derivation against the toy fixture, exact recovery on a small grid, the testability guard |

1. **Exact recovery.** Every effect is re-derived in plain numpy, without
   `shared.worlds`, `shared.effects`, `shared.allocation` or `shared.ordering`,
   and must match the engine row for row (rows, ranks, strata, effects within 1e-12).
2. **Effect sanity.** Directional expectations declared in `EXPECTATIONS` before
   the run: a weak retriever makes the gate cost more (loss on non-survivors), a weak reranker helps less
   and harms more queries, the boundary probe finds near-buried documents but
   misses far-buried ones, budget-stress outsiders reach the top-k and push
   retained candidates down.

    python -m experiments_shubham.synthetic_e1                  # full size, ~15 min
    python -m experiments_shubham.synthetic_e1 --n-queries 10   # quick, ~2 min

Writes `e1_report.md`, `e1_checks.csv`, `e1_stats.csv` and `e1_recovery.csv` to
`effects/synthetic/`, and the committed record `effects/e1_manifest.json`.
Exits non-zero if any check fails. E1 runs the primary mode only: every
synthetic document is judged, so condensing would remove nothing.

## H7 power simulation (H2: reference-subset size)

| file | what it does |
|---|---|
| `power_h7.py` | synthetic query pool → diagnosis label, G_q and Δ DCG per (K_c, policy, t) → power of the H7 test per reference size |
| `tests/test_power_h7.py` | pool shape and reproducibility, both tests' recommendations, untestable subsets counted as misses |

Two tests of H7, at universe sizes 2000 (synthetic config) and the frozen
universe's median |U_q| (from `data/universe_manifest.json`), reference sizes
from `config.power` plus all 97 queries, recommended at the headline K_c from
`analysis/feasibility_manifest.json`:

- **group** (the proposal's): gate-sensitive vs reranking-sensitive Δ DCG.
  `unconditional_power` counts a subset with fewer than two queries in a group
  as a miss; `shared.diagnosis.required_reference_size` reports power over
  testable subsets only, which overstates it when one group is rare.
- **continuous** (proposed): one-sided Spearman(G_q, Δ DCG) over every
  reference query.

    python -m experiments_shubham.power_h7                                  # ~30 min
    python -m experiments_shubham.power_h7 --pool-size 100 --n-docs 2000    # quick

Writes the pool and power tables to `analysis/power/` and the committed record,
with label shares, headline power and both recommendations, to
`analysis/h7_power_manifest.json`.

## Effect analysis and H7 (Week 2)

| file | what it does |
|---|---|
| `analysis.py` | from effect tables alone: distributions by stratum (all and judged-only), the additivity figure, document labels and gating-failure examples, the query diagnosis, H7 (group and continuous tests) per policy and t, the unjudged share of each world's top-k, bootstrap intervals for the headline numbers (`intervals.csv`), and with `--contrast-with` the paired (exact McNemar) test of whether gate-sensitive queries are more frequent under one Stage-1 system, plus paired intervals for every system difference (`contrast_intervals_vs_<other>.csv`) |
| `tests/test_analysis.py` | diagnosis and budget outcomes match `shared.diagnosis` exactly; unjudged shares match a direct count; interval points equal the reported numbers, paired intervals keep the pairing; end-to-end run on a small grid |

It reads only what `engine.run_grid` writes, so it runs unchanged on the real
effect tables in Week 3. K_c defaults to the feasibility report's headline.
Every step runs once per ranking mode, every output row carries `condensed`,
and the additivity figure is drawn per mode (`additivity.png`,
`additivity_condensed.png`).

Intervals are percentile bootstraps over queries (`shared.metrics.bootstrap_ci`,
`predictors.bootstrap_iterations` and `bootstrap_alpha` from the config, seed
`cfg.seed`), never over documents. All statistics share one seed, so they come
from the same resamples. For the H7 group difference, a resample with fewer
than two queries in a group is dropped, as `h7_test` would refuse it;
`n_draws` below the iteration count says how many, and the interval is then
conditional on the group being testable.

    python -m experiments_shubham.engine --regime weak_retriever
    python -m experiments_shubham.analysis --effects-dir effects/synthetic/weak_retriever --system synthetic

## Sensitivity and the BM25-vs-DPR comparison (Week 4)

| file | what it does |
|---|---|
| `real.py --utility/--k/--k-c` | the sensitivity grids: only the frozen kinds (`utility.sensitivity_kinds`: top-k, RBP) and cutoffs (`cascade.cutoff_sensitivity`: 5, 20), each into its own directory (`effects/real_topk_k10`, `effects/real_dcg_k5`, ...) |
| `compare.py` | per (system, K_c, utility, k, mode): mean utility, G_q, R_q, query labels, reranker-harms, gating failures and mean Δ per (policy, t); then BM25 and DPR side by side, one axis away from the headline at a time |
| `tests/test_compare.py` | only controlled comparisons survive, missing settings are reported rather than invented, headline numbers equal the diagnosis |

    # after the headline run (effects/real), the four sensitivity settings at the headline K_c:
    python -m experiments_shubham.real --system bm25 --utility topk --k-c 100     # and rbp, --k 5, --k 20; and dpr
    python -m experiments_shubham.compare        # -> analysis/compare/compare_one_axis.csv

Query labels use the frozen tau_query, set on the DCG@10 scale, so outside the
headline setting they are descriptive. H7 is pre-registered at the headline and
is not recomputed for the sensitivity settings.

## Qualitative figure: real gating failures (Week 4)

| file | what it does |
|---|---|
| `gating_examples.py` | `select` applies a rule fixed before the full run (worst failure per query, largest losses first, 3 queries, headline K_c, uncondensed) and lists the passage ids to fetch; `render` writes the examples with their text |
| `tests/test_gating_examples.py` | the toy failure is selected; worst per query, largest first; passages Stage 1 never returned are flagged as such |

    python -m experiments_shubham.gating_examples select --effects-dir effects/real --system bm25
    # passage text for analysis/gating_examples/doc_ids.txt (the repo has none), then:
    python -m experiments_shubham.gating_examples render --passages passages.tsv
