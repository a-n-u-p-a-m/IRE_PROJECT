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
| `tests/test_engine.py` | toy-fixture numbers, seed handling, refusal cases, every synthetic regime in strict mode |

Every (query, K_c) passes `verify_isolation`, every t-grid passes
`verify_t_baseline`, and every table passes the schema's sign invariants before
it is written. Only the random outsider policy repeats over seeds; boundary and
alternative run once with `seed=NA`.

    python -m experiments_shubham.engine --regime weak_retriever            # ~2 min, ~2 GB
    python -m experiments_shubham.engine --regime strong_stages --n-queries 10

Writes one effect table per K_c (stage effects) and per (K_c, policy) (budget
effects), a summary CSV and a manifest to `effects/synthetic/<regime>/`.

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
Exits non-zero if any check fails.

## H7 power simulation (H2: reference-subset size)

| file | what it does |
|---|---|
| `power_h7.py` | synthetic query pool → diagnosis label, G_q and Δ DCG per (K_c, policy, t) → power of the H7 test per reference size |
| `tests/test_power_h7.py` | pool shape and reproducibility, both tests' recommendations, untestable subsets counted as misses |

Two tests of H7, at universe sizes 2000 (synthetic config) and 8000 (realistic
|U_q|), reference sizes from `config.power` plus all 97 queries:

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
