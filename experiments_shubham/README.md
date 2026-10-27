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
