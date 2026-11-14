# Reproducing every number and figure

Everything below runs on a laptop CPU except the one Stage-2 cache fill, which
needs a GPU and is already done. The whole project is frozen under config
fingerprint **`46d0807d8ed3`** (`make freeze`). Every run record in the repo
carries it: 25 of 25 committed manifests at the time of writing.

## What is in git, and what is not

| in git | not in git (regenerable, or too large) |
|---|---|
| configs, code, tests, the frozen query split (`data/query_split.json`) | the universe and Stage-1 tables (`data/*.parquet`) |
| both Stage-1 runs (`runs/bm25.parquet`, `runs/dpr.parquet`) and their manifests | the Stage-2 cache (`cache/stage2.sqlite`, 173 MB) |
| every run record (`*manifest*.json`): versions, seeds, inputs, digests | effect tables (`effects/real*/`, ~1.2 GB in all) |
| the figures and the exact numbers they plot (`figures/`, `figures/data/`) | result CSVs (`analysis/*/`); their digests are in `docs/design_log.md` |
| `docs/design_log.md`: every decision and every headline number | MS MARCO passage text |

## Environment

- Python 3.13 with `pip install -r requirements.txt`. The versions each run
  used are recorded in its manifest (`python`, `numpy`, `torch`, `platform`).
- CPU steps were run on an Apple M1 with 16 GB. Peak memory is about 3.5 GB,
  in `real.py`.
- The cache fill ran on Tesla T4 GPUs (Colab and Kaggle) with torch 2.11–2.14
  and CUDA 13. It needs `torch` and `sentence-transformers` on top of the
  requirements.

**Seeds.**
- `seed: 42` everywhere a seed is taken: the query split, permutation tests,
  bootstraps and model fits.
- Budget seeds 42, 43 and 44 for the random outsider policy.
- No step reads a clock or an unseeded RNG.

## The pipeline, in order

Each step names what it writes and how to check it. The times are for an M1.

**1. Queries, qrels, split, universe, allocations** (no GPU, about 5 min)

```bash
python -m experiments_shubham.fetch_trec_dl
python -m experiments_anupam.data_prep --queries data/queries.tsv --qrels-txt data/qrels.txt
python -m experiments_anupam.build_universe --qrels data/qrels.parquet
python -m experiments_anupam.stage1_over_universe
python -m experiments_anupam.allocations
```

Checks:
- `data/universe_manifest.json`: 97 queries, 872,455 pairs, fingerprint
  `46d0807d8ed3`.
- `analysis/allocation_manifest.json`: `all_allocations_valid: true`, 11,640
  allocations.
- `data/query_split.json` is regenerated identically: 49 / 19 / 29, seed 42.

**2. The Stage-2 cache** (GPU, done once)

The merged cache is `cache/stage2.sqlite`: 872,455 rows, model
`cross-encoder/ms-marco-MiniLM-L-12-v2`, float32. Its fingerprint and the three
shard fingerprints it contains are:

| | rows | digest |
|---|---|---|
| merged | 872,455 | `b7669b8dde3b328c` |
| shard anupam | 295,531 | `9f2cfe600f0e49aa` |
| shard shubham | 287,058 | `def33b7a41b4f1d0` |
| shard chaitanya | 289,866 | `ab09ad44960cdaf5` |

There are two ways to get it:
- **Download** the team's Drive copy and check it against those digests.
  `experiments_shubham.real.cache_digest` computes them.
- **Rebuild** it: fill the three shards with
  `experiments_shubham/cache_fill_colab.ipynb` (set `OWNER`) or
  `experiments_anupam/cache_fill_kaggle.ipynb`, then merge:

  ```bash
  python -m experiments_chaitanya.merge_cache
  ```

  This takes 52–57 T4-minutes in all, at 257–282 pairs/s. Scores are
  bit-identical across sessions, and `shared/canary.py` checks a new
  environment against reference digest `e1121c9d61e14fd5` before any fill.

**3. Effects** (CPU, about 10.5 min per system; 3 min per sensitivity setting)

```bash
python -m experiments_shubham.real --system bm25
python -m experiments_shubham.real --system dpr
for s in "--utility topk" "--utility rbp" "--k 5" "--k 20"; do
  python -m experiments_shubham.real --system bm25 $s --k-c 100
  python -m experiments_shubham.real --system dpr  $s --k-c 100
done
```

Check: each `effects/real*/<system>_manifest.json` has `stage2_digest:
b7669b8dde3b328c` and `allocation_check.ok: true`, meaning world D equals the
published allocations at all 4,365 per system.

**4. Analyses** (CPU, about 20 min in all)

```bash
python -m experiments_shubham.analysis --effects-dir effects/real --system bm25 --contrast-with dpr
python -m experiments_shubham.analysis --effects-dir effects/real --system dpr  --contrast-with bm25
python -m experiments_shubham.compare
python -m experiments_anupam.feasibility_report
python -m experiments_shubham.outsiders
python -m experiments_shubham.predict
python -m experiments_shubham.gating_examples select --effects-dir effects/real --system bm25
```

Every one of these is deterministic. Re-running gives byte-identical CSVs, and
`docs/design_log.md` lists their sha256 digests (2026-11-13 and 2026-11-14
entries), so a re-run is checked rather than trusted:

```bash
shasum -a 256 analysis/real_bm25_kc100/h7.csv   # compare the first 16 hex with the log
```

**5. Passage text for the qualitative figure** (one-off, any machine with internet)

The repository holds no passage text. Fetch the six passages listed in
`analysis/gating_examples/doc_ids.txt` by streaming the MS MARCO collection:

```python
# Colab, CPU runtime, about 3-5 min
!pip -q install ftfy
import csv, subprocess, ftfy
IDS = set(open("doc_ids.txt").read().split())
URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
proc = subprocess.Popen(f"curl -sL {URL} | tar -xzOf - collection.tsv", shell=True,
                        stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
found = {}
for line in proc.stdout:
    pid, _, text = line.rstrip("\n").partition("\t")
    if pid in IDS:
        found[pid] = ftfy.fix_text(text)
        if len(found) == len(IDS):
            break
proc.kill()
with open("passages.tsv", "w", newline="", encoding="utf-8") as fh:
    csv.writer(fh, delimiter="\t").writerows(sorted(found.items()))
```

Then put the file in place and render:

```bash
cp passages.tsv analysis/gating_examples/passages.tsv
python -m experiments_shubham.gating_examples render --passages analysis/gating_examples/passages.tsv
```

**6. Figures** (CPU, about 30 s; one command each)

| figure | command |
|---|---|
| `figures/effect_distributions.png` | `python -m experiments_shubham.figures --only effect_distributions` |
| `figures/utility_vs_t.png` | `python -m experiments_shubham.figures --only utility_vs_t` |
| `figures/h7.png` | `python -m experiments_shubham.figures --only h7` |
| `figures/cost_fidelity.png` | `python -m experiments_shubham.figures --only cost_fidelity` |
| `figures/gating_examples.png` | `python -m experiments_shubham.figures --only gating_examples` |
| additivity, per system and mode | written by step 4 to `analysis/real_<system>_kc100/additivity*.png` |

Each figure's plotted numbers are in `figures/data/<name>.csv`, and
`figures/figures_manifest.json` records the commit that drew them.

**7. Tests**

```bash
make test-all     # 550+ tests, about 90 s, no GPU
make lint
```

## Where each reported number comes from

| result | record | design-log entry |
|---|---|---|
| universe, feasibility, Recall@K_c | `data/universe_manifest.json`, `analysis/feasibility_manifest.json` | 2026-10-28, 2026-11-13 (recall) |
| allocations and their audit | `analysis/allocation_manifest.json` | 2026-11-14 (outsiders) |
| effects, invariants, world-D check | `effects/real*/*_manifest.json` | 2026-11-13 |
| labels, H7, contrast, intervals | `analysis/real_<system>_kc100/analysis_manifest.json` | 2026-11-13, 2026-11-13 (later) |
| sensitivity, BM25 vs DPR one axis at a time | `analysis/compare/compare_manifest.json` | 2026-11-13 |
| what policies promote, false negatives, seed variation | `analysis/outsiders/outsiders_manifest.json` | 2026-11-14 (outsiders) |
| predictors and probes | `analysis/prediction/prediction_manifest.json` | 2026-11-14 |
| E1 engine validation | `effects/e1_manifest.json` | 2026-10-27 |
| H2 power | `analysis/h7_power_manifest.json` | 2026-10-29 |
