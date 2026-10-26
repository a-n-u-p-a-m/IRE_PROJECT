# Roadmap and 4-Week Plan

Counterfactual Attribution and Budget-Aware Diagnosis of Cascaded Information Retrieval
· CS4.406 · Anupam Dwivedi, Shubham Paliwal, Chaitanya K

The single execution reference: who does what, in which week, in what order,
and what each person depends on. The method itself — worlds A–D, the three
effects, strata, metrics, hypotheses H1–H7 — is defined in the proposal
`modified_retrieval_attribution.pdf` (the DPR revision). Read the two together.
Both sit in the repo root. This PDF is the version of record: the proposal's
original source file still says ANCE, so never re-export from it without making
the same DPR changes first.

---

## Team and ownership

| Member | Owns | Produces |
|---|---|---|
| **Anupam** | Project setup and design freeze; BM25 retrieval; the frozen universe; reference subset; outsider allocation; final integration | design log, config freezes, BM25 run, frozen `U_q`, allocation tables, candidate and coverage reports, reproducibility pack |
| **Shubham** | DPR retrieval (existing index); counterfactual engine (worlds A–D, `E_rerank`, `E_gate`, `Δ_t`); synthetic benchmark; power simulation; H7 | DPR run, effect tables, synthetic results, diagnosis labels, H7 test |
| **Chaitanya** | Reranker wrapper and Stage-2 score cache; feature extraction; effect prediction | the cache, feature tables, fidelity tables, cost-fidelity curves |

Everyone fills one shard of the Stage-2 cache in Week 3 and drafts the report
sections for what they ran.

## Decisions already made

- **Dense Stage 1 is DPR** (`castorini/cosdpr-distil`, trained on MS MARCO),
  replacing ANCE. Its 89-shard index already exists on Shubham's Google Drive
  and is verified against Pyserini (TREC-DL19 nDCG@10 0.706 vs published 0.725;
  R@1000 0.822 vs 0.820). Stage-1 systems and universe sources are BM25 and DPR;
  DPR is also the alternative retriever for outsider selection.
- **Stage 2** is the pointwise MiniLM cross-encoder; **data** is TREC-DL 2019+2020
  on MS MARCO passages; **gain map** gives grades 0 and 1 zero gain.

**Still to settle in the Week-1 meeting** (each gets a design-log entry):
headline `K_c` (provisionally 1000, confirmed by the judged-non-survivor counts
in Week 2) · `tau_query` (gates H7 groups; fix it on synthetic, then never tune
it) · `tau_document` · reference-subset size and sampling strategy · whether the
headline NDCG uses official gains · whether B = `K_c` · SPLADE and hashing in or
out (decided at the end of Week 2) · listwise reranker stays out of scope.

## How nobody waits

- **Contracts first.** The table schemas in `shared/schema.py` and the frozen
  `config/base.yaml` are agreed in Week 1; everyone codes against them.
- **Synthetic substitutes.** `shared/synthetic.py` and `shared/fixtures.py`
  supply complete fake cascades, so every stream can build and test its whole
  pipeline before real data exists.
- **No file hand-offs.** Every table is a deterministic function of (universe,
  cache, config). If you need someone's table, run their command. The config
  fingerprint is stamped on every output and results from different freezes
  are never combined.

The only real hand-offs:

| | What | From → to | Needed | Ready by |
|---|---|---|---|---|
| **H1** | DPR run, top-5,000, all 97 queries | Shubham → Anupam | Week 2 universe build | end of Week 1 |
| **H2** | Reference-subset size from the power simulation | Shubham → Anupam | Week 2 reference subset | end of Week 1 |
| **H3** | Frozen universe, reference subset, allocation tables | Anupam → Chaitanya, Shubham | Week 3 cache fill | middle of Week 2 |
| **H4** | Reranker wrapper, shardable cache filler, shard plan | Chaitanya → everyone | Week 3, day 1 | end of Week 2 |
| **H5** | Stage-2 cache (worlds A/B first, then C/D) | Chaitanya → Shubham | Week 3 effects | during Week 3 |

---

## Week 1 — Contracts, both Stage-1 runs, everything working on synthetic

**Order**
1. **Day 1, all three (~1 hour), led by Anupam:** each reads the proposal and
   names what is most likely wrong; confirm ownership; hand-compute the toy
   fixture's effects together; settle the open decisions above.
2. **Anupam** records freeze v1 and creates the repo (`shared/` reviewed-only,
   one folder and branch per member, `docs/design_log.md`). Nobody waits:
   `shared/` already runs.
3. Everything below runs in parallel. Each person confirms a green test suite
   and a three-line smoke script on their own machine.

| Anupam | Shubham | Chaitanya |
|---|---|---|
| Freeze v1, repo, design log | **First (H1):** port the DPR retriever into the repo so it writes the `run` table (`system=dpr`, checkpoint recorded); run top-5,000 for all 97 queries on Colab; validate the table; share the Drive index read-only | Reranker wrapper with lazy loading, plus a listwise stub that refuses to run; passage-text fetcher that **raises** on a missing passage |
| Query and qrels loaders; train/val/test split **by query** | World runner (A/B/C with isolation checks) and budget runner (t-grid, baseline check), both on synthetic, strict mode by default | Resumable, shardable cache filler with merge; throughput script at the frozen length, batch size and precision |
| Retriever interface, BM25 and precomputed-run wrappers, a Stage-1 runner that refuses depth below M and records the checkpoint | Synthetic runner over all six regimes plus gate and budget stress; **exact-recovery report (E1)** | Hashing go/no-go criteria written down before any hashing code |
| **BM25 run, top-5,000, all 97 queries** (Pyserini prebuilt index, CPU) | **By end of week (H2):** H7 power simulation → reference-subset size. If no size reaches target power, say so now | Conventional features from the factual run only, with extra Stage-2 calls counted as cost |
| Universe builder (refuses to overwrite a frozen universe) on mock runs | | |

**Checkpoint:** freeze v1 recorded · BM25 and DPR runs validated · E1 done ·
reference size proposed · cache filler works on synthetic.

---

## Week 2 — Feasibility gate and the frozen universe

**Order**
1. **Anupam first, target mid-week — Week 3 depends on it (H3):** validate both
   runs (contiguous ranks, depth, no duplicates) → coverage audit of `U/C/D`,
   judged non-survivors per query and `K_c`, BM25-vs-DPR candidate overlap
   (if the median Jaccard exceeds 0.9 the cross-paradigm comparison needs
   rethinking) → build and freeze the universe, check every candidate set is
   inside it → sample the reference subset by the frozen rule, never by hand
   → alternative-rank map → allocation tables for every (`K_c`, t, policy,
   seed) → publish.
2. Shubham and Chaitanya continue on synthetic; Chaitanya starts the cache plan
   as soon as H3 lands.
3. **End of week, all three (~1 hour):** compute the cache bill; decide the
   headline `K_c`, reference-subset size, SPLADE and hashing scope. Anupam
   records freeze v2; each person re-runs their synthetic results under it.

| Anupam | Shubham | Chaitanya |
|---|---|---|
| The universe chain above, plus Stage-1 scores over `U_q` (unretrieved documents get a floor below the observed minimum, recorded in the manifest) | Additivity demonstration figure; document labels and synthetic gating-failure examples | Passage-length check, **real throughput**, and the **shuffle-invariance test on the real reranker** (the whole `E_gate` derivation depends on it) |
| Freeze v2 after the decision meeting | Seed the cache with the existing DPR → MiniLM scores for DPR's top-1,000, after re-scoring a 1,000-pair sample to confirm they match | Predictors (fit on training queries only; three targets kept separate), stratified metrics with all three baselines, cost-fidelity harness that converges to the exact reference at full budget, synthetic fidelity table |
| | Write the analysis scripts (distributions, diagnosis, H7) against synthetic effect tables | After H3: real passage fetcher, exact required document set and call count, **shard plan (H4)** |

**Checkpoint:** frozen universe, reference subset and allocations published ·
freeze v2 · scope decided · shard plan ready.

---

## Week 3 — Fill the cache, compute real effects

The only GPU-heavy week: about 0.6–0.9 million query-passage pairs in total,
under an hour on one T4 at the measured ~400 pairs/s, or 15–20 minutes each
when split three ways. Part of DPR's share is already seeded.

**Order**
1. **Chaitanya first, day 1:** shuffle-invariance gate on real passages (if it
   fails, stop and tell everyone); split the required set into three shards,
   **worlds A/B first**, then world C (reference subset) and world D
   (budget outsiders).
2. **All three fill their shard in parallel**; an interrupted run resumes,
   never restarts. Chaitanya merges each shard as it lands and checks
   completeness for the reference subset and for all candidate sets.
3. Then in parallel:

| Anupam | Shubham | Chaitanya |
|---|---|---|
| Fill shard 1 | Fill shard 2 | Fill shard 3; merge, completeness checks, export with call count and wall time |
| No cache needed: final candidate report at the locked `K_c`; allocation audit (every allocation totals B, strata partition `U_q`); seed-variation report | **When A/B is merged:** `E_rerank` for BM25 and DPR at every `K_c`; invariant checks (a failure is a bug, not a finding) | Real features, recording generation time and memory |
| After the cache completes: what each outsider policy actually promotes; boundary-probe false-negative rate (re-running Shubham's commands) | **When C/D is merged:** `E_gate` on the reference subset; `Δ_t` over the full t × policy × seed grid | First predictor fits on the real effect tables (re-running Shubham's commands) |

**Checkpoint:** cache complete and verified · `E_rerank`, `E_gate`, `Δ_t` for
BM25 and DPR · all invariants pass.

---

## Week 4 — Analysis, H7, report

All parallel; inputs are commands, not files. **Results freeze on day 5** so
the last two days go to figures, packaging and the report.

| Anupam | Shubham | Chaitanya |
|---|---|---|
| Coverage caveats from the Week-2 audit; provenance check that every manifest carries the same fingerprint | Effect distributions by stratum with stratum sizes; utility (top-k, RBP) and cutoff (k = 5, 20) sensitivity; BM25-vs-DPR comparison, one axis changed at a time | Predictors on held-out queries for each target; stratified fidelity tables with all three baselines; cost-fidelity curves for every budget and policy |
| Figures: effect distributions, cost-fidelity, utility vs t, H7 comparison, gating-failure example | Diagnosis labels (reporting reranker-harms separately) → **H7 test** with group sizes beside every p-value (Anupam reviews) | Hashing only if in scope, otherwise one paragraph on why not; full compute accounting; check that predictors and references agree on the largest effects |
| Reproducibility pack: configs, manifests, seeds, reference subset, cached scores, versions, one command per figure | Qualitative figure: two or three real gating failures with passage text | |
| Report: setup, feasibility, threats; merge the final report | Report: effects and H7 sections | Report: prediction and cost sections |

**Checkpoint:** every number traces to a manifest with one fingerprint ·
figures done · report drafted.

---

## Only if the Week-2 decision includes them

- **SPLADE-v3** as a third Stage-1 system — Anupam, before the universe is
  frozen (adding it later means regenerating the universe and every effect).
- **Hashing features** — Chaitanya, Week 4, only if the go/no-go criteria pass.
- **Listwise reranker** — out of scope.

## Cache cost

```
pairs = n_queries × K_c                         (worlds A, B)
      + n_reference × (|U_q| − K_c)             (world C)
      + n_queries × distinct outsiders over the t × policy × seed grid   (world D)
hours = pairs / measured pairs-per-second / 3600
```

If this is too large, the lever is the reference-subset size, set by the power
simulation rather than by guesswork.

## Working rules

- A 30-minute sync at the start of each week to confirm that week's hand-offs.
- Join records by IDs, never row order; keep features and targets in separate
  tables; never use qrels or counterfactual outputs as predictor inputs.
- Any change to the universe, `K_c`, B, t grid, qrels policy, utility,
  checkpoints or hashing method gets a design-log entry and a new config version.
- Run the fixture smoke test before every push.
- If an input is late, continue on synthetic or mock data and mark the result
  provisional.
