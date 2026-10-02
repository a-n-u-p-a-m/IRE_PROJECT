# Roadmap: Intervention-Based Benchmark for Stage-Level Attribution in Multi-Stage Retrieval Pipelines

**Team:** Anupam Dwivedi, Shubham Paliwal, Chaitanya K  
**Course:** CS4.406 — Information Retrieval & Extraction  
**Domain:** Explainable AI + Information Retrieval

---

## Part 1: Shared Design Choices

Every member owns their own retrievers, methods, and experiments end-to-end. For results to be comparable and mergeable, the following choices must be identical across all three members. Agree on these before writing code.

---

### 1.1 Data Contract

All pipeline outputs, attribution results, and intervention results must follow a single schema so anyone can run anyone else's methods on anyone else's pipelines by swapping files.

**Score matrix (one row per query × document):**

| Column | Type | Description |
|---|---|---|
| query_id | str | TREC-DL query identifier |
| doc_id | str | MS MARCO passage ID |
| stage_1_rank | int | Document rank after Stage 1 (1-indexed) |
| stage_1_score | float | Raw Stage 1 score |
| in_candidate_set | bool | Did document survive Stage 1's top-K cut? |
| stage_2_rank | int | Document rank after Stage 2 (1-indexed, within candidates only) |
| stage_2_score | float | Raw Stage 2 score |
| final_rank | int | Final output rank |
| relevance_label | int | TREC qrel label (0/1/2/3) |

**Attribution result (one row per query × document × method):**

| Column | Type | Description |
|---|---|---|
| query_id | str | |
| doc_id | str | |
| method_name | str | One of: rank_delta, inclusion_ordering, stage_shapley, reranker_lime |
| stage_1_attribution | float | Attributed contribution of Stage 1 (normalized 0–1) |
| stage_2_attribution | float | Attributed contribution of Stage 2 (normalized 0–1) |
| gating_attribution | float | (inclusion-ordering only) Binary gating component |
| ordering_attribution | float | (inclusion-ordering only) Continuous ordering component |

**Intervention result (one row per query × document × intervention type):**

| Column | Type | Description |
|---|---|---|
| query_id | str | |
| doc_id | str | |
| intervention_type | str | One of: remove_stage2, randomize_stage1_scores, remove_gating |
| rank_before | int | Rank in the full pipeline |
| rank_after | int | Rank after intervention |
| displacement | int | rank_after − rank_before |

**File format:** Parquet. Naming convention:

```
results/
  scores/
    bm25_minilm.parquet
    dpr_minilm.parquet
    colbert_minilm.parquet
    splade_minilm.parquet
  attributions/
    bm25_minilm_rank_delta.parquet
    bm25_minilm_stage_shapley.parquet
    ...
  interventions/
    bm25_minilm_remove_stage2.parquet
    bm25_minilm_randomize_stage1.parquet
    ...
  synthetic/
    linear_mix_alpha_0.2.parquet
    gated_topk_500.parquet
    adversarial.parquet
```

---

### 1.2 Corpus and Queries

| Decision | Choice | Rationale |
|---|---|---|
| Corpus | MS MARCO Passage v1 (8.8M passages) | Standard benchmark, most retriever implementations support it natively |
| Evaluation queries | TREC-DL 2019 (43 queries) + TREC-DL 2020 (54 queries) | Graded NIST relevance judgments, same corpus |
| Development queries | MS MARCO dev small (6,980 queries, sparse binary labels) | For sanity checks and larger-scale attribution analysis |
| Extension dataset | One BEIR subset (e.g., SciFact or TREC-COVID) | Cross-domain generalization check |

Everyone indexes the same corpus. Everyone evaluates on the same query set. No exceptions.

---

### 1.3 Candidate Set Size (Stage 1 Top-K)

| Decision | Choice | Rationale |
|---|---|---|
| Default top-K | 1000 | Standard in the literature |
| Ablation values | 100, 500, 1000 | For E6 (candidate set size experiment) |

Every retriever must produce top-1000 candidates with scores. The reranker receives all 1000. If a retriever returns fewer than 1000 for a query, pad with a sentinel score and mark in_candidate_set=false.

---

### 1.4 Reranker Configuration

| Decision | Choice | Rationale |
|---|---|---|
| Primary reranker | cross-encoder/ms-marco-MiniLM-L-12-v2 | Fast, well-understood, widely benchmarked |
| Reranker input | (query, passage) pair, truncated to 512 tokens | Model's max sequence length |
| Reranker batch size | 64 (adjust per GPU memory) | |
| Extension reranker | castorini/monot5-base-msmarco | For E7 (reranker strength) |
| No-reranker baseline | Stage 1 scores passed through as final scores | Control condition |

The same reranker, same weights, same truncation, same batch size across all four retrievers. If anyone changes the reranker differently, the cross-paradigm comparison is invalid.

---

### 1.5 Retriever-Specific Decisions

Each retriever has its own parameters (BM25's k1/b, DPR's embedding dimension, ColBERT's nprobe, SPLADE's expansion budget). These can differ because they're retriever-specific, but each person must:

1. Use published default parameters wherever they exist
2. Document exact parameters in a config file committed to the repo
3. Verify nDCG@10 on TREC-DL matches published baselines (within ±0.02) before running any experiments

Published baselines for reference (approximate nDCG@10 on TREC-DL 2019):

| Retriever | Expected nDCG@10 (approx.) | Source |
|---|---|---|
| BM25 (Pyserini defaults) | 0.50–0.52 | Pyserini docs |
| DPR (cosDPR-distil: DPR architecture trained on MS MARCO) | 0.725 (DL20: 0.703) | Pyserini 2CR, Faiss flat |
| ColBERTv2 | 0.68–0.71 | Santhanam et al. |
| SPLADE v3 | 0.69–0.72 | Formal et al. |

If your numbers are wildly off, debug before proceeding — attribution on a broken pipeline is meaningless.

---

### 1.6 Score Normalization

Raw scores from different retrievers are on different scales (BM25 unbounded, DPR cosine in [−1, 1], cross-encoder logits).

| Decision | Choice |
|---|---|
| Score storage | Raw, unnormalized scores in the score matrix |
| Rank storage | 1-indexed integer ranks (rank 1 = best) |
| Attribution normalization | Methods operate on ranks, not scores. Stage contribution expressed as fraction in [0, 1] where stage_1_attribution + stage_2_attribution = 1.0 |

Exception: the synthetic linear-mix setting (Setting 1) uses scores because the planted ground truth α is defined in score space. For that setting only, methods may optionally operate on scores.

---

### 1.7 Attribution Method Specifications

Each method must follow the same API:

```python
def compute_attribution(score_matrix: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Input:  score matrix in agreed schema
    Output: attribution result in agreed schema (only documents in final top-K)
    """
```

**Method-specific decisions (must be identical across whoever runs them):**

| Method | Decision Point | Agreed Choice |
|---|---|---|
| Rank-delta | Normalization | Δ = (rank_1 − rank_2) / rank_1. Positive = Stage 2 promoted. Rescale so stage_1_attr + stage_2_attr = 1.0 |
| Inclusion-ordering | Gating boundary | top-K candidate set boundary (K=1000). Documents near boundary (rank 900–1000): high gating_attribution |
| Stage-Shapley | "Stage 2 alone" semantics | v({}) = 0, v({S2}) = 0, v({S1}) = Stage 1 ranking only, v({S1,S2}) = full pipeline. Run sensitivity check with alternative (v({S2}) = Stage 2 on full corpus) |
| Reranker-only (Rank-LIME/SHAP) | Perturbation scheme | Rank-LIME defaults: perturb input features to the reranker, observe score changes. Black-box access to cross-encoder |

---

### 1.8 Intervention Specifications

Interventions generate the empirical ground truth. Must be identical across pipelines.

| Intervention | Exact Procedure |
|---|---|
| Remove Stage 2 | Final ranking = Stage 1 ranking. Displacement = stage_1_rank − final_rank_full_pipeline (i.e. rank_after − rank_before; positive = document moved down) |
| Randomize Stage 1 scores | Keep same candidate set (top-1000). Replace stage_1_scores with uniform random. Re-run Stage 2. Compute displacement |
| Remove gating | Expand Stage 2 input to top-5000 (or full corpus if feasible). Re-run Stage 2. Documents in top-K only with gating → Stage 1's gating was critical |

Seed for randomization: fixed seed = 42. Run with 3 seeds (42, 123, 456), report mean displacement ± std.

---

### 1.9 Evaluation Metrics

One shared `evaluation.py` module. Identical implementations across all members.

**Synthetic evaluation:**

| Metric | Definition |
|---|---|
| Attribution error | mean(\|predicted_stage_1_attribution − true_α\|) across all documents in top-K |
| Rank correlation | Spearman ρ between predicted per-document stage_1_attribution and true per-document contribution |
| Calibration error | Mean absolute error of predicted attribution, binned by true α. Reported as a calibration curve |

**Real-data evaluation:**

| Metric | Definition |
|---|---|
| Displacement correlation | Spearman ρ between predicted stage importance and measured intervention displacement |
| Comprehensiveness | Fraction of documents where removing the attributed-as-important stage causes displacement > median displacement |
| Sufficiency | Fraction of documents where keeping only the attributed-as-important stage preserves the document in top-K |
| Recovery error | mean(\|predicted_attribution − normalized_displacement\|) |

**All metrics with bootstrap 95% CI.** Resample queries with replacement, 1000 iterations, report 2.5th and 97.5th percentiles.

---

### 1.10 Codebase Structure

```
repo/
├── README.md
├── Makefile
├── requirements.txt
├── config/
│   ├── bm25.yaml
│   ├── dpr.yaml
│   ├── colbert.yaml
│   └── splade.yaml
├── shared/
│   ├── schema.py              # Data contract definitions
│   ├── reranker.py            # MiniLM cross-encoder wrapper
│   ├── logger.py              # Score/rank logger
│   ├── evaluation.py          # All metrics + bootstrap CI
│   └── utils.py
├── retrievers/
│   ├── bm25.py                # Anupam
│   ├── splade.py              # Anupam
│   ├── dpr.py                 # Shubham
│   └── colbert.py             # Shubham
├── attribution/
│   ├── rank_delta.py          # Anupam
│   ├── inclusion_ordering.py  # Anupam
│   ├── stage_shapley.py       # Shubham
│   └── reranker_lime.py       # Shubham
├── synthetic/
│   ├── linear_mix.py          # Anupam
│   ├── gated_pipeline.py      # Anupam
│   └── adversarial.py         # Anupam
├── interventions/
│   ├── remove_stage2.py
│   ├── randomize_stage1.py
│   └── remove_gating.py
├── analysis/
│   ├── cross_paradigm.py      # Chaitanya
│   ├── query_type_slicing.py  # Chaitanya
│   ├── method_comparison.py   # Chaitanya
│   └── visualization.py       # Chaitanya
├── debugger/
│   └── app.py                 # Chaitanya
├── experiments/
│   ├── run_pipeline.py
│   ├── run_attribution.py
│   ├── run_intervention.py
│   └── run_evaluation.py
├── results/                   # gitignored
├── tests/
│   ├── test_schema_compliance.py
│   ├── test_attribution_sanity.py
│   └── test_baseline_ndcg.py
└── query_annotations/
    └── trec_dl_query_types.csv  # Chaitanya
```

**Git rules:** no large files (indexes, model weights, result parquets). Code must be importable: `from attribution.rank_delta import compute_attribution`.

---

### 1.11 Hardware and Environment

| Decision | Choice |
|---|---|
| Python version | 3.10+ |
| Package manager | pip with requirements.txt (pinned versions) |
| Key dependencies | pyserini, faiss-cpu, sentence-transformers, transformers, pandas, pyarrow, scipy, scikit-learn, streamlit |
| GPU | Free-tier (Colab T4 / Kaggle P100) for dense retriever indexing and cross-encoder reranking |
| Seed | Global random seed = 42 for all non-intervention randomness. Intervention seeds: 42, 123, 456 |

---

## Part 2: Parallel Roadmap (Vertical Slicing)

Each member owns their own retrievers, their own methods, and their own experiments end-to-end. Nobody waits.

---

### Day-One Setup (all three, one meeting)

1. Agree on the data contract (Section 1.1)
2. Anupam writes `schema.py`, `reranker.py`, `logger.py` (~150 lines total, shared components)
3. Shubham writes `evaluation.py` (~200 lines, shared metrics module)
4. Chaitanya creates `trec_dl_query_types.csv` (manual annotation, no code dependency)
5. Push all shared components. Everyone pulls and codes against them from day one.

---

### Member 1: Anupam — BM25 + SPLADE + Synthetic + Two Methods

**Vertical A: BM25 pipeline (weeks 1–4)**
- Download MS MARCO, build BM25 index via Pyserini
- Wire BM25 + reranker + score logger → BM25 score matrices on TREC-DL
- Verify nDCG@10 against published baselines
- Implement interventions for BM25 pipeline (remove_stage2, randomize_stage1, remove_gating with 3 seeds)
- Push BM25 score matrices + intervention results to shared repo

**Vertical B: Synthetic ground truth (weeks 1–3, concurrent with A)**
- No indexing needed — works on generated score distributions
- Setting 1 (linear mix): generate synthetic scores, combine as α·s1 + (1-α)·s2 for α ∈ {0.1, 0.2, ..., 0.9}
- Setting 2 (gated pipeline): hard top-K filter on score_1, rerank survivors with score_2
- Setting 3 (adversarial): extreme Stage 1 / Stage 2 disagreement cases
- **Push synthetic parquets to shared repo by end of week 2 — Shubham and Chaitanya can use immediately**

**Vertical C: Two attribution methods (weeks 1–4, concurrent)**
- Implement rank-delta. Unit test on hand-crafted edge cases
- Implement inclusion-vs-ordering decomposition. Unit test
- Run both on own synthetic data + BM25 real pipeline
- **Push methods to shared repo as importable modules**

**Vertical D: SPLADE pipeline (weeks 5–7)**
- Build SPLADE index (naver/splade-v3)
- Wire SPLADE + reranker + score logger, verify baselines
- Run interventions on SPLADE
- Run all four methods (own two + Shubham's two from repo) on SPLADE
- Push all results

**Vertical E: Extension (weeks 8–9)**
- MonoT5 reranker wrapper
- Re-run BM25→MonoT5 and SPLADE→MonoT5 with interventions and all methods (E7)

**Week 10:** Code cleanup, contribute to report (pipeline infra section, synthetic GT section)

---

### Member 2: Shubham — DPR + ColBERT + Two Methods + Method Comparison

**Vertical A: DPR pipeline (weeks 1–4)**
- Download DPR weights, build FAISS index over MS MARCO
- Wire DPR + reranker + score logger → DPR score matrices on TREC-DL
- Verify nDCG@10 against published baselines
- Implement interventions for DPR pipeline
- Push DPR score matrices + intervention results

**Vertical B: Two attribution methods (weeks 1–4, concurrent)**
- Implement stage-Shapley: enumerate stage subsets (4 for 2-stage), compute marginal contributions. Handle "Stage 2 alone" semantics. Unit test
- Implement reranker-only feature attribution: wrap Rank-LIME or RankSHAP for the reranker in isolation. Unit test
- Run both on Anupam's synthetic data (available from week 2) + own DPR real pipeline
- **Push methods to shared repo as importable modules**

**Vertical C: ColBERT pipeline (weeks 3–6, concurrent — ColBERT indexing is slower)**
- Set up ColBERTv2, build index
- Wire ColBERT + reranker + score logger, verify baselines
- Run interventions on ColBERT
- Run all four methods on ColBERT
- Push all results

**Vertical D: Synthetic evaluation + method comparison (weeks 5–9)**
- E1: Run all four methods on all three synthetic settings × all α values. Compute attribution error, rank correlation, calibration
- E2/E3: Evaluate all four methods on all real pipelines against intervention ground truth (own DPR + ColBERT, Anupam's BM25 + SPLADE as they arrive)
- E4: Head-to-head method comparison tables. Which method recovers ground truth best? On which paradigm? On which synthetic setting?
- Failure mode analysis: where does rank-delta break? Is reranker-only surprisingly good?
- Sensitivity analysis: test Shapley with alternative v({S2}) definition

**Vertical E: Literature review (weeks 1–3, concurrent)**
- Rank-LIME, RankingSHAP: confirm no multi-stage handling
- ERASER benchmark: comprehensiveness/sufficiency adaptation
- Deterministic Horizon: impossibility bound implications
- Câmara & Hauff: probing methodology

**Week 10:** Code cleanup, contribute to report (attribution methods section, method comparison section)

---

### Member 3: Chaitanya — Hybrid Retriever + Analysis + Debugger + Report

**Vertical A: Query categorization (weeks 1–2, no dependencies)**
- Annotate all 97 TREC-DL queries: entity/factoid, semantic/conceptual, rare-term, ambiguous, long natural language
- Store as `query_annotations/trec_dl_query_types.csv`
- Build query-type slicing module: given any results parquet + annotations, produce per-type breakdowns

**Vertical B: Visualization pipeline (weeks 1–4, works on mock data)**
- Build all chart/table generators against the agreed schema:
  - Attribution distribution histograms
  - Calibration plots (predicted vs. true)
  - Method comparison tables with CI error bars
  - Cross-paradigm heatmaps
  - Per-query-type attribution breakdowns
- Test everything on mock data. Swap in real results as they arrive

**Vertical C: Interactive debugger skeleton (weeks 2–5, works on mock data)**
- Build Streamlit app:
  - Query input → per-stage ranking display
  - Per-document attribution bar chart
  - Stage toggle (remove/add stages)
  - Paradigm comparison panel (side-by-side across retrievers)
- Wire to mock backend initially

**Vertical D: Own retriever pipeline (weeks 3–6)**
- Build hybrid retriever (BM25 + DPR fusion via reciprocal rank fusion) or take a remaining retriever
- Wire hybrid→MiniLM pipeline, verify baseline
- Run interventions on own pipeline
- Run all four methods on own pipeline
- Push all results

**Vertical E: Cross-paradigm + query-type analysis (weeks 6–9)**
- E4: For each paradigm, compute attribution distributions. Compare across all retrievers
- Paradigm agreement: which documents are shared vs. unique across paradigms' top-K? Which stage explains?
- E5: Slice all results by query type. Test hypotheses about stage dominance per query type
- E6: Vary candidate set size (100, 500, 1000), measure attribution shifts
- E8 (extension): Run one BEIR subset for cross-domain check
- Wire debugger to real pipeline backends

**Vertical F: Literature review (weeks 1–3, concurrent)**
- Anand et al. (2022) survey, SIGIR 2023 tutorial
- ExDocS, RAGXplain: differentiate from existing tools
- Saha, Majumdar & Mitra (ACM Computing Surveys, 2026)
- Agarwal et al. (BioNLP 2026)

**Weeks 9–10: Report + presentation (lead role)**
- Write final report structured around RQ1–RQ4
- All figures from own visualization pipeline
- Prepare presentation with live debugger demo
- Code cleanup, README, one-command reproduce

---

### Parallelism Map

```
Week:        1    2    3    4    5    6    7    8    9   10

Anupam:     [shared infra + schema.py]
            [BM25 index + pipeline---]  [---SPLADE index + pipeline-]  [MonoT5]
            [synthetic GT-----------]
            [rank-δ + inc/ordering--]   [run on SPLADE--------------]
            [BM25 interventions-----]   [SPLADE interventions------]

Shubham:    [evaluation.py]
            [DPR index + pipeline---]   [ColBERT index + pipeline---]
            [Shapley + reranker-LIME]   [synthetic eval (E1)--------]
            [DPR interventions------]   [ColBERT interv]
                                                 [method comparison E2-E4-]

Chaitanya:  [query annotations]
            [viz pipeline on mock---]   [hybrid pipeline---]
            [--debugger skeleton----]   [wire to real------]  [polish]
                                              [cross-paradigm E4-------]
                                                    [query-type E5, E6-]
                                                                [report]
```

---

## Part 3: Experiments

### Core (Must Complete)

| # | Experiment | What it tests | Owner |
|---|---|---|---|
| E1 | Synthetic recovery: all four methods × three settings × multiple α values | Which method recovers planted ground truth? | Shubham |
| E2 | Real intervention recovery: all four methods × BM25→cross-encoder | Which method predicts intervention displacements? | Shubham |
| E3 | Cross-paradigm: repeat E2 with DPR, ColBERT, SPLADE | Does method accuracy differ across paradigms? | Shubham + Chaitanya |
| E4 | Attribution distribution per paradigm | Do paradigms produce structurally different patterns? | Chaitanya |

### Extension

| # | Experiment | What it tests | Owner |
|---|---|---|---|
| E5 | Query-type slicing | Do entity queries differ from semantic queries? | Chaitanya |
| E6 | Candidate set size (100, 500, 1000) | How does gatekeeper boundary affect attribution? | Chaitanya |
| E7 | Reranker strength (MiniLM vs. MonoT5) | Does stronger reranker change patterns? | Anupam |
| E8 | Cross-domain (BEIR subset) | Do findings generalize? | Chaitanya |

---

## Part 4: Minimum Viable Project

```
MVP:         1 real pipeline (BM25→cross-encoder)
             + synthetic ground truth (linear mix)
             + 2 attribution methods (rank-delta + one other)
             + recovery evaluation on synthetic + real
             (E1 + E2 only)

Extension 1: All four methods + all four retrievers
             (+ E3, E4)

Extension 2: Query-type analysis + candidate set ablation
             (+ E5, E6)

Extension 3: Interactive debugger
```

---

## Part 5: Final Deliverables

### Scientific
- **The benchmark:** synthetic ground truth construction protocol (three settings) + intervention-based empirical ground truth protocol + evaluation metrics. Primary contribution.
- **Method comparison:** four attribution methods evaluated across four paradigms with calibrated recovery scores.
- **Cross-paradigm and query-type findings.**

### Engineering
- **Modular retrieval platform** with four+ retrievers, shared reranker, per-stage score logging, intervention automation.
- **Synthetic pipeline generator** for controlled ground truth.
- **Interactive retrieval debugger:** live query → per-stage rankings → attribution visualization → stage toggle.

### Written
- Final report structured around RQ1–RQ4.
- Presentation with live demo.
- Clean, documented, reproducible codebase.

---

## Part 6: Key Risks

| Risk | Mitigation |
|---|---|
| Synthetic ground truth (linear mix) too unrealistic | Include gated pipeline setting; report on both |
| All methods perform similarly | "Use the cheapest (rank-delta)" is a useful practical finding |
| Reranker-only baseline surprisingly good | Means cross-stage attribution is unnecessary — valid finding |
| ColBERT indexing slow / fails on free-tier GPU | Fall back to Contriever (simpler bi-encoder). Three paradigms is sufficient |
| Schema divergence across members | `test_schema_compliance.py` runs on every push |
| Retriever nDCG doesn't match baselines | Debug before any experiments. Attribution on a broken pipeline is meaningless |
| Intervention ground truth is noisy | Use multiple intervention types, check consistency, report agreement |
| Existing work found in lit review that already does this | Pivot to extending/evaluating that work using the benchmark |
| Team member falls behind on their retriever | Other retrievers + synthetic still produce a complete benchmark |

---

## Part 7: Papers to Survey

| Paper | Why |
|---|---|
| Rank-LIME (Chowdhury et al., ICTIR 2023) | Single-stage attribution baseline |
| RankingSHAP (Heuss, de Rijke & Anand, SIGIR 2025) | Shapley axioms for potential stage-level adaptation |
| ERASER (DeYoung et al., 2020) | Comprehensiveness/sufficiency metrics |
| Câmara & Hauff (ECIR 2020) | Probing BERT rankers |
| Anand et al. (2022) survey | ExIR taxonomy, confirms pipeline gap |
| Anand et al. (SIGIR 2023 tutorial) | "Retrieval cascades" as open challenge |
| Saha, Majumdar & Mitra (ACM Computing Surveys, 2026) | Most recent survey, confirms gap |
| BEIR (Thakur et al., NeurIPS 2021) | Cross-paradigm benchmark methodology |
| Agarwal et al. (BioNLP 2026) | Stage-instrumented clinical retrieval — closest existing work |
| The Deterministic Horizon (arXiv 2605.23024) | Formal impossibility bound for k-stage attribution |
| ExDocS (2021) | "Why is document X at rank Y" |
| RAGXplain (arXiv 2505.13538, 2025) | Component-level RAG diagnostics |
| WSDM Cup 2026 (Hao & Wang) | Stage ablation in a real multi-stage pipeline |
