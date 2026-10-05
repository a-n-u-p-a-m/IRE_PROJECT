# Project Overview, Methodology, and Open Questions

**Project:** Intervention-Based Benchmark for Stage-Level Attribution in Multi-Stage Retrieval Pipelines
**Course:** CS4.406 — Information Retrieval & Extraction
**Team:** Anupam Dwivedi, Shubham Paliwal, Chaitanya K
**Status as of:** 2026-10-05 — shared infrastructure complete, all verticals pending
**Purpose of this document:** a plain-language explanation of what we are building and why, how the synthetic and real halves of the benchmark relate to each other, and a consolidated list of unresolved decisions to discuss with teammates and the supervisor.

---

## 1. The problem in plain terms

Modern search does not work in one pass. It works in stages.

**Stage 1 (the retriever)** scans the entire corpus — 8.8 million MS MARCO passages — using a cheap scoring function, and shortlists the top 1,000 candidates.

**Stage 2 (the reranker)** is a transformer cross-encoder far too expensive to run on 8.8M documents. It only ever sees those 1,000, and reorders them. What the user sees is the output of that reordering.

An analogy that holds up well: a company hiring from a million applicants. HR skims every résumé and forwards 1,000 to the hiring manager, who reads those properly and ranks them. When a strong candidate doesn't get hired, whose fault is it? Maybe HR discarded them and the manager never saw them. Maybe HR forwarded them fine and the manager ranked them low. These are completely different failures with completely different fixes — but from outside, both look identical.

**The question this project answers: when a document ends up at the position it does, how much of that is Stage 1's doing and how much is Stage 2's?**

### Why existing work does not answer it

Tools like Rank-LIME, RankingSHAP and ExDocS explain a *single* ranker — "this document scored highly because it contains these query terms." That problem is addressed. None of them attribute responsibility *across* stages of a cascade. The Anand et al. (2022) survey, the SIGIR 2023 tutorial, and Saha et al. (ACM Computing Surveys, 2026) each independently list retrieval cascades as an open challenge.

### The distinction that makes this more than bookkeeping

Stage 1 exercises two different kinds of power, and they are routinely conflated:

- **Gating** — deciding who enters the candidate set at all. A document at rank 1,001 is invisible to the reranker no matter how relevant it is. This is a hard, binary veto.
- **Ordering** — assigning scores that partially propagate into the final ranking. A soft nudge.

Collapsing these into a single "Stage 1 importance" number discards the most actionable information, because "your document was vetoed at the door" and "your document was mildly downweighted" demand entirely different remedies. The inclusion-vs-ordering decomposition method exists to separate them.

---

## 2. Why there is no ground truth, and what our actual contribution is

The reason stage-level attribution is unsolved is not that the methods are hard to write — rank-delta is roughly ten lines. **The reason is that there is no answer key.** If a method reports "Stage 1 was 70% responsible," there is nothing in the world to check that against. You cannot grade a method when the correct answer does not exist.

So the primary contribution is not a method. It is a **benchmark**: two independent routes to manufacturing ground truth where none previously existed.

### Route 1 — Synthetic pipelines with planted truth

Build a pipeline where *we* define the combination rule, so the true answer is known by construction.

### Route 2 — Interventions on real pipelines

Surgically break one stage of a real pipeline and measure what actually moves. The measured displacement is empirical, causal ground truth — not an opinion about what a stage contributed, but a measurement of what happens when it is removed.

With both in place, the real question becomes answerable: **do the four attribution methods recover the ground truth, and which does it best?**

---

## 3. The synthetic ("fake") pipeline

### There is no dataset — we generate it

No MS MARCO. No TREC queries. No relevance judgments. No index. This is not a shortcut; it is the point. The only way to know the true answer is to be the one who constructs it.

We invent N pretend queries. For each, we invent M pretend documents (`doc_0001`…`doc_1000`) that correspond to no real text anywhere. We draw two numbers per document — a synthetic Stage 1 score and a synthetic Stage 2 score — and combine them with a chosen rule:

```
final_score = α · stage_1_score + (1 − α) · stage_2_score
```

Set α = 0.3 and Stage 1 is *definitionally* 30% responsible.

**Relevance labels are not needed.** We are not measuring search quality on synthetic data (no nDCG), only attribution accuracy. Nobody asks "was document 47 actually relevant?", only "did the method recover α = 0.3?". This is why this vertical is unblocked from day one while the retriever verticals wait on indexes.

### Why the data contract mattered

Synthetic output obeys the same schema as real output, so the attribution methods run on either unchanged — `rank_delta.py` needs no special synthetic mode. Had the schemas diverged we would be maintaining two versions of every method.

### The three settings, and what each one probes

| Setting | Construction | Failure mode it catches |
|---|---|---|
| `linear_mix` | Smooth blend, α swept over {0.1, 0.2, … 0.9} | A method that reports "both stages mattered equally" regardless of input. Plausible on any single run; obviously broken across the full sweep. |
| `gated` | Hard top-K cut; documents below the Stage 1 threshold never reach Stage 2 | Whether a method can detect gating at all. The smooth setting cannot reveal this because it has no gate. |
| `adversarial` | Stage 1 and Stage 2 deliberately disagree — documents one loves and the other hates | Methods that quietly assume the two stages roughly agree. |

Output files per the roadmap convention: `synthetic/linear_mix_alpha_0.2.parquet`, `gated_topk_500.parquet`, `adversarial.parquet`.

---

## 4. The real pipeline and its interventions

| Intervention | Procedure | What it isolates |
|---|---|---|
| `remove_stage2` | Discard the reranker; final ranking = Stage 1 ranking | Total Stage 2 contribution |
| `randomize_stage1_scores` | Keep the same candidate set, replace Stage 1 scores with uniform random, re-run Stage 2 | Stage 1's ordering contribution — **see Doubt T1, this may be a no-op as currently specified** |
| `remove_gating` | Expand Stage 1 output from top-1,000 to top-5,000 and re-run Stage 2 | Stage 1's gating contribution — documents that appear only now were being vetoed |

Each runs with three seeds (42, 123, 456); report mean displacement ± std. Displacement convention is fixed as `rank_after − rank_before`, so positive means the rank worsened.

---

## 5. How the synthetic and real halves are compared

**We do not compare the synthetic pipeline to the real pipeline.** That comparison would be meaningless — different data, different score scales, different everything.

What gets compared is the **four attribution methods**. The two pipelines are two separate exams each method sits. The methods are the students; the pipelines are the exam halls.

### Two exams, two necessarily different grading schemes

**The synthetic exam has a marked answer sheet.** We built the pipeline, so the truth is exactly 0.3. A method says 0.42 and we score it precisely: off by 0.12. Three things get graded — average distance from truth (`attribution_error`), whether documents are at least *ordered* correctly even if absolute values drift (`rank_correlation`), and whether errors are systematic and correctable rather than random (`calibration_error`).

**The real exam has no answer sheet — only a before-and-after measurement.** On a real pipeline nobody can state "Stage 1 was 30% responsible," because that number does not exist in the world. All we have is what happened when a stage was broken: this document fell 312 places, that one did not move.

So a method is graded on the real pipeline by **whether its predictions are consistent with what actually happened**:

- If it says "this document owes its position to the reranker," deleting the reranker should wreck that document. Does it?
- If it says "the reranker is irrelevant here," deleting the reranker should barely move it. Does it?

That is a correlation, not an exact match. We check whether the method's story is consistent with reality, not whether it hit a target.

This split is reflected directly in the code: synthetic data goes through `evaluate_all_synthetic` (distance from known truth), real data through `evaluate_all_real` (agreement with measured displacement). Different functions, different metrics, by necessity.

### The actual comparison: a 4 × 2 outcome matrix

Each method ends up with two grades. **This matrix is the central scientific output of the project.**

| | Passes the real exam | Fails the real exam |
|---|---|---|
| **Passes the synthetic exam** | Method is trustworthy — recommend it | Sound in theory, breaks on real pipelines. *Why?* |
| **Fails the synthetic exam** | Suspicious — likely exploiting a correlation, not attributing | Method does not work |

The off-diagonal boxes are the interesting ones.

**Passes synthetic, fails real** is the most likely and most informative result. It means the method is mathematically sound but depends on an assumption real pipelines violate — most plausibly that the two stages combine smoothly, when in reality the candidate-set cut is an all-or-nothing veto. That is not a failure of the project; it is a finding, and it tells practitioners exactly when not to trust the method.

**Fails synthetic, passes real** should prompt suspicion of ourselves rather than celebration. A method that cannot recover a value we literally planted, yet predicts real displacements well, is probably picking up an incidental correlation. That is a signal to check whether the intervention measurements are confounded.

The synthetic exam is a **gate** — fail it and nothing else is credible. The real exam is the **validity check** — pass it and the method is actually useful. Both are required, which is precisely why the benchmark has two ground-truth routes rather than one.

### The diagnostic bridge between the halves

The three synthetic settings are not arbitrary. Each imitates a specific feature of real pipelines: `linear_mix` imitates smooth score blending, `gated` imitates the hard top-1,000 cut, `adversarial` imitates stage disagreement.

So when a method fails on real data, we can ask *which synthetic setting it also failed on*. If a method bombs on real data specifically for documents near the rank-1,000 boundary, and also bombs on `gated`, the failure is localized to gating detection. We learn the cause, not just the symptom.

That is the diagnostic loop: synthetic settings isolate individual failure modes one at a time, so when the real pipeline fails messily we have a vocabulary for *what kind* of failure it is.

### The unit-mismatch wrinkle (worth stating in the report)

The two sides do not share units. Synthetic yields a fraction in [0, 1]. Real yields displacement in rank positions — possibly 3, possibly 800.

Bridging them requires squashing displacement into [0, 1], which is what `recovery_error` does: it divides measured displacement by the maximum displacement that was possible. This was one of the bugs caught in review — the original divided by the largest displacement *observed in the sample*, making results dependent on which queries happened to be in the batch. It now divides by an explicit stated maximum.

This normalization is a reasonable convention, not a law of nature. Two methods that look equivalent under it could rank differently under a different squashing choice. Stating that plainly strengthens the benchmark's credibility rather than weakening it.

---

## 6. Real-world significance

**RAG debugging.** Every production retrieval-augmented system is a multi-stage pipeline feeding an LLM. When the model answers wrongly because a critical passage was never retrieved, an engineer must guess: raise top-K? swap the embedding model? retrain the reranker? fix chunking? Today that is trial and error — change something, rerun the eval set, watch the number. Stage attribution converts guessing into diagnosis. This is what RAGXplain reaches for at the component level and what our debugger deliverable does at the ranking level.

**Cost and latency.** The reranker is the expensive component — a transformer forward pass per candidate. If attribution shows it contributes little for certain query types (E5 tests exactly this), those queries can bypass Stage 2 entirely. That is a direct infrastructure saving, discoverable only if per-stage contribution can be measured per query type.

**Auditing and accountability.** If a hiring-search or medical-literature system systematically buries certain documents, "the algorithm did it" is not an acceptable answer to a regulator or clinician. Stage attribution localizes responsibility to a component someone owns and can fix. Agarwal et al. (BioNLP 2026) on stage-instrumented clinical retrieval is the closest existing work and is narrow to one domain.

**A cheap practical shortcut, if it exists.** A likely outcome is that rank-delta — essentially free to compute — performs nearly as well as stage-Shapley, which requires enumerating stage subsets and rerunning the pipeline. If true, that is a genuinely useful result: a diagnostic practitioners can afford to run in production.

**Honest ceiling.** The Deterministic Horizon paper (arXiv 2605.23024) proves a formal impossibility bound on k-stage attribution. Part of the benchmark's value is mapping where attribution is reliable versus where it is provably not — a stronger contribution than claiming it always works.

---

## 7. Work division

The structure is **vertical slicing**: each member owns complete pipelines end-to-end rather than everyone owning one horizontal layer. Nobody blocks on anyone after day one.

| | Anupam | Shubham | Chaitanya |
|---|---|---|---|
| **Shared module** | `schema.py`, `reranker.py`, `logger.py` | `evaluation.py` | query-type annotations CSV |
| **Retrievers** | BM25 (wk 1–4), SPLADE (wk 5–7) | DPR (wk 1–4), ColBERT (wk 3–6) | Hybrid BM25+DPR via RRF (wk 3–6) |
| **Attribution methods** | rank-delta, inclusion-ordering | stage-Shapley, reranker-LIME | — |
| **Synthetic ground truth** | all three settings (due end wk 2) | — | — |
| **Experiments owned** | E7 (MonoT5 reranker strength) | E1, E2, E3, method comparison E4 | E4, E5, E6, E8 |
| **Other** | MonoT5 wrapper, interventions on own pipelines | literature review, Shapley sensitivity analysis | visualization pipeline, Streamlit debugger, **report + presentation lead** |

**Two dependencies matter.** Anupam's synthetic parquets by end of week 2 unblock Shubham's E1 and Chaitanya's visualization work — neither can produce anything real until controlled data exists. And Chaitanya deliberately builds against mock data in weeks 1–4 so his tooling is ready the moment real results arrive.

---

## 8. Current status

### Complete

The shared infrastructure layer — nine files, on `main` as of commit `94561f7`.

- `shared/schema.py` — data contract for all four table types (score matrix, attribution, intervention, synthetic ground truth), with validators and Parquet I/O
- `shared/evaluation.py` — all eight metrics plus bootstrap CI
- `shared/logger.py` — four loggers emitting schema-compliant rows
- `shared/reranker.py` — MiniLM and MonoT5 wrappers
- `shared/data.py` — corpus, query and qrel loading
- `shared/utils.py` — seeding, config, timing, top-K padding
- `experiments/run_pipeline.py` — retriever → reranker → logger, with nDCG verification
- `tests/test_schema_compliance.py` — 30 tests
- `config/base.yaml`, `config/bm25.yaml`, pinned `requirements.txt`

Shubham reviewed all of it and raised 20 issues; all 20 are resolved. The eight evaluation bugs were the substantive ones — `rank_correlation` was being fed a constant array, intervention types were being merged across a bare `doc_id` join, and `comprehensiveness`/`sufficiency` returned single floats which made bootstrap CI impossible. Four open design questions were decided: displacement is `rank_after − rank_before`, interventions carry a `seed` column, and synthetic ground truth received its own schema with validator and logger.

This matters more than it sounds: three people working in parallel on separate pipelines only works if their outputs are perfectly interchangeable. That is now guaranteed by code and tests rather than by convention.

### Not started

Everything vertical. No retriever exists. No index is built. No attribution method is written. No intervention is implemented. No experiment has been run. `retrievers/`, `attribution/`, `synthetic/`, `interventions/`, `analysis/`, `debugger/` and `query_annotations/` do not yet exist, nor do `run_attribution.py`, `run_intervention.py` or `run_evaluation.py`. README is one line.

Realistically: **week 1 of a 10-week plan.** The workbench is built; the woodwork has not started.

### Immediate next steps (Anupam)

1. Add a `synthetic:` section to `config/base.yaml` pinning the generation parameters (see Doubt T2) — before writing the generator, not after.
2. Build `synthetic/linear_mix.py`. No infrastructure needed, unblocks two teammates, and validates `evaluation.py` against data where the answer is already known.
3. Build the BM25 index and `retrievers/bm25.py`, and confirm nDCG@10 lands in 0.50–0.52. This is a gate, not a suggestion — attribution on a quietly broken pipeline tells us nothing.

---

## 9. Open design decisions requiring team agreement

These are load-bearing and currently written down nowhere.

### T1 — `randomize_stage1_scores` may be a no-op as specified (highest priority)

The roadmap specifies: keep the same candidate set, replace Stage 1 scores with uniform random, re-run Stage 2, compute displacement.

**The problem:** a cross-encoder scores a `(query, passage)` pair. It never reads the Stage 1 score. So if the candidate set is held fixed and the pipeline is a pure cascade where `final_rank = stage_2_rank`, re-running Stage 2 produces *identical* scores and *identical* ranks. Displacement would be zero for every document, by construction.

This is not a hypothetical problem. `_STAGE_INTERVENTION_MAP` in `evaluation.py` pairs `stage_1_attribution` with `randomize_stage1_scores`, so `displacement_correlation_stage1` would correlate predictions against an all-zero vector — NaN or meaningless. **This directly compromises E2.**

Candidate resolutions to discuss:
- **(a)** Randomize scores *and* reselect the candidate set from the randomized scores. Then the candidate set genuinely changes and displacement is real — but this conflates ordering with gating, which is precisely what we are trying to separate.
- **(b)** Define the pipeline with score fusion (`final = λ·s₁ + (1−λ)·s₂` rather than pure replacement), so Stage 1 scores do influence final order. More realistic for some production systems, and makes the intervention meaningful — but it is a change to the pipeline definition, not just the intervention.
- **(c)** Replace this intervention with a *candidate-set perturbation* (randomly drop or swap a fraction of candidates) which is well-defined under a pure cascade.
- **(d)** Accept it as a deliberate null control — if Stage 1's ordering provably cannot affect a pure cascade's output, that is itself a finding worth reporting, and the attribution methods should recover ≈0 for Stage 1 ordering.

Option (d) is intellectually clean and cheap, but means we lose one of three interventions as a ground-truth source. **This needs resolving before Shubham implements E2.**

### T2 — Synthetic generation parameters are unspecified

`config/base.yaml` has no `synthetic:` section. The roadmap is explicit that shared choices must be identical across members, but right now none of the following live anywhere — meaning if Shubham regenerates the synthetic data on his machine he gets different numbers:

- **Score distributions.** Drawing both stages from uniform noise builds a world no real pipeline resembles. Real BM25 scores are right-skewed with a long tail; cross-encoder logits pile up at the extremes. A method could pass the synthetic exam purely because uniform noise is easy, then fail on real data for reasons unrelated to attribution.
- **Inter-stage correlation.** This is the one to worry about most. In reality a genuinely good document tends to score well on *both* stages — the signals overlap heavily. If synthetic Stage 1 and Stage 2 scores are drawn independently, we have built a world where the stages carry unrelated information and attribution is artificially easy, because there is no ambiguity to resolve. Correlation should be an explicit swept knob, not an accident of the generator. **Arguably it deserves to be a fourth setting.**
- **Scale.** Number of queries and documents per query, which determines whether the bootstrap CIs mean anything. 100 × 1,000 is a reasonable start but should be a stated parameter, not a magic number.

Optional upgrade worth considering: once the real BM25 run exists in week 4, fit score distributions *from it* and regenerate synthetic data to match. That turns "plausible-looking fake scores" into "fake scores statistically calibrated to the real thing" — a strong sentence for the report. Timing is chicken-and-egg, so ship the plausible version by week 2 to unblock teammates, then recalibrate if there is room.

### T3 — Attribution is computed on top-10, but gating happens at rank ~1,000

There is a structural tension here. `eval_top_k: 10` means attribution is computed for documents in the final top 10. But the gating phenomenon — the inclusion-vs-ordering distinction that is one of our conceptual contributions — operates at the candidate-set boundary around rank 1,000. **A document excluded by gating is by definition not in the final top 10.**

So gating effects may be structurally unobservable in our attribution outputs. Options: evaluate attribution on a wider set (top-100? all candidates?) for the gating analysis specifically; or define a separate evaluation population for inclusion-ordering; or restrict gating claims to the `remove_gating` intervention results rather than the attribution outputs. Needs a decision.

### T4 — Displacement is unbounded when documents leave the evaluated set

If a document sits at rank 3 before intervention and falls out of the top 1,000 entirely afterwards, what is `rank_after`? Options: cap at `top_k + 1`; cap at candidate-set size; record as null and exclude; or use a reciprocal-rank-style transform that handles the tail gracefully. This interacts directly with `max_possible_displacement` in `recovery_error`, so the two must be decided together.

### T5 — `remove_gating` changes the candidate pool size

Expanding Stage 1 output from top-1,000 to top-5,000 means `rank_before` and `rank_after` are measured in differently-sized pools. Two sub-questions: is displacement comparable across differently-sized pools, and for a document that was *not* in the top-1,000 before but appears after expansion, what is `rank_before`? (Undefined? 1,001? The pool size?) Also a compute question — reranking 5,000 candidates per query is 5× the Stage 2 cost.

### T6 — Shapley's `v({S2})` is computationally infeasible as the natural definition

"Stage 2 alone" would mean running the cross-encoder over all 8.8M passages per query. Not possible. The roadmap sets `v({S2}) = 0` as primary with a sensitivity check against the alternative. **Is `v({S2}) = 0` defensible, or does it bias Shapley's attribution toward Stage 1 by construction?** Worth working through the arithmetic before Shubham commits to the implementation, since an artefactual bias here would invalidate the Shapley column of the whole comparison.

### T7 — How do `gating_attribution` and `ordering_attribution` relate to the stage attributions?

The schema validates `stage_1_attribution + stage_2_attribution = 1.0`. The inclusion-ordering method also emits `gating_attribution` and `ordering_attribution`. Is the intended relationship `gating + ordering = stage_1_attribution`? Something else? Currently unconstrained by the validator, so two implementations could diverge silently. Should be pinned and enforced in `schema.py`.

### T8 — Nulls in `stage_2_rank` for non-candidates

Documents outside the candidate set have no Stage 2 rank. Every attribution method needs a defined behaviour for those rows — skip, impute, or treat as a sentinel. Should be specified once, centrally, rather than four times inconsistently.

### T9 — Ownership of `interventions/` is unassigned

In the roadmap's directory listing, `retrievers/`, `attribution/`, `synthetic/` and `analysis/` all carry owner names. `interventions/` carries none. Part 2 implies each member runs interventions on their own pipelines, but someone must own the shared implementation so three variants don't appear. Suggest assigning it explicitly.

### T10 — Compute and storage budget

A ColBERT index over 8.8M passages is substantial on disk (plausibly 100 GB+ depending on configuration) and free-tier Colab/Kaggle may not accommodate it. Worth an inventory: who has what GPU access, what local disk, and what the fallback is. The roadmap already names Contriever as the ColBERT fallback — the trigger condition for taking it should be agreed in advance rather than in week 6 under pressure.

---

## 10. Questions for the supervisor

### S1 — Does the Deterministic Horizon bound limit what we can claim?

The roadmap cites arXiv 2605.23024 as a formal impossibility bound for k-stage attribution. We need help interpreting its scope precisely. If attribution is provably impossible in general, what remains achievable, and how should the report frame our positive results against it? This affects how we position the entire contribution and should be settled early rather than discovered in week 9.

### S2 — Is intervention displacement legitimate ground truth, or is the argument circular?

This is the sharpest methodological worry. We define a stage's importance by what happens when we remove it, then grade attribution methods on how well they predict that. A skeptical reviewer could argue we have simply defined the answer to be our own measurement, and that a method scoring well has only learned to imitate our intervention procedure rather than recovering anything true.

We believe the causal framing defends this — an intervention is a counterfactual measurement, not another attribution heuristic — but we would like help making that argument rigorously, and knowing where it genuinely does not hold.

### S3 — Is the benchmark alone a sufficient contribution?

Our primary contribution is the benchmark plus an evaluation of four *existing* methods adapted to the multi-stage setting. Is that sufficient for the project's standards, or does it need a novel attribution method of its own? If the latter, the inclusion-vs-ordering decomposition is the strongest candidate for novelty and should be positioned accordingly from the start.

### S4 — Novelty check against Agarwal et al. (BioNLP 2026)

Our literature review flags this as the closest existing work — stage-instrumented clinical retrieval. We would like a read on whether it preempts our contribution or whether domain-generality plus the synthetic ground-truth protocol is sufficient differentiation.

### S5 — Is the scope realistic for 10 weeks?

Full scope is 4 retrievers × 4 methods × 3 interventions × 3 seeds × 2 query sets, plus 3 synthetic settings × 9 α values, plus 4 extension experiments. With three people and free-tier compute this is ambitious. Which of E5–E8 does the supervisor consider essential versus optional? The roadmap's MVP is one pipeline (BM25→cross-encoder) + linear-mix synthetic + 2 methods + E1/E2 — **is the MVP acceptable as a floor if the extensions do not land?**

### S6 — Guidance on T1 (the `randomize_stage1_scores` problem)

Of the four candidate resolutions in T1, we would value a view on whether the null-control framing (option d) is a legitimate finding or reads as avoiding the problem, and whether moving to a score-fusion pipeline definition (option b) makes the study more or less representative of real deployed systems.

### S7 — Should synthetic distributions be calibrated to real ones?

Calibrating synthetic score distributions to the measured BM25/MiniLM distributions makes the synthetic setting more realistic, but arguably weakens the claim that it is a *controlled* setting independent of any particular real pipeline. Which is the better trade-off for a benchmark paper?

### S8 — Is department GPU access available?

Dense retriever indexing (DPR, ColBERT, SPLADE) and cross-encoder reranking across 97 queries × 1,000 candidates × 4 retrievers × 3 interventions × 3 seeds is the dominant compute cost. If institutional compute is available it materially changes what we can attempt, and the decision is better made now than in week 6.

---

## 11. Summary

We are building the first benchmark for attributing responsibility between stages of a multi-stage retrieval pipeline. The core difficulty is the absence of ground truth, which we address two ways: synthetic pipelines where the answer is planted by construction, and interventions on real pipelines where the answer is measured causally by breaking a stage and observing displacement. Four attribution methods are then graded on both, and the agreement pattern between the two grades is the scientific payload.

The shared infrastructure is complete and reviewed. The experimental work has not begun. The highest-priority open item is **T1** — `randomize_stage1_scores` appears to be a no-op under a pure cascade, which would compromise E2 — followed by **T2**, pinning the synthetic generation parameters before any synthetic data is produced.
