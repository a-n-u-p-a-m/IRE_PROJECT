# Project Overview, Methodology, and Open Questions

**Project:** Intervention-Based Benchmark for Stage-Level Attribution in Multi-Stage Retrieval Pipelines
**Course:** CS4.406 — Information Retrieval & Extraction
**Team:** Anupam Dwivedi, Shubham Paliwal, Chaitanya K
**Status as of:** 2026-10-05 — shared infrastructure complete through commit `eb6f207`, all verticals pending
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

## 5. The four attribution methods

All four implement the same interface, so any method runs on any pipeline's score matrix:

```python
def compute_attribution(score_matrix: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Input: score matrix in agreed schema. Output: attribution result."""
```

They sit on a deliberate spectrum from trivially cheap to theoretically principled, plus one control. The point of the comparison is to find out whether the expensive ones earn their cost.

---

### 5.1 Rank-delta (Anupam)

**The idea.** Compare where a document sat after Stage 1 with where it sat after Stage 2. If it moved a long way, Stage 2 did the work. If it barely moved, Stage 1 had already decided the outcome and Stage 2 merely agreed.

**The computation**, per the roadmap:

```
Δ = (rank_1 − rank_2) / rank_1        # positive ⇒ Stage 2 promoted it
```

then rescale so `stage_1_attribution + stage_2_attribution = 1.0`.

A document BM25 ranked 800th that the reranker lifted to 3rd gives Δ ≈ 0.996 — almost all Stage 2. A document ranked 2nd by both gives Δ = 0 — Stage 2 contributed nothing beyond confirmation.

**Why it is in the study.** It costs nothing. No model access, no reruns, only the score matrix we already log. If it performs comparably to stage-Shapley, that is the single most practically useful finding the project could produce, because it is a diagnostic anyone can afford to run in production on every query.

**Known weaknesses, which are part of what we are measuring:**

- **Positional only.** It reads ranks, never scores. Two documents with identical rank movement get identical attribution even if one moved on a hairline score difference and the other on a chasm.
- **The normalization is asymmetric and unbounded below.** 1000 → 1 gives Δ = 0.999; 2 → 1 gives Δ = 0.5; 1 → 2 gives Δ = −1.0; 1 → 100 gives Δ = −99. Positive values are bounded by 1 but negative values are not bounded at all. *How* we squash this into [0, 1] is a free design choice that will visibly affect results — see T12.
- **It conflates agreement with irrelevance.** Δ = 0 means the stages agreed, which rank-delta reports as a 50/50 split. But "both stages independently ranked this first" and "neither stage had any influence" are completely different situations receiving identical attribution.
- **It is structurally blind to gating.** A document outside the candidate set has no `stage_2_rank` at all, so Δ is undefined. Rank-delta can therefore say nothing about the half of Stage 1's power that matters most.

---

### 5.2 Inclusion-vs-ordering decomposition (Anupam)

**The idea, and the project's most likely original contribution.** Stage 1 exercises two separable powers (§1). This method splits its attribution accordingly:

- `gating_attribution` — how much did Stage 1's *decision to admit* this document determine its fate?
- `ordering_attribution` — given that it was admitted, how much did Stage 1's *score* influence the final position?

**The mechanism** keys on proximity to the candidate boundary at K = 1000. A document at Stage 1 rank 4 was going to be admitted under any plausible K; its gating attribution is near zero, because admission was never in question. A document at rank 987 was one small perturbation away from being invisible; gating was nearly the whole story. So `gating_attribution` rises sharply as Stage 1 rank approaches K, and `ordering_attribution` carries the residual influence.

**Why it matters.** It is the only method producing a *directly actionable* answer. "Gating was the problem" means raise K or improve recall. "Ordering was the problem" means improve Stage 1's scoring function. Every other method returns a number that tells you which stage to blame but not what to do about it. Nothing in the surveyed literature performs this decomposition, which is why it is our strongest novelty claim.

**Known weaknesses:**

- **Its schema relationship is undefined.** Is `gating + ordering = stage_1_attribution`? Something else? Currently unconstrained — see T7.
- **It cannot be validated on `linear_mix`**, which has no gate. It needs the `gated` synthetic setting, making that setting load-bearing rather than supplementary.
- **The evaluation population may exclude exactly the cases it explains.** Documents genuinely killed by gating are absent from the final top-10 by definition — see T3. This is the sharpest threat to the method's evaluability.
- **Boundary proximity is a proxy, not a measurement.** There is a real risk the method merely re-derives "distance from rank 1000", which is trivially available from the score matrix and requires no attribution machinery. The `remove_gating` intervention is the actual measurement; the method must beat the trivial baseline to be worth anything.

---

### 5.3 Stage-Shapley (Shubham)

**The idea.** Borrow from cooperative game theory. Treat each stage as a player; a stage's attribution is its average marginal contribution across all coalition orderings. Shapley values are the *unique* allocation satisfying efficiency, symmetry, dummy and additivity — the only method here with a uniqueness guarantee.

With two stages there are four coalitions, so the full enumeration is cheap. The roadmap's value function:

| Coalition | Value |
|---|---|
| `v({})` | 0 |
| `v({S2})` | 0 |
| `v({S1})` | Stage 1 ranking only |
| `v({S1, S2})` | Full pipeline |

Working the Shapley formula through for two players:

```
φ_S1 = ½[v({S1}) − v({})]      + ½[v({S1,S2}) − v({S2})]  =  ½·v({S1}) + ½·v({S1,S2})
φ_S2 = ½[v({S2}) − v({})]      + ½[v({S1,S2}) − v({S1})]  =  ½[v({S1,S2}) − v({S1})]
```

Efficiency checks out: `φ_S1 + φ_S2 = v({S1,S2})`.

**But look at what `v({S2}) = 0` does.** Stage 1 alone is not worthless — BM25 alone reaches nDCG@10 ≈ 0.50. So `v({S1})` is substantially positive, and it appears **twice**: added to `φ_S1` and subtracted from `φ_S2`, each at half weight. The attribution is therefore biased toward Stage 1 by `½·v({S1})` *by construction*, and the size of that bias grows with how good the retriever is alone.

This is not a rounding artifact. It is a systematic bias proportional to Stage 1's standalone quality, and it would make Shapley look like it "discovers" that stronger retrievers matter more — a conclusion that is purely an artifact of the value function. **This needs resolving before the Shapley column of the comparison table can be trusted (T6).**

The honest definition, `v({S2}) =` Stage 2 over the full corpus, requires running a cross-encoder across 8.8M passages per query. Infeasible. A promising middle route: `remove_gating` already reranks the top 5,000, which is a partial observation of Stage 2 operating under weakened Stage 1 influence, and could seed an estimate of `v({S2})` rather than assuming zero.

**Other weaknesses:**

- **Expensive relative to rank-delta** — requires actually rerunning the pipeline per coalition.
- **Shapley allocates a scalar, but our schema wants per-document attribution.** This means `v` must be defined *per document* — document `d`'s reciprocal rank under coalition `S`, say, or `1/rank_d(S)`. The roadmap does not specify this, and the choice materially changes the output (T13).

---

### 5.4 Reranker-only attribution / Rank-LIME (Shubham)

**This is not a cross-stage method, deliberately.** It is the control, and it exists to attack the project's own premise.

It takes the cross-encoder in isolation and applies established single-ranker attribution — Rank-LIME or RankingSHAP — perturbing inputs and fitting a local surrogate to explain the reranker's scores. It knows nothing about Stage 1 and makes no attempt to.

**Why include a method that cannot answer the research question.** Because it is the null hypothesis. If reranker-only attribution predicts intervention displacements as well as the genuinely cross-stage methods, then **cross-stage attribution is unnecessary** and practitioners should keep using the tools they already have. The roadmap correctly lists this under risks as a valid finding rather than a failure. A benchmark that cannot fail is not a benchmark.

**Two structural problems to resolve before implementation:**

- **It may be unable to fill the schema.** The contract requires `stage_1_attribution + stage_2_attribution = 1.0`. A reranker-only method has nothing to put in `stage_1_attribution`. If it always emits (0.0, 1.0) it is a constant predictor, and `rank_correlation` returns NaN for constant input — which the E8 fix now handles gracefully, but a constant baseline also cannot be meaningfully ranked against the others on correlation metrics. We need to decide what it emits (T14).
- **Rank-LIME attributes to *features*, but a cross-encoder consumes text.** Perturbing "features" means perturbing query or passage tokens, which yields *term-level* attribution. The mapping from term-level importance to a single stage-level number is undefined and is not something Rank-LIME provides. This needs specifying, and it is more work than "wrap the existing library" suggests.

---

### 5.5 What the comparison is really testing

Read as a set, the four methods span the hypothesis space:

| Method | Cost | What it would mean if this one wins |
|---|---|---|
| Rank-delta | Free | Stage attribution is easy; use the cheap proxy everywhere |
| Inclusion-ordering | Low | The gating/ordering split is the right abstraction for cascades |
| Stage-Shapley | High | Axiomatic rigour is necessary; approximations lose real information |
| Reranker-only | Medium | Cross-stage attribution is unnecessary — the project's premise is wrong |

Every one of those four outcomes is publishable. That is the mark of a well-posed comparison.

---

## 6. How the synthetic and real halves are compared

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

The two sides do not share units. Synthetic yields a fraction in [0, 1]. Real yields displacement in rank positions — possibly 3, possibly 800. Bridging them requires putting displacement onto the same scale as attribution.

As of `eb6f207`, `recovery_error` does this by comparing `stage_1_attribution` against **Stage 1's share of the total observed displacement**:

```
normalized_displacement = |d1| / (|d1| + |d2|)
```

where `d1` and `d2` are the displacements from the interventions that remove Stage 1 and Stage 2 respectively. Because attribution already satisfies `stage_1 + stage_2 = 1`, a *share* is the natural counterpart, and the error is guaranteed bounded in [0, 1]. Documents that neither intervention moved carry no signal and are skipped.

This replaced two earlier attempts, both of which were unbounded: dividing by the largest displacement observed in the sample (made results depend on batch composition), then dividing by `top_k` (let the error reach values like 49.5).

**This definition is a proposal, not settled.** The roadmap specifies "normalized displacement" without defining it, and Shubham flagged the share formulation explicitly as a team decision — see Doubt T10. Two methods that look equivalent under one normalization could rank differently under another, so whichever we adopt needs to be stated and justified in the report rather than left as an implementation detail.

---

## 7. Real-world significance

**RAG debugging.** Every production retrieval-augmented system is a multi-stage pipeline feeding an LLM. When the model answers wrongly because a critical passage was never retrieved, an engineer must guess: raise top-K? swap the embedding model? retrain the reranker? fix chunking? Today that is trial and error — change something, rerun the eval set, watch the number. Stage attribution converts guessing into diagnosis. This is what RAGXplain reaches for at the component level and what our debugger deliverable does at the ranking level.

**Cost and latency.** The reranker is the expensive component — a transformer forward pass per candidate. If attribution shows it contributes little for certain query types (E5 tests exactly this), those queries can bypass Stage 2 entirely. That is a direct infrastructure saving, discoverable only if per-stage contribution can be measured per query type.

**Auditing and accountability.** If a hiring-search or medical-literature system systematically buries certain documents, "the algorithm did it" is not an acceptable answer to a regulator or clinician. Stage attribution localizes responsibility to a component someone owns and can fix. Agarwal et al. (BioNLP 2026) on stage-instrumented clinical retrieval is the closest existing work and is narrow to one domain.

**A cheap practical shortcut, if it exists.** A likely outcome is that rank-delta — essentially free to compute — performs nearly as well as stage-Shapley, which requires enumerating stage subsets and rerunning the pipeline. If true, that is a genuinely useful result: a diagnostic practitioners can afford to run in production.

**Honest ceiling.** The Deterministic Horizon paper (arXiv 2605.23024) proves a formal impossibility bound on k-stage attribution. Part of the benchmark's value is mapping where attribution is reliable versus where it is provably not — a stronger contribution than claiming it always works.

---

## 8. Work division

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

## 9. Current status

### Complete

The shared infrastructure layer, on `main` as of commit `eb6f207`.

- `shared/schema.py` — data contract for all four table types (score matrix, attribution, intervention, synthetic ground truth), with validators and Parquet I/O
- `shared/evaluation.py` — all eight metrics plus bootstrap CI
- `shared/logger.py` — four loggers emitting schema-compliant rows
- `shared/reranker.py` — MiniLM and MonoT5 wrappers
- `shared/data.py` — corpus, query and qrel loading
- `shared/utils.py` — seeding, config, timing, top-K padding
- `experiments/run_pipeline.py` — retriever → reranker → logger, with nDCG verification
- `tests/test_schema_compliance.py` (30 tests) and `tests/test_evaluation.py` (17 tests) — 47 total
- `config/base.yaml`, `config/bm25.yaml`, pinned `requirements.txt`

**Review round 1 (`94561f7`).** Shubham reviewed the initial infrastructure and raised 20 issues; all 20 are resolved. The eight evaluation bugs were the substantive ones — `rank_correlation` was being fed a constant array, intervention types were being merged across a bare `doc_id` join, and `comprehensiveness`/`sufficiency` returned single floats which made bootstrap CI impossible. Four design questions were decided: displacement is `rank_after − rank_before`, interventions carry a `seed` column, and synthetic ground truth received its own schema with validator and logger.

**Review round 2 (`eb6f207`, Shubham).** Follow-up fixes to `evaluation.py` and shared config:

- `average_over_seeds` collapses the three seed replicates into one row per `(query_id, doc_id, intervention_type)` before any metric runs. Previously the seeds tripled the joined rows and `comprehensiveness` silently kept only the last seed.
- The Stage 1 intervention became a parameter with a degeneracy warning — see T1.
- `recovery_error` was redefined as a displacement *share*, bounded in [0, 1] — see T10.
- Synthetic `attribution_error` now uses per-document truth where available, and raises on missing or duplicated truth rows rather than silently substituting.
- `requirements.txt`: torch → 2.2.2 and faiss-cpu → 1.8.0, since 2.1.2 / 1.7.4 have no Python 3.12 wheels and current Colab is on 3.12. All pins verified resolvable from wheels on Python 3.10–3.12 (Linux, macOS arm64).
- `run_pipeline` warns when `--max-passages` leaves candidates without text, which were previously reranked against empty strings.
- Roadmap §1.8 displacement sign corrected to match the schema, closing Q2.

This matters more than it sounds: three people working in parallel on separate pipelines only works if their outputs are perfectly interchangeable. That is now guaranteed by code and tests rather than by convention.

### Not started

Everything vertical. No retriever exists. No index is built. No attribution method is written. No intervention is implemented. No experiment has been run. `retrievers/`, `attribution/`, `synthetic/`, `interventions/`, `analysis/`, `debugger/` and `query_annotations/` do not yet exist, nor do `run_attribution.py`, `run_intervention.py` or `run_evaluation.py`. README is one line.

Realistically: **week 1 of a 10-week plan.** The workbench is built; the woodwork has not started.

### Immediate next steps (Anupam)

1. Add a `synthetic:` section to `config/base.yaml` pinning the generation parameters (see Doubt T2) — before writing the generator, not after.
2. Build `synthetic/linear_mix.py`. No infrastructure needed, unblocks two teammates, and validates `evaluation.py` against data where the answer is already known.
3. Build the BM25 index and `retrievers/bm25.py`, and confirm nDCG@10 lands in 0.50–0.52. This is a gate, not a suggestion — attribution on a quietly broken pipeline tells us nothing.

---

## 10. Open design decisions requiring team agreement

These are load-bearing and currently written down nowhere.

### T1 — `randomize_stage1_scores` is a no-op as specified (highest priority)

The roadmap specifies: keep the same candidate set, replace Stage 1 scores with uniform random, re-run Stage 2, compute displacement.

**The problem:** a cross-encoder scores a `(query, passage)` pair. It never reads the Stage 1 score. So if the candidate set is held fixed and the pipeline is a pure cascade where `final_rank = stage_2_rank`, re-running Stage 2 produces *identical* scores and *identical* ranks. Displacement is zero for every document, by construction — which would have left `displacement_correlation_stage1` correlating predictions against an all-zero vector, and **compromised E2**.

**Status:** Anupam and Shubham identified this independently, which is reassuring about the diagnosis. `eb6f207` has already defused the silent-failure half: the Stage 1 intervention is now a parameter (`stage1_intervention`, defaulting to `randomize_stage1_scores`), and `_warn_if_degenerate` logs a warning when an intervention never moves any document. So switching the decision is a one-argument change and the degenerate case is no longer silent.

**The substantive decision is still open.** Candidate resolutions:
- **(a)** Randomize scores *and* reselect the candidate set from the randomized scores. Then the candidate set genuinely changes and displacement is real — but this conflates ordering with gating, which is precisely what we are trying to separate.
- **(b)** Define the pipeline with score fusion (`final = λ·s₁ + (1−λ)·s₂` rather than pure replacement), so Stage 1 scores do influence final order. More realistic for some production systems, and makes the intervention meaningful — but it is a change to the pipeline definition, not just the intervention.
- **(c)** Replace this intervention with a *candidate-set perturbation* (randomly drop or swap a fraction of candidates) which is well-defined under a pure cascade.
- **(d)** Accept it as a deliberate null control — if Stage 1's ordering provably cannot affect a pure cascade's output, that is itself a finding worth reporting, and the attribution methods should recover ≈0 for Stage 1 ordering.

Option (d) is intellectually clean and cheap, but means we lose one of three interventions as a ground-truth source. Shubham's code comment suggests `remove_gating` as the natural substitute, which is close to option (a) but keeps gating and ordering distinguishable because gating is what it measures by design. **This needs resolving before E2 runs.**

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

### T10 — Is `|d1| / (|d1| + |d2|)` the right normalized displacement?

`eb6f207` defines normalized displacement as Stage 1's share of total observed displacement, which is bounded in [0, 1] and mirrors the `stage_1 + stage_2 = 1` constraint on attribution. Shubham flagged it explicitly as a proposal for the team rather than a settled choice. Points to discuss:

- A *share* discards magnitude. A document where both interventions move it 2 places and one where both move it 400 places produce the same share of 0.5, despite the second being far more stage-sensitive. Should magnitude re-enter as a weight — e.g. weighting each document's contribution to the mean by `|d1| + |d2|`?
- Documents that neither intervention moves are skipped. Those are arguably the most interesting cases (the pipeline's output was robust to removing *either* stage), and silently dropping them biases the population toward stage-sensitive documents. How many such documents are there in practice, and should they be reported separately?
- Taking absolute values discards direction. A document promoted by Stage 2 and one demoted by it are treated identically. Is that intended?

Whichever definition we adopt needs to be stated and defended in the report, since the method ranking may depend on it.

### T11 — Compute and storage budget

A ColBERT index over 8.8M passages is substantial on disk (plausibly 100 GB+ depending on configuration) and free-tier Colab/Kaggle may not accommodate it. Worth an inventory: who has what GPU access, what local disk, and what the fallback is. The roadmap already names Contriever as the ColBERT fallback — the trigger condition for taking it should be agreed in advance rather than in week 6 under pressure.

### T12 — How is rank-delta's Δ squashed into [0, 1]?

`Δ = (rank_1 − rank_2) / rank_1` is bounded above by 1 but unbounded below: 1 → 100 gives Δ = −99. The schema requires `stage_1_attribution + stage_2_attribution = 1.0` with both in [0, 1], so Δ must be mapped into that range, and the roadmap says only "rescale."

The choice is consequential. A linear rescale against the observed min/max makes results depend on batch composition — the same bug already fixed twice in `recovery_error`. A sigmoid or `tanh` squash is batch-independent but introduces a temperature parameter that controls how aggressively large demotions saturate. Clipping at some bound discards information about catastrophic demotions, which are arguably the most diagnostically interesting cases.

Suggest a bounded, batch-independent transform with the parameter stated in `base.yaml`, plus a sensitivity check showing the method ranking does not flip under a reasonable alternative.

### T13 — What is Shapley's per-document value function?

Shapley values allocate a *scalar*. Our schema demands per-document attribution, so `v` must be defined per document — `v_d(S)` = document `d`'s reciprocal rank under coalition `S`, or `1/rank_d(S)`, or its nDCG contribution, or its relevance-weighted gain. The roadmap specifies the coalition semantics (T6) but not this.

The choice is not cosmetic. Reciprocal rank heavily weights the top of the list, so a document moving 1 → 2 registers a larger value change than one moving 50 → 100, which will systematically inflate Stage 2's attribution for top-ranked documents. A linear rank-based `v` does the opposite. *ShaRP* ([10.14778/3749646.3749682](https://dl.acm.org/doi/10.14778/3749646.3749682)) may already address this and is worth reading before implementing.

### T14 — What does the reranker-only method put in `stage_1_attribution`?

A reranker-only method has no view of Stage 1, but the schema requires the two attributions to sum to 1.0. Options: emit a constant (0.0, 1.0), which makes it a constant predictor whose `rank_correlation` is NaN and which cannot be meaningfully ranked on correlation metrics; derive a pseudo-Stage-1 signal from the residual the reranker cannot explain; or exempt it from the sum constraint and compare it only on the metrics where it is defined.

The third is probably most honest but requires relaxing the schema validator for this one method — which should be an explicit, documented exemption rather than a silently loosened constraint. Relatedly, Rank-LIME produces *term-level* attribution over text, and the mapping from term importance to a stage-level scalar is undefined and not something the library provides. This is more implementation work than "wrap the existing library" suggests, and should be scoped before week 4.

---

## 11. Questions for the supervisor

### S1 — Can we cite the Deterministic Horizon as motivation rather than a limitation?

Having now read the source (see 12.1), we believe the roadmap mischaracterized it. Guo's Construct Conflation Impossibility Theorem bounds *single-score diagnosis* of multi-stage pipelines — a k-stage pipeline provably needs ≥ k independent metrics — not stage-level attribution itself. On that reading it is formal justification for why this project is necessary and belongs in the introduction, not the limitations.

We would like confirmation of that reading before we build the framing on it. Two specific concerns: the paper appears to be a thesis rather than a peer-reviewed venue paper, and we have read Chapter 4 via its abstract and section summaries rather than the full proof in Appendix A. Is it solid enough to carry an introduction's motivating claim, and is there a better-established citation for the same point?

### S1b — Is there a genuine impossibility result we should know about?

Our original understanding was that a formal bound on k-stage attribution exists. Having failed to find one, we would like to know whether the supervisor is aware of such a result. If attribution across cascade stages is provably underdetermined in some regime, we would rather build that into the benchmark's design from the start than discover it in week 9.

### S2 — Is intervention displacement legitimate ground truth, or is the argument circular?

This is the sharpest methodological worry. We define a stage's importance by what happens when we remove it, then grade attribution methods on how well they predict that. A skeptical reviewer could argue we have simply defined the answer to be our own measurement, and that a method scoring well has only learned to imitate our intervention procedure rather than recovering anything true.

We believe the causal framing defends this — an intervention is a counterfactual measurement, not another attribution heuristic — but we would like help making that argument rigorously, and knowing where it genuinely does not hold.

### S3 — Is the benchmark alone a sufficient contribution?

Our primary contribution is the benchmark plus an evaluation of four *existing* methods adapted to the multi-stage setting. Is that sufficient for the project's standards, or does it need a novel attribution method of its own? If the latter, the inclusion-vs-ordering decomposition is the strongest candidate for novelty and should be positioned accordingly from the start.

### S4 — Is our distinction from Agarwal et al. (BioNLP 2026) defensible?

Having read it (see 12.2), their pipeline instruments four ablated stages with per-stage diagnostic metrics but does not attribute a *specific document's* final rank to a stage, and does not evaluate competing attribution methods against ground truth.

Our proposed distinction is per-stage **performance measurement** (theirs) versus per-document, per-stage **credit assignment** validated against planted and interventional ground truth (ours). Does that hold up to a skeptical reviewer, or is it too fine a line? We would rather sharpen the framing now than defend it at submission.

### S5 — Is the scope realistic for 10 weeks?

Full scope is 4 retrievers × 4 methods × 3 interventions × 3 seeds × 2 query sets, plus 3 synthetic settings × 9 α values, plus 4 extension experiments. With three people and free-tier compute this is ambitious. Which of E5–E8 does the supervisor consider essential versus optional? The roadmap's MVP is one pipeline (BM25→cross-encoder) + linear-mix synthetic + 2 methods + E1/E2 — **is the MVP acceptable as a floor if the extensions do not land?**

### S6 — Guidance on T1 (the `randomize_stage1_scores` problem)

Of the four candidate resolutions in T1, we would value a view on whether the null-control framing (option d) is a legitimate finding or reads as avoiding the problem, and whether moving to a score-fusion pipeline definition (option b) makes the study more or less representative of real deployed systems.

### S7 — Should synthetic distributions be calibrated to real ones?

Calibrating synthetic score distributions to the measured BM25/MiniLM distributions makes the synthetic setting more realistic, but arguably weakens the claim that it is a *controlled* setting independent of any particular real pipeline. Which is the better trade-off for a benchmark paper?

### S8 — Is department GPU access available?

Dense retriever indexing (DPR, ColBERT, SPLADE) and cross-encoder reranking across 97 queries × 1,000 candidates × 4 retrievers × 3 interventions × 3 seeds is the dominant compute cost. If institutional compute is available it materially changes what we can attempt, and the decision is better made now than in week 6.

---

## 12. Related work — annotated, with citations verified

All thirteen roadmap citations were checked against primary sources on 2026-10-05. Eleven are accurate. **Two are materially mischaracterized and both changes are in our favour** — details in 12.1 and 12.2, which should be read before the related-work section of the report is drafted.

### 12.1 Correction: The Deterministic Horizon does not bound what we are doing

**Roadmap says:** "Formal impossibility bound for k-stage attribution" with "impossibility bound implications."

**What the paper actually says.** Dongxin Guo, *The Deterministic Horizon: Impossibility Results as Design Specifications for Trustworthy AI Systems* ([arXiv 2605.23024](https://arxiv.org/abs/2605.23024)) is a multi-domain compendium of impossibility results — reasoning depth, preference learning, auction design, zero-knowledge verification, circuit complexity — of which multi-stage retrieval is one chapter. The relevant result is the **Construct Conflation Impossibility Theorem** (Thm. 4.3, Ch. 4), derived from psychometric measurement-validity theory:

> "Retrieval pipelines with more than one stage cannot be diagnosed by any single score: at least as many independent metrics as stages are mathematically required."

**This bounds single-score diagnosis, not attribution.** It does not say stage-level attribution is impossible. It says a *blended* metric (RAGAS and similar) provably cannot diagnose a multi-stage pipeline, and that a k-stage pipeline needs ≥ k independent, orthogonal metrics.

**That motivates this project rather than limiting it.** For a 2-stage retrieve-then-rerank pipeline the theorem requires exactly two independent stage-level metrics — which is what stage attribution supplies. We should cite it as formal justification for why the work is necessary, in the introduction, not as a caveat in the limitations. Worth confirming the chapter and theorem numbering against the full PDF before citing, and noting it appears to be a thesis rather than a peer-reviewed venue paper.

### 12.2 Correction: Agarwal et al. is a weaker preemption than feared

**Roadmap says:** "Stage-instrumented clinical retrieval — closest existing work."

**What the paper actually is.** Shubham Agarwal, Thomas Searle, Richard Dobson, Ninoslav Majkic, Niko Moller-Grell, *A Deterministic Multi-Stage Retrieval Pipeline for Longitudinal EHR Question Answering*, BioNLP 2026 ([ACL Anthology 2026.bionlp-1.53](https://aclanthology.org/2026.bionlp-1.53/)). It decomposes retrieval into four ablated stages, each instrumented with diagnostic metrics, reporting a 22–23% relative recall gain on clinical data.

**It ablates stages; it does not attribute across them.** Its diagnostic metrics measure *per-stage performance* — how well each stage does its job. It does not measure which stage is responsible for a *specific document's* final rank, and it does not evaluate competing attribution methods against ground truth.

The distinction matters for how we position the contribution: per-stage *performance measurement* (their work) versus per-document, per-stage *credit assignment* evaluated against planted and interventional ground truth (ours). It is the nearest neighbour and must be cited and distinguished explicitly — the overlap is real enough that a reviewer will ask — but it does not preempt us. It is also strong supporting evidence that stage instrumentation is a live practical need. Interesting coincidence worth noting: its first author shares a name with our own Shubham, so cite carefully to avoid confusion.

### 12.3 The four methods' source papers

| Paper | Verified identifier | Role in our work |
|---|---|---|
| Chowdhury et al., *Rank-LIME: Local Model-Agnostic Feature Attribution for Learning to Rank*, ICTIR 2023 | [10.1145/3578337.3605138](https://dl.acm.org/doi/10.1145/3578337.3605138) | Direct basis for method 4. Single-stage baseline; confirms no multi-stage handling exists. |
| Heuss, de Rijke & Anand, *RankingSHAP: Listwise Feature Attribution Explanations for Ranking Models* | [arXiv 2403.16085](https://www.alphaxiv.org/overview/2403.16085); reference implementation at [MariaHeuss/RankingShap](https://github.com/MariaHeuss/RankingShap) | Shapley axioms for method 3; alternative basis for method 4. **Roadmap lists "SIGIR 2025" — venue unconfirmed, the preprint is 2024. Verify before citing.** |
| DeYoung et al., *ERASER*, 2020 | Established | Source of comprehensiveness and sufficiency, which we adapt from rationale extraction to stage removal. The adaptation is ours and should be described as such. |
| Câmara & Hauff, ECIR 2020 | Established | Probing methodology for BERT rankers; informs the perturbation design in method 4. |

A reference implementation existing for RankingSHAP is worth acting on — Shubham should check whether it can be adapted directly rather than reimplemented.

### 12.4 Gap-establishing surveys

| Paper | Verified identifier | Why cited |
|---|---|---|
| Saha, Majumdar & Mitra, *Explainability of Text Processing and Retrieval Methods: A Survey* | [arXiv 2212.07126](https://arxiv.org/abs/2212.07126); ACM CSUR version [10.1145/3801957](https://dl.acm.org/doi/10.1145/3801957) | Most recent survey; confirms the cascade gap. Roadmap's "2026" is consistent with the CSUR version, but the preprint is 2022 — cite the CSUR record. |
| Anand et al. (2022) survey + SIGIR 2023 tutorial | Established | ExIR taxonomy; names "retrieval cascades" as an open challenge. The load-bearing citation for our gap claim. |
| Thakur et al., *BEIR*, NeurIPS 2021 | Established | Cross-paradigm benchmark methodology; dataset source for E8. |

### 12.5 Adjacent systems

| Paper | Verified identifier | Relationship |
|---|---|---|
| *RAGXplain: From Explainable Evaluation to Actionable Guidance of RAG Pipelines* | [arXiv 2505.13538](https://arxiv.org/abs/2505.13538) | Component-level RAG diagnostics — the same instinct one level up the stack. Our debugger is the ranking-level analogue. Confirmed accurate. |
| *WSDM Cup 2026 Multilingual Retrieval: A Low-Cost Multi-Stage Retrieval Pipeline* | [arXiv 2602.16989](https://arxiv.org/abs/2602.16989); [10.1145/3773966.3778021](https://dl.acm.org/doi/10.1145/3773966.3778021) | Stage ablation in a deployed competition pipeline. **Roadmap attributes it to "Hao & Wang" — author list unconfirmed, verify before citing.** |
| ExDocS (2021) | Established | "Why is document X at rank Y" — single-stage framing of our question. |

### 12.6 Worth adding to the review

Three items surfaced during verification that are not in the roadmap and look relevant:

- *Rectify: An Interactive Workbench for Post-Evaluation RAG Diagnosis, Repair, and Verification* ([arXiv 2609.16764](https://arxiv.org/html/2609.16764)) — overlaps our debugger deliverable; check before building to avoid reinventing it.
- *RAGExplorer: A Visual Analytics System for the Comparative Diagnosis of RAG Systems* ([arXiv 2601.12991](https://arxiv.org/html/2601.12991v1)) — comparative diagnosis across systems, close to our cross-paradigm analysis (E4).
- *ShaRP: Explaining Rankings and Preferences with Shapley Values* ([10.14778/3749646.3749682](https://dl.acm.org/doi/10.14778/3749646.3749682)) — another Shapley-for-rankings formulation; may inform the document-level value function question (T13).

---

## 13. Summary

We are building the first benchmark for attributing responsibility between stages of a multi-stage retrieval pipeline. The core difficulty is the absence of ground truth, which we address two ways: synthetic pipelines where the answer is planted by construction, and interventions on real pipelines where the answer is measured causally by breaking a stage and observing displacement. Four attribution methods are then graded on both, and the agreement pattern between the two grades is the scientific payload.

The shared infrastructure is complete and has been through two review rounds. The experimental work has not begun.

The highest-priority open items are **T1** — `randomize_stage1_scores` is a no-op under a pure cascade, now parameterized and warned about in code but still needing a substantive decision before E2 runs — and **T2**, pinning the synthetic generation parameters before any synthetic data is produced. **T6** matters nearly as much: setting `v({S2}) = 0` biases Shapley toward Stage 1 by `½·v({S1})` by construction, which would corrupt one full column of the comparison table. **T10** (normalized-displacement definition) and **T3** (gating effects being structurally unobservable in a top-10 population) most affect how results can be interpreted.

On the literature, two roadmap citations were mischaracterized and both corrections help us: the Deterministic Horizon bounds single-score *diagnosis* rather than attribution, so it motivates the work instead of limiting it, and Agarwal et al. ablate stages without attributing across them, so the novelty claim is safer than assumed. Three unlisted and apparently relevant papers were also found — see 12.6, particularly *Rectify*, which overlaps the debugger deliverable and should be read before Chaitanya builds it.

Worth noting that T1 was found independently by two people working from different directions, and that both review rounds caught real bugs that would have silently corrupted results rather than crashing. That is the data contract and test suite earning their keep. The same scrutiny should be applied to the attribution methods and interventions as they land, since those have no equivalent safety net yet.
