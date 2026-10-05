# Revised Project Plan

**Project:** Reachability and Ordering — Architecture-Dependent Stage Attribution in Multi-Stage Retrieval
**Course:** CS4.406 — Information Retrieval & Extraction
**Team:** Anupam Dwivedi, Shubham Paliwal, Chaitanya K
**Date:** 2026-10-05 (scope reduced same day — see §5)
**Supersedes:** the problem framing in `roadmap_final.md`. That document's shared design choices (§1.1–1.11) remain in force except where §4 below overrides them.
**Companion:** `docs/project_overview_and_open_questions.md` — holds the detailed methodology, the four methods, the open doubts T1–T14 and the supervisor questions S1–S8.

---

## 1. Problem statement

Multi-stage retrieval is universal in deployed search and RAG systems: a cheap retriever shortlists candidates from a large corpus, and an expensive reranker reorders the shortlist. When a document ends up where it does, practitioners need to know which stage is responsible, because the remedies are disjoint — raise K, swap the embedding model, retrain the reranker, change fusion weights.

Existing explainability methods for ranking explain a *single* ranker. The multi-stage literature treats stages as a **cost/quality allocation** problem and never as a **credit-assignment** problem. The closest work instruments pipeline stages with aggregate per-stage diagnostics but never attributes an individual document's outcome to a stage.

### Why the obvious formulation is ill-posed

The natural formulation — *decompose a document's final position into a Stage 1 share and a Stage 2 share summing to one* — **is undefined for a standard cascade.**

In a pure cascade:

```
final_rank(d) = |{d′ ∈ C(q) : s₂(q,d′) > s₂(q,d)}| + 1 ,    C(q) = top-K of D by s₁
```

This is invariant to `s₁` once `C(q)` is fixed. A document's Stage 1 rank being 4 or 847 has **provably zero** effect on its final position. Stage 1 influences the output through exactly one channel: **membership in `C(q)`**.

Two consequences:

1. **"Stage 1's ordering contribution" is exactly zero**, not merely small. Any method reporting a continuous Stage 1 ordering share on a cascade is reporting an artifact.
2. **The two stages contribute in incommensurable currencies.** Stage 1 determines what is *reachable* (necessity); Stage 2 determines *ordering among the reachable* (displacement). There is no exchange rate between them, so constraining them to sum to 1.0 manufactures a trade-off that does not exist.

The α-decomposition framing is valid — but it is imported from **score-fusion** architectures, where `final = f(s₁, s₂)` and α genuinely parameterises the system. It does not transfer to cascades.

This also explains why existing Shapley-propagation machinery does not apply. Chen, Lundberg & Lee (*Nature Communications* 2022) propagate Shapley values through a series of models under the assumption of function composition `f = h_k ∘ ⋯ ∘ h₁`. A cascade's top-K cut is a **set-selection operation, not function composition on a fixed input space**, so that framework cannot represent cascade gating.

---

## 2. Research questions

**RQ1 — Well-posedness.** Under which pipeline architectures is per-document stage attribution well-posed? Is Stage 1's contribution in a pure cascade reducible to candidate-set membership, and can this be confirmed empirically — randomising Stage 1 scores within a fixed candidate set should yield identically zero displacement?

**RQ2 — Measurement.** Can per-document stage contribution be measured as a **pair of distinct counterfactual quantities** — *reachability contingency* (how close was this document to exclusion) and *ordering dependence* (how far would it fall without the reranker) — rather than a zero-sum decomposition? Watson et al. (UAI 2021) establish necessity and sufficiency as formally non-complementary explanation measures.

**RQ3 — Method recovery.** Which of four attribution methods recovers these quantities, judged against planted synthetic truth and measured interventional displacement? Where do the two verdicts disagree, and does the disagreement localise to a known failure mode?

**RQ4 — Architecture and paradigm dependence.** Do the answers to RQ1–RQ3 change across pipeline **architectures** (pure cascade, score fusion, permeable gate) and across retrieval **paradigms** (sparse, dense, late-interaction or learned-sparse)?

---

## 3. Contributions, with honest novelty assessment

A targeted adversarial literature search was run on 2026-10-05. Ratings reflect what it found and did not find.

| # | Contribution | Novelty | Basis for the rating |
|---|---|---|---|
| C1 | **Cascade degeneracy result** — Stage 1's contribution reduces to membership; α-decomposition is undefined for cascades; confirmed empirically by zero displacement under Stage 1 randomisation | **High** | No paper found stating this. Exists as practitioner folklore ("recall ceiling") in blogs and vendor guides, never formalised as an attribution impossibility. Has a technical hook: Chen/Lundberg/Lee's composition assumption fails for set selection. |
| C2 | **Architecture-dependence of attribution** — the answer and its *type* change across cascade, fusion and permeable-gate topologies | **High** | Nothing found. |
| C3 | **Per-document attribution inside the ranking stack** | **Moderate–high** | Pothuru (2026) attributes per-hop in agentic RAG; Agarwal et al. (2026) report aggregates. Nobody does per-document within retriever→reranker. |
| C4 | **Necessity/sufficiency measurement model for retrieval stages** | **Moderate–high** | Watson et al. supply the theory; no application to retrieval pipelines found. |
| C5 | **Benchmark of four methods** across architectures and paradigms, with bootstrap CIs | **Moderate** | Valuable, but see the warning below. |
| C6 | **Order-independent alternative to cumulative ablation** — cumulative ablation (current practice, including Agarwal et al.) yields order-dependent credit; stage-Shapley does not | **Moderate** | A concrete, nameable improvement over the closest existing work. |
| C7 | **Interactive retrieval debugger** — query → per-stage rankings → per-document reachability and ordering diagnostics → truncation slider | Engineering | Check *Rectify* (arXiv 2609.16764) before building; it overlaps. |
| C8 | **Reranker score calibration study** — how miscalibrated cross-encoder scores are, whether calibration transfers across query sets, and what it buys downstream | **Moderate–high** | Searching found only blog posts and vendor docs. Same "practitioner pain, no academic supply" signature as the main gap. Methods are standard; the measurement study and downstream-utility demonstration are the contribution. |

### Warning: the original roadmap's primary contribution is the weakest part

`roadmap_final.md` positions **the intervention-based benchmark protocol** as the primary contribution. That claim can no longer stand.

Pothuru, *When Failures Propagate: Causal Failure Attribution in Agentic Retrieval-Augmented Generation* (arXiv 2608.20627, **August 2026**) already provides an interventional benchmark for multi-component RAG: ground truth by corrupting a trace prefix at a known stage, regenerating downstream, and scoring competing diagnosis methods against the injected intervention. That is our methodological template, published two months ago, and arguably in a stronger form — injected faults at a known location on a real system combine both of our ground-truth routes into one.

**Action required:** remove any claim of being "the first ground truth for stage attribution." The protocol becomes a method we adopt with citation; the contribution is the axis we apply it to (the ranking stack), the granularity (per-document), and C1–C2, which are structural claims that survive papers like this.

Pothuru does not touch retriever-vs-reranker and does not do per-document attribution, so the project is not scooped. But the question *"what is new relative to Pothuru?"* must have a prepared answer.

### Papers that must be cited and distinguished

| Paper | Why it threatens us | How we distinguish |
|---|---|---|
| Pothuru, arXiv 2608.20627 (2026) | Same interventional-benchmark template | Different axis (ranking stack vs agent hops), different granularity (per-document vs per-hop) |
| Chen, Lundberg & Lee, *Nat. Commun.* 13:4512 (2022) | Shapley through a pipeline of models, high-visibility venue | They attribute *input features through* stages; we attribute *to* stages. Their composition assumption fails for set selection — which is our RQ1 |
| Agarwal et al., BioNLP 2026 | Stage-instrumented multi-stage clinical retrieval | Aggregate recall metrics, cumulative (order-dependent) ablation, no per-document attribution, no ground truth for stage responsibility |

### Evidence of practitioner demand, no academic supply

Searches for retriever-vs-reranker attribution return **only blog posts and vendor guides** — including one titled "The Vector Search Was Right. The Reranker Was Wrong." Demonstrated demand with no academic treatment is strong gap evidence and worth citing in the introduction.

---

## 4. What changed from the original roadmap

| Item | Original | Revised | Reason |
|---|---|---|---|
| **Framing** | Decompose responsibility into shares summing to 1 | Measure two incommensurable counterfactual quantities | Shares are undefined for a cascade (§1) |
| **Schema constraint** | `stage_1 + stage_2 = 1.0`, enforced in `schema.py` | Two independent measures; sum-to-one validator relaxed | Category error; needs team sign-off (T7) |
| **Primary experimental axis** | Retriever paradigm | Paradigm **and** architecture | C2 is the novelty; a 4th paradigm only adds a table row |
| **Retrievers** | BM25, DPR, ColBERT, SPLADE | **BM25 only** for the core result; DPR as optional O1 | The degeneracy claim is architecture-level, not paradigm-level. Dropping dense retrieval from the core removes the entire indexing risk |
| **Architectures** | Pure cascade only (implicit) | **Cascade + stage-score interpolation**, swept over the mixing weight λ | Cascade is the degenerate limiting case. Interpolation is free (cached scores), gives a continuum, and plants a known λ on real data. Truncated reranking was tried and rejected — it collapses to the K-ablation for top-10 metrics (§5) |
| **Pipeline count** | Up to 9 (3 paradigms × 3 architectures) | **2, sharing one index** | Scope reduction; see §5 |
| **Fusion architecture** | — | **Dropped** | Another team is doing adaptive hybrid fusion. `linear_mix` is retained as a synthetic fusion control, which makes the architectural point without building the pipeline |
| **Extension** | MonoT5, query-type slicing, BEIR | **Reranker score calibration (X1–X4)**, with the earlier extensions demoted to optional | Calibration is free, unclaimed in the class, and load-bearing: P2 interpolates two score scales, which is meaningless until they are comparable |
| **Primary synthetic setting** | `linear_mix` | `gated` — `linear_mix` reclassified as a **fusion-architecture control** | `linear_mix` models fusion, not cascade |
| `randomize_stage1_scores` | Stage 1 ground truth | **Null control**; its zero result is evidence for RQ1 | Provably a no-op in a pure cascade (T1) |
| `remove_gating` | Clean gating measurement | **Confounded**; must be reported as such | Widening K degrades the reranker (Jacob et al., ReNeuIR 2025: scaling K made Recall@10 worse than retrieval alone in ~50% of settings), so displacement mixes de-gating with Stage 2 degradation |
| **Inclusion-ordering method** | A decomposition | A **detector** — ordering component pinned to zero in a cascade; nonzero output diagnoses fusion, Stage 2 truncation or tie-break leakage | Follows from §1 |
| **Deterministic Horizon citation** | A limitation on what we can claim | **Motivation** — the Construct Conflation theorem requires ≥ k independent metrics for k stages, which is what this work supplies | Roadmap mischaracterised the paper |
| **Benchmark protocol claim** | Primary contribution, "first of its kind" | A method adopted with citation | Preempted by Pothuru (2026) |

Unchanged: the corpus and query sets (§1.2), candidate set size and its ablation (§1.3), reranker configuration (§1.4), score normalisation storage (§1.6), the four attribution methods (§1.7), displacement convention, bootstrap CI protocol, and the vertical-slicing team structure.

---

## 5. Experiment plan (reduced scope)

### Pipelines: two core, sharing one index

The original plan had up to nine pipelines (3 paradigms × 3 architectures). The core result needs **two**, and they share a single BM25 index:

| | Architecture | Construction | Is `s₁` ordering inert? |
|---|---|---|---|
| **P1** | Pure cascade | BM25 top-1000 → MiniLM reranks all 1000; final order = `s₂` alone | **Yes** — provably |
| **P2** | Stage-score interpolation | BM25 top-1000 → MiniLM reranks all 1000; final order = `λ·norm(s₁) + (1−λ)·norm(s₂)`, with λ swept over {0, 0.1, …, 1.0} | **No** — continuously |

### Why interpolation and not truncation

An earlier draft of this plan used *truncated reranking* (rerank only the top-N, leave the tail in Stage 1 order) as P2. **That does not work.** Under the natural merge convention the reranked head occupies final ranks 1…N and the tail falls below it, so for N ≥ 10 no tail document can enter the final top-10. For top-10 attribution, truncating to N is then observationally equivalent to setting K = N — the existing candidate-set ablation, not a distinct architecture.

Truncation is retained, correctly labelled, as part of the gating analysis (E5).

Stage-score interpolation is a genuine second architecture, and it is strictly better on four counts:

1. **`s₁` propagates continuously.** At λ = 0, P2 *is* P1 and Stage 1's ordering is inert. As λ rises, `s₁` scores influence the final order of every document, not just the tail. The λ sweep is the continuum from "Stage 1 ordering irrelevant" to "Stage 1 ordering dominant."
2. **λ is planted ground truth on real data.** We set λ, so we *know* the true mixing weight — on real MS MARCO passages, with real BM25 and real MiniLM scores. This is stronger than either original ground-truth route and closes the gap between them: the synthetic settings have known truth but artificial data, the interventions have real data but no known truth. P2 has both.
3. **It makes `linear_mix` correct rather than a control.** The `linear_mix` synthetic setting models score fusion, which the original plan did not study — so it was a control for an architecture that appeared nowhere. Under P2 it models the real pipeline exactly, and synthetic-vs-real comparison becomes like-for-like.
4. **Still free.** Both `s₁` and `s₂` are already computed for all 1000 candidates. Every λ is arithmetic over cached scores — no additional reranking, no additional indexing.

**Caveat to resolve first: score normalisation.** Interpolating raw `s₁` and `s₂` is meaningless — BM25 is unbounded, cross-encoder outputs are logits. λ is only interpretable once both are on a common scale, which is exactly what the calibration extension provides. **X1–X2 therefore gate P2**, which makes the extension load-bearing rather than optional. Use min-max or z-score normalisation per query as an interim measure so P2 is not blocked, and report λ results under both the interim and the calibrated scales.

**Collision check needed.** Another team is doing *adaptive hybrid fusion*. The distinction we would claim is **stage fusion** (interpolating two stages of one pipeline) versus **retriever fusion** (RRF over a lexical and a dense retriever, which is what "hybrid" normally denotes). That is a real difference, but it is close enough that it should be confirmed with the TA or that team before week 3 rather than argued at the presentation.

**No dense retrieval is required for the core result.** That removes the entire indexing risk identified in §7.

### Core experiments

| # | Experiment | Tests | Cost | Owner |
|---|---|---|---|---|
| **E0** | **Degeneracy confirmation** — randomise Stage 1 scores within P1's fixed candidate set; verify displacement is identically zero for every document | RQ1; empirical anchor for C1 | **Free** (scores provably unchanged) | Anupam |
| **E1** | Synthetic recovery: four methods × `gated` (primary) and `adversarial`; `linear_mix` as a fusion-architecture control | RQ3 | CPU only | Shubham |
| **E2** | Interventional recovery on P1: four methods vs `remove_stage2` (primary ground truth) and `remove_gating` (confound-flagged) | RQ3 | ~2 h once | Shubham |
| **E3** | **λ sweep** — repeat E2 on P2 at λ ∈ {0, 0.1, …, 1.0}; show Stage 1 ordering attribution rising from zero as λ grows, and test whether each method recovers the λ we planted | **RQ4 — the core novelty experiment.** Doubles as planted-truth recovery on real data | **Free** (cached scores) | Chaitanya |
| **E4** | Necessity/sufficiency characterisation: joint distribution of reachability contingency and ordering dependence per document, per architecture | RQ2 | Free | Anupam |
| **E5** | Gating analysis: candidate-set ablation (K ∈ {100, 500, 1000}) and reranker truncation, which are equivalent for top-10 metrics and reported together | Boundary sensitivity; the necessity half of RQ2 | Free (subsets of the cached run) | Chaitanya |

### Extension: reranker score calibration

Not a bolt-on. Attribution asks *how much did Stage 2 contribute*, which is measured against Stage 2's score scale — and a cross-encoder's scores are not probabilities. If that scale is miscalibrated, the attribution is measured against a ruler with uneven markings, which is a candidate explanation for why methods disagree.

Searching found **only blog posts and vendor documentation** on reranker calibration — the same "practitioner pain, no academic supply" signature as the project's main gap.

| # | Experiment | Tests | Cost | Owner |
|---|---|---|---|---|
| **X1** | Measure MiniLM calibration against graded TREC qrels: reliability diagram, expected calibration error | How miscalibrated is Stage 2 in the first place? | Free (reuses `calibration_error`) | Anupam |
| **X2** | Fit Platt scaling, isotonic regression and temperature scaling on DL19; evaluate on DL20 | Can it be fixed, and does the fix transfer across query sets? | Free | Anupam |
| **X3** | Re-run E2 with calibrated Stage 2 scores | Does displacement correlation improve? Does inter-method disagreement shrink? | Free | Shubham |
| **X4** | Use calibrated scores as the common scale for P2's interpolation, and compare λ recovery under calibrated vs min-max normalised scores | **Closes the loop** — λ is only interpretable on a common scale, so calibration is what makes the second architecture well-defined | Free | Chaitanya |

X4 is why this is an extension rather than a second project. P2 interpolates `s₁` and `s₂`, which is only meaningful once both sit on a comparable scale — and cross-encoder scores are not probabilities. So calibration is not a nice-to-have bolted on at the end; it is what makes the second architecture well-defined at all. The core result shows attribution depends on how the stages are composed; the extension supplies the measurement scale that composition requires.

The extension is also strategically useful: calibrated reranker scores are an implicit dependency of at least three other teams' projects (adaptive fusion needs comparable scores to fuse, latency-SLA design needs thresholds, candidate-depth prediction needs a confidence signal).

### Optional, only if time permits

| # | Experiment | Why it is optional |
|---|---|---|
| O1 | Second paradigm — DPR→MiniLM cascade, confirming degeneracy is not BM25-specific | Strengthens generality but needs a dense index, the one real compute risk |
| O2 | Permeable gate via adaptive re-ranking (`pyterrier_adaptive`) as a third architecture | Conceptually the richest architecture, but **check first whether a precomputed MS MARCO corpus graph is downloadable** — building one requires dense embeddings over 8.8M passages and would reintroduce the indexing cost |
| O3 | Query-type slicing over the 97 annotated TREC-DL queries | Nice analysis, not load-bearing |
| O4 | MonoT5 as a stronger Stage 2 | ~2 h GPU; tests whether reranker strength shifts the balance |
| O5 | Cross-domain check on one BEIR subset | Generalisation |

### Minimum viable project

**P1 + E0 + `gated` synthetic + two methods + `remove_stage2`.** One BM25 index, one reranking pass, no dense retrieval. That delivers RQ1, RQ2 and contribution C1 — the degeneracy result, which is the strongest novel claim.

Adding **E3** (free) delivers C2, architecture-dependence, plus planted-λ recovery on real data. Adding **X1–X4** (free) supplies the common score scale E3 depends on. The entire core and extension run on one Lucene index and one cached reranking pass.

---

## 6. Work division

| | Anupam | Shubham | Chaitanya |
|---|---|---|---|
| **Shared** | `schema.py`, `reranker.py`, `logger.py` ✅ | `evaluation.py` ✅ | query-type annotations |
| **Pipelines** | BM25 index + **P1 (cascade)**, Stage 2 score cache | — | **P2 (stage interpolation)** + the λ sweep |
| **Methods** | rank-delta, inclusion-ordering | stage-Shapley, reranker-LIME | — |
| **Synthetic** | all settings; `gated` first | — | — |
| **Core experiments** | E0, E4 | E1, E2 | E3, E5 |
| **Extension** | X1, X2 (calibration measurement + fitting) | X3 (attribution under calibrated scores) | X4 (calibrated scale for P2's interpolation) |
| **Optional** | O1 (DPR), O2 (adaptive re-ranking), O4 (MonoT5) | — | O3, O5 |
| **Other** | — | **Literature review — now urgent, see §9** | Visualisation, debugger, **report + presentation lead** |

**Changes from the original division.** Nobody now owns a dense index as a core deliverable. Anupam drops SPLADE, ColBERT and the adaptive-re-ranking architecture, picking up the BM25 cascade, the score cache, E0, E4 and the calibration measurement. Shubham drops DPR, ColBERT and Contriever from the core and concentrates on the two methods, E1, E2 and the literature review. Chaitanya owns P2 and the λ sweep — which is the core novelty experiment — plus E5's gating analysis, instead of a fourth retriever.

**Critical path, now much shorter.** Anupam's BM25 index and cached Stage 2 scores unblock everything downstream, including Chaitanya's entire λ sweep, which is arithmetic over that one cached run. Anupam's X1–X2 normalisation work also feeds P2, so it should not be left to the end. The `gated` synthetic generator unblocks Shubham's E1 in parallel and needs no index at all. Two artifacts gate the whole project, both deliverable in week 2–3.

**Load concern worth flagging.** Shubham's verticals are now lighter than the others', while his literature review has become more urgent given the Pothuru finding. If the team wants to rebalance, O1 (the DPR cascade) is the natural thing to hand him, since it is the one optional item that strengthens a core claim rather than adding an analysis.

---

## 7. Hardware and compute requirements

> **Under the reduced scope of §5, the core result and the full extension need one 2.6 GB prebuilt Lucene index and one ~5-minute reranking pass on a T4.** Everything else — E0, E3's entire λ sweep, E4, E5 and X1–X4 — is free, because it reads cached Stage 2 scores. The estimates below describe the full original scope and now apply only to the optional items O1–O5.

### Headline: indexing is the bottleneck, not reranking

Reranking the full evaluation set is cheap. MiniLM-L-12 at batch 64 processes roughly 300 pairs/sec on a T4, so one complete pipeline run over 97 TREC-DL queries × 1000 candidates is about **5 minutes**. The expensive part is building corpus indexes.

**Use Pyserini's prebuilt indexes.** Pyserini distributes prebuilt Lucene and dense indexes for MS MARCO passage. Downloading them eliminates most GPU-hours and most risk. Build from scratch only if a prebuilt index for the exact configuration does not exist.

### Storage

| Artifact | Approximate size | Notes |
|---|---|---|
| Corpus text | ~3.2 GB | Required — the reranker needs passage text, not just IDs |
| BM25 Lucene index (prebuilt) | ~2.6 GB | CPU only |
| DPR flat index, fp16 | ~13.5 GB | 8.8M × 768 × 2 bytes. **Will not fit in free-tier RAM** |
| DPR with IVF-PQ compression | ~1–2 GB | Required for free tier; costs some recall |
| Contriever | similar to DPR | |
| ColBERTv2 compressed index | **~25 GB** | The blocker |
| Corpus graph (adaptive re-ranking) | ~1–2 GB | `pyterrier_adaptive` provides a precomputed graph |
| Results parquets | < 1 GB | |
| **Total, without ColBERT** | **~25–30 GB** | Feasible |
| **Total, with ColBERT** | **~50–55 GB** | Not feasible on free tier |

### RAM is the binding constraint

Free Colab provides roughly 12–13 GB RAM. A 13.5 GB fp16 flat dense index does not fit. **Dense retrieval must use a compressed (IVF-PQ) or memory-mapped index** — this is not optional and should be decided before indexing begins, not discovered at load time.

### GPU time estimates (T4)

| Task | Estimate |
|---|---|
| Encoding 8.8M passages with a bi-encoder | 3–8 hours, assuming no prebuilt index |
| ColBERTv2 indexing | Many GPU-hours; **likely infeasible on free tier within a 12-hour session limit** |
| Full cascade run (97 queries × 1000, MiniLM) | ~5 min |
| All three architectures × three paradigms | ~45 min |
| `remove_gating` at K=5000 (4 pipelines, deterministic, no seeds) | ~1.8 hours |
| E3 λ sweep, and E5's K ablation | **Free** — every λ is arithmetic over the cached scores; every smaller K is a subset of them |
| O4 MonoT5 (220M params, seq2seq, ~30–60 pairs/sec) | ~1.5–2 hours |
| `remove_stage2` | **Free** — reuses Stage 1 ranks, no reranking |
| E0 degeneracy confirmation | **Free** — scores are provably unchanged |

Total recurring GPU need is roughly **4–6 hours**, which free tier handles comfortably. One-time indexing is the risk.

### Minimum viable hardware

**A laptop plus free Colab covers the entire reduced scope — core and extension.** Synthetic generation needs no GPU. BM25 runs on CPU with a 2.6 GB prebuilt index. MiniLM reranking of 97 queries × 1000 candidates runs on a free T4 in about five minutes, or on CPU overnight. P2's λ sweep, E0, E4, E5 and all of X1–X4 read the cached scores and need no GPU at all.

Institutional compute (S8) now matters only for the optional items — the DPR cascade (O1), adaptive re-ranking (O2) and MonoT5 (O4). It is no longer a prerequisite for the project's main claims, which is the point of the reduction.

### Recommendations

1. **Ask about institutional compute now** (supervisor question S8). IIIT-H lab or cluster access would remove the ColBERT constraint entirely and is worth asking about in week 2, not week 6.
2. **Pre-commit the ColBERT decision.** Agree now that ColBERT is attempted only with institutional compute, and that Contriever is the default. Deciding under pressure in week 6 is how projects lose a fortnight.
3. **Persist indexes outside the session.** Colab storage is ephemeral. Mount Google Drive (15 GB free; ~100 GB for a small monthly fee) or use institutional storage. Re-downloading a 13 GB index every session will dominate everyone's time.
4. **Cache Stage 2 scores keyed by `(query_id, doc_id, reranker)`.** The reranker's score is independent of Stage 1 entirely — which is the whole point of §1 — so every architecture, every K ablation and every intervention that preserves the candidate set can reuse the same cached scores. This single optimisation is what makes E3's entire λ sweep, E0, E4, E5 and X1–X4 free, and it is the reason the reduced scope needs only one reranking pass.

---

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Literature velocity** — three closely related papers appeared in six months (CausalFlow May 2026, Causal Agent Replay June 2026, Pothuru Aug 2026) | High | Forward-citation chaining on Rank-LIME, RankingSHAP and the Anand survey **in the next two weeks**, not at week 8. C1 and C2 are structural claims that survive most new empirical papers. |
| **Scope reduction leaves the result looking thin** — one retriever, one index | Medium | The λ sweep (E3) produces a *curve* with planted ground truth at every point, which is more evidence than a second paradigm would give. O1 (DPR cascade) is the designated answer if a reviewer or the supervisor asks for paradigm generality, and it is one prebuilt index away. |
| Dense indexing infeasible on free tier | **Now low** | Removed from the core result entirely (§5). Affects only O1 and O2. |
| **Collision with the *adaptive hybrid fusion* team** | **Medium–high** | P2 is stage fusion (two stages of one pipeline), not retriever fusion (lexical + dense via RRF). Defensible, but **confirm with the TA or that team before week 3** |
| O2's corpus graph may not be downloadable | Low | Check before attempting; P2 already provides the non-cascade architecture, so O2 is purely additive |
| `remove_gating` confound not separable | Medium | Report as confounded; use `remove_stage2` as primary ground truth; consider a Stage-2-degradation control that reranks 5000 candidates but measures only the original 1000 |
| Relaxing the sum-to-one constraint breaks teammates' code | Medium | Decide in the week-2 meeting before anyone writes against it |
| Shapley's `v({S2}) = 0` bias (T6) invalidates one comparison column | Medium | Resolve before implementation; consider seeding `v({S2})` from the K=5000 rerank already computed for `remove_gating` |
| Adaptive re-ranking is unfamiliar work | Low | `pyterrier_adaptive` is a working public implementation |
| Methods perform indistinguishably | Low | "Use the cheapest" is a useful practical finding |
| Supervisor rejects the reframe | Medium | Original framing remains runnable; the cost is a weaker contribution. Raise in week 2. |

---

## 9. Decisions required before week 2

### Team

1. **T1** — what replaces `randomize_stage1_scores` as a Stage 1 intervention, or is E0 (null control) the answer? Gates E2.
2. **T2** — pin synthetic generation parameters in `config/base.yaml`: score distributions, inter-stage correlation, query and document counts. Gates all synthetic work and must precede the generator.
3. **T7** — relax the sum-to-one validator, and define the relationship between `gating_attribution`, `ordering_attribution` and `stage_1_attribution`. Touches everyone's code.
4. **T6** — Shapley's `v({S2})` definition. Gates Shubham's implementation.
5. **SPLADE → adaptive re-ranking swap, and ColBERT → Contriever.** Changes two people's verticals.
6. **Stage 2 score caching** as a shared utility (§7, recommendation 4).

### Supervisor

1. **S1** — can we cite the Deterministic Horizon as motivation rather than limitation? Is a thesis solid enough to carry an introduction's motivating claim?
2. **S2** — is intervention displacement legitimate ground truth, or circular?
3. **S3/S4** — given Pothuru (2026), is the reframed contribution sufficient, and is our distinction from Agarwal et al. defensible to a skeptical reviewer?
4. **S5** — is the revised scope realistic? Is the MVP acceptable as a floor?
5. **S8** — is institutional GPU and storage available? This is the single highest-leverage question and determines whether ColBERT is possible at all.

---

## 10. Summary

The project measures, for each document, **which stage's behaviour its final position was contingent on** — not by decomposing a score into additive shares, but by intervening on each stage and observing what moves.

The central finding available to us is that this question is **architecture-dependent**: in a pure cascade, Stage 1's contribution collapses to candidate-set membership and the conventional α-decomposition is undefined. Interpolate the two stages' scores, as hybrid systems routinely do, and Stage 1's ordering becomes live again — continuously, as a function of the mixing weight. Because we set that weight, it is also planted ground truth on real data. That claim is structural rather than empirical, which makes it robust to the fast-moving literature around us, and it was not in the original roadmap. It emerged from interrogating why the reranker never reads the retriever's scores.

The extension asks whether Stage 2's scores are even on a meaningful scale — which the interpolation architecture requires, since mixing two incomparable scales is meaningless. Diagnosis supplies the question; calibration supplies the ruler.

**Scope, after reduction:** two pipelines sharing one prebuilt index, one cached reranking pass, no dense retrieval. The entire core result and the full extension run on a laptop plus free Colab. The two artifacts that gate everything — the BM25 index with cached Stage 2 scores, and the `gated` synthetic generator — are both week 2–3 deliverables, and neither depends on the other.

The infrastructure is built and reviewed. Next: those two artifacts, E0, and the decisions in §9.
