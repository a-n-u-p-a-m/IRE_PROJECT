# Revised Project Plan

**Project:** Reachability and Ordering — Architecture-Dependent Stage Attribution in Multi-Stage Retrieval
**Course:** CS4.406 — Information Retrieval & Extraction
**Team:** Anupam Dwivedi, Shubham Paliwal, Chaitanya K
**Date:** 2026-10-05
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
| C7 | **Interactive retrieval debugger** — query → per-stage rankings → per-document reachability and ordering diagnostics → stage toggle | Engineering | Check *Rectify* (arXiv 2609.16764) before building; it overlaps. |

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
| **Retrievers** | BM25, DPR, ColBERT, SPLADE | BM25, DPR, Contriever (ColBERT only if compute allows) — **SPLADE dropped** | Freed budget buys the architecture axis, which tests the core claim |
| **Architectures** | Pure cascade only (implicit) | Cascade + fusion (RRF) + permeable gate (adaptive re-ranking) | Cascade is the degenerate limiting case, not the canonical one |
| **Primary synthetic setting** | `linear_mix` | `gated` — `linear_mix` reclassified as a **fusion-architecture control** | `linear_mix` models fusion, not cascade |
| `randomize_stage1_scores` | Stage 1 ground truth | **Null control**; its zero result is evidence for RQ1 | Provably a no-op in a pure cascade (T1) |
| `remove_gating` | Clean gating measurement | **Confounded**; must be reported as such | Widening K degrades the reranker (Jacob et al., ReNeuIR 2025: scaling K made Recall@10 worse than retrieval alone in ~50% of settings), so displacement mixes de-gating with Stage 2 degradation |
| **Inclusion-ordering method** | A decomposition | A **detector** — ordering component pinned to zero in a cascade; nonzero output diagnoses fusion, Stage 2 truncation or tie-break leakage | Follows from §1 |
| **Deterministic Horizon citation** | A limitation on what we can claim | **Motivation** — the Construct Conflation theorem requires ≥ k independent metrics for k stages, which is what this work supplies | Roadmap mischaracterised the paper |
| **Benchmark protocol claim** | Primary contribution, "first of its kind" | A method adopted with citation | Preempted by Pothuru (2026) |

Unchanged: the corpus and query sets (§1.2), candidate set size and its ablation (§1.3), reranker configuration (§1.4), score normalisation storage (§1.6), the four attribution methods (§1.7), displacement convention, bootstrap CI protocol, and the vertical-slicing team structure.

---

## 5. Experiment plan

### Core

| # | Experiment | Tests | Owner |
|---|---|---|---|
| **E0** | **Degeneracy confirmation** — randomise Stage 1 scores within a fixed candidate set on the cascade; verify displacement is identically zero for all documents | RQ1. Cheap, requires no reranking (scores are unchanged by construction), and is the empirical anchor for C1 | Anupam |
| E1 | Synthetic recovery: four methods × `gated` (primary) and `adversarial`; `linear_mix` run separately as the fusion control | RQ3 | Shubham |
| E2 | Interventional recovery on the cascade: four methods vs `remove_stage2` (primary ground truth) and `remove_gating` (confound-controlled) | RQ3 | Shubham |
| E3 | Architecture sweep: repeat E2 on fusion and permeable-gate pipelines | **RQ4 — the core novelty experiment** | Chaitanya + Shubham |
| E4 | Paradigm sweep: repeat E2 across BM25, DPR, Contriever | RQ4 | Shubham |
| E5 | Necessity/sufficiency characterisation: distribution of reachability contingency vs ordering dependence per document, per architecture | RQ2 | Anupam |

### Extension

| # | Experiment | Tests | Owner |
|---|---|---|---|
| E6 | Candidate set size (K = 100, 500, 1000) — **free if Stage 2 scores are cached**, since smaller K is a subset of the K=1000 run | Gating boundary sensitivity | Chaitanya |
| E7 | Query-type slicing across the 97 annotated TREC-DL queries | Does stage dependence vary by query type | Chaitanya |
| E8 | Reranker strength (MiniLM vs MonoT5) | Does a stronger Stage 2 shift the balance | Anupam |
| E9 | Cross-domain (one BEIR subset) | Generalisation | Chaitanya |

### Minimum viable project

One cascade (BM25→MiniLM) + one fusion pipeline (RRF over BM25+DPR) + `gated` synthetic + two methods + `remove_stage2` as ground truth + **E0**.

That alone establishes RQ1 and RQ2 and demonstrates architecture-dependence, which is the novel claim. Everything else is extension. E0 in particular is nearly free and delivers C1.

---

## 6. Work division

| | Anupam | Shubham | Chaitanya |
|---|---|---|---|
| **Shared** | `schema.py`, `reranker.py`, `logger.py` ✅ | `evaluation.py` ✅ | query-type annotations |
| **Retrievers** | BM25 | DPR, Contriever (ColBERT only if compute allows) | Fusion (RRF over BM25+DPR) |
| **Architectures** | Cascade (reference), **adaptive re-ranking (permeable gate)** | — | **Fusion** |
| **Methods** | rank-delta, inclusion-ordering | stage-Shapley, reranker-LIME | — |
| **Synthetic** | all settings; `gated` first | — | — |
| **Experiments** | E0, E5, E8 | E1, E2, E4 | E3, E6, E7, E9 |
| **Other** | MonoT5 wrapper | **Literature review — now urgent, see §9** | Visualisation, debugger, **report + presentation lead** |

**Changes from the original division.** Anupam drops SPLADE and picks up adaptive re-ranking instead, plus E0 and E5. Shubham drops ColBERT to a stretch goal and substitutes Contriever. Chaitanya's hybrid retriever is promoted from "another paradigm" to "the fusion architecture," which makes it load-bearing for E3 rather than supplementary.

**Critical path.** Anupam's `gated` synthetic data by end of week 2 unblocks Shubham's E1 and Chaitanya's visualisation work. Chaitanya's fusion pipeline is now on the critical path for E3, the core novelty experiment, so it should move earlier than the original week 3–6 slot.

---

## 7. Hardware and compute requirements

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
| E6 (K = 100, 500, 1000) | **Free** — smaller K is a subset of the cached K=1000 scores |
| E8 MonoT5 (220M params, seq2seq, ~30–60 pairs/sec) | ~1.5–2 hours |
| `remove_stage2` | **Free** — reuses Stage 1 ranks, no reranking |
| E0 degeneracy confirmation | **Free** — scores are provably unchanged |

Total recurring GPU need is roughly **4–6 hours**, which free tier handles comfortably. One-time indexing is the risk.

### Minimum viable hardware

**A laptop plus free Colab is sufficient for the MVP.** Synthetic generation needs no GPU at all. BM25 runs on CPU with a 2.6 GB prebuilt index. MiniLM reranking of 97 queries runs on free T4 in minutes, or on CPU overnight. The fusion architecture adds nothing — it reuses BM25 and DPR output.

### Recommendations

1. **Ask about institutional compute now** (supervisor question S8). IIIT-H lab or cluster access would remove the ColBERT constraint entirely and is worth asking about in week 2, not week 6.
2. **Pre-commit the ColBERT decision.** Agree now that ColBERT is attempted only with institutional compute, and that Contriever is the default. Deciding under pressure in week 6 is how projects lose a fortnight.
3. **Persist indexes outside the session.** Colab storage is ephemeral. Mount Google Drive (15 GB free; ~100 GB for a small monthly fee) or use institutional storage. Re-downloading a 13 GB index every session will dominate everyone's time.
4. **Cache Stage 2 scores keyed by `(query_id, doc_id, reranker)`.** The reranker's score is independent of Stage 1 entirely — which is the whole point of §1 — so every architecture, every K ablation and every intervention that preserves the candidate set can reuse the same cached scores. This single optimisation makes E6 free and cuts the architecture sweep substantially.

---

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Literature velocity** — three closely related papers appeared in six months (CausalFlow May 2026, Causal Agent Replay June 2026, Pothuru Aug 2026) | High | Forward-citation chaining on Rank-LIME, RankingSHAP and the Anand survey **in the next two weeks**, not at week 8. C1 and C2 are structural claims that survive most new empirical papers. |
| ColBERT indexing infeasible on free tier | High | Pre-committed Contriever fallback (§7). Three paradigms is sufficient. |
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

The central finding available to us is that this question is **architecture-dependent**: in a pure cascade, Stage 1's contribution collapses to candidate-set membership and the conventional α-decomposition is undefined, while under score fusion and permeable-gate architectures the full attribution question is live. That claim is structural rather than empirical, which makes it robust to the fast-moving literature around us, and it was not in the original roadmap — it emerged from interrogating why the reranker never reads the retriever's scores.

The infrastructure is built and reviewed. The next two weeks are the `gated` synthetic generator, the BM25 index, E0, and the decisions in §9.
