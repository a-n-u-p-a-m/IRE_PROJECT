# Counterfactual Attribution and Budget-Aware Diagnosis of Cascaded IR

Shared layer for the CS4.406 project. Written from scratch against the current
proposal: four worlds, three effect types, conditional invariants.

**This is the common code only.** Each member's experiment runners live in their
own `experiments_mN/` directory and import from here. `ROADMAP.md` has the
ordered step list and who does what.

```bash
pip install -r requirements.txt
pytest                                    # 207 tests, ~10s, no GPU, no index
python -c "from shared.fixtures import smoke_test; smoke_test()"
```

---

## What the code computes

```
Corpus ──Stage 1 (s1)──▶  C_q = TopKc(U_q, s1)  ──Stage 2 (s2)──▶  top-k
                          D_q = U_q \ C_q
```

For a pointwise cross-encoder Stage 2 never reads `s1`, so for a fixed candidate
set the final rank of a document contains no `s1` term at all. **Stage 1's only
causal channel into the output is membership in `C_q`** — except under a budget
constraint, where Stage-1 rank decides who survives the cut and becomes causally
active again. That second case is what world D is for, and
`tests/test_worlds.py` pins both halves.

| world | ranking | isolates |
|---|---|---|
| A `π_act` | `Sort(C_q, s2)` | the actual cascade |
| B `π_no2` | `C_q` in Stage-1 order | replacing Stage 2, **gate preserved** |
| C `π_no1` | `Sort(U_q, s2)` | removing the gate within a fixed universe |
| D `π_t` | `Sort(retained ∪ promoted, s2)` | budget-constrained reallocation |

```
E_rerank(q,d) = U_d(π_act) − U_d(π_no2)
E_gate(q,d)   = U_d(π_act) − U_d(π_no1)
Δ_t(q,d)      = U_d(π_t)   − U_d(π_{t=1})
```

Signed, bounded, and **not** an additive decomposition — each compares against a
different reference world, so `E_gate + E_rerank ≠ U_d(π_act)`.
`effects.check_additivity` demonstrates that rather than enforcing it.

World B ranks `C_q`, not `U_q`. Sorting the universe by `s1` would remove the
reranker *and* the gate at once, confounding the two estimands.
`worlds.verify_isolation` asserts this on every query.

---

## The three invariant sets

They are not the same, which is why `schema.validate_effect` dispatches on
`effect_type`. One unconditional validator would either reject correct `Δ_t`
rows or wave through incorrect `E_gate` rows.

**E_rerank** — `d ∉ C_q ⟹ effect = 0` exactly. Both worlds rank the same set.

**E_gate** — `d ∈ C_q ⟹ effect ≥ 0`; `d ∉ C_q ⟹ effect ≤ 0`. Holds for a
*pointwise* Stage 2 under a *monotone* rank utility. A listwise reranker
re-scores survivors when the set changes, so validate it with
`pointwise=False` and say in the write-up that the estimand changed.

**Δ_t** — four membership strata, three with fixed signs:

| stratum | at t=1 | at t<1 | sign |
|---|---|---|---|
| retained (top ⌊tB⌋ of C_q) | ranked | ranked | **either** |
| removed (C_q, below the cut) | ranked | dropped | ≤ 0 |
| selected outsider (from D_q) | absent | ranked | ≥ 0 |
| unselected outsider | absent | absent | = 0 |

---

## Why every statistic is stratified

Three of four `Δ_t` strata have signs fixed by allocation membership, and
`sign(E_gate)` is fixed by survivorship. So a predictor that does nothing but
detect the partition earns a large **pooled** correlation while predicting no
magnitude at all.

`metrics.evaluate_within_strata` returns one row per stratum and no pooled row.
`metrics.pooled_fidelity` exists but labels its own output `pooled_secondary`
and attaches the warning. `baselines.py` gives you the bars:

| baseline | what it knows | what it demonstrates |
|---|---|---|
| `zero` | nothing | most documents barely move; beating it is not free |
| `membership_only` | survivor or not | scores ρ > 0.3 pooled against `E_gate` knowing no magnitude |
| `stratum_mean` | which of the four strata | the same trap on the budget experiment |

`tests/test_metrics.py` demonstrates both traps on generated data.

---

## The gain map is explicit and dataset-specific

In TREC Deep Learning passage ranking **grade 1 means "Related"** — on topic,
does not answer the query. The usual `2^rel − 1` awards it a gain of 1. Because
every estimand here is a *difference* of utilities, that error does not cancel:
it inflates every effect in which a merely-related passage moves.

So the map lives in the frozen config, not in a formula:

```yaml
gain_map: {0: 0.0, 1: 0.0, 2: 3.0, 3: 7.0}
```

An unmapped grade raises rather than silently scoring zero. The headline NDCG
may use official conventions instead; if so both maps are recorded and the
difference gets a sentence in the paper.

---

## Layout

```
config/base.yaml     the design freeze, hashed to a 12-char fingerprint
shared/              17 modules, no torch/pyserini imported at module level
tests/               207 tests
ROADMAP.md           ordered steps, owners, and the non-blocking protocol
```

`shared/` in dependency order: `schema`, `config`, `utility`, `ordering`,
`provenance`, `qrels`, `universe`, `cache`, `allocation`, `worlds`, `strata`,
`effects`, `diagnosis`, `metrics`, `baselines`, `synthetic`, `fixtures`.

Six tables are the interface between the three streams (proposal §14):

```
run           (query_id, doc_id, system, rank, score, checkpoint)
universe      (query_id, doc_id, sources)
qrels         (query_id, doc_id, relevance, judged)
stage2_cache  (query_id, doc_id, model, score)
effect        long format with effect_type; sign checks dispatch on it
feature       predictor inputs, kept strictly apart from targets
```

---

## What the code refuses to do

Each refusal is a way the frame breaks silently — producing results that still
run, still look coherent, and mean nothing.

| refusal | why |
|---|---|
| a Stage-1 system that is not a universe source | `C_q` can escape `U_q` |
| `M ≤ max(K_c)` | `U_q == C_q`, so `E_gate` is identically zero |
| `qrels.unjudged: nonrelevant` | fabricates the evidence `E_gate` rests on |
| a missing Stage-2 score | the cache is incomplete and `π_no1` would be truncated |
| a t grid without 1.0 | `Δ_t` has no baseline |
| an allocation that does not total exactly B | turns a budget comparison into a compute comparison |
| a document collection with no declared segmentation | a cross-encoder cannot read a long document, and passage-level calls are what must be budgeted |
| convenience selection of the reference subset | small `U_q` means the retrievers agreed, which is where gate effects are smallest |
| a non-decreasing-violating gain map | the utility stops being monotone in relevance |
| rebuilding or re-fingerprinting a frozen universe | every earlier effect becomes incomparable |
| a pooled `E_gate` headline | there is no API for one |

---

## The synthetic arm

```
y                           latent relevance
rel                         graded label from fixed normal quantiles
f1 = a1·y + √(1−a1²)·z      Stage-1 preference
f2 = a2·y + √(1−a2²)·w      Stage-2 preference
s1 = f1 + σ1·ε1             ε1 ⊥ ε2
s2 = f2 + σ2·ε2
```

`s1` and `s2` correlate because both depend on `y` — a **common cause**, never
communication. `s2` receives neither `s1` nor `rank1(d)`.
`test_stages_correlate_only_through_the_common_cause` residualises both on `y`
and asserts the partial correlation collapses to noise, so a future edit that
introduces a coupling fails loudly.

Six regimes, plus two stress manipulations. `apply_gate_stress(..., far=True)`
buries relevant documents well below the cutoff on purpose: that is the
documented false-negative mode of a narrow boundary probe, generated so the
error rate can be *measured* rather than discovered.

---

## H7 and why its sample size comes first

Every other result here is internal. H7 is the only one that points outward:
the diagnosis comes from the reference worlds, the outcome comes from the
budget intervention, and those are genuinely different interventions — which
they would not be if the "repair" were simply widening `K_c`.

It runs only on the reference subset, so it is also the result most likely to
come back null for want of sample size. `diagnosis.required_reference_size`
estimates the needed size on synthetic data **before** GPU budget is committed;
if nothing reaches the target power, that is worth knowing in week one rather
than at the end.

One caveat the frozen rule carries: `R_q` is an absolute value, so a query the
reranker *damages* earns the same label as one it helps, with the opposite
repair. `query_diagnosis` records `rerank_signed` and `reranker_harms` so the
two can be separated without touching the labelling rule.

---

## Known edge in the labelling rule

The rule uses `G_q > R_q` for gate-sensitive but `R_q >= max(G_q, 0)` for
reranking-sensitive, so an exact tie breaks towards reranking-sensitive. Ties
are vanishingly unlikely on real graded data, but the asymmetry is arbitrary and
should be a recorded choice rather than an artefact of which comparison happened
to be written strictly. `test_toy_query_has_equal_gate_and_rerank_sensitivity`
pins the current behaviour.
