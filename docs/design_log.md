# Design decision log

Append-only. Newest entry at the bottom, one dated block each, so merges never
conflict on anything but the final line.

Every change to the universe, `K_c`, `B`, the `t` grid, the qrels policy, the
utility or gain map, a model checkpoint, or the hashing method gets an entry
here **and** a new config fingerprint. A result produced under one fingerprint
is never combined with one produced under another.

How to read an entry: **what** was decided, **why**, and — where the value
depends on data not yet collected — the **rule** that will decide it later.
Pre-registering the rule is what makes the freeze mean anything; picking a
value now and "confirming" it after seeing the data is tuning with extra steps.

---

## 2026-10-26 — Freeze v1

**Fingerprint:** `46d0807d8ed3` (`config/base.yaml`)

**Decided**

- **Dense Stage 1 is DPR**, `castorini/cosdpr-distil`, trained on MS MARCO.
  Replaces ANCE. Verified locally against Pyserini: TREC-DL19 nDCG@10 0.706 vs
  published 0.725, R@1000 0.822 vs 0.820. The gap is consistent with float16
  exact search over the 89-shard index. BM25's published DL19 nDCG@10 is ~0.506,
  so the lexical-vs-semantic contrast the mismatch axis needs is real.
- Stage-1 systems and universe sources: **BM25 and DPR**. DPR is also the
  alternative retriever for outsider selection.
- Stage 2: pointwise **MiniLM cross-encoder**.
- Data: **TREC-DL 2019 + 2020** on MS MARCO passages — 43 + 54 = **97 queries**.
- **Gain map** gives grades 0 and 1 zero gain. TREC DL grade 1 means "Related",
  not relevant; because every estimand is a *difference* of utilities, treating
  it as relevant inflates effects systematically rather than cancelling.

**Still open — to settle at the Week-1 meeting**

Decide the *value* now:

| decision | value | note |
|---|---|---|
| `tau_document` | | presentation labels only |
| B = `K_c`? | | if no, `t=1` is a candidate-only baseline and every budget result must say so |
| headline NDCG gain map | | official TREC gains, or the attribution map; if they differ, both get reported |
| listwise reranker | out of scope | |

Decide the *rule* now, the value arrives later:

| decision | rule | value from |
|---|---|---|
| headline `K_c` | | Week-2 judged-non-survivor counts |
| `tau_query` | fixed on synthetic, then never tuned | Shubham, end of Week 1 |
| reference-subset size | whatever the power simulation returns | Shubham, end of Week 1 (H2) |
| reference sampling strategy | | |
| SPLADE in/out | | end of Week 2, after the Jaccard check |
| hashing in/out | Chaitanya's written go/no-go criteria | Week 2 |

**Also to settle at that meeting**

- **Do not seed the Stage-2 cache from the existing DPR → MiniLM scores.**
  It saves ~97,000 pairs out of a 0.6–0.9M bill — about four minutes of T4 time —
  and it interleaves two scoring runs everywhere except DPR's own candidate
  sets. BM25's `C_q` partially overlaps DPR's top-1,000, world C mixes by
  definition, and world D's outsiders mix. A BM25-vs-DPR comparison where one
  system's candidates come from a single run and the other's from a blend is
  exactly the asymmetry H3 cannot absorb. Decision: ____
- Throughput measurement must report the **observed mean sequence length**
  beside pairs/sec. If the cross-encoder pads to `max_length` 512 rather than to
  the longest in batch, the bill is several times larger than it needs to be —
  MS MARCO passages average around 55 words. Decision: ____

---

## 2026-10-27 — Week-1 code, and a shared-module correction

**Anupam's Week-1 modules landed:** `data_prep.py`, `retrievers.py`,
`run_stage1.py`, `build_universe.py`, `smoke_pipeline.py`, with 59 tests. The
one Week-1 row still open is the BM25 run itself, which needs Pyserini, a JDK
and the prebuilt index.

**`shared/diagnosis.py`: the power denominator was conditional.**
`required_reference_size` divided detections by the number of *testable* draws.
A reference subset with fewer than two queries in a group cannot be tested at
all — which is an H7 failure, not a draw to be discarded — so the old `power`
answered "given that the subset was testable" and overstated what a real subset
would deliver. It now divides by every simulation drawn, and the conditional
figure is available separately as `power_given_testable`.

`n_usable` was **removed** rather than kept, deliberately. Shubham was already
correcting for this downstream (`power_h7.py` multiplied by
`n_usable / n_simulations`), and the two corrections together would have
cancelled the fix silently. Removing the column made his line raise `KeyError`
instead, which is how it got found and deleted in the same change. His recorded
H7 numbers are unchanged: his old expression came to
`detections / n_simulations`, which is exactly what `power` now is.

**Consequence for H7.** The honest figure is lower than the one reported
before, and with gate-sensitive queries at 2–7% of the synthetic pool the group
test may not reach target power at any feasible reference size. That is a
finding to state in advance, not a problem to tune away: if no size reaches it,
H7 is declared descriptive rather than confirmatory. The continuous
(Spearman) variant is the fallback and is not yet frozen.

**Containment is a tautology inside one build.** `config.py` already refuses
any Stage-1 system that is not also a universe source, so inside a single
`build_universe` call every candidate set is a subset of a union built from
that same run. The check is kept as defence against that config rule being
loosened, and a test now asserts the subset relation so the day it stops
holding is noticed. The check's real use is Week 2 and after: a frozen universe
against a **regenerated** run, where it does fire.

**The `|U_q| > max(K_c)` guard needs two layers.** The config catches a
retrieval depth `M` below the largest `K_c` at load time. It cannot catch
`universe.max_size` capping `|U_q|` below `max(K_c)`, because that depends on
the union's actual size — so that one is caught in `build_universe` after the
union is built. Both produce the same failure (world C equals world A and
`E_gate` is identically zero) and both now have a test.

---

## 2026-10-28 — BM25 run, frozen universe, and three things the numbers revealed

**Week 1 closed.** BM25 top-5,000 over all 97 queries (485,000 rows, matching
the DPR run exactly), universe frozen at fingerprint `46d0807d8ed3`:
`|U_q|` 7,045–9,986, median 9,097, **872,455 total pairs**.

**The Stage-2 bill is 872,455 pairs for the entire project, not per world.**
The cache keys on `(query, doc)` and every world re-ranks a subset of `U_q` —
world A sorts `C_q`, B reuses A's scores, D sorts retained ∪ promoted, C sorts
all of `U_q`. So the union across both Stage-1 systems, all three `K_c`, every
`t` and every outsider policy *is* `U_q`, once. Pending Chaitanya's throughput
figure, this may put full-universe scoring within reach, which would make the
reference-subset sampling unnecessary and remove H2 as a blocker. Decision
deferred to that measurement; it is the first thing to ask him for.

**BM25 and DPR barely agree at depth.** Median overlap 903 documents of 5,000
(Jaccard ≈ 10%); the least-agreeing query shares 14. Good for the project — two
systems selecting the same documents would leave the cross-paradigm comparison
nothing to separate. The decision-relevant number is the Jaccard at each `K_c`,
not at 5,000, and `feasibility_report.py` now computes it.

**Judgment coverage is 1.65% of `U_q` (median 149 judged of ~9,097), minimum
0.26%.** This is a validity threat to `E_gate`, not just a reporting detail.
World C sorts all of `U_q` while world A ranks within `C_q`, and TREC pools were
built from the top-~100 of participating systems, so the head is far denser in
judgments than the tail. Unjudged documents earn 0 gain, so
`E_gate = U(π_act) − U(π_no1)` can come out large and positive because world C's
top-10 is full of zeros rather than because gating helped. A mitigation must be
stated before Stage-2 scoring — restrict the utility to judged documents, or
report both and show the conclusion survives either way. Decision: ____

**A second, worse asymmetry.** The same pooling history means BM25's candidate
set is likely judged more densely than DPR's: lexical runs contributed to the
2019/2020 pools and a modern dense retriever did not. Comparing `E_gate` across
the two paradigms would then compare judgment coverage as much as retrieval
paradigm. `feasibility_report.py` measures this between-system ratio and warns
above 3×. If it fires, the headline cross-paradigm claim needs rework.
Decision: ____

**`shared/qrels.py` and the gain map disagree about "relevant".** Lines 135 and
174 count `relevance > 0`, which includes TREC-DL grade 1; the frozen gain map
scores grade 1 as **0.0**. So a grade-1 non-survivor registers as a relevant
one while excluding it costs no utility — it is not a gate failure. This matters
because the headline `K_c` is to be chosen from the judged-non-survivor count,
and the looser definition inflates it. `feasibility_report.py` uses the gain map
for its primary counts and reports the shared definition beside them so the gap
is visible on real data. The shared helpers should be brought into line; not
done here because it is a shared module and the report makes the discrepancy
observable first.

**Headline `K_c` rule, declared before magnitudes were inspected:** the largest
`K_c` at which *every* Stage-1 system has a median gain-carrying non-survivor
count ≥ 1. Per system, not pooled — a median across both would average a dense
head with a sparse one and describe neither.

**trec_eval's nDCG uses linear gain.** `verify_bm25.py` reproduces the published
BM25 figures (DL19 0.5058, DL20 0.4796) and therefore has to match trec_eval's
definition, which is `gain = rel`, **not** the exponential `2^rel − 1` used by
several other toolkits and by this project's own gain map. Established by
probing `pytrec_eval` directly, not from memory; the exponential form gave
0.0799 against trec_eval's 0.1116 on the first random case. Pinned by tests
that cross-check against the real trec_eval C code where it is installed.

---

## 2026-10-28 (later) — Feasibility results, and the one open decision

**BM25 verified exact.** DL19 nDCG@10 0.5058 / R@1000 0.7501, DL20 0.4796 /
0.7863 — delta 0.0000 on all four against the Pyserini figures. The run is the
canonical one.

**The between-system asymmetry does NOT exist.** Flagged earlier as a possible
headline-breaker; measured, it is a ratio of 1.06 (BM25 `C_q` 51% judged, DPR
48%) and 1.15 at `K_c=1000`. The TREC-DL 2019/2020 pools included neural runs,
so they are not lexically biased. **Retired as a concern.** The
cross-paradigm comparison is not confounded by judgment coverage.

**Leverage confirmed.** Candidate Jaccard 0.16 / 0.14 / 0.11 at
`K_c` 100 / 500 / 1000.

**The grade-1 inflation was 4×–9×**, and `shared/qrels.py` is now fixed.
Measured: at `K_c=100` BM25 showed 28 relevant non-survivors under
`relevance > 0` against 6 under the gain map; at `K_c=1000` DPR showed **9
against 1**. A headline `K_c` chosen on the looser count would have been
chosen on a number nine times too large. `coverage_report` and
`judged_non_survivor_counts` now take the gain map as a **required** argument
and return `n_gainful` / `n_gainful_non_survivors`. Required, not defaulted, so
the definition always travels with the config that set it — and so the change
broke its callers loudly rather than silently returning different numbers under
the same column name.

**The headline-`K_c` rule was mis-specified and is replaced.** The old rule —
"the largest `K_c` with median gain-carrying non-survivors ≥ 1" — selected
`K_c=1000`, where supply is median 1–2 with **36–39 of 97 queries having none
at all**.

The error was conflating two jobs in one expression. `≥ 1` is a
**non-degeneracy floor**: below it a depth has no gate failure to detect. It
says nothing about which of several usable depths to prefer, and used as a
selection rule it chose the deepest — exactly backwards, because a TREC pool is
built from the top ~100 of the systems of the day, so a deeper `K_c` swallows
more of the judged pool into `C_q` and leaves *less* supply outside it.

Separated, and now implemented in `select_headline_k_c`: among depths clearing
the floor for **every** Stage-1 system, take the one with the greatest supply —
the weakest system's supply, since the comparison needs both — breaking ties
towards the deeper depth for external validity. The configured
`candidate_depths` bound the shallow end, so the config's own realism judgement
still applies. On this collection it selects **`K_c=100`** (median supply 6,
17–23 queries empty), with 500 and 1000 reported as the realism ladder.

Recorded as a post-hoc change to a pre-declared rule. It is defensible only
because gate-failure supply is measurable *before* any effect is computed, so
the choice cannot be driven by effect sizes. **No threshold was tuned to
produce a particular answer**: the floor is unchanged at 1, and only the
direction of the preference changed. Three tests pin the direction, including
one with a synthetic collection whose supply *grows* with depth, where the rule
correctly selects a deeper headline — the check that it is a rule and not a
constant dressed as one.

**THE OPEN DECISION: unjudged competitors and `E_gate`.** Judged density of
`C_q` against `U_q` is 28.6× / 10.9× / 6.4× at the three depths. This is a
consequence of a deliberate rule, recorded in `qrels.py`: *unjudged documents
still occupy ranks as competitors, because the policy being simulated really
would have ranked them.* That rule is principled. The problem is that at this
collection's density it may make the estimand degenerate: world C sorts ~9,000
documents of which ~1.65% are judged, so nearly every judged survivor is
displaced out of the top-10 in world C, `U_d(π_no1) → 0`, and
`E_gate → DCG(π_act)` — the gate would appear to account for *all* retrieved
utility, as an artefact of 2019 pooling rather than as a finding.

Three options, to settle before the Stage-2 cache is filled:

1. **Keep the rule**, and report the judged fraction of world C's top-10 beside
   every result, treating `E_gate` as an upper bound on the gate's benefit.
2. **Add a condensed-list secondary analysis** and report both, showing whether
   the conclusion survives either way. `shared.utility.condense` now exists for
   this — a pure, tested function that drops unjudged documents before ranks are
   assigned (Sakai, "Alternatives to Bpref", SIGIR 2007). **Nothing calls it**;
   it deliberately contradicts the primary rule, so applying it is a team
   decision and belongs to whoever owns the world runner.
3. **Restrict `U_q` to the judged pool** for the gate analysis specifically.
   Cleanest statistically and nearly free computationally (~20,700 pairs rather
   than 872,455), but it narrows the claim from "retrievable documents" to
   "judged documents".

Recommendation: (2), because it preserves the primary estimand while bounding
the artefact. **Whichever is chosen carries a config fingerprint**, and the
cheapest moment to change the config is now — before the cache exists. One
measurement settles how much it matters: the judged fraction of world C's
top-10, available as soon as any Stage-2 scores exist. Decision: ____

**Report output grouped by finding kind.** It was emitting one message per
(`K_c`, system) — twelve lines for two distinct problems. Now one entry per kind
listing everywhere it fires.

---

<!-- next entry below -->
