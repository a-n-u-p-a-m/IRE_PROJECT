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
  published 0.725, R@1000 0.822 vs 0.820. BM25's published DL19 nDCG@10 is
  ~0.506, so the lexical-vs-semantic contrast the mismatch axis needs is real.

  **Correction (2026-10-31): the float16 explanation was wrong.** This entry
  originally attributed the nDCG gap to float16 exact search over the 89-shard
  index. Re-scoring in float32 returns the same nDCG, so precision is not the
  cause. The gap holds on both years -- DL19 0.7063 vs 0.7250, DL20 0.6889 vs
  0.7025, about 0.014-0.019 -- while **recall is at or above published**
  (0.8224 vs 0.8201; 0.8587 vs 0.8533). Retrieving the right documents and
  ordering them slightly worse near the top is a different failure from
  retrieving fewer of them, and it has a different set of candidate causes
  (query encoding, normalisation, or the tie-break at equal scores). Cause
  still unknown; recorded as open rather than explained, because the original
  explanation reads as settled and is not.
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
| `tau_document` | **0.05** | presentation labels only; already frozen in `config/base.yaml` |
| B = `K_c`? | **yes** (`b_equals_kc: true`) | so `t=1` reproduces the conventional cascade exactly, and budget results need no candidate-only caveat |
| headline NDCG gain map | **both, labelled** | trec_eval's linear gains for any number compared to published work; the attribution map for every estimand. See 2026-10-28. |
| listwise reranker | out of scope | |

These three were **already binding before they were recorded**: the values live
in the frozen config and its fingerprint `46d0807d8ed3` is baked into the
frozen universe. Recording them ratifies what the fingerprint already enforces.
Changing any of them now means a new fingerprint and a universe rebuild —
cheap today, and not cheap once the Stage-2 cache is filled.

Decide the *rule* now, the value arrives later:

| decision | rule | value |
|---|---|---|
| headline `K_c` | greatest gate-failure supply among non-degenerate depths | **100** (2026-10-28 feasibility report) |
| `tau_query` | fixed on synthetic, then never tuned | **0.05**, frozen in config |
| reference-subset size | whatever the power simulation returns | **all 97 queries** (Shubham, H2) |
| reference sampling strategy | moot — every query is a reference query | — |
| SPLADE in/out | **out** | a third Stage-1 system would require rebuilding the frozen universe, and the Jaccard check (0.11–0.16) already shows the two systems give ample leverage |
| hashing in/out | Chaitanya's written criteria | **out** unless he reports all four met; `hashing.enabled: false` |

**Also to settle at that meeting**

- **Do not seed the Stage-2 cache from the existing DPR → MiniLM scores.**
  It saves ~97,000 pairs out of a 0.6–0.9M bill — about four minutes of T4 time —
  and it interleaves two scoring runs everywhere except DPR's own candidate
  sets. BM25's `C_q` partially overlaps DPR's top-1,000, world C mixes by
  definition, and world D's outsiders mix. A BM25-vs-DPR comparison where one
  system's candidates come from a single run and the other's from a blend is
  exactly the asymmetry H3 cannot absorb.
  **Decision (freeze v2): dropped.** At the measured 535 pairs/sec it saves
  about three minutes of a 27-minute bill. Shubham removed it from his Week-2
  scope and the ROADMAP row was corrected.
- Throughput measurement must report the **observed mean sequence length**
  beside pairs/sec. If the cross-encoder pads to `max_length` 512 rather than to
  the longest in batch, the bill is several times larger than it needs to be —
  MS MARCO passages average around 55 words.
  **Decision (freeze v2): partly settled, one question left.** Measured mean
  71 tokens, max 183 — far inside `max_length=512`, and robust even to a 2x
  error in the `words x 1.3` estimate behind it. What is still unanswered is
  whether the tokeniser pads to the longest sequence in each batch or to 512,
  which is a ~2.8x factor on the bill. `sentence-transformers`' `CrossEncoder`
  pads to longest-in-batch by default, so this is probably already fine;
  Chaitanya to log the actual padded tensor width per batch and close it. Not
  a blocker: it affects how long the fill takes, not what it produces.

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
report both and show the conclusion survives either way.
**Decision (freeze v2): compute both, always.** See the freeze v2 entry for
the reasoning; in short, condensing is a ranking transform over the same
scores, so it costs no Stage-2 calls, and the measurement that would settle
which mode is right does not exist until the cache is filled.

**A second, worse asymmetry.** The same pooling history means BM25's candidate
set is likely judged more densely than DPR's: lexical runs contributed to the
2019/2020 pools and a modern dense retriever did not. Comparing `E_gate` across
the two paradigms would then compare judgment coverage as much as retrieval
paradigm. `feasibility_report.py` measures this between-system ratio and warns
above 3×. If it fires, the headline cross-paradigm claim needs rework.
**Decision (freeze v2): retired — it does not fire.** Measured 1.06x at
`K_c=100` (BM25 `C_q` 51% judged, DPR 48%) and 1.15x at `K_c=1000`. The
TREC-DL 2019/2020 pools included neural runs, so they are not lexically
biased. The floor is symmetric too: both systems retrieved exactly 5,000, so
each floors `|U_q| - 5000` and the floored fraction is identical at 0.450.
The cross-paradigm comparison is unconfounded on both axes that could have
spoiled it.

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
top-10, available as soon as any Stage-2 scores exist.
**Decision (freeze v2): option (2), compute both modes always.** And it needs
no config field after all — see the freeze v2 entry.

**Report output grouped by finding kind.** It was emitting one message per
(`K_c`, system) — twelve lines for two distinct problems. Now one entry per kind
listing everywhere it fires.

---

## 2026-10-29 — Throughput, H2 resolved, and the decisions that must precede the cache fill

Chaitanya's PR #3 and Shubham's Week-2 branch both landed as proposals. Three
things are now settled, two are newly urgent, and one is a defect.

**Throughput: 535 pairs/sec on a Colab T4 → ~27 min for the whole universe.**
The full 872,455 pairs is affordable, and with the 3-way shard plan it is ~9
minutes each. Two caveats on the number rather than the conclusion:

- It was measured at **`float32`** with **MiniLM-L-12-v2** (`FROZEN_PRECISION`
  in `throughput.py`). The reranker already supports `float16`, and a T4's
  half-precision path would roughly double this. **This is the newly urgent
  decision**, because precision changes the scores, which changes the config
  fingerprint, which invalidates the cache. It has to be settled *before* the
  fill, not after.
  **Decision (freeze v2): stay `float32`.** No config change. See the freeze
  v2 entry for the arithmetic; the saving is 14 minutes once, against an
  unquantified risk of rank flips in a rank-based estimand.
- The mean sequence length (71 tokens, max 183) is a `words × 1.3` estimate,
  not real tokenisation. It does not threaten the conclusion — even a 2×
  error leaves us far below `max_length=512` — but it does not answer the
  question this log asked on 2026-10-26 either: whether the cross-encoder pads
  to the longest sequence in each batch or to 512. That is a ~2.8× factor on
  the bill. `sentence-transformers`' `CrossEncoder` pads to longest-in-batch by
  default, so it is probably fine; the report should log the actual padded
  tensor width per batch and settle it.

**Shuffle-invariance passed on the real reranker.** The cross-encoder is
pointwise and batch-order invariant. This is the assumption the whole `E_gate`
derivation rests on, and it is now verified rather than assumed.

**H2 is resolved, and the reference-subset design is retired.** Shubham re-ran
the power simulation at `K_c=100` with the real median `|U_q|` of 9,097: **no
subset smaller than all 97 queries reaches 0.80 power**, and even at 97 only
the *alternative* outsider policy clears it (0.935 continuous / 0.875 group).
Boundary falls short (0.67 / 0.41).

So: the reference subset is every query, which is exactly what the 27-minute
figure makes affordable, and the sampling-strategy decision disappears.
**H7 is confirmatory for the alternative policy and descriptive for boundary**,
declared now rather than after the effects are seen. This is the outcome the
2026-10-27 entry anticipated when the power denominator was corrected.

**Cache seeding from the old DPR → MiniLM scores is dropped.** Shubham has
removed it from his Week-2 scope, per the 2026-10-26 note: it saved ~97,000
pairs of a 872,455-pair bill — about three minutes at the measured
throughput — and in exchange interleaved two scoring runs everywhere except
DPR's own candidate sets. At three minutes the trade was never close. The
ROADMAP Week-2 row that still assigned it to Shubham has been corrected.

**Defect in PR #3: the predictors read world-C scores and count them free.**
`features.py:352` builds `doc_ids_per_query` from the whole top-5,000 run
rather than from `rank <= k_c`, so `extract_stage2_features` pulls Stage-2
scores for every universe document while its docstring states "Scores come
from world A (the factual run)" and "Extra Stage-2 calls = 0".

Those cannot both hold. For a document outside `C_q` a cached Stage-2 score
exists *only because world C was computed*, and world C is the expensive
counterfactual the predictors exist to avoid. The consequence is stronger than
mis-stated cost. For a non-survivor `d`, `rank_{π_act}(d)` is undefined, so

    e_gate(d) = U_d(π_act) − U_d(π_no1) = −U_d(π_no1)

and `U_d(π_no1)` is fixed by `rank_C(d)`, which the `s2` scores over `U_q`
determine exactly. Given those scores and `d`'s label, `e_gate(d)` is not
approximately recoverable — it is **determined**. A predictor with this feature
is not predicting the effect, it is reading it.

The fix is one line (restrict to `rank <= k_c`), but every synthetic fidelity
number in the PR has to be regenerated afterwards, and the three baselines will
move. Note the 25 tests pass either way: the suite covers survivor flags in
`extract_stage1_features` and never asserts the `C_q` boundary for Stage-2
features, which is why this survived review-by-test.

**Smaller items in PR #3**, recorded so they are not relitigated:
`IrDatasetsPassageFetcher` materialises all ~8.8M MS MARCO passages (~3–4 GB)
when the universe needs perhaps 600–800k of them (~320 MB) via
`docs_store().get_many()`; its `cache_size` parameter is documented, assigned
and never read. `run_real_benchmark` requires `data/universe.parquet`, which is
deliberately not committed — the universe is rebuilt from the committed run
tables, verified bit-identical, rather than passed between machines.

**One correction to a proposed change.** The `--ignore` for
`test_dpr.py` in the Makefile cannot be removed yet: the lazy `torch` import is
on `shubham/week2`, not on `main`. Removing it before that branch merges breaks
`make test-all` on every CPU-only machine. The Makefile now says so, with the
condition for removing it.

**Still open, and all three gate the cache fill:** the `float16` decision above,
the unjudged-competitor question from 2026-10-28, and the PR #3 leakage fix.
The first two change the config fingerprint. There is no cheaper moment than
now.

---

## 2026-10-29 — Freeze v2

**Fingerprint: `46d0807d8ed3` — unchanged. Freeze v2 is identical to freeze
v1.** Every open decision resolved either to keep the current value or to
something that does not belong in the config at all. Nothing downstream needs
re-stamping: the frozen universe, its sidecar, and every manifest written
against v1 remain valid.

That is the headline, because the two live decisions both looked like config
changes and neither turned out to be one.

### Precision: stay `float32`

`reranker_precision: float32` is unchanged.

The case for `float16` was speed: 535 pairs/sec becomes roughly 1,100 on a T4,
so the 872,455-pair fill drops from ~27 minutes to ~13. **That saving is 14
minutes, once, for the entire project.**

Against it: every estimand here is a *difference* of rank-based utilities.
`float16` perturbs scores at ~1e-3, and in a 9,000-document ranking that will
flip near-ties. How many, and whether the flips land where they matter, is
unquantified — and quantifying it costs a GPU session that would itself take
longer than the 14 minutes being saved.

So the decision needs no further evidence: the cost side is already measured
and it is negligible. Trading an unquantified risk to the estimands for a
quarter of an hour is not a trade worth making. Revisit only if the bill grows
by an order of magnitude.

A consequence for Chaitanya: `throughput.py` hardcodes `FROZEN_MODEL`,
`FROZEN_MAX_LENGTH`, `FROZEN_BATCH_SIZE` and `FROZEN_PRECISION` as module
constants duplicating `cfg.reranker_*`. They agree today. They should read from
the config, or a future precision change will leave the throughput report
describing a precision the project is not using.

### Unjudged competitors: compute both modes, always

The open question was whether to keep the primary rule — *unjudged documents
occupy ranks as competitors, because the policy being simulated really would
have ranked them* — given that `U_q` is 1.65% judged while `C_q` is 6–29%
judged, which risks driving `E_gate` towards `DCG(π_act)` as an artefact of
2019 pooling.

**Decision: compute every effect both ways and report both.** Three reasons,
in order of weight.

**It costs no GPU.** Condensing is a transform on *rankings*, not on scores:
drop unjudged documents before ranks are assigned. Same cache, same
872,455 calls. The second mode is CPU only. A robustness check that costs
nothing should not be argued about.

**It is the only option that does not decide in the dark.** The measurement
that would settle which mode is right — the judged fraction of world C's
top-10 — does not exist until the cache is filled. Every other option commits
now to a choice whose evidence arrives later. This one defers the choice to
when the evidence exists, which is what the pre-registration discipline is
actually for: fixing the *procedure* in advance, not guessing the answer.

**It needs no config field, so the fingerprint does not move.** This is the
part worth noticing. If "which mode is headline" were a config value, freeze v2
would change the fingerprint and invalidate the frozen universe. But if *both*
are always computed, there is no choice to encode. The config governs what is
computed; the design log governs what is presented. The cost is a `condensed`
column on the effect table — a schema change in Shubham's area, not a config
one — and **that is the one piece of this decision that needs his agreement.**

Headline presentation, declared now: the **primary** estimand is the
uncondensed one, because it is what the deployed policy would really do. The
condensed figure is reported beside it as the robustness check. If the two
disagree in sign or in which stratum dominates, that disagreement *is* the
finding and gets its own paragraph rather than a footnote.

### Everything else resolved

- **Cache seeding from the old DPR → MiniLM scores: dropped.** ~3 minutes of a
  27-minute bill, in exchange for interleaving two scoring runs everywhere
  except DPR's own candidate sets.
- **Between-system judgment asymmetry: retired.** Measured 1.06×, not the 3×+
  that would have confounded the cross-paradigm claim. The floor is symmetric
  too — both systems floor exactly `|U_q| − 5000`, an identical 0.450.
- **Headline `K_c` = 100**, by greatest gate-failure supply among
  non-degenerate depths. 500 and 1000 are the realism ladder.
- **Reference subset = all 97 queries.** H7 is **confirmatory for the
  `alternative` policy only** and **descriptive for `boundary`**; `random` is
  the control and is not expected to reach power. Recorded in
  `data/reference_subset_manifest.json`, derived mechanically from Shubham's
  power manifest rather than from a chat message.
- **SPLADE: out.** A third Stage-1 system would require rebuilding the frozen
  universe, and the candidate Jaccard (0.11–0.16) already gives ample leverage.
- **Hashing: out**, `hashing.enabled: false`, per Chaitanya's own written
  criteria.
- **Sequence-length padding:** one narrow question left with Chaitanya — log
  the actual padded tensor width per batch. It affects how long the fill takes,
  not what it produces, so it is not a blocker.

### State at freeze v2

| | |
|---|---|
| Fingerprint | `46d0807d8ed3` (unchanged from v1) |
| Queries | 97, split 49/19/29 by query, seed 42 |
| BM25 | verified exact against published figures, delta 0.0000 on all four |
| Universe | `\|U_q\|` 7,045–9,986, median 9,097, **872,455 pairs**; reproduced bit-identical in three environments across two OSes |
| Stage-1 over `U_q` | floored fraction 0.450; identical for both systems **by construction**, not as a measurement |
| Reference subset | all 97 queries |
| Allocations | 11,640 audited; spend, partition, `t=1` baseline and random-control sensitivity all hold |
| Stage-2 bill | 872,455 unique pairs, ~27 min on one T4 *(superseded: see 2026-11-09)* |
| Tests | 406 passing, 1 skipped |

**Open, and owned by others:** Chaitanya's feature-leakage fix and the padded
width; Shubham's agreement on the `condensed` column. Nothing blocks the cache
fill once the leakage fix lands.

---

## 2026-11-09 — Week-3 rebuild, and the candidate divergence measured at depth M

`stage1_over_universe` and `allocations` re-ran clean on the frozen universe:
top-`K_c` agreement at every depth, exact coverage, floor ordering intact,
every outsider policy supplied at `B = K_c`. 11,640 allocations, 1,319,200
promoted documents — verified independently rather than trusted, since the
total is fully determined:

    per system  =  97 × 5 × Σ_{t<1} (1−t)·B   summed over the three K_c
                =  97 × [5(10+25+50) + 5(50+125+250) + 5(100+250+500)]
                =  659,600          ×2 systems  =  1,319,200  ✓

### The floored fraction is not a retriever property

Freeze v2 already recorded that both systems floor exactly `|U_q| − 5000`.
Restating it here because the *report* must not present it otherwise: with both
retrievers at a fixed depth M=5000 and `U_q` their union, the floored fraction
is `(|U_q| − 5000)/|U_q|` — a deterministic function of `|U_q|` alone. It cannot
differ between systems, so it measures nothing about either. Confirmed on all
97 queries: `|U_q| = 10000 − |BM25 ∩ DPR|` holds exactly, with no exceptions.

Consequence for `stage1_over_universe`'s own docstring, which claims "a
retriever that floors most of `U_q` is doing far less work than the comparison
assumes": true in general, **unmeasurable in this configuration**. It would
become measurable only with unequal retrieval depths, which we are not doing.

### What the same numbers do measure

The identity above means the floored fraction is a re-encoding of candidate
divergence, which *is* a result. Computed from the committed run tables:

| BM25 ∩ DPR at depth 5000 | min | median | max |
|---|---|---|---|
| overlap (documents) | 14 | **903** | 2,955 |
| as fraction of each top-5000 | 0.003 | **0.181** | 0.591 |
| Jaccard | 0.0014 | **0.0993** | 0.4194 |

The two paradigms nominate almost disjoint sets: a median of 903 documents in
common out of 5000, and on one query **14**. This is consistent with — and
deeper than — the candidate-depth Jaccard of 0.11–0.16 already recorded under
the SPLADE decision.

It also supplies a mechanism rather than just a magnitude for the headline
estimand. `E_gate` is expected to be large not because the reranker is weak but
because gate *membership* is nearly determinative when the two candidate
generators agree on under a fifth of their output. That is an argument the
report can make from measurement instead of assertion.

Reproduce with the committed run tables alone, no GPU and no corpus:

```
python3 - <<'PY'
import pandas as pd, numpy as np
b = pd.read_parquet('runs/bm25.parquet').groupby('query_id')['doc_id'].apply(set)
d = pd.read_parquet('runs/dpr.parquet').groupby('query_id')['doc_id'].apply(set)
q = sorted(set(b.index) & set(d.index))
ov = np.array([len(b[k] & d[k]) for k in q]); un = np.array([len(b[k] | d[k]) for k in q])
print('overlap', ov.min(), int(np.median(ov)), ov.max())
print('jaccard', (ov/un).min().round(4), np.median(ov/un).round(4), (ov/un).max().round(4))
print('identity holds:', bool(np.all(un == 10000 - ov)))
PY
```

### Throughput: ~52 min, not ~27

The freeze-v2 estimate is superseded by measurement. Shubham's shard fill ran
at **282.3 pairs/sec** on a T4 (re-timed; the fill session's own output was lost
to a disconnect), so the full 872,455-pair bill is **≈51.5 minutes**, not 27.
The estimate was optimistic by 1.9×. Nothing downstream depends on it — the
cache is keyed on `(query_id, doc_id, model)` and the fill is resumable — but
the progress report's "~27 min (est.)" is wrong and should read ~52 min.

### Effects do not need a GPU

Confirmed by inspection: `experiments_shubham/real.py`, `engine.py`,
`shared/cache.py`, `shared/worlds.py` and `shared/effects.py` import no torch.
Only the cache *fill* needs an accelerator; computing worlds and effects from a
filled shard is sqlite reads plus pandas. This matters operationally — Colab
GPU scarcity blocks the fill but not the analysis, so a preview run on
Shubham's completed 32-query shard can proceed on a laptop.

**Still open:** the judged fraction of world C's top-10, which decides whether
the headline `E_gate` is the uncondensed or the condensed figure. It must be
measured before any analysis prose is written. Candidate-set Recall@K remains
the one unmet §12.1 pre-run check and is computable from
`n_relevant_cq / n_relevant_uq` already on disk.

---

## 2026-11-10 — Shard 1 filled, and the unjudged question answered

Anupam's shard: 33 queries, 295,531 pairs, 0 errors, digest `9f2cfe600f0e49aa`,
19.1 min on a Kaggle T4 at 257 pairs/s. Two of three shards in — 582,589 of
872,455 pairs, 66.8%; the remainder is exactly Chaitanya's 289,866.

The canary found no drift to report. Kaggle and Colab both served
`torch 2.11.0+cu130`, CUDA 13.0, Python 3.13.16 on a Tesla T4, differing only
on numpy, which takes no part in the forward pass. `REFERENCE_DIGEST`
`e1121c9d61e14fd5` is committed so the third fill can be checked against it
without coordinating with anyone.

### The unjudged share of world C's top-10: 3.87%

**This closes the decision left open at freeze v2.** Measured independently on
two disjoint query sets:

| | queries | mean | median | max |
|---|---|---|---|---|
| shard shubham | 32 | 3.8% | 0 | 40% |
| shard anupam | 33 | 3.94% | 0 | 40% |
| combined | 65 of 97 | **3.87%** | 0 | 40% |

They agree to 0.14 percentage points, with identical medians and identical
maxima. One subset could be a sampling accident; two disjoint subsets agreeing
this closely is a property of the collection. The median of **0** is the
striking figure: on more than half of queries, world C's top-10 contains no
unjudged document at all, drawn from a pool that is 1.78% judged. The reranker
concentrates pooled documents at the top almost perfectly, which follows from
NIST having pooled what good systems ranked highly.

The freeze-v2 worry was that unjudged competitors would displace judged
survivors out of world C's top-k and drag `E_gate` towards `DCG(pi_act)` as an
artefact of pooling. At 3.87% that mechanism has almost nothing to act on.

**Decision: the UNCONDENSED figure is the headline; the condensed analysis
becomes the robustness check.** Recommended on this evidence and pending
Shubham's and Chaitanya's confirmation, per the freeze-v2 note that the mode
is a team decision. Nothing about the fingerprint changes either way — both
modes are already computed on every run, which was the point of deciding to
compute both rather than choosing early.

The stronger argument for the decision is not the 3.87% at all. It is that on
33 queries **the two modes agree on every qualitative conclusion** — the same
policies significant, the same null, effect sizes within roughly 20%. The
conclusion does not depend on the mode, so the choice of headline is a
presentational one rather than a load-bearing one.

### H7 on 33 queries, bm25, K_c=100 — preview only

`alternative` is significant on BOTH tests at every `t` and in both modes
(group p 0.0001–0.0005; Spearman rho 0.72–0.87, p 3e-11 to 2e-07). `random`
is null throughout — p 0.21 to 1.00, rho negative at two of three `t` — which
is what makes the `alternative` result mean something other than "reranking
more documents helps". `boundary` is null at t=0.9 and 0.75 and significant at
t=0.5 (p 0.0035, rho 0.46), consistent with boundary documents resembling the
worst survivors until enough are promoted.

Three qualifications, all of which must survive into the report:

  * **The p-values sit at the permutation floor.** With 10,000 permutations
    the smallest achievable is ~1e-4. Report `p < 0.001`, never `p = 0.0002`.
  * **This is 33 of 97 queries, and it is a preview.** The confirmatory
    analysis is pre-registered on the full set. Having now seen this, nothing
    about the test may be adjusted; that is what preserves the registration.
  * **The group test is not weak**, contrary to the Week-3 note. That note saw
    2–4 gate-sensitive queries of 32; this shard has 5 (uncondensed) and 6
    (condensed) of 33, and the group test clears at every `t`. So no switch
    between the group and continuous tests is needed, and the question of
    justifying one post hoc does not arise. The 29 Oct power manifest had
    already put `continuous|alternative` above `group|alternative` at n=97
    (0.935 against 0.875); both were adequately powered then and both pass now.

Observed gate-sensitivity, 15–18% across the two shards, runs about three
times the 5.6% the H2 simulation assumed at `U=9097, K_c=100`, and the
observed effect is far larger than the simulation's: power at n=30 was put at
0.525, against an observed p of 3e-11. The power analysis was conservative,
which is the direction one wants to be wrong in.

### Non-additivity, measured

83.0% of documents (uncondensed; 81.4% condensed) lie off the additive
diagonal, with a maximum absolute residual of **7.0** — the entire gain of a
grade-3 document. `E_rerank + E_gate` is not `U_d(pi_act)`, by margins as
large as a perfectly-placed relevant passage. The decision to report no pooled
statistic now rests on a measurement instead of an argument.

---

## 2026-11-10 (later) — DPR has no gate-sensitive queries, and why that is the result

> **SUPERSEDED — see 2026-11-12.** The "zero" below is a 33-query preview
> artefact. On the full 97 queries dpr has FOUR gate-sensitive queries against
> bm25's 14. The contrast and the mechanism survive; the absolute does not.
> Do not quote this entry's headline.

Running the same preview for `dpr` on the same 33 queries gives **0
gate-sensitive queries** against bm25's 5 (6 condensed). The naive reading -
"dpr retrieves better, so it discards less" - is wrong, and the committed
full-97 feasibility run says so directly. At `K_c=100`:

| | median gainful non-survivors | queries with supply |
|---|---|---|
| bm25 | 6.0 | 80/97 |
| dpr | **6.0** | 74/97 |

**The two retrievers discard the same quantity of gain-carrying documents.**
What differs is whether those documents matter: bm25 discards passages the
cross-encoder would have placed in the top-10, and dpr does not. Both
retrievers throw away about six judged, gainful documents per query; only one
of them throws away the ones that would have changed the answer.

The mechanism is paradigm alignment. Stage 2 is a dense cross-encoder, so
dpr's ordering already correlates with what Stage 2 wants and dpr's top-100
contains world C's top-10. bm25 is lexical, and its discards include
semantically relevant passages that the cross-encoder then ranks highly. This
is the lexical-vs-semantic comparison the project set out to make, and it is a
stronger claim than a recall difference, because recall of gainful documents
is the same.

Three signals in the dpr run agree, which is what makes it a mechanism rather
than a null:

  * 0 gate-sensitive queries;
  * `alternative` returns nothing (rho 0.03-0.05, p ~0.40), where bm25 gave
    rho 0.72-0.87 at p < 0.001 - the second opinion has nothing to find
    because nothing worth finding was discarded;
  * `random` actively HURTS (rho -0.44, -0.35, p -> 1), where for bm25 it was
    roughly null. Removing dpr candidates to make room for random outsiders is
    strictly bad, which can only hold if the candidates were already right.

### What this does and does not establish

0 of 33 is weak alone: under the simulated 5.6% rate, P(0 of 33) = 0.149, so
the observation is unsurprising even if dpr's true rate matched bm25's. The
claim rests on the CONTRAST, 5/33 against 0/33, Fisher one-sided p = 0.027.
Both ran on the same 33 queries, so that treats paired data as unpaired;
**the full-97 analysis must use McNemar on the per-query pairs instead.**

### Supply is not sensitivity

The headline `K_c=100` was chosen by greatest gate-failure SUPPLY among
non-degenerate depths. dpr demonstrates that supply does not convert into
sensitivity: 74 of 97 queries carry gainful non-survivors, and none of the 33
tested is gate-sensitive. Supply is necessary and not sufficient, and the
writeup must keep the two apart or it will read as a contradiction between the
feasibility report and the results.

### A correction to the previous entry

That entry recorded that the two modes agree on every qualitative conclusion.
That was measured on bm25 only. In dpr, `boundary` at t=0.5 flips sign between
modes - +0.398 (p 0.011) uncondensed against -0.194 condensed. With zero
gate-sensitive queries the H7 frame is degenerate there and it should not be
read as a finding, but the claim as written was broader than the evidence and
is narrowed here to bm25.

---

---

### 2026-11-11 — Hashing: final go/no-go decision — OUT (Chaitanya)

Decision: `hashing.enabled` stays `false`.  One paragraph will appear in the
Week-4 prediction/cost section explaining why.

**All four written criteria were reviewed at the Week-2 meeting and none were
fully satisfied by any candidate approach:**

| Criterion | Required | Status |
|---|---|---|
| 1. Method pre-specified | `hashing.method` ≠ null before any code | Not committed — no method was agreed before the fill |
| 2. Representation source independent of MiniLM | BM25 TF, DPR embeddings, or separate neural hash | Not specified — DPR embeddings are plausible but were not committed |
| 3. Bit-width pre-specified | `hashing.bits` set before any experiments | Not committed |
| 4. Within-stratum fidelity clears membership baseline on test queries | ≥ 2 of 3 effect types improved | Untested — cannot be evaluated without (1)–(3) |

**Rationale for firm close-out:** Criteria 1–3 form a logical dependency chain;
4 cannot be evaluated without them.  Keeping hashing nominally "open" while
doing the Week-4 predictor runs would create pressure to retrofit a method post
hoc, which Criterion 1 explicitly prohibits.  The conventional-feature
predictors carry the prediction section on their own, as designed.

**Report paragraph (verbatim, to be inserted in Chaitanya's prediction/cost
section):**

> Hashing features were not included. The go/no-go criteria required a
> concrete method specified in `config/base.yaml` before any hashing code was
> written, a representation source independent of the Stage-2 model, a
> pre-specified bit-width, and a within-stratum fidelity improvement over the
> membership-only baseline on held-out queries. At the Week-2 decision meeting
> none of the four criteria were fully satisfied (method, representation source
> and bit-width were not pre-specified in the config freeze), so
> `hashing.enabled` remained `false`. All prediction results use conventional
> features only.

---

### 2026-11-11 — Throughput tokenizer instrumentation fix (Chaitanya)

**Bug:** `throughput.py` attempted to intercept the AutoTokenizer's
`__call__` by setting `tokenizer.__call__ = _patched_tokenize` on the
*instance*.  Python resolves dunder methods on the *type*, never the instance,
so this assignment was silently ignored and `padded_widths` stayed empty.  The
`except Exception: pass` guard hid the absence of an error, so the code
reported success while doing nothing.

**Fix (committed in this entry's commit):** Instead of patching an attribute
on the existing tokenizer object, the code now wraps the tokenizer in a
`_TokenizerWrapper` proxy whose *class* defines `__call__`.  After measuring,
`inner.tokenizer` is restored to the original object.  The fix is confined to
`experiments_chaitanya/throughput.py` and does not affect any cached score or
any effect computation — it is instrumentation only.

**Impact on existing results:** None.  The fallback (word_count × 1.3
estimate) was always used in Colab, and the reported `pairs_per_second` and
`estimated_hours` are unaffected.  The fix makes the `mean_padded_tensor_width`
field accurate for future runs.

---

## 2026-11-12 — CORRECTION: DPR has four gate-sensitive queries, not zero

**The entry dated 2026-11-10 (later) is wrong in its title and in its central
claim.** It reads "DPR has no gate-sensitive queries, and why that is the
result". On the full 97 queries DPR has **four**. That entry was written from a
33-query preview and the zero did not survive.

From the committed manifests, `analysis/real_<system>_kc100/analysis_manifest.json`:

| gate-sensitive | uncondensed | condensed | rerank-sensitive | mixed-stable |
|---|---|---|---|---|
| bm25 | **14** / 97 | 16 | 83 | 0 |
| dpr | **4** / 97 | 3 | 86 | 7 |

Checked before concluding anything: `tau_query` and `tau_document` are both
0.05 in each run, `k_c` is 100, the fingerprint is `46d0807d8ed3`, and
`config/base.yaml` has not been modified since the original upload. Nothing
about the labelling rule changed between the preview and the full run, so the
difference is sampling and nothing else.

### What the preview got right, and what it got wrong

The caution recorded at the time was correct, and is the only reason this is a
correction rather than a retraction:

> 0 of 33 is weak alone: under the simulated 5.6% rate, P(0 of 33) = 0.149, so
> the observation is unsurprising even if dpr's true rate matched bm25's. The
> claim rests on the CONTRAST.

That is what happened. DPR's observed rate is 4/97 = **4.1%**, close to the
5.6% the H2 simulation assumed. It was never a zero-rate system; 33 queries
simply had none. The outlier is bm25 at 14/97 = **14.4%**, about 2.5x the
simulated rate.

**The mechanism survives, restated quantitatively.** Both retrievers discard
the same quantity of gain-carrying documents — median 6.0 each at K_c=100, per
the feasibility run — and bm25's discards cost it gate-sensitivity 3.5x as
often. That is still the lexical-versus-semantic finding, and still stronger
than a recall difference, because recall of gainful documents is identical.
What is no longer available is the clean "DPR discards nothing that matters":
DPR discards things that matter on four queries in ninety-seven.

**The report must not say "zero".** The contrast is paired — both systems on
the same 97 queries — so the statistic is the exact McNemar that `analysis.py`
now implements, reported from `contrast_vs_dpr.csv`; not a rate comparison and
not Fisher.

A label appears on the full set that the preview never showed: dpr has seven
`mixed_stable` queries (nine condensed) against the preview's one. A category
that was nearly invisible at n=33 is 7% of the collection.

### Non-additivity holds at full scale

| | n documents | off-diagonal | max abs residual |
|---|---|---|---|
| bm25 uncondensed | 795 | 80.9% | 7.0 |
| bm25 condensed | 796 | 80.0% | 7.0 |
| dpr uncondensed | 760 | 79.3% | 7.0 |
| dpr condensed | 772 | 80.2% | 7.0 |

The preview measured 83.0% on 264 documents; the full run gives 79–81% on
~780. The finding is unchanged and now rests on three times the evidence. The
maximum residual is still exactly 7.0, the entire gain of a grade-3 document,
in all four cells.

### The general lesson, which is why this is written down

A preview on a third of the data reproduced every qualitative conclusion except
one, and the one it got wrong was the one stated as an absolute. "Zero" is a
claim a subset cannot support: a rate can be estimated from 33 queries, an
absence cannot. Preview findings belong in the log as previews, and any
absolute in one should be read as "none observed yet".

### What this entry does NOT record

The H7 p-values, the McNemar result and the full-run `unjudged_at_k` live in
`analysis/real_<system>_kc100/*.csv`, which are correctly gitignored. This
entry records only what the committed manifests carry. Whoever writes the
results section should put the headline H7 and McNemar figures into a log entry
of their own, because at present no committed artefact states them.

---

## 2026-11-13 — Full-run results on 97 queries: H7, the BM25–DPR contrast, Δ_t, sensitivity (Shubham)

This is the entry the 2026-11-12 correction asked for: the headline numbers,
written into a committed file so that the report and the video quote one
source. Everything is K_c = 100, DCG@10, **uncondensed** (the headline mode)
unless a line says otherwise; condensed is given where it could change a
reading. Nothing below came from a preview.

**Provenance.** `cache/stage2.sqlite`, 872,455 rows, merged digest
`b7669b8dde3b328c`; each shard inside it matches its committed digest
(anupam `9f2cfe600f0e49aa`, shubham `def33b7a41b4f1d0`, chaitanya
`ab09ad44960cdaf5`) and covers the frozen universe exactly. World D equals the
published allocations at **4,365 of 4,365** allocations for each system.
Manifests: `effects/real*/<system>_manifest.json`,
`analysis/real_<system>_kc100/analysis_manifest.json`,
`analysis/compare/compare_manifest.json` (commit `7d3b18c`; code at `5808ed1`;
fingerprint `46d0807d8ed3`). The analysis is deterministic: a second run wrote
byte-identical CSVs. Their digests are at the end of this entry, so a re-run
can be checked against these numbers rather than trusted.

### Query labels and the per-query quantities

| | bm25 | dpr |
|---|---|---|
| gate-sensitive (condensed) | **14** (16) | **4** (3) |
| rerank-sensitive (condensed) | 83 (81) | 86 (85) |
| mixed-stable (condensed) | 0 (0) | 7 (9) |
| reranker harms the query | 17 | 29 |
| mean G_q (median) | 1.333 (0.0) | −0.118 (0.0) |
| queries with G_q > 0 / < 0 | 33 / 26 | 10 / 19 |
| mean R_q (median) | 6.146 (3.881) | 2.204 (1.431) |
| mean DCG@10 of the cascade | 14.603 | 16.055 |

G_q is negative more often than positive under dpr: removing the gate admits
documents the reranker places above the survivors but which carry less gain.
Part of that is pooling, since unjudged documents count as zero; condensed,
dpr's mean G_q is −0.022 (9 positive, 16 negative). Paired per query, the dpr cascade has the
higher DCG@10 on 38 queries, bm25 on 23, and they tie on 36.

### H7 — confirmatory for `alternative`, descriptive for `boundary`

H7 as registered: Δ_t (DCG@10 under the budget world minus the actual cascade)
is larger for gate-sensitive than for rerank-sensitive queries. Two tests:
**group** (difference of means, two-sided permutation, 10,000 permutations,
seed 42) and **continuous** (Spearman of G_q against Δ_t over all 97, one-sided).
`t` is the share of the Stage-2 budget B = K_c kept for the top of C_q; 1 − t
goes to outsiders.

**`alternative` (confirmatory)**, uncondensed, groups gate / rerank beside
every p-value:

| system | t | groups | difference | group p | Spearman rho | rho p |
|---|---|---|---|---|---|---|
| bm25 | 0.9 | 14 / 83 | 6.87 | **< 0.001** | 0.854 | < 1e-20 |
| bm25 | 0.75 | 14 / 83 | 7.68 | **< 0.001** | 0.814 | < 1e-20 |
| bm25 | 0.5 | 14 / 83 | 8.83 | **< 0.001** | 0.805 | < 1e-20 |
| dpr | 0.9 | 4 / 86 | 0.55 | 0.084 | 0.124 | 0.11 |
| dpr | 0.75 | 4 / 86 | 1.46 | 0.039 | 0.087 | 0.20 |
| dpr | 0.5 | 4 / 86 | 2.49 | **0.0007** | 0.247 | **0.0074** |

Condensed gives the same reading: bm25 at the floor at every t (rho
0.81–0.87); dpr significant on both tests at t = 0.5 only (group p 0.0005,
rho 0.249, p 0.0069) and on the group test alone at 0.75 (p 0.035).

**H7 holds for bm25 at every t, in both modes, on both tests. For dpr it
holds at t = 0.5 on both tests and nowhere else on both.**

**`boundary` (descriptive)**: bm25 is null at t = 0.9 and 0.75 and
significant at t = 0.5 (group p 0.016, rho 0.232, p 0.011). The two shard
previews disagreed here (p 0.0035 on anupam's 33, 0.71 on shubham's 32); the
full run sides with the first. dpr's
boundary is stronger than bm25's: t = 0.75 rho 0.317 (p 0.0008), t = 0.5
group p 0.0011 and rho 0.425 (p 7e-6).

**`random` (control)** is null for both systems at every t (group p 0.08–0.88,
rho between −0.29 and −0.03). That is what makes the `alternative` result mean
something other than "reranking more documents helps".

Qualifications, which must survive into the report:

  * **Report the floor as `p < 0.001`**, never as 0.0001: with 10,000
    permutations the smallest achievable p is ~1e-4.
  * **dpr's group test rests on four queries.** It is valid (the permutation
    test needs no normality), but one query moves it. Quote the group size
    with the p-value every time.
  * **No correction across t was pre-registered.** Under Holm across the
    three t values, bm25 is unchanged; dpr keeps t = 0.5 only, on both tests
    (adjusted 0.0021 group, 0.022 continuous). This is a reading aid, not a
    change to the registered test.

### Δ_t magnitudes

Mean Δ DCG@10 over 97 queries, uncondensed, with the number of queries
improved / worsened. The median is 0 in every cell: most queries are
untouched by any budget shift.

| policy | t | bm25 mean | bm25 +/− | dpr mean | dpr +/− |
|---|---|---|---|---|---|
| alternative | 0.9 | +1.332 | 35 / 17 | +0.021 | 7 / 7 |
| alternative | 0.75 | +1.444 | 36 / 20 | +0.019 | 9 / 12 |
| alternative | 0.5 | +1.075 | 38 / 27 | +0.028 | 16 / 15 |
| boundary | 0.9 | −0.022 | 12 / 6 | +0.004 | 3 / 3 |
| boundary | 0.75 | +0.037 | 19 / 15 | −0.017 | 5 / 10 |
| boundary | 0.5 | −0.371 | 25 / 29 | +0.002 | 10 / 15 |
| random | 0.9 | −0.071 | 8 / 5 | +0.024 | 5 / 1 |
| random | 0.75 | −0.208 | 14 / 15 | −0.001 | 9 / 4 |
| random | 0.5 | −0.753 | 23 / 30 | −0.001 | 14 / 8 |

The bm25 `alternative` gain is carried almost entirely by the gate-sensitive
queries: their mean Δ is +7.2 / +8.0 / +8.6 at t = 0.9 / 0.75 / 0.5, against
+0.34 / +0.34 / −0.20 for rerank-sensitive queries. Giving away even 10% of
the budget to dpr's documents buys most of the gain. Under dpr the
corresponding figures are +0.55 / +1.42 / +2.41 for its four gate-sensitive
queries and about zero for the rest. These are means of signed per-query
differences and no pooled statistic is reported, per freeze v2.

### BM25 against DPR, paired — the contrast

Exact McNemar on the gate-sensitive label, same 97 queries under both systems
(`contrast_vs_dpr.csv`):

| mode | bm25 only | dpr only | both | one-sided p | two-sided p |
|---|---|---|---|---|---|
| uncondensed | 13 | 3 | 1 | 0.011 | **0.021** |
| condensed | 15 | 2 | 1 | 0.0012 | **0.0024** |

The one query gate-sensitive under both is 87452; the three dpr-only ones are
1037798, 23849 and 405163. **Quote the two-sided p (0.021).** The direction
was hypothesised on the 33-query preview (2026-11-10, later), and those 33
queries are inside the 97, so the one-sided test is not a clean confirmation.

The same direction shows elsewhere: bm25's reranker moves more (R_q 6.15
against 2.20) and harms fewer queries (17 against 29); bm25 has 140 gating
failures over 49 queries, dpr 21 over 14.

### Unjudged documents in the top 10

Mean share of unjudged documents in each world's top 10 (median 0 throughout):

| world | bm25 | dpr |
|---|---|---|
| A (cascade) | 2.06% | 2.78% |
| B (no rerank: Stage-1 order) | 0.31% | **8.87%** |
| C (no gate) | 4.02% | 4.02% |

World C is identical for the two systems **by construction**: with no gate it
reranks all of U_q, which does not depend on the Stage-1 system. 4.02%
replaces the 3.87% the 65-query preview reported. dpr's Stage-1 top 10 is
nearly 9% unjudged against bm25's 0.3%, consistent with TREC-DL pools built
from runs closer to bm25 than to cosDPR. Unjudged documents count as zero
gain, so this lowers world B under dpr and biases dpr's E_rerank upward. dpr's
R_q still comes out below bm25's. The condensed mode removes the issue (R_q 1.96)
and changes no conclusion.

### Strata and document labels

Uncondensed, both systems: 9,700 survivors (97 × 100) and 862,755
non-survivors. Condensed (judged only): bm25 5,116 / 10,685, dpr 4,723 /
11,078. bm25's top 100 holds more judged documents than dpr's, consistent
with the pool bias above.

| | bm25 | dpr |
|---|---|---|
| non-survivors with E_gate ≠ 0 | 0.016% | 0.0024% |
| gating failures (documents) | 140 | 21 |
| survivors: gate-sensitive / jointly / rerank-sensitive | 10 / 184 / 418 | 3 / 68 / 571 |

### One axis away from the headline (`compare_one_axis.csv`)

**K_c**, DCG@10, uncondensed:

| K_c | bm25 gate-sens. | bm25 mean G_q | bm25 failures (queries) | bm25 Δ alt t=0.5 | dpr gate-sens. | dpr mean G_q | dpr failures (queries) |
|---|---|---|---|---|---|---|---|
| 100 | 14 | 1.333 | 140 (49) | +1.075 | 4 | −0.118 | 21 (14) |
| 500 | 7 | 0.412 | 56 (26) | +0.198 | 1 | −0.129 | 4 (4) |
| 1000 | 3 | 0.066 | 29 (15) | −0.180 | 0 | −0.060 | 1 (1) |

The gate's cost under bm25 falls by a factor of twenty from K_c = 100 to 1000,
and the `alternative` gain goes with it. dpr's 0 at K_c = 1000 is a count at a
different depth, not a reinstatement of the superseded "zero".

**Utility and cutoff** at K_c = 100, gate-sensitive bm25 / dpr: top-k@10
16 / 3, RBP 13 / 4, DCG@5 13 / 3, DCG@20 18 / 6; condensed 16 / 3. bm25 is
higher in every setting. These labels use the frozen `tau_query`, which is set
on the DCG@10 scale, so outside the headline they are descriptive; H7 is
not recomputed for them.

### The qualitative figure: three real gating failures

Selected by the rule fixed in `gating_examples.py` before the full run (bm25,
K_c = 100, uncondensed, worst failure per query, largest loss first). In all
three, a grade-3 passage that would have ranked **first** without the gate was
cut by bm25:

| query | excluded passage | bm25 rank | ranked first instead |
|---|---|---|---|
| 1110199 "what is wifi vs bluetooth" | 3838638, grade 3 | 177 | 8160520, grade 2 |
| 1114646 "what is famvir prescribed for" | 8117090, grade 3 | not in top 5,000 | 2647994, grade 3 |
| 1132532 "average annual income data analyst" | 6832789, grade 3 | 151 | 6189364, grade 3 |

E_gate = −7.0 for each is the document's effect, not the query's: in the last
two the cascade still put a grade-3 passage first. **Unverified:** passage
8117090 reads "Famvir®", and the raw MS MARCO collection carries mojibake; if
the index holds a garbled token, the famvir case is an encoding artefact rather
than a lexical miss. Check the raw text (Colab cell without `ftfy`) before the
example goes in the report.

### Not here yet

  * **No intervals.** `bootstrap_iterations` / `bootstrap_alpha` have no
    consumer yet; bootstrap CIs on the headline effects are the next code
    change to `analysis.py`, and this entry gets a follow-up when they exist.
  * `headline_metric_gain_map`, candidate-set Recall@K and the
    `merge_cache.py` completeness flag are open as listed by Anupam on
    2026-11-12.

### File digests (sha256, first 16 hex)

| file | digest |
|---|---|
| `analysis/real_bm25_kc100/h7.csv` | `70289914f6d54a0d` |
| `analysis/real_bm25_kc100/contrast_vs_dpr.csv` | `fd35c4a5b7ec1241` |
| `analysis/real_bm25_kc100/budget_outcomes.csv` | `d569aee7be855612` |
| `analysis/real_bm25_kc100/unjudged_at_k.csv` | `fb9c9d7b7c34a47f` |
| `analysis/real_bm25_kc100/diagnosis.csv` | `dd947eda4a28b90a` |
| `analysis/real_dpr_kc100/h7.csv` | `ea018511b179bdac` |
| `analysis/real_dpr_kc100/contrast_vs_bm25.csv` | `06ad7dad142aa8a2` |
| `analysis/real_dpr_kc100/budget_outcomes.csv` | `97213dfa51180da5` |
| `analysis/real_dpr_kc100/unjudged_at_k.csv` | `65b6889c529cd10e` |
| `analysis/real_dpr_kc100/diagnosis.csv` | `03a47f9c0b5d3260` |
| `analysis/compare/compare_one_axis.csv` | `87b5acbe972c70c2` |
| `analysis/gating_examples/examples.csv` | `6e3d18f4bca173e3` |

To regenerate: the commands in `experiments_shubham/README.md` under "Real
effects", "Sensitivity" and "Qualitative figure", with `cache/stage2.sqlite`
in place.

---

## 2026-11-13 (later) — Bootstrap intervals on the headline numbers (Shubham)

The follow-up the previous entry promised. `analysis.py` now writes
`intervals.csv` and, with `--contrast-with`, `contrast_intervals_vs_<other>.csv`.
`predictors.bootstrap_iterations` and `bootstrap_alpha` finally have a
consumer.

**Method.** Percentile bootstrap over **queries** (`shared.metrics.bootstrap_ci`,
unchanged), 1,000 resamples, 95%, seed 42. Every statistic is drawn from the
same 1,000 resamples. Paired system differences resample query **pairs**.
Each point estimate is the number already in the previous entry. Adding the
intervals changed none of the existing files: all ten digests recorded there
still match.

### Per system (K_c = 100, DCG@10, uncondensed; condensed in brackets)

| | bm25 | dpr |
|---|---|---|
| mean DCG@10 | 14.60 [13.03, 16.38] | 16.06 [14.40, 17.76] |
| mean G_q | 1.33 [0.58, 2.23] | −0.12 [−0.44, 0.16] |
| mean R_q | 6.15 [4.98, 7.31] | 2.20 [1.74, 2.71] |
| share gate-sensitive | 14.4% [7.2, 21.6] | 4.1% [1.0, 8.2] |
| share reranker harms | 17.5% [10.3, 24.7] | 29.9% [20.6, 39.2] |

Condensed: bm25 G_q 1.51 [0.77, 2.39], gate-sensitive 16.5% [10.3, 23.7];
dpr G_q −0.02 [−0.22, 0.20], gate-sensitive 3.1% [0.0, 6.2].

### H7, `alternative`

| system | t | mean Δ | group difference (draws) | Spearman rho |
|---|---|---|---|---|
| bm25 | 0.9 | 1.33 [0.67, 2.14] | 6.87 [4.15, 10.15] | 0.854 [0.734, 0.936] |
| bm25 | 0.75 | 1.44 [0.74, 2.29] | 7.68 [5.05, 10.96] | 0.814 [0.681, 0.915] |
| bm25 | 0.5 | 1.08 [0.27, 2.00] | 8.83 [6.24, 11.82] | 0.805 [0.671, 0.889] |
| dpr | 0.9 | 0.02 [−0.09, 0.12] | 0.55 [−0.07, 1.71] (903) | 0.124 [−0.207, 0.470] |
| dpr | 0.75 | 0.02 [−0.14, 0.22] | 1.46 [−0.64, 5.02] (903) | 0.087 [−0.201, 0.411] |
| dpr | 0.5 | 0.03 [−0.18, 0.24] | 2.49 [0.40, 5.54] (903) | 0.247 [−0.025, 0.504] |

For bm25 every interval excludes zero, at every t, in both modes. For dpr the
group difference excludes zero at t = 0.5 only, and the mean gain from the
policy is indistinguishable from zero at every t.

### BM25 minus DPR, paired

| | uncondensed | condensed |
|---|---|---|
| share gate-sensitive | +10.3 pp [2.1, 18.6] | +13.4 pp [6.2, 21.6] |
| share reranker harms | −12.4 pp [−22.7, −2.0] | −13.4 pp [−23.7, −3.1] |
| mean R_q | +3.94 [2.71, 5.18] | +4.20 [2.98, 5.44] |
| mean DCG@10 | −1.45 [−2.32, −0.71] | −1.53 [−2.37, −0.81] |
| mean G_q | +1.45 [0.71, 2.32] | +1.53 [0.81, 2.37] |

### How to read them, which the report must keep

  * **The G_q and DCG rows are one finding, not two.** World C ranks all of
    U_q by the reranker and does not depend on the Stage-1 system, so per
    query G_q(bm25) − G_q(dpr) = DCG_act(dpr) − DCG_act(bm25), exactly.
    "BM25's gate costs 1.45 more" and "DPR's cascade scores 1.45 higher" are
    the same statement. The label contrast is not reducible to this, because
    the labels also compare G_q with R_q.
  * **An interval is not the test.** H7 and the contrast are decided by the
    registered permutation test and the exact McNemar test. The intervals say
    how precise each number is. Where the two disagree, the test stands. bm25
    `random` at t = 0.5 is the case in point: its group-difference interval,
    [0.13, 1.74], excludes zero, but the permutation p is 0.19. Percentile
    intervals for a difference of means are known to be too narrow when one
    group is small and skewed (14 queries, mostly zeros).
  * **95% two-sided is not the one-sided test.** dpr's rho at t = 0.5 has an
    interval of [−0.03, 0.50] and a one-sided p of 0.0074. These do not
    contradict each other: a one-sided 5% test corresponds to the lower bound
    of a 90% interval, not a 95% one.
  * **dpr's group intervals are conditional.** 97 of 1,000 resamples (205
    condensed) drew fewer than two of its four gate-sensitive queries and were
    dropped, as `h7_test` would refuse them. Those intervals hold only for
    resamples where the test is defined; quote them with that.

| file | digest |
|---|---|
| `analysis/real_bm25_kc100/intervals.csv` | `97b5f8289fb14e75` |
| `analysis/real_dpr_kc100/intervals.csv` | `de24638df8570811` |
| `analysis/real_bm25_kc100/contrast_intervals_vs_dpr.csv` | `c609cf950aaa102e` |
| `analysis/real_dpr_kc100/contrast_intervals_vs_bm25.csv` | `c577adaa217c1f05` |

---

## 2026-11-13 (later) — Candidate-set Recall@K_c, and recall is NOT the same for the two systems (Shubham)

The last §12.1 pre-run check, which Anupam handed over: `feasibility_report.py`
now reports Recall@K_c = |C_q ∩ R_q| / |R_q| per (query, system, K_c), with R_q
the query's gain-carrying judged documents (grades 2 and 3). Its summary gives
the mean, the median and the number of queries with a defined recall. Re-run on
the frozen universe, **every pre-existing number in
`analysis/feasibility_manifest.json` is reproduced exactly**, headline K_c = 100
included. The only change is the three added fields per (K_c, system).

| K_c | bm25 mean (median) | dpr mean (median) | dpr − bm25, paired | dpr higher / bm25 higher / tie |
|---|---|---|---|---|
| 100 | 0.529 (0.500) | **0.670** (0.750) | +0.141, Wilcoxon p ≈ 1e-6 | 65 / 18 / 14 |
| 500 | 0.714 (0.833) | 0.806 (0.895) | +0.092, p 0.002 | 46 / 25 / 26 |
| 1000 | 0.770 (0.897) | 0.843 (0.929) | +0.072, p 0.006 | 39 / 24 / 34 |

**Validation.** bm25's mean Recall@1000 over the 97 queries is **0.7703**. That
is exactly the query-weighted mean of its published TREC-DL recall@1000,
(43 × 0.7501 + 54 × 0.7863) / 97 = 0.7703, from `verify_bm25.py`. The measure
is the standard one and the candidate sets are the right ones.

### Correction to the 2026-11-10 (later) and 2026-11-12 entries

Both entries say the bm25-vs-dpr contrast is "a stronger claim than a recall
difference, because recall of gainful documents is the same" (2026-11-12:
"identical"). **It is not the same.** Those entries inferred recall from the
*median* count of gain-carrying non-survivors, which is 6.0 for both. The
means differ (27.9 against 22.7 at K_c = 100), and so does recall: dpr's top
100 keeps 14 points more of each query's relevant documents, on 65 queries
against 18.

The recall gap also tracks the finding itself:

  * on all **13 of 13** queries that are gate-sensitive under bm25 but not
    dpr, dpr's Recall@100 is higher;
  * bm25's 14 gate-sensitive queries have a mean Recall@100 of **0.16**,
    against 0.59 for its other 83 queries;
  * across the 97 queries, the paired recall gap (dpr − bm25) correlates with
    the paired G_q gap (bm25 − dpr): Spearman 0.47, p 1e-6.

What this does and does not change:

  * **Unchanged:** every effect, H7 result and the McNemar contrast. They are
    measured, not inferred from recall.
  * **Changed: the explanation.** "bm25 is gate-sensitive more often because
    its top 100 misses more of the relevant passages" is now a sufficient
    descriptive account, and the data cannot separate it from "paradigm
    alignment with the cross-encoder". The two may be one effect: the
    passages a lexical retriever misses can be exactly the semantically
    relevant ones a dense cross-encoder ranks highly. But the report cannot
    claim the contrast is *beyond* a recall difference. The sentence "stronger
    than a recall difference" should not appear in the report or the video.
  * Both retrievers' Recall@100 belongs beside the gate results as context: at
    the headline depth, bm25's candidate set holds about half of each query's
    relevant passages.

Files: `analysis/feasibility_density.parquet` gains the `recall_cq` column (the
parquets are gitignored; the manifest carries the summary).

---

## 2026-11-13 (gain map) — `headline_metric_gain_map` is descriptive (Shubham, agreed with Anupam)

Freeze v2 left two gain maps: `gain_map` {0: 0, 1: 0, 2: 3, 3: 7}, where grade 1
("Related") earns nothing, and `headline_metric_gain_map` {0: 0, 1: 1, 2: 3,
3: 7}, the official-convention one, with the rule that if they differ both are
reported. They differ, and the second one has never had a consumer.

**Decision: document it as descriptive rather than wire it.** Every reported
number — effects, labels, H7, the contrast, the intervals, the sensitivity
grid — uses `gain_map`. The rule exists for comparing nDCG against published
TREC-DL figures, and the report makes no such comparison. Its only published
comparison is bm25's Recall@1000 (0.7703), which is binary at grade ≥ 2 and
does not go through a gain map. Wiring the second map would add a parallel set
of numbers that nothing in the report reads.

What the report must therefore say once: utilities are DCG@k with grade 1 at
zero gain, so they are not comparable with published nDCG@10. The key stays in
`config/base.yaml` with a comment, as `alternative_source` did. Removing it
would change fingerprint `46d0807d8ed3`; the comment does not, and that was
checked after the edit.

---

<!-- next entry below -->

---

### 2026-11-13 — Chaitanya: E1 engine validation stays; provisional synthetic predictors superseded; merge_cache provenance fix

**Owner:** Chaitanya  
**Commits:** `ac78c2c`, `7ef19bb`, `<next>`

#### Correction: E1 engine validation is valid; only early synthetic predictor trials are superseded

Shubham clarified that `effects/e1_manifest.json` (owner: shubham, 2026-10-27) is
Experiment E1 — the counterfactual engine validation (81.6M rows, independent
re-derivation with max |diff| 0.0, plus the five engine sanity checks across stress
variants). It contains no predictors, features, or fidelity metrics.

**Clarification and decision:**

1. **E1 engine validation stands and stays in the report.**
   `effects/e1_manifest.json` validates the mathematical correctness of the
   counterfactual engine across 81.6 million rows. It has no dependency on the
   Stage-2 cache or feature boundaries, and remains valid and cited in the report.

2. **What is superseded:** Only the provisional synthetic predictor trials (from
   late October in `experiments_chaitanya/synthetic_results/`, which predated the
   survivor-boundary correction `rank ≤ k_c`). Those early synthetic numbers
   leaked non-survivor Stage-2 scores and must not appear in the report or video.

3. **Empirical predictor results:** `run_real_predictors.py` evaluates predictors
   on the real Stage-2 cache (872,455 rows, `row_count_ok: true`) and frozen
   universe under the freeze-v2 survivor boundary, writing to
   `effects/real_predictors/`.

**Action:** `effects/e1_manifest.json` stays in the report as the primary validation
of the counterfactual engine. Empirical predictor fidelity and cost-fidelity curves
are cited exclusively from `effects/real_predictors/`.

#### `merge_cache.py` — provenance fix for `--skip-completeness`

The manifest `note` previously claimed *"Completeness verified against the frozen
universe"* unconditionally, even when `--skip-completeness` was passed. The
manifest now records:

```json
"completeness_checked": true   // or false when --skip-completeness was used
```

and the `note` is conditional: a skipped run now states explicitly that
per-query coverage was not verified. The actual merge that produced the live
`cache/stage2.sqlite` did run the full check (`completeness_checked: true`,
`row_count_ok: true`, 872,455 rows). This fix is forward-looking — it protects
the provenance of any future re-merge.


