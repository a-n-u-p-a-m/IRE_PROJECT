# Week-1 handoff — Anupam → Shubham, Chaitanya

Everything below is on `main` as of `1b0df38`. Two changes touch code you own,
and one decision needs all three of us. The rest is for reference.

## Pull and check

```bash
git pull origin main
pip install -r requirements.txt            # ruff is now in there
python -m pytest tests experiments_anupam experiments_shubham \
    --ignore=experiments_shubham/tests/test_dpr.py
```

Expect **355 collected** — 347 passed and 8 skipped, the skips being
`pytrec_eval` cross-checks that run only if you install that package (you
don't need to). `test_dpr.py` needs torch, hence the ignore.

There is now a `.gitattributes` pinning line endings to LF. If you work across
Windows and WSL on one checkout, this stops every text file showing as
modified; it bit me for half an hour.

## Delivered

| | |
|---|---|
| Queries and qrels | 97 judged TREC-DL 2019+2020 queries, 20,646 judgments |
| Split (seed 42) | train 49 / val 19 / test 29, **by query**, committed as `data/query_split.json` |
| BM25 run | top-5,000, all 97 queries, `runs/bm25.parquet` |
| Frozen universe | fingerprint `46d0807d8ed3`, `\|U_q\|` 7,045–9,986 (median 9,097) |
| Feasibility gate | `make feasibility` |

**BM25 reproduces the published figures exactly** — DL19 nDCG@10 0.5058 /
R@1000 0.7501, DL20 0.4796 / 0.7863, delta 0.0000 on all four. `make
verify-bm25` re-checks this and exits non-zero if it ever stops matching.

**H3 is delivered early**, so Chaitanya is unblocked for the Week-3 cache fill.
Note the universe parquet itself is **not** committed — it is large and
reproducible from the committed run tables plus the config. Rebuild it with:

```bash
python -m experiments_anupam.build_universe --qrels data/qrels.parquet
```

and check the output against `data/universe_manifest.json`. I verified it is
bit-identical across two runs and across Linux and Windows.

## Two shared changes that touch your code

**1. `shared/diagnosis.py` — the power denominator was conditional.**
`required_reference_size` divided detections by the number of *testable* draws,
so it answered "power given the subset was testable" and overstated what a real
subset delivers. It now divides by every simulation drawn; the conditional
figure is `power_given_testable`.

I removed `n_usable` rather than keeping it, deliberately — Shubham was already
rescaling by it in `power_h7.py`, and the two corrections together would have
cancelled the fix silently. Removing the column made that line raise
`KeyError`, which is how it got found. I deleted it and repointed one assertion.

**Shubham: your numbers should be unchanged** — your old expression reduced to
`detections / n_simulations`, which is exactly what `power` now is. But
`analysis/h7_power_manifest.json` was produced by the old code path, so please
re-run and confirm it reproduces identically. If it doesn't, that matters.

**2. `shared/qrels.py` — "relevant" now means gain-carrying.**
`coverage_report` and `judged_non_survivor_counts` counted `relevance > 0`,
which includes TREC-DL grade 1 ("Related"). The frozen gain map scores grade 1
as **0.0**, so a grade-1 document contributes nothing wherever it ranks and
excluding one is not a gate failure.

On the real data the two definitions differ by **4× to 9×**: at `K_c=1000` the
DPR candidate set has a median of 9 excluded documents with `relevance > 0` and
exactly **1** that carries gain. We were about to choose the headline `K_c` from
the first number.

Both functions now take the gain map as a **required** argument and return
`n_gainful` / `n_gainful_non_survivors`. Required rather than defaulted so the
definition travels with the config that set it — and so the change broke its
callers loudly instead of quietly returning different numbers under the same
column name. Nothing of yours calls either function, so nothing of yours broke.

## Feasibility results

Three of the four questions are settled.

**Leverage: yes.** Candidate Jaccard 0.16 / 0.14 / 0.11 at `K_c` 100 / 500 /
1000. The two Stage-1 systems select genuinely different documents, so the
cross-paradigm comparison has something to separate.

**Cross-paradigm confound: no.** I worried the 2019/2020 pools would favour
lexical runs, leaving BM25's candidate set better judged than DPR's. Measured,
it is a ratio of 1.06 (BM25 51% judged, DPR 48%). Those pools included neural
runs. **Retired as a concern.**

**Headline `K_c` = 100**, and the rule changed. My original rule — "largest
`K_c` with median gain-carrying non-survivors ≥ 1" — selected 1000, where
supply is median 1–2 with 36–39 of 97 queries having none at all. The error was
conflating two jobs: `≥ 1` is a *non-degeneracy floor*, not a selection rule,
and used as one it picked the deepest depth, which is backwards — a deeper
`K_c` swallows more of the judged pool into `C_q` and leaves *less* supply
outside it.

Separated, the rule is: among depths clearing the floor for every Stage-1
system, take the one with the greatest supply (the weakest system's), ties to
the deeper depth. That gives `K_c=100`: median supply 6, 17–23 queries empty.
500 and 1000 are still reported as the realism ladder.

**This is a post-hoc change to a pre-declared rule** and is recorded as such in
the design log. It is defensible only because gate-failure supply is measurable
*before* any effect is computed, so the choice cannot be driven by effect
sizes. No threshold was tuned: the floor is unchanged at 1, and only the
direction of the preference changed. Three tests pin the direction, including
one on a synthetic collection whose supply grows with depth, where the rule
correctly selects a deeper headline — the check that it is a rule rather than a
constant in disguise. **Please sanity-check this one rather than trusting the
green tests.**

## The open decision — needs all three of us

Judged density of `C_q` against `U_q` is **28.6× / 10.9× / 6.4×** at the three
depths. `U_q` is 1.65% judged; BM25's top-100 is 51% judged.

This is a consequence of a rule we chose deliberately, recorded in `qrels.py`:
*unjudged documents still occupy ranks as competitors, because the policy being
simulated really would have ranked them.* That rule is principled. The problem
is that at this collection's density it may make `E_gate` degenerate.

World C sorts ~9,000 documents of which ~1.65% are judged. Nearly every judged
survivor is then displaced out of the top-10 there, `U_d(π_no1) → 0`, and
`E_gate → DCG(π_act)` — the gate would appear to account for *all* retrieved
utility, as an artefact of where NIST placed judgments in 2019 rather than as a
finding. Note the bias works through **rank displacement**, so restricting the
analysis to `J_q` does not fix it: the ranking is still over all of `U_q`.

Three options:

1. **Keep the rule**, and report the judged fraction of world C's top-10 beside
   every result, treating `E_gate` as an upper bound on the gate's benefit.
2. **Add a condensed-list secondary analysis** and report both, showing whether
   the conclusion survives either way. Drop unjudged documents before ranks are
   assigned, so a judged document's rank counts only judged documents above it
   (Sakai, "Alternatives to Bpref", SIGIR 2007). `shared.utility.condense` now
   exists for this — pure and tested, and **nothing calls it**, because
   applying it contradicts the primary rule and that is not a one-person
   decision.
3. **Restrict `U_q` to the judged pool** for the gate analysis specifically.
   Cleanest statistically and nearly free (~20,700 pairs rather than 872,455),
   but it narrows the claim from "retrievable documents" to "judged documents".

I lean towards (2): it preserves the primary estimand while bounding the
artefact. Whichever we pick **carries a config fingerprint**, and the cheapest
moment to change the config is now, before the Stage-2 cache exists.

One measurement settles how much this matters — the judged fraction of world
C's top-10 — and it is available as soon as any Stage-2 scores exist. If it
comes back at 20% unjudged this is a footnote; at 95% it is the whole result.

## What I need from each of you

**Chaitanya — the throughput number, first.** This is the highest-leverage
unknown in the project. The Stage-2 bill is **872,455 unique pairs for the
entire project, not per world**: the cache keys on `(query, doc)` and every
world re-ranks a subset of `U_q`, so the union across both Stage-1 systems, all
three `K_c`, every `t` and every outsider policy *is* `U_q`, once. If your
pairs/sec puts that under an hour, we score the whole universe, the
reference-subset sampling becomes unnecessary and H2 stops being a blocker.
Please report observed mean sequence length beside pairs/sec — MS MARCO
passages average ~55 words, so padding to `max_length` 512 would multiply the
bill several times over for nothing.

**Shubham — re-record `analysis/h7_power_manifest.json`** against the fixed
`diagnosis.py` and confirm the reference size you proposed for H2 still stands.
It should; I want it confirmed rather than assumed. Also worth knowing:
`experiments_shubham/dpr.py` imports torch at module top level, so your whole
test module is uncollectable on a CPU-only machine — a lazy import inside the
constructor (as `experiments_anupam/retrievers.py` does for pyserini) would let
us all run your suite.

**Both — fifteen minutes on the open decision above**, before Chaitanya starts
filling the cache.

## Reference

Per-`K_c`, from `make feasibility`:

| `K_c` | Jaccard | density `C_q` (bm25 / dpr) | density ratio | gainful non-survivors | queries with none |
|---|---|---|---|---|---|
| 100 | 0.163 | 0.51 / 0.48 | 28.6× / 26.9× | 6 / 6 | 17 / 23 |
| 500 | 0.136 | 0.178 / 0.158 | 10.9× / 9.3× | 3 / 3 | 33 / 35 |
| 1000 | 0.115 | 0.108 / 0.094 | 6.4× / 5.7× | 2 / 1 | 36 / 39 |

`U_q` density is 0.0165 throughout; 97 queries.

Full reasoning for every decision above, including the ones still marked
`Decision: ____`, is in `docs/design_log.md` under 2026-10-27 and 2026-10-28.
