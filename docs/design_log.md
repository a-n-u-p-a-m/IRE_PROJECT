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

<!-- next entry below -->
