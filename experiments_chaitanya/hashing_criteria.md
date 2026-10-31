# Hashing Go/No-Go Criteria

## Status: Written before any hashing code

Per the ROADMAP: _"Hashing go/no-go criteria written down before any hashing code"_
and from `config/base.yaml`: `hashing.enabled: false` until these pass.

The Week-2 decision meeting will evaluate these criteria. If they are not met,
hashing is excluded and one paragraph in the Week-4 report explains why.

---

## Purpose of Hashing in This Project

Hashing features would map passages to a compact binary representation
(e.g., LSH, LEH, or a neural hash) and use the hash distance as a predictor
of Stage-2 score similarity. The goal is to **predict E_rerank / E_gate /
Delta_t without scoring every document with the full MiniLM cross-encoder**.

---

## Go/No-Go Criteria

All four criteria must be satisfied for hashing to be added (Week 4):

### Criterion 1 — Method is concrete and committed

The exact hashing method (`lsh`, `leh`, or `neural`) must be specified in
`config/base.yaml` under `hashing.method` **before** any code is written,
and a design-log entry must record the rationale.

A method that is chosen post-hoc to maximise fidelity is inadmissible —
it would conflate the hashing contribution with the selection procedure.

**Go condition:** `hashing.method` is set to a specific, non-null value in the
config freeze, with a design-log entry dated before the first hash code commit.

---

### Criterion 2 — Representation source is decided

The representation must come from a document source that is **not** the
MiniLM cross-encoder itself, because using MiniLM's own embeddings as a hash
would collapse into a re-parameterisation of the Stage-2 score, contributing
no new information.

Acceptable sources:
- BM25 term frequency vectors (sparse)
- DPR dense embeddings (already exist in Shubham's index)
- A separately trained neural hash function

**Go condition:** `hashing.representation_source` is set and the source is
confirmed to be independent of the Stage-2 model's parameters.

---

### Criterion 3 — Dimensionality (bits) is pre-specified

The number of hash bits must be set in `hashing.bits` before any experiments
are run. Post-hoc selection of dimensionality is equivalent to feature
engineering and must be treated as such (requires a train/val/test split of
queries, with test kept blind).

**Go condition:** `hashing.bits` is set to a specific integer in the config
freeze before any hash features are computed on held-out queries.

---

### Criterion 4 — Fidelity threshold clears the baseline

Hashing is only included if it improves prediction fidelity over the
**membership-only baseline** from `shared/baselines.py` on held-out
test queries. The comparison must be stratified by effect type (E_rerank,
E_gate, Delta_t) and by stratum.

A global pooled improvement is insufficient — the sign of Delta_t and E_gate
are fixed by membership, so a pooled metric that looks good but degrades
within-stratum performance is inadmissible.

**Go condition:** On test queries, within-stratum prediction fidelity
(Spearman ρ or MSE, as defined in `shared/metrics.py`) exceeds the
membership-only baseline for at least 2 of the 3 effect types.

---

## Current Status (Week 1)

- `hashing.enabled: false` in `config/base.yaml`
- No hashing code exists
- These criteria are committed so the decision meeting can evaluate them
- **Decision point: end of Week-2 meeting** — if all four criteria can be
  committed to, set `hashing.enabled: true` and record in the design log

---

## If Hashing is Out of Scope (Default)

One paragraph in the Week-4 report (Chaitanya's prediction/cost section)
will state:

> Hashing features were not included. The go/no-go criteria required a
> concrete, pre-specified method committed before any experiments, a
> representation source independent of the Stage-2 model, and a fidelity
> improvement over the membership-only baseline on held-out queries. [Reason
> criteria were not met or decision was made at Week-2 meeting.] All
> prediction results in this section use conventional features only.
