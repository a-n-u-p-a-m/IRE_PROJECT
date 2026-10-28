# experiments_anupam

**Owner: Anupam.** Project setup and design freeze; BM25 retrieval; the frozen
universe; reference subset; outsider allocation; final integration.

Import from `shared/`. Never import another member's module — go through the
file schemas in `shared/schema.py`.

## Modules

| Module | What it owns |
|---|---|
| `data_prep.py` | Query and qrels loaders, the gain-map coverage check, the train/val/test split **by query** |
| `retrievers.py` | The `Retriever` interface, `PyseriniBM25`, `PrecomputedRun`, scores over `U_q` and the floor policy |
| `run_stage1.py` | A Stage-1 run to depth M, with the two depth refusals and the checkpoint recorded |
| `build_universe.py` | Builds and **freezes** `U_q`; refuses to overwrite a frozen universe |
| `smoke_pipeline.py` | Drives the whole chain end to end with no retriever |
| `verify_bm25.py` | Reproduces published BM25 effectiveness, or refuses |
| `feasibility_report.py` | The Week-2 gate: leverage, judgment density, gate-failure supply |

Still to come (Weeks 2–4): the reference-subset sampler runner, the
outsider-allocation runner and the reproducibility pack.

## Running it

    make test-all            # 341 tests: shared, Shubham's, mine
    make smoke-week1         # the whole chain, no Pyserini, ~5s
    python -m pytest experiments_anupam     # mine alone, 99 tests

`smoke_pipeline.py` is the one to run after any change to a shared file. The
unit suites prove each module correct; the smoke run proves they **compose** —
that `run_stage1`'s output is what `build_universe` reads, that manifests land
where the next step looks for them, that the fingerprint one step writes is the
one the next step demands. It fakes exactly one thing, the retriever, so when
the real BM25 run lands the retriever is the only untested variable.

## The real sequence

Needs Pyserini, a JDK and the prebuilt index, so it runs on a real machine and
not in CI:

    python -m experiments_anupam.data_prep \
        --queries data/queries.tsv --qrels-txt data/qrels.txt

    python -m experiments_anupam.run_stage1 --system bm25        # top-M, all 97 queries

    python -m experiments_anupam.verify_bm25     # before trusting that run

    # once Shubham's runs/dpr.parquet has landed (H1):
    python -m experiments_anupam.build_universe --qrels data/qrels.parquet
    python -m experiments_anupam.feasibility_report

Order matters. `build_universe` requires **every** configured universe source
to be present, because adding a source afterwards means regenerating `U_q` and
recomputing every effect derived from it. Run it once, after both runs exist.

## Three things that are refusals, not bugs

**`run_stage1` refuses a shallow depth.** Below M for a universe source, or
below the largest `K_c` for a Stage-1 system. A shallower run would make the
frozen universe shallower than the config claims, or leave a system unable to
fill its own candidate set.

**`build_universe` refuses to overwrite a frozen universe.** Every
counterfactual claim in the project is relative to `U_q`. `--force` exists, and
using it means recomputing everything downstream.

**`verify_bm25` exits non-zero when effectiveness misses the published
figures.** Every other check verifies that a run is *well formed*; a run built
against the wrong index or with mangled query text passes all of them. This is
the only one that checks the run is *right*, so it gates the handoff rather
than printing a number.

## Two conventions that are easy to get wrong

**`verify_bm25` does not use `shared.utility`, deliberately.** The project's
gain map differs from trec_eval's nDCG in two independent ways: it scores
TREC-DL grade 1 ("Related") as 0, and its `{2: 3, 3: 7}` is an exponential
`2^g - 1` scale. trec_eval's `ndcg_cut` uses **linear** gain and counts grade 1.
Both choices are right for this project and wrong for reproducing a published
number, so the official measures are implemented separately and cross-checked
against `pytrec_eval`. Recall binarises at grade ≥ 2, which is the TREC DL
track's own convention and why nDCG and recall disagree about "relevant".

**"Relevant" means gain-carrying, not `relevance > 0`.** `shared/qrels.py`
(lines 135 and 174) counts any judged document with `relevance > 0`, which
includes grade 1 — but grade 1 gains 0.0 under the frozen map, so excluding one
is not a gate failure. `feasibility_report.py` uses the gain map and reports
the shared definition beside it, because the headline-`K_c` decision is made
from that count and the looser definition inflates it.

## The floor policy

`scores_from_run(..., missing="floor")` gives any universe document the
retriever did not return a score strictly below everything it did return.
That asserts only what the retriever actually said — *not in my top-M* — and
never lets an unretrieved document outrank a retrieved one. `floor_counts`
reports the floored fraction per query, which belongs beside any
cross-paradigm result: a retriever that floors most of `U_q` is doing far less
work than the comparison assumes. `missing="error"` refuses to invent scores at
all, for the paths where a gap is a bug rather than a fact about depth.
