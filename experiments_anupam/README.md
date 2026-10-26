# experiments_anupam

**Owner: Anupam.** Project setup and design freeze; BM25 retrieval; the frozen
universe; reference subset; outsider allocation; final integration.

Expected here: query and qrels loaders, the by-query split, the retriever
interface and BM25 wrapper, the Stage-1 runner, the synthetic-to-run converter,
the universe builder runner, candidate/coverage reports, the allocation runner,
and the reproducibility pack.

Import from `shared/`. Never import another member's module — go through the
file schemas in `shared/schema.py`.
