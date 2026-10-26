# experiments_chaitanya

**Owner: Chaitanya.** The reranker wrapper and Stage-2 score cache; feature
extraction; effect prediction.

Expected here: the pointwise reranker wrapper (plus a listwise stub that
refuses to run), the passage-text fetcher, the shardable cache filler and
merger, the throughput script, the conventional feature extractors, the
predictor fitting and evaluation, the cost-fidelity harness.

Import from `shared/`. Never import another member's module — go through the
file schemas in `shared/schema.py`.
