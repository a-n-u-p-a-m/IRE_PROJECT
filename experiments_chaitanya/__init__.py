"""
experiments_chaitanya — reranker wrapper, cache filler, feature extraction,
effect prediction and cost-fidelity harness.

Import structure:
  - reranker       PointwiseCrossEncoder, ListwiseCrossEncoderStub
  - passage_fetcher DictPassageFetcher, IrDatasetsPassageFetcher, make_fetcher
  - cache_filler    CacheFiller, FillReport, load_query_texts
  - features        extract_stage1_features, extract_stage2_features, ...
  - predictors      EffectPredictor, fit_all_predictors, evaluate_all_predictors
  - cost_fidelity   compute_cost_fidelity_curve, convergence_check, synthetic_cost_fidelity
  - shard_plan      build_shard_plan
  - throughput      measure_throughput, run_synthetic_benchmark

Import from shared/ only. Never import another member's module directly.
"""
