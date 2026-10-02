"""
shared/ — Common infrastructure for the stage-level attribution benchmark.

Modules:
    schema      Data contract definitions, validation, Parquet I/O
    data        Corpus, query, and qrel loading (MS MARCO + TREC-DL)
    reranker    Cross-encoder reranker wrapper (MiniLM + MonoT5)
    logger      Score/Attribution/Intervention loggers
    evaluation  All metrics + bootstrap CI
    utils       Seeding, config, timing helpers
"""
