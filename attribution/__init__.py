"""
attribution/ — Stage-level attribution methods (roadmap 1.7).

Each module exposes METHOD_NAME and
    compute_attribution(score_matrix, top_k=10) -> DataFrame
returning the agreed attribution schema for documents in the final top-k.
"""
