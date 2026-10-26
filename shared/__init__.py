"""
shared/ - The common layer all three streams build on.

Nothing here imports torch, pyserini or faiss at module level, so the whole
analysis layer, the synthetic arm and the entire test suite run on a laptop.
Only the Stage-2 cache FILLER needs a GPU, and that lives in each member's own
experiment code, not here.

Modules, in dependency order:

    schema      the six-table data contract; sign invariants conditional on
                effect_type, because the three estimands do not share them
    config      the design freeze, its coherence checks and its fingerprint
    utility     U_d(pi) under an explicit, dataset-specific gain map
    ordering    the single deterministic sort every world uses
    provenance  seeding, timing, run manifests, freeze-mismatch detection
    qrels       judgments, the judged/unjudged flag, coverage
    universe    build/freeze/load U_q; the reference-subset sampler
    cache       the persistent Stage-2 score store; shard merge; completeness
    allocation  budget splitting and the three outsider policies
    worlds      the four counterfactual worlds and their isolation checks
    strata      survivorship and allocation strata, and why they are mandatory
    effects     E_rerank, E_gate, Delta_t and the invariant checks
    diagnosis   query-level G_q/R_q labels, the H7 test, the power simulation
    metrics     stratified fidelity, bootstrap over queries
    baselines   zero, membership-only and stratum-mean - the bars to clear
    synthetic   the controlled cascade generator
    fixtures    the shared toy cascade and `smoke_test()`

Four facts the whole package is organised around:

  1. For a pointwise reranker s2(q, d) does not depend on the candidate set, so
     ONE inference pass over U_q serves every K_c, every t, every policy and
     both the actual and no-gate worlds.

  2. Once C_q is fixed the final ranking is invariant to s1. Stage 1's only
     causal channel is MEMBERSHIP in C_q - except under a budget constraint,
     where Stage-1 rank decides who survives the cut and becomes causally
     active again. That is what world D probes.

  3. sign(E_gate) is fixed by survivorship and three of four sign(Delta_t)
     groups are fixed by allocation membership. Every statistic is therefore
     computed within stratum, and the pooled variant is behind an explicit
     call that labels its own output secondary.

  4. Unjudged is unknown, never non-relevant - but unjudged documents still
     occupy ranks as competitors, because the policy being simulated really
     would have ranked them.
"""

__all__ = [
    "allocation", "baselines", "cache", "config", "diagnosis", "effects",
    "fixtures", "metrics", "ordering", "provenance", "qrels", "schema",
    "strata", "synthetic", "universe", "utility", "worlds",
]
