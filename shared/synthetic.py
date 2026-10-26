"""
synthetic.py - The controlled cascade generator (proposal section 7).

Real interventions give well-defined reference effects but no absolute notion
of true responsibility. Synthetic data give exact reference effects because the
whole data-generating mechanism is known.

The generator reproduces the CAUSAL STRUCTURE of a cascade, not an additive
score mixture:

    y                            latent relevance
    rel                          ordinal grade from fixed normal quantiles
    f1 = a1*y + sqrt(1-a1^2)*z   Stage-1 preference
    f2 = a2*y + sqrt(1-a2^2)*w   Stage-2 preference
    s1 = f1 + sigma1*eps1
    s2 = f2 + sigma2*eps2        eps1 independent of eps2

a1 and a2 set how strongly each stage tracks true relevance; sigma1 and sigma2
set its noise. Observed s1 and s2 correlate because both depend on y, which is
a COMMON CAUSE - never communication between the stages. eps1 and eps2 stay
independent on purpose: correlating them would inject a shared-error mechanism
a real cascade does not have.

Critically, s2 never receives s1 or rank1(d). `test_synthetic.py` residualises
both on y and asserts the partial correlation collapses to noise, so a future
edit that introduces a coupling fails loudly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIMES: dict[str, str] = {
    "strong_stages": "both stages track relevance well, little noise",
    "weak_retriever": "Stage 1 noisy, Stage 2 strong: gating is the bottleneck",
    "weak_reranker": "Stage 1 strong, Stage 2 noisy: reranking can do harm",
    "gate_stress": "relevant documents pushed to just below or far below K_c",
    "stage_mismatch": "stage preferences diverge without any score passing",
    "budget_stress": "strong outsiders sit in D_q, stressing candidate replacement",
}

_PARAMS: dict[str, dict[str, float]] = {
    "strong_stages":  dict(a1=0.85, a2=0.85, sigma1=0.15, sigma2=0.15),
    "weak_retriever": dict(a1=0.45, a2=0.85, sigma1=0.60, sigma2=0.15),
    "weak_reranker":  dict(a1=0.85, a2=0.45, sigma1=0.15, sigma2=0.60),
    "gate_stress":    dict(a1=0.70, a2=0.80, sigma1=0.30, sigma2=0.20),
    "stage_mismatch": dict(a1=0.75, a2=0.40, sigma1=0.25, sigma2=0.25),
    "budget_stress":  dict(a1=0.70, a2=0.80, sigma1=0.30, sigma2=0.20),
}


@dataclass(frozen=True)
class SyntheticParams:
    a1: float
    a2: float
    sigma1: float
    sigma2: float
    n_queries: int
    n_docs: int
    relevance_quantiles: tuple[float, ...] = (0.80, 0.93, 0.98)
    regime: str = "strong_stages"
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("a1", "a2"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {v}")
        if self.sigma1 < 0 or self.sigma2 < 0:
            raise ValueError("sigma values must be non-negative")
        q = self.relevance_quantiles
        if list(q) != sorted(q) or not all(0.0 < x < 1.0 for x in q):
            raise ValueError("relevance_quantiles must be increasing and in (0, 1)")


def params_for_regime(
    regime: str,
    n_queries: int,
    n_docs: int,
    quantiles: Sequence[float],
    seed: int = 42,
) -> SyntheticParams:
    if regime not in _PARAMS:
        raise ValueError(f"unknown regime {regime!r}; expected {sorted(_PARAMS)}")
    return SyntheticParams(
        **_PARAMS[regime],
        n_queries=n_queries,
        n_docs=n_docs,
        relevance_quantiles=tuple(quantiles),
        regime=regime,
        seed=seed,
    )


def _ordinal_relevance(y: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """
    Latent relevance to graded labels {0,1,2,3} by FIXED normal quantiles.

    Thresholds come from the standard normal, not from the sample, so the label
    distribution does not drift between regimes or sample sizes.
    """
    from scipy.stats import norm

    cuts = [norm.ppf(q) for q in quantiles]
    rel = np.zeros(len(y), dtype=int)
    for level, cut in enumerate(cuts, start=1):
        rel[y >= cut] = level
    return rel


def generate(params: SyntheticParams) -> pd.DataFrame:
    """
    Generate a synthetic collection.

    Returns one row per (query, document): query_id, doc_id, y, relevance,
    f1, f2, s1, s2.
    """
    rng = np.random.default_rng(params.seed)
    n_q, n_d = params.n_queries, params.n_docs
    total = n_q * n_d

    y = rng.standard_normal(total)
    z = rng.standard_normal(total)
    w = rng.standard_normal(total)
    e1 = rng.standard_normal(total)
    e2 = rng.standard_normal(total)

    a1, a2 = params.a1, params.a2
    f1 = a1 * y + np.sqrt(max(0.0, 1.0 - a1 * a1)) * z
    f2 = a2 * y + np.sqrt(max(0.0, 1.0 - a2 * a2)) * w

    df = pd.DataFrame(
        {
            "query_id": np.repeat([f"sq{i:04d}" for i in range(n_q)], n_d),
            "doc_id": np.tile([f"sd{j:06d}" for j in range(n_d)], n_q),
            "y": y,
            "relevance": _ordinal_relevance(y, params.relevance_quantiles),
            "f1": f1,
            "f2": f2,
            "s1": f1 + params.sigma1 * e1,
            "s2": f2 + params.sigma2 * e2,
        }
    )
    df["query_id"] = df["query_id"].astype("string")
    df["doc_id"] = df["doc_id"].astype("string")

    logger.info(
        "synthetic %s: %d queries x %d docs, a1=%.2f a2=%.2f s1n=%.2f s2n=%.2f, "
        "relevant fraction %.3f",
        params.regime, n_q, n_d, a1, a2, params.sigma1, params.sigma2,
        float((df["relevance"] > 0).mean()),
    )
    return df


# ──────────────────────────────────────────────────────────────────────
# Stress manipulations
# ──────────────────────────────────────────────────────────────────────

def apply_gate_stress(
    df: pd.DataFrame, k_c: int, n_planted: int, seed: int = 42, far: bool = False
) -> pd.DataFrame:
    """
    Push relevant documents below the Stage-1 cutoff by lowering s1 only.

    `far=True` buries them well below the boundary instead of just under it,
    which is the case a narrow boundary probe structurally cannot see - the
    documented false-negative mode, generated deliberately so it can be
    measured rather than discovered.

    No attribution label is planted: the gate failure is still DERIVED from the
    exact counterfactual.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    for _qid, grp in out.groupby("query_id", observed=True):
        rel = grp[grp["relevance"] > 0]
        if rel.empty or len(grp) <= k_c:
            continue
        take = rel.sample(
            min(n_planted, len(rel)), random_state=int(rng.integers(1 << 31))
        )
        order = grp["s1"].sort_values(ascending=False)
        just_below = float(order.iloc[k_c])
        if far:
            floor = float(order.iloc[min(len(order) - 1, int(k_c * 3))])
            out.loc[take.index, "s1"] = floor - abs(
                rng.normal(0.1, 0.05, len(take))
            )
        else:
            out.loc[take.index, "s1"] = just_below - abs(
                rng.normal(0.05, 0.02, len(take))
            )
    return out


def apply_budget_stress(
    df: pd.DataFrame, k_c: int, n_outsiders: int, seed: int = 42
) -> pd.DataFrame:
    """
    Give some discarded documents high s2 while leaving survivors' s2 alone.

    Isolates the competitive-set channel: any movement among survivors comes
    purely from which competitors are admitted, with no score coupling at all.
    This is the regime the t-experiment is designed to probe.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    for _qid, grp in out.groupby("query_id", observed=True):
        order = grp["s1"].sort_values(ascending=False)
        if len(order) <= k_c:
            continue
        excluded = order.index[k_c:]
        if not len(excluded):
            continue
        chosen = rng.choice(
            excluded, size=min(n_outsiders, len(excluded)), replace=False
        )
        top_s2 = float(grp.loc[order.index[:k_c], "s2"].max())
        out.loc[chosen, "s2"] = top_s2 + abs(rng.normal(0.5, 0.2, len(chosen)))
    return out


# ──────────────────────────────────────────────────────────────────────
# Conversion to the structures the engine consumes
# ──────────────────────────────────────────────────────────────────────

def to_lookups(
    df: pd.DataFrame,
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, int]],
]:
    """
    Split a synthetic frame into (universe, s1, s2, relevance), all keyed per
    query.

    Relevance and the scores are keyed PER QUERY because doc_ids repeat across
    queries by construction; a flat doc_id -> value map would silently collapse
    them and quietly destroy the labels.
    """
    universe: dict[str, list[str]] = {}
    s1: dict[str, dict[str, float]] = {}
    s2: dict[str, dict[str, float]] = {}
    rel: dict[str, dict[str, int]] = {}

    for qid, grp in df.groupby("query_id", observed=True):
        q = str(qid)
        docs = grp["doc_id"].astype(str).tolist()
        universe[q] = docs
        s1[q] = dict(zip(docs, grp["s1"].astype(float)))
        s2[q] = dict(zip(docs, grp["s2"].astype(float)))
        rel[q] = dict(zip(docs, grp["relevance"].astype(int)))
    return universe, s1, s2, rel


def alternative_ranks(
    df: pd.DataFrame, a_alt: float = 0.6, sigma_alt: float = 0.3, seed: int = 7
) -> dict[str, dict[str, int]]:
    """
    A third retriever's opinion, for the 'alternative' outsider policy.

    Generated from the same latent relevance with its own alignment and noise,
    so it agrees with Stage 1 only to the extent both track y - which is the
    whole point of asking a second opinion about what Stage 1 discarded.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, int]] = {}
    for qid, grp in df.groupby("query_id", observed=True):
        n = len(grp)
        y = grp["y"].to_numpy(float)
        score = (
            a_alt * y
            + np.sqrt(max(0.0, 1.0 - a_alt * a_alt)) * rng.standard_normal(n)
            + sigma_alt * rng.standard_normal(n)
        )
        docs = grp["doc_id"].astype(str).to_numpy()
        order = np.argsort(-score, kind="stable")
        out[str(qid)] = {str(docs[i]): r for r, i in enumerate(order, start=1)}
    return out


__all__ = [
    "REGIMES", "SyntheticParams", "params_for_regime", "generate",
    "apply_gate_stress", "apply_budget_stress", "to_lookups", "alternative_ranks",
]
