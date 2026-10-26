"""
The controlled cascade generator.

The property that matters most: s2 never receives s1 or rank1(d). Observed s1
and s2 correlate, but only because both depend on the latent relevance y,
which is a COMMON CAUSE. Conditioning on y must remove the association.
`test_stages_correlate_only_through_the_common_cause` residualises both on y
and asserts the partial correlation collapses to noise, so a future edit that
introduces a score coupling fails loudly rather than quietly making the
exact-recovery results meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.synthetic import (
    REGIMES,
    SyntheticParams,
    alternative_ranks,
    apply_budget_stress,
    apply_gate_stress,
    generate,
    params_for_regime,
    to_lookups,
)
from shared.worlds import build_t_grid, build_worlds


def _params(**over):
    base = dict(a1=0.7, a2=0.8, sigma1=0.3, sigma2=0.2,
                n_queries=20, n_docs=200, seed=5)
    base.update(over)
    return SyntheticParams(**base)


def test_shape_and_columns():
    df = generate(_params())
    assert len(df) == 20 * 200
    assert set(df.columns) == {
        "query_id", "doc_id", "y", "relevance", "f1", "f2", "s1", "s2"
    }


def test_relevance_grades_follow_fixed_quantiles():
    """
    Thresholds come from the standard normal, not the sample, so the label
    distribution does not drift between regimes or sample sizes.
    """
    df = generate(_params(n_queries=50, n_docs=400))
    frac = df["relevance"].value_counts(normalize=True).sort_index()
    assert set(df["relevance"].unique()) <= {0, 1, 2, 3}
    assert frac.loc[0] == pytest.approx(0.80, abs=0.03)
    assert frac.loc[3] == pytest.approx(0.02, abs=0.01)


def test_stages_correlate_only_through_the_common_cause():
    df = generate(_params(a1=0.85, a2=0.85, sigma1=0.15, sigma2=0.15,
                          n_queries=40, n_docs=500))
    s1 = df["s1"].to_numpy(float)
    s2 = df["s2"].to_numpy(float)
    y = df["y"].to_numpy(float)

    marginal = float(np.corrcoef(s1, s2)[0, 1])
    assert marginal > 0.4, "the common cause should induce a clear association"

    def residual(x):
        return x - (np.cov(x, y)[0, 1] / np.var(y)) * y

    partial = float(np.corrcoef(residual(s1), residual(s2))[0, 1])
    assert abs(partial) < 0.05, (
        f"partial correlation given y is {partial:.3f}; s2 appears to depend on "
        "s1 beyond the common cause, which a real cascade does not do"
    )


def test_higher_alignment_means_stronger_tracking():
    weak = generate(_params(a1=0.45, sigma1=0.6, n_queries=40, n_docs=400))
    strong = generate(_params(a1=0.85, sigma1=0.15, n_queries=40, n_docs=400))
    assert (
        np.corrcoef(strong["s1"], strong["y"])[0, 1]
        > np.corrcoef(weak["s1"], weak["y"])[0, 1] + 0.2
    )


def test_parameters_are_validated():
    with pytest.raises(ValueError, match="a1 must lie"):
        _params(a1=1.5)
    with pytest.raises(ValueError, match="sigma"):
        _params(sigma1=-0.1)
    with pytest.raises(ValueError, match="relevance_quantiles"):
        SyntheticParams(a1=0.5, a2=0.5, sigma1=0.1, sigma2=0.1,
                        n_queries=2, n_docs=10, relevance_quantiles=(0.9, 0.5))


def test_generation_is_reproducible():
    a, b = generate(_params(seed=99)), generate(_params(seed=99))
    assert np.allclose(a["s1"], b["s1"]) and np.allclose(a["s2"], b["s2"])


@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_every_named_regime_has_parameters(regime):
    p = params_for_regime(regime, 10, 100, (0.8, 0.93, 0.98))
    assert p.regime == regime and 0.0 <= p.a1 <= 1.0 and 0.0 <= p.a2 <= 1.0


def test_unknown_regime_is_rejected():
    with pytest.raises(ValueError, match="unknown regime"):
        params_for_regime("S9", 10, 100, (0.8, 0.93, 0.98))


# ── lookups ───────────────────────────────────────────────────────────

def test_relevance_is_keyed_per_query():
    """
    doc_ids repeat across queries by construction, so a flat doc_id map would
    silently collapse different queries' labels onto one another.
    """
    df = generate(_params(n_queries=5, n_docs=200))
    universe, s1, s2, rel = to_lookups(df)
    queries = sorted(universe)
    shared_docs = set(universe[queries[0]])
    for q in queries[1:]:
        assert set(universe[q]) == shared_docs

    disagreeing = [
        d for d in sorted(shared_docs) if len({rel[q][d] for q in queries}) > 1
    ]
    assert disagreeing, (
        "every doc_id carries the same label in every query; the per-query "
        "keying has been lost and relevance has collapsed across queries"
    )
    d = disagreeing[0]
    assert len({s1[q][d] for q in queries}) == len(queries)


def test_lookups_feed_the_world_engine():
    df = generate(_params(n_queries=4, n_docs=120))
    universe, s1, s2, rel = to_lookups(df)
    for qid, docs in universe.items():
        w = build_worlds(qid, docs, s1[qid], s2[qid], 20)
        assert len(w.candidates) == 20
        assert set(w.pi_no1) == set(docs)
        tw = build_t_grid(w, s1[qid], s2[qid], 20, (1.0, 0.5), "boundary")
        assert set(tw) == {1.0, 0.5}


def test_alternative_ranks_are_a_permutation_per_query():
    df = generate(_params(n_queries=3, n_docs=50))
    alt = alternative_ranks(df)
    for _qid, ranks in alt.items():
        assert sorted(ranks.values()) == list(range(1, 51))


def test_alternative_opinion_differs_from_stage_one():
    df = generate(_params(n_queries=5, n_docs=200))
    _, s1, _, _ = to_lookups(df)
    alt = alternative_ranks(df)
    qid = sorted(alt)[0]
    s1_top = {d for d, _ in sorted(s1[qid].items(), key=lambda kv: -kv[1])[:20]}
    alt_top = {d for d, r in alt[qid].items() if r <= 20}
    assert s1_top != alt_top, "a second opinion identical to the first is useless"


# ── stress regimes ────────────────────────────────────────────────────

def test_gate_stress_pushes_relevant_documents_below_the_cutoff():
    df = generate(_params(n_queries=5, n_docs=200))
    k_c = 50
    stressed = apply_gate_stress(df, k_c, n_planted=10, seed=3)

    assert np.allclose(stressed["s2"], df["s2"]), "gate stress must not touch Stage 2"
    assert not np.allclose(stressed["s1"], df["s1"])

    def relevant_survivors(frame):
        return sum(
            int((grp.nlargest(k_c, "s1")["relevance"] > 0).sum())
            for _, grp in frame.groupby("query_id", observed=True)
        )

    assert relevant_survivors(stressed) < relevant_survivors(df)


def test_far_gate_stress_buries_documents_beyond_a_boundary_probe():
    """
    The documented false-negative mode of a narrow boundary probe, generated
    on purpose so the error rate can be measured rather than discovered.
    """
    df = generate(_params(n_queries=4, n_docs=300))
    k_c = 40
    near = apply_gate_stress(df, k_c, 8, seed=1, far=False)
    far = apply_gate_stress(df, k_c, 8, seed=1, far=True)

    def mean_depth_of_planted(frame):
        depths = []
        for _, grp in frame.groupby("query_id", observed=True):
            order = grp.sort_values("s1", ascending=False).reset_index(drop=True)
            rel_pos = order.index[order["relevance"] > 0].to_numpy()
            below = rel_pos[rel_pos >= k_c]
            if len(below):
                depths.append(float(below.mean()))
        return float(np.mean(depths)) if depths else 0.0

    assert mean_depth_of_planted(far) > mean_depth_of_planted(near)


def test_budget_stress_only_raises_outsider_stage2_scores():
    """
    Isolates the competitive-set channel: survivors' own s2 is untouched, so
    any movement comes purely from which competitors are admitted.
    """
    df = generate(_params(n_queries=5, n_docs=200))
    k_c = 50
    stressed = apply_budget_stress(df, k_c, n_outsiders=20, seed=3)

    assert np.allclose(stressed["s1"], df["s1"]), "budget stress must not touch Stage 1"
    assert (~np.isclose(stressed["s2"].to_numpy(), df["s2"].to_numpy())).sum() > 0

    for _qid, grp in df.groupby("query_id", observed=True):
        survivors = grp.nlargest(k_c, "s1").index
        assert np.allclose(
            stressed.loc[survivors, "s2"], df.loc[survivors, "s2"]
        ), "a survivor's Stage-2 score was modified"
