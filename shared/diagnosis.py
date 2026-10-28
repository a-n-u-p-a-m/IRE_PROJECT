"""
diagnosis.py - Query-level attribution and the H7 validation (section 4.2).

Every other result in this project is internal: effects defined by our own
interventions, predictors scored against those effects. H7 is the only test
that points outward. It asks whether the attribution predicts something it was
not used to compute.

    G_q = DCG@k(pi_no-gate) - DCG@k(pi_actual)      gate-recoverable gain
    R_q = |DCG@k(pi_actual) - DCG@k(pi_no-reranker)| reranking sensitivity

    gate_sensitive     G_q > tau and G_q > R_q
    rerank_sensitive   R_q > tau and R_q >= max(G_q, 0)
    mixed_stable       otherwise

Then: do gate-sensitive queries gain more from a lower t than
reranking-sensitive ones? The diagnosis comes from the reference worlds; the
outcome comes from the budget intervention. Those are genuinely different
interventions, so the test is not circular - which it WOULD be if the repair
were simply "widen K_c", since that is a partial version of the very
intervention that defined G_q.

One caveat the frozen rule carries. R_q is an absolute value, so a query where
the reranker does valuable work and a query where it actively damages the
ranking both land in `rerank_sensitive`, with opposite repairs. The signed
value is recorded alongside as `rerank_signed` so the two can be separated in
analysis without touching the frozen labelling rule.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .utility import GainMap, dcg_at_k
from .worlds import TWorld, Worlds

logger = logging.getLogger(__name__)

GATE_SENSITIVE = "gate_sensitive"
RERANK_SENSITIVE = "rerank_sensitive"
MIXED_STABLE = "mixed_stable"
QUERY_LABELS = (GATE_SENSITIVE, RERANK_SENSITIVE, MIXED_STABLE)


def query_diagnosis(
    worlds: Worlds,
    relevances: Mapping[str, int],
    k: int,
    gain_map: GainMap,
    tau_query: float,
) -> dict:
    """
    Label one query from its reference worlds.

    Requires the full no-gate world, so this is defined only on the reference
    subset. That is also what bounds H7's sample size, which is why
    `required_reference_size` exists.
    """
    dcg_act = dcg_at_k(worlds.pi_act, relevances, k, gain_map)
    dcg_no1 = dcg_at_k(worlds.pi_no1, relevances, k, gain_map)
    dcg_no2 = dcg_at_k(worlds.pi_no2, relevances, k, gain_map)

    g_q = dcg_no1 - dcg_act
    r_signed = dcg_act - dcg_no2
    r_q = abs(r_signed)

    if g_q > tau_query and g_q > r_q:
        label = GATE_SENSITIVE
    elif r_q > tau_query and r_q >= max(g_q, 0.0):
        label = RERANK_SENSITIVE
    else:
        label = MIXED_STABLE

    return {
        "query_id": worlds.query_id,
        "k_c": worlds.k_c,
        "dcg_actual": dcg_act,
        "dcg_no_gate": dcg_no1,
        "dcg_no_reranker": dcg_no2,
        "g_q": g_q,
        "r_q": r_q,
        "rerank_signed": r_signed,
        # A negative signed value means the reranker HURT this query. Labelled
        # rerank_sensitive either way by the frozen rule; split on this column
        # before drawing conclusions about what repair would help.
        "reranker_harms": bool(r_signed < 0),
        "label": label,
    }


def diagnose_queries(
    worlds_by_query: Mapping[str, Worlds],
    relevances_by_query: Mapping[str, Mapping[str, int]],
    k: int,
    gain_map: GainMap,
    tau_query: float,
) -> pd.DataFrame:
    """Diagnosis table over the reference subset."""
    rows = [
        query_diagnosis(w, relevances_by_query.get(qid, {}), k, gain_map, tau_query)
        for qid, w in sorted(worlds_by_query.items())
    ]
    out = pd.DataFrame(rows)
    if len(out):
        counts = out["label"].value_counts().to_dict()
        logger.info("query diagnosis at tau=%.4g: %s", tau_query, counts)
    return out


# ──────────────────────────────────────────────────────────────────────
# H7
# ──────────────────────────────────────────────────────────────────────

def query_delta_dcg(
    t_worlds: Mapping[float, TWorld],
    relevances: Mapping[str, int],
    k: int,
    gain_map: GainMap,
) -> dict[float, float]:
    """
    DCG@k at each t minus DCG@k at t=1, for one query.

    The query-level outcome H7 compares across diagnosis groups.
    """
    if 1.0 not in t_worlds:
        raise ValueError("need the t=1 world as the budget baseline")
    base = dcg_at_k(t_worlds[1.0].pi_t, relevances, k, gain_map)
    return {
        float(t): dcg_at_k(tw.pi_t, relevances, k, gain_map) - base
        for t, tw in t_worlds.items()
    }


def h7_table(
    diagnosis: pd.DataFrame,
    delta_dcg_by_query: Mapping[str, Mapping[float, float]],
) -> pd.DataFrame:
    """Join the diagnosis labels to the budget outcome, one row per (query, t)."""
    rows = []
    label_by_query = dict(zip(diagnosis["query_id"].astype(str), diagnosis["label"]))
    harms_by_query = dict(
        zip(diagnosis["query_id"].astype(str), diagnosis.get("reranker_harms", False))
    )
    for qid, by_t in delta_dcg_by_query.items():
        for t, delta in by_t.items():
            if float(t) == 1.0:
                continue
            rows.append(
                {
                    "query_id": str(qid),
                    "t": float(t),
                    "label": label_by_query.get(str(qid)),
                    "reranker_harms": harms_by_query.get(str(qid), False),
                    "delta_dcg": float(delta),
                }
            )
    return pd.DataFrame(rows)


def h7_test(
    table: pd.DataFrame,
    n_permutations: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Per-t comparison of delta DCG between gate-sensitive and
    reranking-sensitive queries, by permutation test.

    A permutation test rather than a t-test because the group sizes are small
    by construction - the reference subset is compute-bound - and query-level
    DCG differences are not remotely normal.

    Returns one row per t with the observed difference, a two-sided p-value and
    both group sizes. ALWAYS read the group sizes: a null result on groups of
    eight says nothing about the hypothesis.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t, block in table.groupby("t", observed=True):
        gate = block.loc[block["label"] == GATE_SENSITIVE, "delta_dcg"].to_numpy(float)
        rer = block.loc[block["label"] == RERANK_SENSITIVE, "delta_dcg"].to_numpy(float)
        n_g, n_r = len(gate), len(rer)
        if n_g < 2 or n_r < 2:
            rows.append(
                {
                    "t": float(t), "n_gate_sensitive": n_g, "n_rerank_sensitive": n_r,
                    "mean_gate": float(gate.mean()) if n_g else float("nan"),
                    "mean_rerank": float(rer.mean()) if n_r else float("nan"),
                    "difference": float("nan"), "p_value": float("nan"),
                    "note": "too few queries in one group to test",
                }
            )
            continue

        observed = float(gate.mean() - rer.mean())
        pooled = np.concatenate([gate, rer])
        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(pooled)
            diff = perm[:n_g].mean() - perm[n_g:].mean()
            if abs(diff) >= abs(observed):
                count += 1
        rows.append(
            {
                "t": float(t), "n_gate_sensitive": n_g, "n_rerank_sensitive": n_r,
                "mean_gate": float(gate.mean()), "mean_rerank": float(rer.mean()),
                "difference": observed,
                "p_value": (count + 1) / (n_permutations + 1),
                "note": "",
            }
        )
    return pd.DataFrame(rows).sort_values("t", ascending=False)


# ──────────────────────────────────────────────────────────────────────
# Power
# ──────────────────────────────────────────────────────────────────────

def required_reference_size(
    simulate_query,
    candidate_sizes: Sequence[int],
    n_simulations: int = 200,
    alpha: float = 0.05,
    target_power: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    How many reference queries H7 needs, estimated on synthetic data.

    This must be run BEFORE GPU budget is committed. The reference subset is
    the only place the full no-gate world exists, so it is the only place H7
    can run; if its size is chosen purely by compute, H7 inherits whatever
    power happens to fall out, and a null result will be uninterpretable.

    Parameters
    ----------
    simulate_query
        Callable (rng) -> (label, delta_dcg) producing one synthetic query's
        diagnosis label and its budget outcome at the t of interest. Build it
        from `shared.synthetic` so the generating mechanism is known.
    candidate_sizes
        Reference-subset sizes to evaluate.

    Returns one row per size with the estimated power. Take the smallest size
    reaching `target_power` and make that the subset size in the config.

    An UNTESTABLE draw counts as a miss. A reference subset that happens to put
    fewer than two queries in a group yields no p-value, and in a real run at
    that subset size you would simply have no answer - which is a failure to
    detect, not a sample to discard. `power` is therefore detections divided by
    `n_simulations`, never by the number of testable draws.

    That distinction is not cosmetic here. When gate-sensitive queries are rare,
    most draws are untestable and conditioning on the testable ones overstates
    power several-fold. `power_given_testable` is reported alongside as a
    diagnostic, but `meets_target` keys off the unconditional figure.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in candidate_sizes:
        detections = 0
        testable = 0
        for _ in range(n_simulations):
            labels, deltas = [], []
            for _ in range(n):
                label, delta = simulate_query(rng)
                labels.append(label)
                deltas.append(delta)
            frame = pd.DataFrame(
                {"query_id": range(n), "t": 0.5, "label": labels,
                 "reranker_harms": False, "delta_dcg": deltas}
            )
            res = h7_test(frame, n_permutations=400, seed=int(rng.integers(1 << 31)))
            if res.empty or not np.isfinite(res.iloc[0]["p_value"]):
                continue                      # untestable: counted as a miss
            testable += 1
            if res.iloc[0]["p_value"] < alpha:
                detections += 1
        power = detections / n_simulations if n_simulations else float("nan")
        rows.append(
            {
                "reference_size": int(n),
                "n_simulations": n_simulations,
                "n_testable": testable,
                "testable_fraction": (testable / n_simulations) if n_simulations else float("nan"),
                "power": power,
                "power_given_testable": (detections / testable) if testable else float("nan"),
                "meets_target": bool(np.isfinite(power) and power >= target_power),
            }
        )
    out = pd.DataFrame(rows)
    meeting = out[out["meets_target"]]
    if len(meeting):
        logger.info(
            "H7 reaches power %.2f at a reference subset of %d queries",
            target_power, int(meeting["reference_size"].min()),
        )
    else:
        logger.warning(
            "H7 does not reach power %.2f at any tested size up to %d. Either "
            "enlarge the reference subset, widen the t grid, or state in advance "
            "that H7 is descriptive rather than confirmatory.",
            target_power, int(out["reference_size"].max()),
        )
    return out


def smallest_sufficient_size(power_table: pd.DataFrame) -> int | None:
    """The smallest reference size meeting the power target, or None."""
    ok = power_table[power_table["meets_target"]]
    return int(ok["reference_size"].min()) if len(ok) else None


__all__ = [
    "GATE_SENSITIVE", "RERANK_SENSITIVE", "MIXED_STABLE", "QUERY_LABELS",
    "query_diagnosis", "diagnose_queries", "query_delta_dcg", "h7_table",
    "h7_test", "required_reference_size", "smallest_sufficient_size",
]
