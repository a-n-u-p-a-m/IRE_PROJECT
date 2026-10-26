"""
config.py - Load, validate and fingerprint the design freeze.

The coherence checks are the useful part. Each one refuses a configuration
that is internally contradictory in a way that would still run and still
produce plausible-looking numbers that mean nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a frozen configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class FrozenConfig:
    # Dataset
    dataset_name: str
    collection: str
    granularity: str
    segmentation: dict[str, Any] | None
    queries_file: str
    qrels_file: str
    split_fractions: dict[str, float]

    # Universe
    universe_sources: tuple[str, ...]
    optional_sources: tuple[str, ...]
    universe_depth_m: int
    universe_max_size: int

    # Cascade
    stage1_systems: tuple[str, ...]
    candidate_depths: tuple[int, ...]
    final_cutoff_k: int
    cutoff_sensitivity: tuple[int, ...]

    # Stage 2
    reranker_model: str
    reranker_max_length: int
    reranker_batch_size: int
    reranker_precision: str
    reranker_pointwise: bool

    # Budget / world D
    b_equals_kc: bool
    b_absolute: int | None
    t_grid: tuple[float, ...]
    t_grid_optional: tuple[float, ...]
    outsider_policies: tuple[str, ...]
    candidate_portion_rule: str
    alternative_source: str
    budget_seeds: tuple[int, ...]

    # Utility
    utility_kind: str
    sensitivity_kinds: tuple[str, ...]
    rbp_persistence: float
    gain_map: dict[int, float]
    headline_gain_map: dict[int, float]
    unjudged_policy: str

    # Reference subset
    reference_size: int
    reference_strategy: str
    reference_n_strata: int
    reference_seed: int

    # Diagnosis
    tau_query: float
    tau_document: float
    record_signed_rerank_sensitivity: bool

    # Predictors
    boundary_budgets: tuple[int, ...]
    outsider_budgets: tuple[int, ...]
    top_m: int
    bootstrap_iterations: int
    bootstrap_alpha: float
    report_pooled: bool

    # Hashing (optional extension)
    hashing_enabled: bool
    hashing_method: str | None
    hashing_representation: str | None
    hashing_bits: int | None

    # Synthetic + power
    synthetic: dict[str, Any] = field(default_factory=dict)
    power: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    seed: int = 42
    tie_break: str = "doc_id"

    # ── derived ──────────────────────────────────────────────────────

    def fingerprint(self) -> str:
        """Stable 12-character hash of the entire configuration."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def budget_for(self, k_c: int) -> int:
        """
        The reranker budget B at a given candidate depth.

        When b_equals_kc is true, t=1 reproduces the conventional cascade
        exactly. Otherwise t=1 is only a candidate-only budget baseline, and
        every report built on it must say so.
        """
        return int(k_c) if self.b_equals_kc else int(self.b_absolute)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d


_REQUIRED_SECTIONS = [
    "dataset", "universe", "cascade", "stage2", "budget", "utility", "qrels",
    "reference_subset", "diagnosis", "predictors", "synthetic", "power",
]


def load_config(path: str | Path) -> FrozenConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    missing = [k for k in _REQUIRED_SECTIONS if k not in raw]
    if missing:
        raise ConfigError(f"config missing sections: {missing}")

    ds, uni, casc = raw["dataset"], raw["universe"], raw["cascade"]
    s2, bud, util = raw["stage2"], raw["budget"], raw["utility"]
    ref, diag, pred = raw["reference_subset"], raw["diagnosis"], raw["predictors"]
    hsh = raw.get("hashing", {})

    cfg = FrozenConfig(
        dataset_name=ds["name"],
        collection=ds["collection"],
        granularity=ds["granularity"],
        segmentation=ds.get("segmentation"),
        queries_file=ds["queries_file"],
        qrels_file=ds["qrels_file"],
        split_fractions=dict(ds["split_fractions"]),
        universe_sources=tuple(uni["sources"]),
        optional_sources=tuple(uni.get("optional_sources", [])),
        universe_depth_m=int(uni["depth_m"]),
        universe_max_size=int(uni["max_size"]),
        stage1_systems=tuple(casc["stage1_systems"]),
        candidate_depths=tuple(int(x) for x in casc["candidate_depths"]),
        final_cutoff_k=int(casc["final_cutoff_k"]),
        cutoff_sensitivity=tuple(int(x) for x in casc.get("cutoff_sensitivity", [])),
        reranker_model=s2["model"],
        reranker_max_length=int(s2["max_length"]),
        reranker_batch_size=int(s2["batch_size"]),
        reranker_precision=s2.get("precision", "float32"),
        reranker_pointwise=bool(s2.get("pointwise", True)),
        b_equals_kc=bool(bud["b_equals_kc"]),
        b_absolute=(None if bud.get("b_absolute") is None else int(bud["b_absolute"])),
        t_grid=tuple(float(x) for x in bud["t_grid"]),
        t_grid_optional=tuple(float(x) for x in bud.get("t_grid_optional", [])),
        outsider_policies=tuple(bud["outsider_policies"]),
        candidate_portion_rule=bud.get("candidate_portion_rule", "stage1_rank"),
        alternative_source=bud.get("alternative_source", ""),
        budget_seeds=tuple(int(x) for x in bud.get("seeds", [42])),
        utility_kind=util["kind"],
        sensitivity_kinds=tuple(util.get("sensitivity_kinds", [])),
        rbp_persistence=float(util.get("rbp_persistence", 0.8)),
        gain_map={int(k): float(v) for k, v in util["gain_map"].items()},
        headline_gain_map={
            int(k): float(v)
            for k, v in util.get("headline_metric_gain_map", util["gain_map"]).items()
        },
        unjudged_policy=raw["qrels"].get("unjudged", "unknown"),
        reference_size=int(ref["size"]),
        reference_strategy=ref["strategy"],
        reference_n_strata=int(ref.get("n_strata", 5)),
        reference_seed=int(ref.get("seed", 42)),
        tau_query=float(diag["tau_query"]),
        tau_document=float(diag["tau_document"]),
        record_signed_rerank_sensitivity=bool(
            diag.get("record_signed_rerank_sensitivity", True)
        ),
        boundary_budgets=tuple(int(x) for x in pred["boundary_budgets"]),
        outsider_budgets=tuple(int(x) for x in pred["outsider_budgets"]),
        top_m=int(pred.get("top_m", 20)),
        bootstrap_iterations=int(pred.get("bootstrap_iterations", 1000)),
        bootstrap_alpha=float(pred.get("bootstrap_alpha", 0.05)),
        report_pooled=bool(pred.get("report_pooled", False)),
        hashing_enabled=bool(hsh.get("enabled", False)),
        hashing_method=hsh.get("method"),
        hashing_representation=hsh.get("representation_source"),
        hashing_bits=(None if hsh.get("bits") is None else int(hsh["bits"])),
        synthetic=raw["synthetic"],
        power=raw["power"],
        paths=raw.get("paths", {}),
        seed=int(raw.get("seed", 42)),
        tie_break=raw.get("tie_break", "doc_id"),
    )
    _check_coherence(cfg)
    return cfg


def _check_coherence(cfg: FrozenConfig) -> None:
    # Every core Stage-1 retriever must contribute to U_q, or its candidate set
    # can fall outside the universe and C_q subset-of U_q fails.
    orphans = set(cfg.stage1_systems) - set(cfg.universe_sources)
    if orphans:
        raise ConfigError(
            f"Stage-1 systems {sorted(orphans)} are not universe sources. Every "
            "Stage-1 retriever must contribute its top-M to U_q before it is "
            "used, or its candidates can escape the universe. Adding one later "
            "means regenerating U_q and recomputing every effect."
        )

    # U_q == C_q makes the no-gate world identical to the actual one.
    if cfg.universe_depth_m <= max(cfg.candidate_depths):
        raise ConfigError(
            f"universe depth M={cfg.universe_depth_m} must exceed the largest "
            f"candidate depth K_c={max(cfg.candidate_depths)}, else C_q == U_q "
            "and E_gate is identically zero."
        )

    if cfg.final_cutoff_k > min(cfg.candidate_depths):
        raise ConfigError(
            f"final cutoff k={cfg.final_cutoff_k} exceeds the smallest candidate "
            f"depth K_c={min(cfg.candidate_depths)}; the actual ranking would be "
            "shorter than the evaluation cutoff."
        )

    # World D.
    if not cfg.b_equals_kc and cfg.b_absolute is None:
        raise ConfigError(
            "budget.b_equals_kc is false but b_absolute is unset. Either tie B "
            "to K_c so that t=1 is the conventional cascade, or state B."
        )
    if 1.0 not in cfg.t_grid:
        raise ConfigError(
            "budget.t_grid must contain 1.0: it is the baseline every Delta_t "
            "is measured against."
        )
    if any(not 0.0 <= t <= 1.0 for t in cfg.t_grid + cfg.t_grid_optional):
        raise ConfigError("budget t values must lie in [0, 1]")
    if "alternative" in cfg.outsider_policies and not cfg.alternative_source:
        raise ConfigError(
            "outsider policy 'alternative' needs budget.alternative_source: it "
            "selects outsiders a different retriever ranked highly."
        )
    if cfg.alternative_source and cfg.alternative_source not in (
        set(cfg.universe_sources) | set(cfg.optional_sources)
    ):
        raise ConfigError(
            f"budget.alternative_source '{cfg.alternative_source}' is not a "
            "universe source, so its opinion is undefined on most of U_q."
        )

    # Utility and judgments.
    if cfg.utility_kind not in {"dcg", "topk", "rbp"}:
        raise ConfigError("utility.kind must be dcg, topk or rbp")
    if not cfg.gain_map:
        raise ConfigError("utility.gain_map must be given explicitly")
    if any(v < 0 for v in cfg.gain_map.values()):
        raise ConfigError("utility.gain_map contains a negative gain")
    if sorted(cfg.gain_map.values()) != list(cfg.gain_map.values()):
        # Gains must be non-decreasing in grade or the utility is not monotone
        # in relevance and the invariant derivations break.
        ordered = [cfg.gain_map[g] for g in sorted(cfg.gain_map)]
        if any(a > b for a, b in zip(ordered, ordered[1:])):
            raise ConfigError(
                "utility.gain_map must be non-decreasing in relevance grade"
            )
    if not 0.0 < cfg.rbp_persistence < 1.0:
        raise ConfigError("utility.rbp_persistence must lie in (0, 1)")

    if cfg.unjudged_policy != "unknown":
        raise ConfigError(
            "qrels.unjudged must be 'unknown'. The no-gate universe is far "
            "larger than the judgment pool, so treating unjudged as "
            "non-relevant would fabricate the evidence E_gate depends on."
        )

    # Reference subset.
    valid_strategies = {"random", "stratified_universe_size", "stratified_jaccard"}
    if cfg.reference_strategy not in valid_strategies:
        raise ConfigError(
            f"reference_subset.strategy must be one of {sorted(valid_strategies)}. "
            "Convenience selection is refused: picking queries with small |U_q| "
            "biases towards queries where the retrievers agree, which is exactly "
            "where gate effects are smallest."
        )
    if cfg.reference_size < 2:
        raise ConfigError("reference_subset.size must be at least 2")

    # Diagnosis.
    if cfg.tau_query <= 0:
        raise ConfigError("diagnosis.tau_query must be positive")

    # Granularity and cost.
    if cfg.granularity not in {"passage", "document"}:
        raise ConfigError("dataset.granularity must be 'passage' or 'document'")
    if cfg.granularity == "document" and not cfg.segmentation:
        raise ConfigError(
            "a document collection needs dataset.segmentation declared. A "
            "cross-encoder cannot consume a long document in one "
            f"{cfg.reranker_max_length}-token input, and the passage-level call "
            "count is what must be budgeted and reported, not the document count."
        )

    # Hashing.
    if cfg.hashing_enabled and not cfg.hashing_method:
        raise ConfigError(
            "hashing.enabled is true but no method is named. The extension stays "
            "disabled until the exact method and a low-cost feature-generation "
            "path are confirmed."
        )

    # Synthetic.
    for key in ("n_queries", "n_docs_per_query", "relevance_quantiles"):
        if key not in cfg.synthetic:
            raise ConfigError(f"synthetic config missing '{key}'")


__all__ = ["FrozenConfig", "ConfigError", "load_config"]
