"""
utils.py — Shared utilities: seeding, config loading, timing, etc.

Usage:
    from shared.utils import set_global_seed, load_config, Timer
"""

from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────

DEFAULT_SEED = 42
INTERVENTION_SEEDS = [42, 123, 456]


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set random seed for Python, NumPy, and (if available) PyTorch.

    Note: Setting PYTHONHASHSEED here has no effect on the current process
    (it must be set before the Python interpreter starts). It is set for
    any child processes spawned via subprocess/os.system.  To guarantee
    hash reproducibility, set PYTHONHASHSEED in the shell before running
    the pipeline:  export PYTHONHASHSEED=42
    """
    random.seed(seed)
    np.random.seed(seed)
    # U4: Only affects child processes; document the limitation above
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info("Global seed set to %d", seed)


# ──────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from %s", config_path)
    return config


# ──────────────────────────────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = ""):
    """Context manager that logs elapsed time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info("%s took %.2f seconds", label or "Block", elapsed)


class Timer:
    """
    Cumulative timer for profiling multi-step pipelines.

    Usage:
        t = Timer()
        t.start("indexing")
        ... indexing code ...
        t.stop("indexing")

        t.start("retrieval")
        ... retrieval code ...
        t.stop("retrieval")

        t.report()
    """

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def start(self, label: str) -> None:
        self._starts[label] = time.perf_counter()

    def stop(self, label: str) -> float:
        if label not in self._starts:
            raise ValueError(f"Timer '{label}' was never started")
        elapsed = time.perf_counter() - self._starts.pop(label)
        self._totals[label] = self._totals.get(label, 0.0) + elapsed
        self._counts[label] = self._counts.get(label, 0) + 1
        return elapsed

    def report(self) -> dict[str, dict[str, float]]:
        """Print and return timing summary."""
        print("\n  Timing Report")
        print("  " + "-" * 50)
        result = {}
        for label in sorted(self._totals):
            total = self._totals[label]
            count = self._counts[label]
            avg = total / count if count > 0 else 0.0
            print(f"    {label:30s}  {total:8.2f}s  ({count}x, avg {avg:.2f}s)")
            result[label] = {"total": total, "count": count, "avg": avg}
        return result


# ──────────────────────────────────────────────────────────────────────
# Top-K helpers
# ──────────────────────────────────────────────────────────────────────

TOP_K_DEFAULT = 1000
TOP_K_ABLATION = [100, 500, 1000]


def top_k_filter(
    results: list[tuple[str, float]],
    k: int = TOP_K_DEFAULT,
) -> list[tuple[str, float]]:
    """
    Sort (doc_id, score) pairs by score descending, keep top-K.

    Parameters
    ----------
    results : list of (doc_id, score)
    k : int

    Returns
    -------
    list of (doc_id, score), length ≤ k, sorted by score descending.
    """
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return sorted_results[:k]


def pad_to_k(
    results: list[tuple[str, float]],
    k: int = TOP_K_DEFAULT,
    sentinel_score: float = -1e9,
) -> list[tuple[str, float]]:
    """
    If fewer than K results, pad with sentinel-scored entries.

    Sentinel doc_ids are "__PAD_i__" and in_candidate_set should be False.
    """
    if len(results) >= k:
        return results[:k]

    padded = list(results)
    for i in range(len(results), k):
        padded.append((f"__PAD_{i}__", sentinel_score))
    return padded


# ──────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
