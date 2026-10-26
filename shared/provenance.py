"""
provenance.py - Seeding, timing and run manifests.

Every primary result table gets a manifest written beside it. Three people
producing tables independently need to be able to tell, months later, whether
two numbers came from the same design freeze.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Seed Python, NumPy and, if present, PyTorch.

    PYTHONHASHSEED is set for child processes but cannot affect the current
    interpreter - it must be set before Python starts. For full hash
    reproducibility export it in the shell:  export PYTHONHASHSEED=42
    """
    random.seed(seed)
    np.random.seed(seed)
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


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


@contextmanager
def timed(label: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    logger.info("%s took %.1fs", label, time.perf_counter() - start)


@dataclass
class Timer:
    """Cumulative timer for multi-phase runs."""

    _starts: dict[str, float] = field(default_factory=dict)
    _totals: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def start(self, label: str) -> None:
        self._starts[label] = time.perf_counter()

    def stop(self, label: str) -> float:
        if label not in self._starts:
            raise KeyError(f"timer {label!r} was never started")
        dt = time.perf_counter() - self._starts.pop(label)
        self._totals[label] = self._totals.get(label, 0.0) + dt
        self._counts[label] = self._counts.get(label, 0) + 1
        return dt

    def report(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for label in sorted(self._totals):
            total, n = self._totals[label], self._counts[label]
            out[label] = {"total_s": total, "calls": n, "mean_s": total / max(n, 1)}
        return out


def measure_throughput(fn, n_items: int, label: str = "throughput") -> float:
    """
    Items per second for a callable.

    Compute planning must use the rate observed on the actual hardware at the
    frozen sequence length, batch size and precision - never a number borrowed
    from a paper.
    """
    start = time.perf_counter()
    fn()
    elapsed = max(time.perf_counter() - start, 1e-9)
    rate = n_items / elapsed
    logger.info("%s: %d items in %.1fs = %.1f items/sec", label, n_items, elapsed, rate)
    return rate


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def run_manifest(
    config_fingerprint: str,
    owner: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Everything needed to tie a result file back to how it was produced.

    `owner` names the stream that produced it, so a mismatched fingerprint can
    be chased to a person rather than a file.
    """
    rec: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_fingerprint": config_fingerprint,
        "owner": owner,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        rec["torch"] = torch.__version__
        rec["cuda"] = torch.version.cuda
        rec["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        pass
    if extra:
        rec.update(extra)
    return rec


def write_manifest(path: str | Path, record: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_same_freeze(*manifest_paths: str | Path) -> str:
    """
    Check that several result files were produced under one design freeze.

    Run this before combining any two members' outputs. Mixing fingerprints is
    the failure mode that produces a coherent-looking table nobody can
    reproduce.
    """
    fingerprints = {}
    for p in manifest_paths:
        rec = read_manifest(p)
        fingerprints[str(p)] = rec.get("config_fingerprint")
    distinct = set(fingerprints.values())
    if len(distinct) > 1:
        detail = "\n".join(f"  {k}: {v}" for k, v in fingerprints.items())
        raise ValueError(
            "these results were produced under different design freezes and "
            f"cannot be combined:\n{detail}"
        )
    return distinct.pop() if distinct else ""


__all__ = [
    "DEFAULT_SEED", "set_global_seed", "setup_logging", "timed", "Timer",
    "measure_throughput", "run_manifest", "write_manifest", "read_manifest",
    "assert_same_freeze",
]
