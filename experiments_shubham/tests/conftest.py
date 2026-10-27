"""Shared fixtures for experiments_shubham's tests."""

from __future__ import annotations

import pytest
import yaml

from shared.config import load_config


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory):
    """The frozen config with a grid small enough for a unit test."""
    with open("config/base.yaml", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["cascade"]["candidate_depths"] = [20, 50, 100]
    raw["synthetic"]["n_queries"] = 4
    raw["synthetic"]["n_docs_per_query"] = 300
    raw["power"]["candidate_sizes"] = [6, 30]
    raw["power"]["n_simulations"] = 30
    path = tmp_path_factory.mktemp("cfg") / "small.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)
