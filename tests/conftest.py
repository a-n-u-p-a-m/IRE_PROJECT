"""
Shared fixtures.

Tests run against the hand-checkable toy cascade in `shared.fixtures` rather
than against the synthetic generator, so a failure points at the code under
test and not at the generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import fixtures as fx


@pytest.fixture
def toy():
    return fx.toy_cascade()


@pytest.fixture
def worlds():
    return fx.toy_worlds()


@pytest.fixture
def t_worlds():
    return fx.toy_t_worlds()


@pytest.fixture
def effects():
    return fx.toy_effects()


@pytest.fixture
def config_path():
    return str(Path(__file__).resolve().parent.parent / "config" / "base.yaml")
