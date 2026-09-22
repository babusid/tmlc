"""Shared fixtures for the tmlc test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy generator so tests are deterministic."""
    return np.random.default_rng(0)
