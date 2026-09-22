"""
Regression coverage for the runnable programs in examples/.

Each example is imported and its main() is executed; the examples self-check their own numerics
(assertions, np.testing.assert_allclose), so a clean run is the regression signal. logreg_compiled
goes through the C backend, so it is marked c_backend.
"""

import importlib
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(autouse=True)
def _examples_on_path():
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        yield
    finally:
        sys.path.remove(str(EXAMPLES_DIR))


@pytest.mark.parametrize(
    "module_name",
    [
        "autodiff",
        "compute_reduction",
        "fusion_benchmark",
        "logreg",
        pytest.param("logreg_compiled", marks=pytest.mark.c_backend),
        pytest.param("rmsnorm_projection_activation", marks=pytest.mark.c_backend),
    ],
)
def test_example_runs(module_name):
    module = importlib.import_module(module_name)
    module.main()
