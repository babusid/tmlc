"""
InlineReductionFree fuses reduction-free blocks into their consumers: the fused program must match
the graph-interpreter oracle and collapse to fewer blocks than the naive lowering.
"""

from __future__ import annotations

import numpy as np
import pytest

import tmlc
from tmlc.compute.transforms import InlineReductionFree
from tmlc.interpreters.compute_interpreter import ComputeInterpreter
from tmlc.interpreters.graph_interpreter import GraphInterpreter

from support import graph_input_bindings


def _elementwise(rng):
    size = 4096
    x, y, z = (tmlc.input((size,), label=lbl) for lbl in ("x", "y", "z"))
    graph = tmlc.Graph([((x + y) * z) + y])
    values = {t: rng.standard_normal(size, dtype=np.float32) for t in (x, y, z)}
    return graph, values


def _shape_chain(rng):
    a, b, c = 16, 32, 8
    flat = tmlc.input((a * b,), label="flat")
    reshaped = tmlc.reshape(flat, (a, b, 1))
    broadcast = tmlc.broadcast_to(reshaped, (a, b, c))
    graph = tmlc.Graph([tmlc.transpose(broadcast, axes=(0, 1))])
    values = {flat: rng.standard_normal(a * b, dtype=np.float32)}
    return graph, values


def _matmul_epilogue(rng):
    # (a@b)+bias is reduction-free and not the output, so it fuses into the final mul; the matmul (a
    # reduction) stays materialized in the middle.
    m, k, n = 8, 12, 6
    a = tmlc.input((m, k), label="a")
    b = tmlc.input((k, n), label="b")
    bias = tmlc.input((m, n), label="bias")
    graph = tmlc.Graph([((a @ b) + bias) * bias])
    values = {
        a: rng.standard_normal((m, k), dtype=np.float32),
        b: rng.standard_normal((k, n), dtype=np.float32),
        bias: rng.standard_normal((m, n), dtype=np.float32),
    }
    return graph, values


@pytest.mark.parametrize("build", [_elementwise, _shape_chain, _matmul_epilogue])
def test_inline_reduction_free_matches_oracle_and_fuses(rng, build):
    graph, values = build(rng)
    interp = ComputeInterpreter()
    expected = GraphInterpreter().run(graph, values)[0]

    lowered = graph.lower()
    bindings = graph_input_bindings(graph, lowered, values)
    fused = InlineReductionFree()(lowered)

    np.testing.assert_allclose(interp.run(lowered, bindings)[0], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(interp.run(fused, bindings)[0], expected, rtol=1e-6, atol=1e-6)
    assert len(fused.blocks) < len(lowered.blocks), "fusion should reduce the block count"
