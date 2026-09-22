"""
Emit portable C for the Loop IR, compile it, run it through the CProgram dispatch layer, and check
the result against the GraphInterpreter oracle. Marked c_backend: each case invokes the C compiler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

import tmlc
from tmlc import Input
from tmlc.backends.c import CProgram, emit_c
from tmlc.compute import (
    AxisRef,
    Combiner,
    ComputeProgramBuilder,
    Select,
    index_le,
)
from tmlc.compute.transforms import InlineReductionFree
from tmlc.interpreters.graph_interpreter import GraphInterpreter
from tmlc.loop import Buffer, ExternalCall, LoopProgram, ProgramScope, lower_to_loops

pytestmark = pytest.mark.c_backend


def _check_program(
    save_dir,
    program: LoopProgram,
    cases: Sequence[tuple[Mapping[Buffer, np.ndarray], Sequence[np.ndarray]]],
) -> None:
    compiled = CProgram(program, save_dir=save_dir)
    assert (save_dir / "kernel.c").read_text() == compiled.source
    assert (save_dir / "kernel.so").is_file()

    # Every surviving intermediate owns a buffer for the entire native call.
    external = set(program.inputs) | set(program.outputs)
    external.update(constant.buffer for constant in program.constants)
    intermediates = [
        buffer
        for buffer in program.buffers
        if isinstance(buffer.scope, ProgramScope) and buffer not in external and buffer.shape
    ]
    assert compiled.source.count("malloc(") == len(intermediates)
    assert compiled.source.count("free(") == len(intermediates)
    if intermediates:
        assert compiled.source.rfind("malloc(") < compiled.source.index("for (")
        assert compiled.source.index("free(") > compiled.source.rfind("/* store")

    for bindings, expected in cases:
        got = compiled.run(bindings)
        assert len(got) == len(expected)
        for out_expected, out_got in zip(expected, got):
            np.testing.assert_allclose(out_got, out_expected, rtol=1e-5, atol=1e-5)


def _check_graph(save_dir, graph: tmlc.Graph, values: Mapping[tmlc.Tensor, np.ndarray]) -> None:
    program = lower_to_loops(InlineReductionFree()(graph.lower()))
    graph_inputs = [node for node in graph.topo_sort if isinstance(node.op, Input)]
    assert len(graph_inputs) == len(program.inputs)
    bindings = {ct: values[g] for g, ct in zip(graph_inputs, program.inputs)}
    expected = GraphInterpreter().run(graph, values)
    _check_program(save_dir, program, [(bindings, expected)])


def test_external_call_is_representable_but_not_yet_emittable():
    program = LoopProgram(body=(ExternalCall(),), buffers=(), inputs=(), outputs=())
    with pytest.raises(NotImplementedError, match="ExternalCall"):
        emit_c(program)


def _elementwise_const(rng):
    n = 256
    x, y, z = (tmlc.input((n,), label=lbl) for lbl in ("x", "y", "z"))
    graph = tmlc.Graph([((x + y) * z) + 0.5])
    values = {t: rng.standard_normal(n, dtype=np.float32) for t in (x, y, z)}
    return graph, values


def _sum_scale(rng):
    i, j, k = 4, 5, 6
    x = tmlc.input((i, j, k), label="x")
    graph = tmlc.Graph([tmlc.summation(x, axes=2) * 2.0])
    return graph, {x: rng.standard_normal((i, j, k), dtype=np.float32)}


def _logsumexp_ish(rng):
    i, k = 8, 7
    x = tmlc.input((i, k), label="x")
    graph = tmlc.Graph([tmlc.log(tmlc.summation(tmlc.exp(x), axes=1))])
    return graph, {x: rng.standard_normal((i, k), dtype=np.float32) * 0.5}


def _matmul_epilogue(rng):
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


def _two_nests(rng):
    # Sum over axis 1 -> (i,), then sum to a scalar: two trees, with the intermediate malloc'd.
    i, j = 5, 4
    x = tmlc.input((i, j), label="x")
    graph = tmlc.Graph([tmlc.summation(tmlc.summation(x, axes=1))])
    return graph, {x: rng.standard_normal((i, j), dtype=np.float32)}


def _shape_chain(rng):
    x = tmlc.input((12,), label="flat")
    y = tmlc.transpose(tmlc.broadcast_to(tmlc.reshape(x, (3, 4, 1)), (3, 4, 2)), (0, 1))
    return tmlc.Graph([y]), {x: rng.standard_normal(12)}


@pytest.mark.parametrize(
    "build",
    [_elementwise_const, _sum_scale, _logsumexp_ish, _matmul_epilogue, _two_nests, _shape_chain],
)
def test_c_backend_matches_oracle(tmp_path, rng, build):
    graph, values = build(rng)
    _check_graph(tmp_path, graph, values)


def test_predicated_kept_reduction_dimension(tmp_path):
    builder = ComputeProgramBuilder()
    x = builder.declare_input((3, 4), "float32", hint="x")
    i, k = builder.spatial(3, "i"), builder.reduce(4, "k")
    masked = Select(index_le(AxisRef(k), AxisRef(i)), x[AxisRef(i), AxisRef(k)], 0.0)
    y = builder.compute((i, k), masked, (k,), Combiner.SUM, hint="masked_sum")
    program = lower_to_loops(builder.finish((y,)))
    (loop_x,) = program.inputs

    cases = []
    for offset in (0, 10):
        values = np.arange(12, dtype=np.float32).reshape(3, 4) + offset
        expected = [np.tril(values).sum(axis=1, keepdims=True)]
        cases.append(({loop_x: values}, expected))
    _check_program(tmp_path, program, cases)
