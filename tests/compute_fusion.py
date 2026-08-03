"""Show the opportunity for elementwise fusion and shape-index collapsing in Compute IR."""

from __future__ import annotations

import timeit
from collections.abc import Mapping

import numpy as np

import tmlc
from tmlc import Input
from tmlc.compute import AxisRef, ComputeProgram, ComputeProgramBuilder, ComputeTensor
from tmlc.interpreters.compute_interpreter import ComputeInterpreter
from tmlc.interpreters.graph_interpreter import GraphInterpreter

WARMUPS = 3
REPEATS = 7


def lowered_bindings(
    graph: tmlc.Graph,
    program: ComputeProgram,
    values: Mapping[tmlc.Tensor, np.ndarray],
) -> dict[ComputeTensor, np.ndarray]:
    """
    Given a Graph, its lowered ComputeProgram, and the Graph Inputs as ndarrays,
    return a dictionary mapping the ComputeProgram's inputs to the corresponding ndarrays.
    """
    graph_inputs = [node for node in graph.topo_sort if isinstance(node.op, Input)]
    assert len(graph_inputs) == len(program.inputs)
    return {
        computetensor: values[node] for node, computetensor in zip(graph_inputs, program.inputs)
    }


def median_runtime_ms(
    interpreter: ComputeInterpreter,
    program: ComputeProgram,
    bindings: Mapping[ComputeTensor, np.ndarray],
    number: int,
) -> float:
    for _ in range(WARMUPS):
        interpreter.run(program, bindings)
    samples = timeit.repeat(
        lambda: interpreter.run(program, bindings), repeat=REPEATS, number=number
    )
    return float(np.median(samples)) * 1e3 / number


def logical_writes(program: ComputeProgram) -> int:
    return sum(int(np.prod(block.output.shape)) for block in program.blocks)


def elementwise_fusion(rng: np.random.Generator) -> None:
    size = 1_000_000
    x = tmlc.input((size,), label="x")
    y = tmlc.input((size,), label="y")
    z = tmlc.input((size,), label="z")
    output = ((x + y) * z) + y
    graph = tmlc.Graph([output])

    values = {
        x: rng.standard_normal(size, dtype=np.float32),
        y: rng.standard_normal(size, dtype=np.float32),
        z: rng.standard_normal(size, dtype=np.float32),
    }
    expected = GraphInterpreter().run(graph, values)[0]

    lowered = graph.lower()
    lowered_inputs = lowered_bindings(graph, lowered, values)

    builder = ComputeProgramBuilder()
    direct_x = builder.declare_input((size,), "float32", hint="x")
    direct_y = builder.declare_input((size,), "float32", hint="y")
    direct_z = builder.declare_input((size,), "float32", hint="z")
    i = builder.spatial(size, "i")
    index = (AxisRef(i),)
    body = ((direct_x[index] + direct_y[index]) * direct_z[index]) + direct_y[index]
    direct_output = builder.compute(output_axes=(i,), body=body, hint="fused")
    optimized = builder.finish((direct_output,))
    optimized_inputs = {direct_x: values[x], direct_y: values[y], direct_z: values[z]}

    interpreter = ComputeInterpreter()
    lowered_result = interpreter.run(lowered, lowered_inputs)[0]
    optimized_result = interpreter.run(optimized, optimized_inputs)[0]
    np.testing.assert_allclose(lowered_result, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(optimized_result, expected, rtol=1e-6, atol=1e-6)

    lowered_ms = median_runtime_ms(interpreter, lowered, lowered_inputs, number=3)
    optimized_ms = median_runtime_ms(interpreter, optimized, optimized_inputs, number=3)

    print("Elementwise fusion")
    print("  ((x + y) * z) + y  ->  ((x[i] + y[i]) * z[i]) + y[i]")
    print(
        f"  lowered: {len(lowered.blocks)} blocks, {logical_writes(lowered):,} writes, "
        f"{lowered_ms:.3f} ms"
    )
    print(
        f"  fused:   {len(optimized.blocks)} block,  {logical_writes(optimized):,} writes, "
        f"{optimized_ms:.3f} ms ({lowered_ms / optimized_ms:.2f}x faster)"
    )


def shape_index_collapsing(rng: np.random.Generator) -> None:
    # tensor sizing vars
    a_extent = 128
    b_extent = 256
    c_extent = 8

    # do the reshape / broadcast chain only in Graph IR
    flat = tmlc.input((a_extent * b_extent,), label="flat")
    reshaped = tmlc.reshape(flat, (a_extent, b_extent, 1))
    broadcast = tmlc.broadcast_to(reshaped, (a_extent, b_extent, c_extent))
    output = tmlc.transpose(broadcast, axes=(0, 1))
    graph = tmlc.Graph([output])

    # create a input and run it using the Graph Interpreter to get an oracle value
    values = {flat: rng.standard_normal(a_extent * b_extent, dtype=np.float32)}
    expected = GraphInterpreter().run(graph, values)[0]

    # build an equivalent compute program by hand
    builder = ComputeProgramBuilder()
    direct_flat = builder.declare_input((a_extent * b_extent,), "float32", hint="flat")
    a = builder.spatial(a_extent, "a")
    b = builder.spatial(b_extent, "b")
    c = builder.spatial(c_extent, "c")
    source_index = AxisRef(a) * b_extent + AxisRef(b)
    direct_output = builder.compute(
        output_axes=(b, a, c),  # final tensor shape
        body=direct_flat[(source_index,)],  # body is just a indexed read of the flat tensor
        hint="collapsed_shape_chain",
    )
    optimized = builder.finish((direct_output,))
    optimized_inputs = {direct_flat: values[flat]}

    # build a compute interpreter
    interpreter = ComputeInterpreter()

    # use the lowering path to generate a ComputeProgram
    lowered = graph.lower()
    lowered_inputs = lowered_bindings(graph, lowered, values)

    # run lowered and handwritten to compare results
    lowered_result = interpreter.run(lowered, lowered_inputs)[0]
    optimized_result = interpreter.run(optimized, optimized_inputs)[0]
    np.testing.assert_array_equal(lowered_result, expected)
    np.testing.assert_array_equal(optimized_result, expected)

    # run lowered and handwritten to compare speedup
    lowered_ms = median_runtime_ms(interpreter, lowered, lowered_inputs, number=5)
    optimized_ms = median_runtime_ms(interpreter, optimized, optimized_inputs, number=5)

    print("\nShape/index collapsing")
    print("  reshape(A,B,1) -> broadcast(A,B,C) -> transpose(B,A,C)")
    print("  collapsed read: flat[a * B + b]")
    print(
        f"  lowered:   {len(lowered.blocks)} blocks, {logical_writes(lowered):,} writes, "
        f"{lowered_ms:.3f} ms"
    )
    print(
        f"  collapsed: {len(optimized.blocks)} block,  {logical_writes(optimized):,} writes, "
        f"{optimized_ms:.3f} ms ({lowered_ms / optimized_ms:.2f}x faster)"
    )


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"NumPy ComputeInterpreter timings ({WARMUPS} warmups, {REPEATS} repeats)\n")
    elementwise_fusion(rng)
    shape_index_collapsing(rng)


if __name__ == "__main__":
    main()
