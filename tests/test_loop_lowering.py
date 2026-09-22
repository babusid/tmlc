"""Structural checks for the pure recursive Loop IR translation."""

from __future__ import annotations

import pytest

import tmlc
from tmlc.compute.transforms import InlineReductionFree
from tmlc.loop import (
    PROGRAM_SCOPE,
    Accumulate,
    Buffer,
    BufferLayout,
    BufferLoad,
    FloatConst,
    Loop,
    LoopAxis,
    LoopIndex,
    LoopProgram,
    LoopScope,
    LoopVerifyError,
    Store,
    lower_to_loops,
    row_major_strides,
    verify_loop_program,
)


def _lower(graph: tmlc.Graph):
    return lower_to_loops(InlineReductionFree()(graph.lower()))


def _loop_chain(root: Loop) -> tuple[list[Loop], tuple[object, ...]]:
    chain = [root]
    current = root
    while len(current.body) == 1 and isinstance(current.body[0], Loop):
        current = current.body[0]
        chain.append(current)
    return chain, current.body


def test_sum_then_scale_translates_to_independent_loop_trees():
    i, j, k = 4, 5, 6
    x = tmlc.input((i, j, k), label="x")
    program = _lower(tmlc.Graph([tmlc.summation(x, axes=2) * 2.0]))

    assert len(program.body) == 2
    reduction_root, scale_root = program.body
    assert isinstance(reduction_root, Loop)
    assert isinstance(scale_root, Loop)

    reduction_chain, reduction_body = _loop_chain(reduction_root)
    assert [loop.axis.extent for loop in reduction_chain] == [i, j]
    initialize, reduction_loop, final_store = reduction_body
    assert isinstance(initialize, Store)
    assert isinstance(initialize.value, FloatConst)
    assert isinstance(initialize.buffer.scope, LoopScope)
    assert initialize.buffer.scope.axis is reduction_chain[-1].axis
    assert isinstance(reduction_loop, Loop) and reduction_loop.axis.extent == k
    assert len(reduction_loop.body) == 1
    accumulate = reduction_loop.body[0]
    assert isinstance(accumulate, Accumulate)
    assert accumulate.buffer is initialize.buffer
    assert isinstance(final_store, Store)
    assert isinstance(final_store.value, BufferLoad)
    assert final_store.value.buffer is initialize.buffer

    scale_chain, scale_body = _loop_chain(scale_root)
    assert [loop.axis.extent for loop in scale_chain] == [i, j]
    assert len(scale_body) == 1 and isinstance(scale_body[0], Store)
    assert final_store.buffer in program.buffers
    assert scale_body[0].buffer in program.outputs


def test_matmul_epilogue_remains_in_separate_loop_trees():
    m, k, n = 8, 12, 6
    a = tmlc.input((m, k), label="a")
    b = tmlc.input((k, n), label="b")
    bias = tmlc.input((m, n), label="bias")
    program = _lower(tmlc.Graph([((a @ b) + bias) * bias]))

    assert len(program.body) == 2
    matmul_root, epilogue_root = program.body
    assert isinstance(matmul_root, Loop)
    assert isinstance(epilogue_root, Loop)

    matmul_chain, matmul_body = _loop_chain(matmul_root)
    assert [loop.axis.extent for loop in matmul_chain] == [m, n]
    _, reduction_loop, matmul_store = matmul_body
    assert isinstance(reduction_loop, Loop) and reduction_loop.axis.extent == k
    assert isinstance(reduction_loop.body[0], Accumulate)
    assert isinstance(matmul_store, Store)

    epilogue_chain, epilogue_body = _loop_chain(epilogue_root)
    assert [loop.axis.extent for loop in epilogue_chain] == [m, n]
    assert len(epilogue_body) == 1 and isinstance(epilogue_body[0], Store)

    for buffer in program.buffers:
        assert buffer.layout == BufferLayout.dense(buffer.shape), buffer.name
        assert buffer.layout.strides == row_major_strides(buffer.shape), buffer.name


def test_multiple_reduction_axes_become_nested_single_axis_loops():
    a, b, c = 3, 4, 5
    x = tmlc.input((a, b, c), label="x")
    program = _lower(tmlc.Graph([tmlc.summation(x, axes=(1, 2))]))

    (root,) = program.body
    assert isinstance(root, Loop) and root.axis.extent == a
    initialize, first_reduction, _ = root.body
    assert isinstance(initialize, Store)
    assert isinstance(first_reduction, Loop) and first_reduction.axis.extent == b
    (second_reduction,) = first_reduction.body
    assert isinstance(second_reduction, Loop) and second_reduction.axis.extent == c
    assert isinstance(second_reduction.body[0], Accumulate)


def test_different_spatial_domains_remain_separate_loop_trees():
    p = tmlc.input((8,), label="p")
    q = tmlc.input((4, 3), label="q")
    program = _lower(tmlc.Graph([p * 2.0, q + 1.0]))
    assert len(program.body) == 2
    first, second = program.body
    assert isinstance(first, Loop) and first.axis.extent == 8
    assert isinstance(second, Loop) and second.axis.extent == 4


def test_compute_objects_do_not_survive_the_translation_boundary():
    x = tmlc.input((8,), label="x")
    program = _lower(tmlc.Graph([x * 2.0]))

    assert all(isinstance(buffer, Buffer) for buffer in program.buffers)
    assert all(
        isinstance(buffer.scope, (type(PROGRAM_SCOPE), LoopScope)) for buffer in program.buffers
    )
    (root,) = program.body
    assert isinstance(root, Loop)
    assert isinstance(root.axis, LoopAxis)


def test_axis_refs_are_lexically_bound_by_loop_axis_identity():
    defined = LoopAxis(4, "i")
    lookalike = LoopAxis(4, "i")
    source = Buffer("source", (4,), "float32", BufferLayout.dense((4,)))
    output = Buffer("output", (4,), "float32", BufferLayout.dense((4,)))
    program = LoopProgram(
        body=(
            Loop(
                defined,
                (
                    Store(
                        output,
                        (LoopIndex(defined),),
                        BufferLoad(source, (LoopIndex(lookalike),)),
                    ),
                ),
            ),
        ),
        buffers=(source, output),
        inputs=(source,),
        outputs=(output,),
    )

    with pytest.raises(LoopVerifyError, match="outside its loop scope"):
        verify_loop_program(program)


def test_loop_local_buffer_cannot_escape_its_scope():
    axis = LoopAxis(4, "i")
    local = Buffer("local", (), "float32", BufferLayout.dense(()), LoopScope(axis))
    output = Buffer("output", (), "float32", BufferLayout.dense(()))
    program = LoopProgram(
        body=(
            Loop(axis, (Store(local, (), FloatConst(1.0)),)),
            Store(output, (), BufferLoad(local, ())),
        ),
        buffers=(local, output),
        inputs=(),
        outputs=(output,),
    )

    with pytest.raises(LoopVerifyError, match="outside its scope"):
        verify_loop_program(program)
