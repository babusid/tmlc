"""Translate Compute IR into an independent, self-contained Loop IR program."""

from __future__ import annotations

from collections.abc import Iterable

from tmlc.compute.axis import Axis, AxisKind
from tmlc.compute.index import (
    AxisRef,
    CompareOp,
    IndexAdd as ComputeIndexAdd,
    IndexCompare as ComputeIndexCompare,
    IndexExpr as ComputeIndexExpr,
    IndexFloorDiv as ComputeIndexFloorDiv,
    IndexMod as ComputeIndexMod,
    IndexMul as ComputeIndexMul,
    IntConst,
)
from tmlc.compute.program import Combiner, ComputeBlock, ComputeProgram, ComputeTensor
from tmlc.compute.scalar import Read, ScalarConst, ScalarExpr, ScalarExprBase, ScalarOpKind, Select
from tmlc.loop.axis import LoopAxis
from tmlc.loop.buffer import (
    PROGRAM_SCOPE,
    Buffer,
    BufferConstant,
    BufferLayout,
    LoopScope,
)
from tmlc.loop.expr import (
    BufferLoad,
    FloatConst,
    IndexAdd,
    IndexCompare,
    IndexConst,
    IndexFloorDiv,
    IndexMod,
    IndexMul,
    LoopCompareOp,
    LoopIndex,
    LoopIndexExpr,
    LoopValue,
    LoopValueOp,
    ValueExpr,
    ValueSelect,
)
from tmlc.loop.program import (
    Accumulate,
    Loop,
    LoopCombiner,
    LoopProgram,
    LoopStatement,
    Store,
)
from tmlc.loop.verify import verify_loop_program

_COMPARE_OP = {
    CompareOp.LT: LoopCompareOp.LT,
    CompareOp.LE: LoopCompareOp.LE,
    CompareOp.GT: LoopCompareOp.GT,
    CompareOp.GE: LoopCompareOp.GE,
    CompareOp.EQ: LoopCompareOp.EQ,
    CompareOp.NE: LoopCompareOp.NE,
}

_VALUE_OP = {
    ScalarOpKind.ADD: LoopValueOp.ADD,
    ScalarOpKind.SUB: LoopValueOp.SUB,
    ScalarOpKind.MUL: LoopValueOp.MUL,
    ScalarOpKind.DIV: LoopValueOp.DIV,
    ScalarOpKind.NEG: LoopValueOp.NEG,
    ScalarOpKind.EXP: LoopValueOp.EXP,
    ScalarOpKind.LOG: LoopValueOp.LOG,
    ScalarOpKind.TANH: LoopValueOp.TANH,
    ScalarOpKind.MAX: LoopValueOp.MAX,
    ScalarOpKind.POW: LoopValueOp.POW,
}

_COMBINER = {
    Combiner.SUM: LoopCombiner.SUM,
    Combiner.PROD: LoopCombiner.PROD,
    Combiner.MAX: LoopCombiner.MAX,
}


def _wrap_axes(
    axes: Iterable[LoopAxis], body: tuple[LoopStatement, ...]
) -> tuple[LoopStatement, ...]:
    statements = body
    for axis in reversed(tuple(axes)):
        statements = (Loop(axis=axis, body=statements),)
    return statements


def _translate_index(expr: ComputeIndexExpr, axes: dict[Axis, LoopAxis]) -> LoopIndexExpr:
    if isinstance(expr, AxisRef):
        return LoopIndex(axes[expr.axis])
    if isinstance(expr, IntConst):
        return IndexConst(expr.value)
    if isinstance(expr, ComputeIndexAdd):
        return IndexAdd(_translate_index(expr.lhs, axes), _translate_index(expr.rhs, axes))
    if isinstance(expr, ComputeIndexMul):
        return IndexMul(_translate_index(expr.lhs, axes), _translate_index(expr.rhs, axes))
    if isinstance(expr, ComputeIndexFloorDiv):
        return IndexFloorDiv(_translate_index(expr.lhs, axes), expr.divisor)
    if isinstance(expr, ComputeIndexMod):
        return IndexMod(_translate_index(expr.lhs, axes), expr.modulus)
    if isinstance(expr, ComputeIndexCompare):
        return IndexCompare(
            _COMPARE_OP[expr.op],
            _translate_index(expr.lhs, axes),
            _translate_index(expr.rhs, axes),
        )
    raise TypeError(f"unknown Compute IndexExpr: {type(expr).__name__}")


def _translate_value(
    expr: ScalarExprBase,
    axes: dict[Axis, LoopAxis],
    buffers: dict[ComputeTensor, Buffer],
) -> LoopValue:
    if isinstance(expr, ScalarConst):
        return FloatConst(expr.value)
    if isinstance(expr, Read):
        return BufferLoad(
            buffers[expr.tensor],
            tuple(_translate_index(coordinate, axes) for coordinate in expr.index),
        )
    if isinstance(expr, Select):
        return ValueSelect(
            _translate_index(expr.cond, axes),
            _translate_value(expr.if_true, axes, buffers),
            _translate_value(expr.if_false, axes, buffers),
        )
    if isinstance(expr, ScalarExpr):
        return ValueExpr(
            _VALUE_OP[expr.kind],
            tuple(_translate_value(argument, axes, buffers) for argument in expr.args),
        )
    raise TypeError(f"unknown Compute ScalarExpr: {type(expr).__name__}")


def _lower_block(
    block: ComputeBlock,
    tensor_buffers: dict[ComputeTensor, Buffer],
) -> tuple[tuple[LoopStatement, ...], Buffer | None]:
    compute_spatial = tuple(axis for axis in block.output_axes if axis.kind is AxisKind.SPATIAL)
    all_axes = (*compute_spatial, *block.reduce_axes)
    axes = {axis: LoopAxis(axis.extent, axis.name) for axis in all_axes}
    spatial = tuple(axes[axis] for axis in compute_spatial)
    reductions = tuple(axes[axis] for axis in block.reduce_axes)
    value = _translate_value(block.body, axes, tensor_buffers)
    store_index = tuple(
        LoopIndex(axes[axis]) if axis.kind is AxisKind.SPATIAL else IndexConst(0)
        for axis in block.output_axes
    )
    output = tensor_buffers[block.output]

    if block.combiner is None:
        inner: tuple[LoopStatement, ...] = (Store(output, store_index, value),)
        accumulator = None
    else:
        scope = LoopScope(spatial[-1]) if spatial else PROGRAM_SCOPE
        accumulator = Buffer(
            name=f"acc_{block.output.name}",
            shape=(),
            dtype=block.output.dtype,
            layout=BufferLayout.dense(()),
            scope=scope,
        )
        combiner = _COMBINER[block.combiner]
        reduction = _wrap_axes(
            reductions,
            (Accumulate(accumulator, (), value, combiner),),
        )
        inner = (
            Store(accumulator, (), FloatConst(combiner.identity)),
            *reduction,
            Store(output, store_index, BufferLoad(accumulator, ())),
        )

    return _wrap_axes(spatial, inner), accumulator


def lower_to_loops(program: ComputeProgram) -> LoopProgram:
    """Translate Compute IR objects into Loop-owned buffers, axes, expressions, and statements."""
    materialized = set(program.inputs)
    materialized |= {tensor for tensor, _ in program.constants}
    materialized |= {block.output for block in program.blocks}
    materialized |= set(program.outputs)

    tensor_buffers = {
        tensor: Buffer(
            name=tensor.name,
            shape=tensor.shape,
            dtype=tensor.dtype,
            layout=BufferLayout.dense(tensor.shape),
            scope=PROGRAM_SCOPE,
        )
        for tensor in program.tensors
        if tensor in materialized
    }

    body: list[LoopStatement] = []
    local_buffers: list[Buffer] = []
    for block in program.blocks:
        statements, accumulator = _lower_block(block, tensor_buffers)
        body.extend(statements)
        if accumulator is not None:
            local_buffers.append(accumulator)

    loop_program = LoopProgram(
        body=tuple(body),
        buffers=(*tensor_buffers.values(), *local_buffers),
        inputs=tuple(tensor_buffers[tensor] for tensor in program.inputs),
        outputs=tuple(tensor_buffers[tensor] for tensor in program.outputs),
        constants=tuple(
            BufferConstant(tensor_buffers[tensor], value) for tensor, value in program.constants
        ),
    )
    verify_loop_program(loop_program)
    return loop_program
