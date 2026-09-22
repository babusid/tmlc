"""Structural, lexical-scope, and buffer-interface verification for pure Loop IR."""

from __future__ import annotations

from collections.abc import Iterator

from tmlc.loop.axis import LoopAxis
from tmlc.loop.buffer import Buffer, LoopScope, ProgramScope
from tmlc.loop.expr import (
    BufferLoad,
    FloatConst,
    IndexAdd,
    IndexCompare,
    IndexConst,
    IndexFloorDiv,
    IndexMod,
    IndexMul,
    LoopIndex,
    LoopIndexExpr,
    LoopValue,
    ValueExpr,
    ValueSelect,
)
from tmlc.loop.program import Accumulate, ExternalCall, Loop, LoopProgram, LoopStatement, Store


class LoopVerifyError(Exception):
    """Raised when a Loop IR program violates lexical or storage invariants."""


def _index_axes(expr: LoopIndexExpr) -> Iterator[LoopAxis]:
    if isinstance(expr, LoopIndex):
        yield expr.axis
    elif isinstance(expr, IndexConst):
        return
    elif isinstance(expr, (IndexAdd, IndexMul, IndexCompare)):
        yield from _index_axes(expr.lhs)
        yield from _index_axes(expr.rhs)
    elif isinstance(expr, (IndexFloorDiv, IndexMod)):
        yield from _index_axes(expr.lhs)
    else:
        raise TypeError(f"unknown LoopIndexExpr: {type(expr).__name__}")


def _value_axes(value: LoopValue) -> Iterator[LoopAxis]:
    if isinstance(value, FloatConst):
        return
    if isinstance(value, BufferLoad):
        for coordinate in value.index:
            yield from _index_axes(coordinate)
        return
    if isinstance(value, ValueExpr):
        for argument in value.args:
            yield from _value_axes(argument)
        return
    if isinstance(value, ValueSelect):
        yield from _index_axes(value.condition)
        yield from _value_axes(value.if_true)
        yield from _value_axes(value.if_false)
        return
    raise TypeError(f"unknown LoopValue: {type(value).__name__}")


def _value_loads(value: LoopValue) -> Iterator[BufferLoad]:
    if isinstance(value, BufferLoad):
        yield value
    elif isinstance(value, FloatConst):
        return
    elif isinstance(value, ValueExpr):
        for argument in value.args:
            yield from _value_loads(argument)
    elif isinstance(value, ValueSelect):
        yield from _value_loads(value.if_true)
        yield from _value_loads(value.if_false)
    else:
        raise TypeError(f"unknown LoopValue: {type(value).__name__}")


def _check_axes(index: tuple[LoopIndexExpr, ...], axes: set[LoopAxis]) -> None:
    for coordinate in index:
        for axis in _index_axes(coordinate):
            if axis not in axes:
                raise LoopVerifyError(f"axis {axis.name!r} is referenced outside its loop scope")


def _check_buffer_access(buffer: Buffer, index: tuple[LoopIndexExpr, ...]) -> None:
    if len(index) != buffer.rank:
        raise LoopVerifyError(
            f"access to {buffer.name!r} has {len(index)} coordinates, expected {buffer.rank}"
        )


def _buffer_visible(buffer: Buffer, axes: set[LoopAxis]) -> bool:
    return isinstance(buffer.scope, ProgramScope) or buffer.scope.axis in axes


def _check_value(value: LoopValue, axes: set[LoopAxis], program: LoopProgram) -> None:
    for axis in _value_axes(value):
        if axis not in axes:
            raise LoopVerifyError(f"axis {axis.name!r} is referenced outside its loop scope")
    for load in _value_loads(value):
        if not program.has_buffer(load.buffer):
            raise LoopVerifyError(f"load source {load.buffer.name!r} is not declared")
        if not _buffer_visible(load.buffer, axes):
            raise LoopVerifyError(f"buffer {load.buffer.name!r} is accessed outside its scope")
        _check_buffer_access(load.buffer, load.index)


def _verify_statements(
    statements: tuple[LoopStatement, ...],
    axes: set[LoopAxis],
    seen_axes: set[LoopAxis],
    program: LoopProgram,
) -> None:
    readonly = set(program.inputs) | {constant.buffer for constant in program.constants}
    for statement in statements:
        if isinstance(statement, Loop):
            if statement.axis in seen_axes:
                raise LoopVerifyError(f"axis {statement.axis.name!r} is defined by multiple loops")
            seen_axes.add(statement.axis)
            _verify_statements(statement.body, axes | {statement.axis}, seen_axes, program)
        elif isinstance(statement, (Store, Accumulate)):
            if not program.has_buffer(statement.buffer):
                raise LoopVerifyError(f"store target {statement.buffer.name!r} is not declared")
            if statement.buffer in readonly:
                raise LoopVerifyError(f"buffer {statement.buffer.name!r} is read-only")
            if not _buffer_visible(statement.buffer, axes):
                raise LoopVerifyError(
                    f"buffer {statement.buffer.name!r} is accessed outside its scope"
                )
            _check_buffer_access(statement.buffer, statement.index)
            _check_axes(statement.index, axes)
            _check_value(statement.value, axes, program)
        elif isinstance(statement, ExternalCall):
            continue
        else:
            raise TypeError(f"unknown LoopStatement: {type(statement).__name__}")


def verify_loop_program(program: LoopProgram) -> None:
    """Verify that a LoopProgram is self-contained and respects lexical buffer scopes."""
    if len(set(program.buffers)) != len(program.buffers):
        raise LoopVerifyError("a Buffer may appear only once in LoopProgram.buffers")
    for buffer in (*program.inputs, *program.outputs):
        if not program.has_buffer(buffer):
            raise LoopVerifyError(f"interface buffer {buffer.name!r} is not declared")
        if not isinstance(buffer.scope, ProgramScope):
            raise LoopVerifyError(f"interface buffer {buffer.name!r} must have program scope")
    for constant in program.constants:
        if not program.has_buffer(constant.buffer):
            raise LoopVerifyError(f"constant buffer {constant.buffer.name!r} is not declared")
    _verify_statements(program.body, set(), set(), program)

    defined_axes = set()

    def collect(statements: tuple[LoopStatement, ...]) -> None:
        for statement in statements:
            if isinstance(statement, Loop):
                defined_axes.add(statement.axis)
                collect(statement.body)

    collect(program.body)
    for buffer in program.buffers:
        if isinstance(buffer.scope, LoopScope) and buffer.scope.axis not in defined_axes:
            raise LoopVerifyError(
                f"buffer {buffer.name!r} is scoped to an axis absent from the program"
            )
