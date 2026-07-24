"""
Verifier for the Compute IR: cheap structural checks on each ComputeBlock.
"""

from __future__ import annotations

from collections.abc import Iterator

from tmlc.compute.axis import Axis, AxisKind
from tmlc.compute.compute import ComputeBlock, ComputeProgram, Read
from tmlc.compute.index import (
    AxisRef,
    IndexAdd,
    IndexExpr,
    IndexFloorDiv,
    IndexMod,
    IndexMul,
    IntConst,
    index_axes,
)
from tmlc.compute.scalar import ScalarConst, ScalarExpr, ScalarExprBase


class VerifyError(Exception):
    pass


def _reads(body: ScalarExprBase) -> Iterator[Read]:
    if isinstance(body, Read):
        yield body
    elif isinstance(body, ScalarConst):
        return
    elif isinstance(body, ScalarExpr):
        for arg in body.args:
            yield from _reads(arg)
    else:
        raise TypeError(f"unknown ScalarExprBase: {type(body).__name__}")


def _body_axes(body: ScalarExprBase) -> Iterator[Axis]:
    for read in _reads(body):
        for coord in read.index:
            yield from index_axes(coord)


def _bounds(expr: IndexExpr) -> tuple[int, int]:
    # Inclusive [lo, hi] range of an index expression, with each axis ranging over [0, extent).
    if isinstance(expr, AxisRef):
        return (0, expr.axis.extent - 1)
    if isinstance(expr, IntConst):
        return (expr.value, expr.value)
    if isinstance(expr, IndexAdd):
        lo1, hi1 = _bounds(expr.lhs)
        lo2, hi2 = _bounds(expr.rhs)
        return (lo1 + lo2, hi1 + hi2)
    if isinstance(expr, IndexMul):
        lo1, hi1 = _bounds(expr.lhs)
        lo2, hi2 = _bounds(expr.rhs)
        corners = (lo1 * lo2, lo1 * hi2, hi1 * lo2, hi1 * hi2)
        return (min(corners), max(corners))
    if isinstance(expr, IndexFloorDiv):
        lo, hi = _bounds(expr.lhs)
        return (lo // expr.divisor, hi // expr.divisor)
    if isinstance(expr, IndexMod):
        # x % m in [0, m-1]. TODO: if ever too conservative, refine to [lo%m, hi%m] when the
        # operand range doesn't wrap a full period.
        return (0, expr.modulus - 1)
    raise TypeError(f"unknown IndexExpr: {type(expr).__name__}")


def verify_block(block: ComputeBlock) -> None:
    name = block.output.name
    domain_axes = set(block.domain)  # Axis is eq=False, so membership is by identity

    # every axis referenced in the body must belong to the block's domain
    for axis in _body_axes(block.body):
        if axis not in domain_axes:
            raise VerifyError(f"block {name!r}: axis {axis.name!r} used in body but not in domain")

    # a combiner is present iff the domain has a reduce axis
    has_reduce = any(axis.kind is AxisKind.REDUCE for axis in block.domain)
    if has_reduce and block.combiner is None:
        raise VerifyError(f"block {name!r}: reduce axis present but no combiner")
    if not has_reduce and block.combiner is not None:
        raise VerifyError(f"block {name!r}: combiner present but no reduce axis")

    # output shape is exactly the spatial extents, in domain order
    spatial_extents = tuple(a.extent for a in block.domain if a.kind is AxisKind.SPATIAL)
    if block.output.shape != spatial_extents:
        raise VerifyError(
            f"block {name!r}: output shape {block.output.shape} "
            + f"!= spatial extents {spatial_extents}"
        )

    # each read must index its tensor with one in-bounds coordinate per dimension
    for read in _reads(block.body):
        if len(read.index) != len(read.tensor.shape):
            raise VerifyError(
                f"block {name!r}: read of {read.tensor.name!r} has {len(read.index)} "
                + f"coordinates but tensor is rank {len(read.tensor.shape)}"
            )
        for dim, coord in enumerate(read.index):
            lo, hi = _bounds(coord)
            extent = read.tensor.shape[dim]
            if lo < 0 or hi >= extent:
                raise VerifyError(
                    f"block {name!r}: read of {read.tensor.name!r} dim {dim} index range "
                    + f"[{lo}, {hi}] out of bounds [0, {extent})"
                )


def verify_program(program: ComputeProgram) -> None:
    for block in program.blocks:
        verify_block(block)
