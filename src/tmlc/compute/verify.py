"""
Verifier for the Compute IR: cheap structural checks on each ComputeBlock.
"""

from __future__ import annotations

from collections.abc import Iterator

from tmlc.compute.axis import Axis, AxisKind
from tmlc.compute.index import (
    AxisRef,
    CompareOp,
    IndexAdd,
    IndexCompare,
    IndexExpr,
    IndexFloorDiv,
    IndexMod,
    IndexMul,
    IntConst,
    index_axes,
)
from tmlc.compute.program import ComputeBlock, ComputeProgram
from tmlc.compute.scalar import Read, ScalarConst, ScalarExpr, ScalarExprBase, Select


class VerifyError(Exception):
    pass


def _reads(body: ScalarExprBase) -> Iterator[Read]:
    if isinstance(body, Read):
        yield body
    elif isinstance(body, ScalarConst):
        return
    elif isinstance(body, Select):
        yield from _reads(body.if_true)
        yield from _reads(body.if_false)
    elif isinstance(body, ScalarExpr):
        for arg in body.args:
            yield from _reads(arg)
    else:
        raise TypeError(f"unknown ScalarExprBase: {type(body).__name__}")


def _body_axes(body: ScalarExprBase) -> Iterator[Axis]:
    # A Select condition can reference a domain axis that appears in no read, so walk the body
    # directly rather than only over reads.
    if isinstance(body, Read):
        for coord in body.index:
            yield from index_axes(coord)
    elif isinstance(body, ScalarConst):
        return
    elif isinstance(body, Select):
        yield from index_axes(body.cond)
        yield from _body_axes(body.if_true)
        yield from _body_axes(body.if_false)
    elif isinstance(body, ScalarExpr):
        for arg in body.args:
            yield from _body_axes(arg)
    else:
        raise TypeError(f"unknown ScalarExprBase: {type(body).__name__}")


_NEGATION: dict[CompareOp, CompareOp] = {
    CompareOp.LT: CompareOp.GE,
    CompareOp.LE: CompareOp.GT,
    CompareOp.GT: CompareOp.LE,
    CompareOp.GE: CompareOp.LT,
    CompareOp.EQ: CompareOp.NE,
    CompareOp.NE: CompareOp.EQ,
}


def _compare_holds(op: CompareOp, lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    """
    Whether `op` holds (is true) for every (lhs, rhs) pair drawn from the two inclusive ranges.
    """
    (lo1, hi1), (lo2, hi2) = lhs, rhs
    match op:
        case CompareOp.LT:
            return hi1 < lo2
        case CompareOp.LE:
            return hi1 <= lo2
        case CompareOp.GT:
            return lo1 > hi2
        case CompareOp.GE:
            return lo1 >= hi2
        case CompareOp.EQ:
            return lo1 == hi1 == lo2 == hi2
        case CompareOp.NE:
            return hi1 < lo2 or hi2 < lo1


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
    if isinstance(expr, IndexCompare):
        # "totally true" means returning 1 across the whole range. vv for "totally false"
        # A comparison is totally true only when its negation is totally false.
        # And, it is false exactly when its negation is totally true.
        lhs, rhs = _bounds(expr.lhs), _bounds(expr.rhs)
        # check if the op is true across the entire range
        totally_true = _compare_holds(expr.op, lhs, rhs)
        # check if the op's inverse is true across the entire range
        totally_false = _compare_holds(_NEGATION[expr.op], lhs, rhs)
        # if op is totally true (tt=1, tf=0) return (1,1)
        # if op is totally false (tt=0, tf=1), return (0,0)
        # if op is in the middle ie (tt=0, tf=0), return (0, 1)
        # note that tt and tf aren't complements!
        return (int(totally_true), int(not totally_false))
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
