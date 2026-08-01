"""
Index expressions for the Compute IR: the affine core.

Index expressions are integer-valued and defined over axes. They are the coordinates at
which a block's operands are read.

Deliberately a SEPARATE hierarchy from ScalarExpr (see `scalar`). Indices are integer-valued;
scalar values are dtype-valued and may be transcendental. Sharing a base class would let `exp(i)`
typecheck as an index.

An affine index expression, rather than a bare Axis, is what lets one mechanism cover every access
pattern:

    elementwise            x[i, j]
    transpose              x[j, i]
    broadcast (1,N)->(M,N) x[0, j]        -- an integer constant  (IntConst)
    slice                  x[off + i]     -- needs +              (IndexAdd)
    reshape                x[i*N + j]     -- needs *              (IndexMul)

A bare Axis cant express any of the last three as it's just an iterator over an extent.
As such, we have the IndexExpr dialect where `AxisRef` becomes a leaf bound to an axis
(the other leaf type being a constant).

Constants are their own leaf (`IntConst`), not an offset folded onto the Axis. A constant plays
three roles across the patterns above -- a coordinate (broadcast `0`), an additive offset (slice
`off + i`), and a multiplicative coefficient (reshape `i*N`) -- and an axis offset only covers the
additive one. It also belongs to the read, not the loop: matmul's `i` maps into different operands
with different coefficients, so scale/offset live on the IndexExpr, not the shared Axis. Folding a
constant into an extent-1 axis would further pollute the domain (phantom output dims), break the
domain-membership check, and clash with the Axis identity semantics constants don't want.

Comparison nodes live in `comparators`; they subclass `BinaryIndex` here so `index_axes` traverses
them without importing that module.
"""

from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass

from tmlc.compute.axis import Axis
from tmlc.util.types import PositiveInt, StrictInt


class IndexExpr:
    """Base for integer-valued index expressions."""

    def __add__(self, other: IndexExpr | StrictInt) -> IndexAdd:
        return IndexAdd(self, as_index(other))

    def __radd__(self, other: IndexExpr | StrictInt) -> IndexAdd:
        return IndexAdd(as_index(other), self)

    def __mul__(self, other: IndexExpr | StrictInt) -> IndexMul:
        return IndexMul(self, as_index(other))

    def __rmul__(self, other: IndexExpr | StrictInt) -> IndexMul:
        return IndexMul(as_index(other), self)

    def __floordiv__(self, divisor: PositiveInt) -> IndexFloorDiv:
        return IndexFloorDiv(self, divisor)

    def __mod__(self, modulus: PositiveInt) -> IndexMod:
        return IndexMod(self, modulus)


def as_index(value: IndexExpr | StrictInt) -> IndexExpr:
    if isinstance(value, IndexExpr):
        return value
    return IntConst(value)


class BinaryIndex(IndexExpr):
    """
    Shared base for two-operand index nodes (`IndexAdd`, `IndexMul`, and `IndexCompare` in
    `comparators`). Lets `index_axes` recurse over any of them with one `isinstance` branch, and
    lets a comparator subclass live in another module without this one importing it.
    """

    lhs: IndexExpr
    rhs: IndexExpr


@dataclass(frozen=True)
class AxisRef(IndexExpr):
    axis: Axis


@dataclass(frozen=True)
class IntConst(IndexExpr):
    value: int


@dataclass(frozen=True)
class IndexAdd(BinaryIndex):
    lhs: IndexExpr
    rhs: IndexExpr


@dataclass(frozen=True)
class IndexMul(BinaryIndex):
    lhs: IndexExpr
    rhs: IndexExpr


@dataclass(frozen=True)
class IndexFloorDiv(IndexExpr):
    """Floor division by a positive constant, used to unflatten reshape coordinates."""

    lhs: IndexExpr
    divisor: PositiveInt


@dataclass(frozen=True)
class IndexMod(IndexExpr):
    """Modulo by a positive constant, used to unflatten reshape coordinates."""

    lhs: IndexExpr
    modulus: PositiveInt


def index_axes(expr: IndexExpr) -> Iterator[Axis]:
    """Yield every Axis referenced by an index expression."""
    if isinstance(expr, AxisRef):
        yield expr.axis
    elif isinstance(expr, IntConst):
        return
    elif isinstance(expr, BinaryIndex):
        yield from index_axes(expr.lhs)
        yield from index_axes(expr.rhs)
    elif isinstance(expr, (IndexFloorDiv, IndexMod)):
        yield from index_axes(expr.lhs)
    else:
        raise TypeError(f"unknown IndexExpr: {type(expr).__name__}")
