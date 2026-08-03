"""
Integer-valued coordinate expressions over block axes.

The language provides axis and constant leaves with addition, multiplication, floor division,
modulo, and comparisons. These expressions describe tensor access patterns such as elementwise
reads, transposes, broadcasts, slices, and reshapes. They are separate from dtype-valued scalar
expressions used to compute tensor values.
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
    """Shared base for binary index expressions."""

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
    """Floor division by a positive constant."""

    lhs: IndexExpr
    divisor: PositiveInt


@dataclass(frozen=True)
class IndexMod(IndexExpr):
    """Modulo by a positive constant."""

    lhs: IndexExpr
    modulus: PositiveInt


def index_axes(expr: IndexExpr) -> Iterator[Axis]:
    """Yield the axes referenced by an index expression."""
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
