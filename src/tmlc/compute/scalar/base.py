"""
Dtype-valued scalar expressions used as ComputeBlock bodies.

Trees contain `Read` and `ScalarConst` leaves, `Select` nodes, and `ScalarExpr` operations.
Operators on `ScalarExprBase` build these trees and coerce numeric operands to constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from tmlc.util.types import StrictInt


class ScalarOpKind(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()
    EXP = auto()
    LOG = auto()
    TANH = auto()
    MAX = auto()
    POW = auto()


class ScalarExprBase:
    """Common base for nodes in a scalar expression tree."""

    def __add__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarAdd((self, as_scalar(other)))

    def __radd__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarAdd((as_scalar(other), self))

    def __sub__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarSub((self, as_scalar(other)))

    def __rsub__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarSub((as_scalar(other), self))

    def __mul__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarMul((self, as_scalar(other)))

    def __rmul__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarMul((as_scalar(other), self))

    def __truediv__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarDiv((self, as_scalar(other)))

    def __rtruediv__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarDiv((as_scalar(other), self))

    def __pow__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarPow((self, as_scalar(other)))

    def __rpow__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarPow((as_scalar(other), self))

    def __neg__(self) -> ScalarExpr:
        return ScalarNeg((self,))

    def exp(self) -> ScalarExpr:
        return ScalarExp((self,))

    def log(self) -> ScalarExpr:
        return ScalarLog((self,))

    def tanh(self) -> ScalarExpr:
        return ScalarTanh((self,))


@dataclass(frozen=True)
class ScalarConst(ScalarExprBase):
    value: float


@dataclass(frozen=True, init=False)
class ScalarExpr(ScalarExprBase):
    kind: ScalarOpKind
    args: tuple[ScalarExprBase, ...]

    def __new__(cls) -> ScalarExpr:
        raise TypeError("ScalarExpr cannot be constructed directly; use a typed scalar builder")


def _scalar_expr(kind: ScalarOpKind, args: tuple[ScalarExprBase, ...]) -> ScalarExpr:
    expr = object.__new__(ScalarExpr)
    object.__setattr__(expr, "kind", kind)
    object.__setattr__(expr, "args", args)
    return expr


def ScalarAdd(args: tuple[ScalarExprBase, ScalarExprBase]) -> ScalarExpr:
    return _scalar_expr(ScalarOpKind.ADD, args)


def ScalarSub(args: tuple[ScalarExprBase, ScalarExprBase]) -> ScalarExpr:
    return _scalar_expr(ScalarOpKind.SUB, args)


def ScalarMul(args: tuple[ScalarExprBase, ScalarExprBase]) -> ScalarExpr:
    return _scalar_expr(ScalarOpKind.MUL, args)


def ScalarDiv(args: tuple[ScalarExprBase, ScalarExprBase]) -> ScalarExpr:
    return _scalar_expr(ScalarOpKind.DIV, args)


def ScalarNeg(args: tuple[ScalarExprBase]) -> ScalarExpr:
    return _scalar_expr(ScalarOpKind.NEG, args)


def ScalarExp(args: tuple[ScalarExprBase]) -> ScalarExpr:
    return _scalar_expr(ScalarOpKind.EXP, args)


def ScalarLog(args: tuple[ScalarExprBase]) -> ScalarExpr:
    return _scalar_expr(ScalarOpKind.LOG, args)


def ScalarTanh(args: tuple[ScalarExprBase]) -> ScalarExpr:
    return _scalar_expr(ScalarOpKind.TANH, args)


def ScalarMax(args: tuple[ScalarExprBase, ScalarExprBase]) -> ScalarExpr:
    return _scalar_expr(ScalarOpKind.MAX, args)


def ScalarPow(args: tuple[ScalarExprBase, ScalarExprBase]) -> ScalarExpr:
    return _scalar_expr(ScalarOpKind.POW, args)


def as_scalar(value: ScalarExprBase | float | StrictInt) -> ScalarExprBase:
    if isinstance(value, ScalarExprBase):
        return value
    return ScalarConst(float(value))
