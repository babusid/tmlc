"""
Scalar expressions for the Compute IR.

Scalar expressions are dtype-valued and form the body of a ComputeBlock. They are built over tensor
reads (the `Read` leaf, added once ComputeTensor lands) and scalar constants.

Deliberately a SEPARATE hierarchy from IndexExpr (see `index.py`). Index expressions are integer-
valued; scalar values are dtype-valued and may be transcendental. Sharing a base class would let
`exp(i)` typecheck as an index.

Naming note: the base of the hierarchy is `ScalarExprBase`; the operation node -- an application of
a `ScalarOpKind` to argument expressions -- is `ScalarExpr`.

The operator sugar lives on `ScalarExprBase` so every scalar expression (`Read`, `ScalarConst`,
`ScalarExpr`) inherits it and trees compose to any depth: `(x[i, k] * w[k, j] - m[i]).exp()`.
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
    """
    Base for dtype-valued scalar expressions (the block body).

    Numeric literals in an operand position are coerced to `ScalarConst`, so `x[i] * 0.5` works.
    """

    def __add__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.ADD, (self, as_scalar(other)))

    def __radd__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.ADD, (as_scalar(other), self))

    def __sub__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.SUB, (self, as_scalar(other)))

    def __rsub__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.SUB, (as_scalar(other), self))

    def __mul__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.MUL, (self, as_scalar(other)))

    def __rmul__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.MUL, (as_scalar(other), self))

    def __truediv__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.DIV, (self, as_scalar(other)))

    def __rtruediv__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.DIV, (as_scalar(other), self))

    def __pow__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.POW, (self, as_scalar(other)))

    def __rpow__(self, other: ScalarExprBase | float | StrictInt) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.POW, (as_scalar(other), self))

    def __neg__(self) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.NEG, (self,))

    def exp(self) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.EXP, (self,))

    def log(self) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.LOG, (self,))

    def tanh(self) -> ScalarExpr:
        return ScalarExpr(ScalarOpKind.TANH, (self,))


@dataclass(frozen=True)
class ScalarConst(ScalarExprBase):
    value: float


@dataclass(frozen=True)
class ScalarExpr(ScalarExprBase):
    kind: ScalarOpKind
    args: tuple[ScalarExprBase, ...]


def as_scalar(value: ScalarExprBase | float | StrictInt) -> ScalarExprBase:
    if isinstance(value, ScalarExprBase):
        return value
    return ScalarConst(float(value))
