"""Index and scalar-value expressions evaluated inside explicit Loop IR loops."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from tmlc.loop.axis import LoopAxis
from tmlc.loop.buffer import Buffer


class LoopIndexExpr:
    """Base class for integer buffer coordinates over enclosing loop axes."""


@dataclass(frozen=True)
class LoopIndex(LoopIndexExpr):
    axis: LoopAxis


@dataclass(frozen=True)
class IndexConst(LoopIndexExpr):
    value: int


@dataclass(frozen=True)
class IndexAdd(LoopIndexExpr):
    lhs: LoopIndexExpr
    rhs: LoopIndexExpr


@dataclass(frozen=True)
class IndexMul(LoopIndexExpr):
    lhs: LoopIndexExpr
    rhs: LoopIndexExpr


@dataclass(frozen=True)
class IndexFloorDiv(LoopIndexExpr):
    lhs: LoopIndexExpr
    divisor: int


@dataclass(frozen=True)
class IndexMod(LoopIndexExpr):
    lhs: LoopIndexExpr
    modulus: int


class LoopCompareOp(Enum):
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    EQ = auto()
    NE = auto()


@dataclass(frozen=True)
class IndexCompare(LoopIndexExpr):
    op: LoopCompareOp
    lhs: LoopIndexExpr
    rhs: LoopIndexExpr


class LoopValue:
    """Base class for scalar values computed by Loop IR statements."""


@dataclass(frozen=True)
class FloatConst(LoopValue):
    value: float


@dataclass(frozen=True)
class BufferLoad(LoopValue):
    buffer: Buffer
    index: tuple[LoopIndexExpr, ...]


class LoopValueOp(Enum):
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


@dataclass(frozen=True)
class ValueExpr(LoopValue):
    op: LoopValueOp
    args: tuple[LoopValue, ...]


@dataclass(frozen=True)
class ValueSelect(LoopValue):
    condition: LoopIndexExpr
    if_true: LoopValue
    if_false: LoopValue
