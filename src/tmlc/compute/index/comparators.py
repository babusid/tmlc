"""
Comparison index expressions for the Compute IR.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from tmlc.compute.index.base import BinaryIndex, IndexExpr, as_index
from tmlc.util.types import StrictInt


class CompareOp(Enum):
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    EQ = auto()
    NE = auto()


@dataclass(frozen=True)
class IndexCompare(BinaryIndex):
    """A comparison built with the `index_*` functions below."""

    op: CompareOp
    lhs: IndexExpr
    rhs: IndexExpr


def index_lt(lhs: IndexExpr | StrictInt, rhs: IndexExpr | StrictInt) -> IndexCompare:
    return IndexCompare(CompareOp.LT, as_index(lhs), as_index(rhs))


def index_le(lhs: IndexExpr | StrictInt, rhs: IndexExpr | StrictInt) -> IndexCompare:
    return IndexCompare(CompareOp.LE, as_index(lhs), as_index(rhs))


def index_gt(lhs: IndexExpr | StrictInt, rhs: IndexExpr | StrictInt) -> IndexCompare:
    return IndexCompare(CompareOp.GT, as_index(lhs), as_index(rhs))


def index_ge(lhs: IndexExpr | StrictInt, rhs: IndexExpr | StrictInt) -> IndexCompare:
    return IndexCompare(CompareOp.GE, as_index(lhs), as_index(rhs))


def index_eq(lhs: IndexExpr | StrictInt, rhs: IndexExpr | StrictInt) -> IndexCompare:
    return IndexCompare(CompareOp.EQ, as_index(lhs), as_index(rhs))


def index_ne(lhs: IndexExpr | StrictInt, rhs: IndexExpr | StrictInt) -> IndexCompare:
    return IndexCompare(CompareOp.NE, as_index(lhs), as_index(rhs))
