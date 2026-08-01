"""
The `Select` scalar node: a value chosen by an index-domain condition.

Like `Read`, `Select` bridges index into scalar -- its `cond` is an `IndexExpr` while its branches
are scalar expressions -- but it needs no `ComputeTensor`, so it sits cleanly in the scalar package.
"""

from __future__ import annotations

from dataclasses import dataclass

from tmlc.compute.index.base import IndexExpr, as_index
from tmlc.compute.scalar.base import ScalarExprBase, as_scalar
from tmlc.util.types import StrictInt


@dataclass(frozen=True, init=False)
class Select(ScalarExprBase):
    """
    A value chosen by an index-domain condition: `if_true` where `cond` is non-zero, else
    `if_false`. `cond` is an IndexExpr (typically an `IndexCompare`), the branches are scalar
    expressions.

    The index->scalar bridge, mirroring `Read`: a scalar node whose child is an index expression.
    Value selection only, never control flow -- both branches are defined over the whole domain, so
    each is verified in-bounds unconditionally; a provably dead branch is removed by a pruning pass,
    not excused at verify time.
    """

    cond: IndexExpr
    if_true: ScalarExprBase
    if_false: ScalarExprBase

    def __init__(
        self,
        cond: IndexExpr | StrictInt,
        if_true: ScalarExprBase | float | StrictInt,
        if_false: ScalarExprBase | float | StrictInt,
    ) -> None:
        object.__setattr__(self, "cond", as_index(cond))
        object.__setattr__(self, "if_true", as_scalar(if_true))
        object.__setattr__(self, "if_false", as_scalar(if_false))
