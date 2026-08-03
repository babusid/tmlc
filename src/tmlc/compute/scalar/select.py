"""Index predication in scalar expression trees."""

from __future__ import annotations

from dataclasses import dataclass

from tmlc.compute.index.base import IndexExpr, as_index
from tmlc.compute.scalar.base import ScalarExprBase, as_scalar
from tmlc.util.types import StrictInt


@dataclass(frozen=True, init=False)
class Select(ScalarExprBase):
    """Index-predicated value: choose `if_true` when `cond` is nonzero, else `if_false`."""

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
