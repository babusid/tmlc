"""`Read` and `ScalarConst` are the leaf nodes of scalar expression trees."""

from __future__ import annotations

from dataclasses import dataclass

from tmlc.compute.index.base import IndexExpr, as_index
from tmlc.compute.program.tensor import ComputeTensor
from tmlc.compute.scalar.base import ScalarExprBase
from tmlc.util.types import StrictInt


@dataclass(frozen=True, init=False)
class Read(ScalarExprBase):
    """Read a tensor at an index, coercing integer coordinates to `IntConst`."""

    tensor: ComputeTensor
    index: tuple[IndexExpr, ...]

    def __init__(self, tensor: ComputeTensor, index: tuple[IndexExpr | StrictInt, ...]) -> None:
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "index", tuple(as_index(i) for i in index))
