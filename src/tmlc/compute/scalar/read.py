"""
The `Read` scalar leaf: a tensor read at an affine coordinate.

`Read` bridges the index domain into the scalar domain -- a `ScalarExprBase` whose children are
`IndexExpr` coordinates -- so it imports `ComputeTensor`. That is the one edge that would cycle if
`Read` lived beside `ComputeTensor`, so it lives here and `ComputeTensor.__getitem__` reaches back
for it with a deferred import.
"""

from __future__ import annotations

from dataclasses import dataclass

from tmlc.compute.index.base import IndexExpr, as_index
from tmlc.compute.program.tensor import ComputeTensor
from tmlc.compute.scalar.base import ScalarExprBase
from tmlc.util.types import StrictInt


@dataclass(frozen=True, init=False)
class Read(ScalarExprBase):
    """
    The ScalarExprBase leaf: a read of `tensor` at an affine coordinate. `len(index)` must equal
    the tensor's rank. Bare ints in the coordinate are coerced to
    IntConst, so `Read(x, (0, AxisRef(j)))` is a broadcast read.
    """

    tensor: ComputeTensor
    index: tuple[IndexExpr, ...]

    def __init__(self, tensor: ComputeTensor, index: tuple[IndexExpr | StrictInt, ...]) -> None:
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "index", tuple(as_index(i) for i in index))
