"""
`ComputeTensor`: a named buffer in the Compute IR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tmlc.compute.index.base import IndexExpr
from tmlc.util.types import StrictInt

if TYPE_CHECKING:
    from tmlc.compute.scalar.read import Read
else:
    # forward ref: avoids the Read <-> ComputeTensor cycle
    # at runtime, but keeps beartype runtime check
    Read = "tmlc.compute.scalar.read.Read"


@dataclass(frozen=True, eq=False)
class ComputeTensor:
    """
    A named buffer. eq=False gives identity semantics, so tensors are safe dict keys and two
    same-shaped intermediates never alias. There is exactly one ComputeTensor per block output,
    created by the builder and handed back.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def rank(self) -> int:
        return len(self.shape)

    def __getitem__(self, index: IndexExpr | StrictInt | tuple[IndexExpr | StrictInt, ...]) -> Read:
        """
        Sugar for a Read of this tensor at `index`. A single index is wrapped to a 1-tuple, and
        the arity must match the tensor's rank. Bare ints are coerced to IntConst by Read.
        """
        from tmlc.compute.scalar.read import Read

        coords = index if isinstance(index, tuple) else (index,)
        if len(coords) != self.rank:
            raise ValueError(f"index arity {len(coords)} does not match tensor rank {self.rank}")
        return Read(self, coords)
