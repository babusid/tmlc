"""Named buffers in the Compute IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tmlc.compute.index.base import IndexExpr
from tmlc.util.types import StrictInt

if TYPE_CHECKING:
    from tmlc.compute.scalar.read import Read
else:
    # Keep runtime type checks without importing Read here.
    Read = "tmlc.compute.scalar.read.Read"


@dataclass(frozen=True, eq=False)
class ComputeTensor:
    """A named buffer with identity-based equality."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def rank(self) -> int:
        return len(self.shape)

    def __getitem__(self, index: IndexExpr | StrictInt | tuple[IndexExpr | StrictInt, ...]) -> Read:
        """Build a rank-matched `Read`; single coordinates are wrapped and integers are coerced."""
        from tmlc.compute.scalar.read import Read

        coords = index if isinstance(index, tuple) else (index,)
        if len(coords) != self.rank:
            raise ValueError(f"index arity {len(coords)} does not match tensor rank {self.rank}")
        return Read(self, coords)
