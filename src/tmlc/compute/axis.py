"""
Iteration axes for Compute IR blocks.

Spatial axes define output dimensions; reduce axes are either eliminated by the block's combiner or
are turned into singletons (if they are present in the output shape).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class AxisKind(Enum):
    SPATIAL = auto()
    REDUCE = auto()


@dataclass(frozen=True, eq=False)
class Axis:
    """A block-scoped iteration axis with identity-based equality. Do not reuse across blocks."""

    kind: AxisKind
    extent: int
    name: str
