"""
Iteration axes for the Compute IR.

An axis is a single dimension of a ComputeBlock's iteration domain. A block has exactly ONE
iteration domain, which is the index space of the block's OUTPUT: spatial axes survive into the
output (in domain order, identity write map), and reduce axes are combined away by the block's
combiner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class AxisKind(Enum):
    SPATIAL = auto()
    REDUCE = auto()
    # SCAN = auto()   # sequential carry (cumsum). Not implemented.


@dataclass(frozen=True, eq=False)
class Axis:
    """
    An iteration axis.
    eq=False is load-bearing: identity semantics, NOT value semantics. Axes are all individual
    objects, even if they have the same kind and extent.

    Axes are block-scoped: mint fresh axes per block, never reuse across blocks. Since axes are
    identity-based, a reused axis would appear in two domains and make "which block does this axis
    belong to" ambiguous for later passes.
    """

    kind: AxisKind
    extent: int
    name: str
