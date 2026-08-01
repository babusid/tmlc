"""The IndexExpr dialect: affine `base` plus `comparators`."""

from __future__ import annotations

from .base import (
    AxisRef,
    BinaryIndex,
    IndexAdd,
    IndexExpr,
    IndexFloorDiv,
    IndexMod,
    IndexMul,
    IntConst,
    as_index,
    index_axes,
)
from .comparators import (
    CompareOp,
    IndexCompare,
    index_eq,
    index_ge,
    index_gt,
    index_le,
    index_lt,
    index_ne,
)

__all__ = [
    "IndexExpr",
    "BinaryIndex",
    "AxisRef",
    "IntConst",
    "IndexAdd",
    "IndexMul",
    "IndexFloorDiv",
    "IndexMod",
    "as_index",
    "index_axes",
    "CompareOp",
    "IndexCompare",
    "index_lt",
    "index_le",
    "index_gt",
    "index_ge",
    "index_eq",
    "index_ne",
]
