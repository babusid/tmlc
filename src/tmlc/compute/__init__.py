"""
Compute IR for tmlc.

Layering:
    Graph IR  --lower-->  Compute IR (this package)  --lower-->  Loop IR  --emit-->  C / MSL / CUDA
"""

from __future__ import annotations

from .axis import Axis, AxisKind
from .index import (
    AxisRef,
    CompareOp,
    IndexAdd,
    IndexCompare,
    IndexExpr,
    IndexFloorDiv,
    IndexMod,
    IndexMul,
    IntConst,
    as_index,
    index_axes,
    index_eq,
    index_ge,
    index_gt,
    index_le,
    index_lt,
    index_ne,
)
from .program import (
    Combiner,
    ComputeBlock,
    ComputeProgram,
    ComputeProgramBuilder,
    ComputeTensor,
    DenseConst,
)
from .scalar import Read, ScalarConst, ScalarExpr, ScalarExprBase, ScalarOpKind, Select
from .verify import VerifyError, verify_block, verify_program

__all__ = [
    "Axis",
    "AxisKind",
    "IndexExpr",
    "AxisRef",
    "IntConst",
    "IndexAdd",
    "IndexMul",
    "IndexFloorDiv",
    "IndexMod",
    "IndexCompare",
    "CompareOp",
    "index_lt",
    "index_le",
    "index_gt",
    "index_ge",
    "index_eq",
    "index_ne",
    "index_axes",
    "as_index",
    "ScalarOpKind",
    "ScalarExprBase",
    "ScalarConst",
    "ScalarExpr",
    "ComputeTensor",
    "Combiner",
    "ComputeBlock",
    "ComputeProgram",
    "ComputeProgramBuilder",
    "DenseConst",
    "Read",
    "Select",
    "VerifyError",
    "verify_block",
    "verify_program",
]
