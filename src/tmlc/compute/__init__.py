"""
Compute IR for tmlc.

Layering:
    Graph IR  --lower-->  Compute IR (this package)  --lower-->  Loop IR  --emit-->  C / MSL / CUDA
"""

from __future__ import annotations

from .axis import Axis, AxisKind
from .compute import (
    Combiner,
    ComputeBlock,
    ComputeProgram,
    ComputeProgramBuilder,
    ComputeTensor,
    DenseConst,
    Read,
)
from .index import (
    AxisRef,
    IndexAdd,
    IndexExpr,
    IndexFloorDiv,
    IndexMod,
    IndexMul,
    IntConst,
    as_index,
    index_axes,
)
from .scalar import ScalarConst, ScalarExpr, ScalarExprBase, ScalarOpKind
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
    "VerifyError",
    "verify_block",
    "verify_program",
]
