"""Self-contained explicit-loop IR and the Compute-to-Loop translation boundary."""

from __future__ import annotations

from .axis import LoopAxis
from .buffer import (
    PROGRAM_SCOPE,
    Buffer,
    BufferConstant,
    BufferLayout,
    LoopScope,
    ProgramScope,
    row_major_strides,
)
from .expr import (
    BufferLoad,
    FloatConst,
    IndexAdd,
    IndexCompare,
    IndexConst,
    IndexFloorDiv,
    IndexMod,
    IndexMul,
    LoopCompareOp,
    LoopIndex,
    LoopIndexExpr,
    LoopValue,
    LoopValueOp,
    ValueExpr,
    ValueSelect,
)
from .lower import lower_to_loops
from .program import Accumulate, ExternalCall, Loop, LoopCombiner, LoopProgram, Store
from .verify import LoopVerifyError, verify_loop_program

__all__ = [
    "LoopAxis",
    "Buffer",
    "BufferConstant",
    "BufferLayout",
    "ProgramScope",
    "LoopScope",
    "PROGRAM_SCOPE",
    "row_major_strides",
    "LoopIndexExpr",
    "LoopIndex",
    "IndexConst",
    "IndexAdd",
    "IndexMul",
    "IndexFloorDiv",
    "IndexMod",
    "LoopCompareOp",
    "IndexCompare",
    "LoopValue",
    "FloatConst",
    "BufferLoad",
    "LoopValueOp",
    "ValueExpr",
    "ValueSelect",
    "Store",
    "Accumulate",
    "ExternalCall",
    "LoopCombiner",
    "Loop",
    "LoopProgram",
    "LoopVerifyError",
    "verify_loop_program",
    "lower_to_loops",
]
