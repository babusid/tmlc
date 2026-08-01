"""
`ComputeProgram`: a flat, ordered list of blocks plus its inputs, outputs, and constants.
"""

from __future__ import annotations

from dataclasses import dataclass

from tmlc.compute.program.block import ComputeBlock
from tmlc.compute.program.tensor import ComputeTensor

# A dense constant payload: a scalar, or a (possibly nested) tuple of them. Splat constants are the
# Fill op (a block).
type DenseConst = float | tuple[DenseConst, ...]


@dataclass(frozen=True)
class ComputeProgram:
    """
    Blocks in dependency order. Fields are tuples, not lists: frozen=True is shallow, so a list
    field would still be mutable via .append().
    """

    tensors: tuple[ComputeTensor, ...]
    blocks: tuple[ComputeBlock, ...]
    inputs: tuple[ComputeTensor, ...]
    outputs: tuple[ComputeTensor, ...]
    constants: tuple[tuple[ComputeTensor, DenseConst], ...] = ()
