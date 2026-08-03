"""A flat, ordered Compute IR program."""

from __future__ import annotations

from dataclasses import dataclass

from tmlc.compute.program.block import ComputeBlock
from tmlc.compute.program.tensor import ComputeTensor

# A scalar or nested tuple payload. Splats are represented by compute blocks.
type DenseConst = float | tuple[DenseConst, ...]


@dataclass(frozen=True)
class ComputeProgram:
    """A program with dependency-ordered blocks and immutable tuple collections."""

    tensors: tuple[ComputeTensor, ...]
    blocks: tuple[ComputeBlock, ...]
    inputs: tuple[ComputeTensor, ...]
    outputs: tuple[ComputeTensor, ...]
    constants: tuple[tuple[ComputeTensor, DenseConst], ...] = ()
