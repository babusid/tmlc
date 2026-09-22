"""
Base class for Compute IR passes, mirroring ``tmlc.graph.graph.GraphTransform``.

A ``ComputeProgramTransform`` maps a ``ComputeProgram`` to a new ``ComputeProgram``. Like
``GraphTransform`` it is a class rather than a bare callable so that each pass carries its own
configuration (thresholds, cost-model hooks) on the instance while presenting a uniform, easily
type-checked ``__call__`` signature. Passes must be pure: build and return a new program, never
mutate the input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import reduce

from tmlc.compute.program import ComputeProgram


class ComputeProgramTransform(ABC):
    @abstractmethod
    def __call__(self, program: ComputeProgram) -> ComputeProgram:
        raise NotImplementedError("ComputeProgramTransform subclasses must implement __call__")


def apply_transforms(
    program: ComputeProgram, transforms: Sequence[ComputeProgramTransform]
) -> ComputeProgram:
    """Run a pipeline of Compute IR passes left-to-right."""
    return reduce(lambda prog, fn: fn(prog), transforms, program)
