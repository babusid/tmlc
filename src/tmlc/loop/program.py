"""
Pure Loop IR: recursive single-axis loops over explicit buffers and value expressions.

The IR deliberately owns every construct it uses. Compute axes, tensors, reads, scalar expressions,
and combiners are translated at the lowering boundary rather than surviving into this dialect.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from tmlc.loop.axis import LoopAxis
from tmlc.loop.buffer import Buffer, BufferConstant
from tmlc.loop.expr import LoopIndexExpr, LoopValue


class LoopCombiner(Enum):
    SUM = 0.0
    PROD = 1.0
    MAX = float("-inf")

    @property
    def identity(self) -> float:
        return float(self.value)


@dataclass(frozen=True)
class Store:
    """Write one value to an explicit buffer coordinate."""

    buffer: Buffer
    index: tuple[LoopIndexExpr, ...]
    value: LoopValue


@dataclass(frozen=True)
class Accumulate:
    """Combine one value into an explicit buffer coordinate."""

    buffer: Buffer
    index: tuple[LoopIndexExpr, ...]
    value: LoopValue
    combiner: LoopCombiner


@dataclass(frozen=True)
class ExternalCall:
    """Opaque low-level external operation; its ABI and effects will be defined later."""


@dataclass(frozen=True)
class Loop:
    """One induction axis and its ordered, lexically scoped statements."""

    axis: LoopAxis
    body: tuple[LoopStatement, ...]


LoopStatement: TypeAlias = Loop | Store | Accumulate | ExternalCall


@dataclass(frozen=True)
class LoopProgram:
    """A standalone Loop IR statement forest and its complete buffer interface."""

    body: tuple[LoopStatement, ...]
    buffers: tuple[Buffer, ...]
    inputs: tuple[Buffer, ...]
    outputs: tuple[Buffer, ...]
    constants: tuple[BufferConstant, ...] = ()

    def has_buffer(self, buffer: Buffer) -> bool:
        return any(candidate is buffer for candidate in self.buffers)
