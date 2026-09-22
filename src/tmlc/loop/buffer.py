"""Explicit Loop IR buffers, layouts, and allocation scopes."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TypeAlias

from tmlc.loop.axis import LoopAxis


def row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Dense C-order strides in elements: ``strides[i] = prod(shape[i+1:])``."""
    return tuple(prod(shape[dim + 1 :]) for dim in range(len(shape)))


@dataclass(frozen=True)
class BufferLayout:
    """A concrete strided layout whose strides and offset are measured in elements."""

    strides: tuple[int, ...]
    offset: int = 0

    @classmethod
    def dense(cls, shape: tuple[int, ...]) -> BufferLayout:
        """Construct a row-major dense layout over ``shape``."""
        return cls(strides=row_major_strides(shape))


@dataclass(frozen=True)
class ProgramScope:
    """Storage whose lifetime spans the entire LoopProgram invocation."""


@dataclass(frozen=True)
class LoopScope:
    """Storage allocated once per iteration of ``axis`` and visible in its subtree."""

    axis: LoopAxis


BufferScope: TypeAlias = ProgramScope | LoopScope
PROGRAM_SCOPE = ProgramScope()


@dataclass(frozen=True, eq=False)
class Buffer:
    """A typed, shaped storage object owned entirely by Loop IR."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    layout: BufferLayout
    scope: BufferScope = PROGRAM_SCOPE

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return prod(self.shape) if self.shape else 1


ConstantValue: TypeAlias = float | int | tuple["ConstantValue", ...]


@dataclass(frozen=True)
class BufferConstant:
    """Initializer for an immutable program-scoped buffer."""

    buffer: Buffer
    value: ConstantValue
