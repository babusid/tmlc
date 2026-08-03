"""
Compute blocks evaluate a scalar body over explicitly retained and reduced axes.

``output_axes`` directly defines the output dimensions in order. A spatial axis contributes its
extent; a reduction axis present here contributes a singleton dimension. ``reduce_axes`` are
quantified by the block's combiner, so a reduction axis can be omitted from ``output_axes``
(reduced away) or included in it (kept as a singleton dimension). This way, we no longer
need an explicit keepdims flag or infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tmlc.compute.axis import Axis
from tmlc.compute.program.tensor import ComputeTensor
from tmlc.compute.scalar.base import ScalarExprBase, ScalarOpKind


class Combiner(Enum):
    """A reduction operation and its float accumulator identity."""

    SUM = (ScalarOpKind.ADD, 0.0)
    PROD = (ScalarOpKind.MUL, 1.0)
    MAX = (ScalarOpKind.MAX, float("-inf"))

    def __init__(self, op: ScalarOpKind, identity: float) -> None:
        self.op = op
        self.identity = identity


@dataclass(frozen=True)
class ComputeBlock:
    output: ComputeTensor
    output_axes: tuple[Axis, ...]
    reduce_axes: tuple[Axis, ...]
    body: ScalarExprBase
    combiner: Combiner | None  # non-None iff reduce_axes is nonempty
