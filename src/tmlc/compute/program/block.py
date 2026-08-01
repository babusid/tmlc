"""
`Combiner` and `ComputeBlock`: one iteration domain, one scalar body, at most one reduction.

A ComputeBlock has exactly ONE iteration domain, which is the index space of the block's OUTPUT.
Operands are read at affine index expressions over that domain (see `index`), and the body is a
scalar expression over those reads (see `scalar`).

    output.shape == tuple(a.extent for a in domain if a.kind is SPATIAL)

Spatial axes survive into the output in domain order (identity write map). Reduce axes are combined
away by the block's `combiner`; a block has at most ONE combiner shared by all reduce axes, so an op
needing two different reductions (e.g. logsumexp: max then sum) becomes multiple blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tmlc.compute.axis import Axis
from tmlc.compute.program.tensor import ComputeTensor
from tmlc.compute.scalar.base import ScalarExprBase, ScalarOpKind


class Combiner(Enum):
    """
    Reduction combiner. `op` is the binary scalar op applied across a reduce axis; `identity` is
    the accumulator's init value, so the emitter looks it up rather than switching on the enum.

    `identity` is the float32 identity. It becomes dtype-dependent once integer dtypes exist (int
    max wants INT_MIN, not -inf).
    """

    SUM = (ScalarOpKind.ADD, 0.0)
    PROD = (ScalarOpKind.MUL, 1.0)
    MAX = (ScalarOpKind.MAX, float("-inf"))

    def __init__(self, op: ScalarOpKind, identity: float) -> None:
        self.op = op
        self.identity = identity


@dataclass(frozen=True)
class ComputeBlock:
    output: ComputeTensor
    domain: tuple[Axis, ...]  # ordered; spatial axes map to output dims in order
    body: ScalarExprBase
    combiner: Combiner | None  # non-None iff domain contains a REDUCE axis
