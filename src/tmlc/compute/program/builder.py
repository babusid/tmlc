"""
`ComputeProgramBuilder`: the accumulation channel used while lowering.

`compute()` records the block and its output tensor internally and returns only the output tensor,
the handle a consumer needs. Multi-block ops leave every block in the builder and return just their
final tensor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tmlc.compute.axis import Axis, AxisKind
from tmlc.compute.program.block import Combiner, ComputeBlock
from tmlc.compute.program.program import ComputeProgram, DenseConst
from tmlc.compute.program.tensor import ComputeTensor
from tmlc.compute.scalar.base import ScalarExprBase


@dataclass
class ComputeProgramBuilder:
    """Accumulates blocks and tensors while lowering, then emits a ComputeProgram via `finish`."""

    _blocks: list[ComputeBlock] = field(default_factory=list)
    _tensors: list[ComputeTensor] = field(default_factory=list)
    _inputs: list[ComputeTensor] = field(default_factory=list)
    _constants: list[tuple[ComputeTensor, DenseConst]] = field(default_factory=list)
    _counters: dict[str, int] = field(default_factory=dict)

    def _fresh(self, hint: str) -> str:
        self._counters[hint] = self._counters.get(hint, 0) + 1
        return f"{hint}_{self._counters[hint]}"

    def spatial(self, extent: int, name: str = "spatial") -> Axis:
        return Axis(AxisKind.SPATIAL, extent, self._fresh(name))

    def reduce(self, extent: int, name: str = "reduce") -> Axis:
        return Axis(AxisKind.REDUCE, extent, self._fresh(name))

    def declare_input(
        self, shape: tuple[int, ...], dtype: str, hint: str = "input"
    ) -> ComputeTensor:
        t = ComputeTensor(self._fresh(hint), shape, dtype)
        self._tensors.append(t)
        self._inputs.append(t)
        return t

    def declare_constant(
        self, shape: tuple[int, ...], dtype: str, value: DenseConst, hint: str = "const"
    ) -> ComputeTensor:
        t = ComputeTensor(self._fresh(hint), shape, dtype)
        self._tensors.append(t)
        self._constants.append((t, value))
        return t

    def compute(
        self,
        domain: tuple[Axis, ...],
        body: ScalarExprBase,
        combiner: Combiner | None = None,
        dtype: str = "float32",
        hint: str = "t",
    ) -> ComputeTensor:
        shape = tuple(a.extent for a in domain if a.kind is AxisKind.SPATIAL)
        out = ComputeTensor(self._fresh(hint), shape, dtype)
        block = ComputeBlock(output=out, domain=domain, body=body, combiner=combiner)
        # deferred import: verify depends on the program types, so a top-level import would cycle
        from tmlc.compute.verify import verify_block

        verify_block(block)
        self._blocks.append(block)
        self._tensors.append(out)
        return out

    def finish(self, outputs: Sequence[ComputeTensor]) -> ComputeProgram:
        return ComputeProgram(
            tensors=tuple(self._tensors),
            blocks=tuple(self._blocks),
            inputs=tuple(self._inputs),
            outputs=tuple(outputs),
            constants=tuple(self._constants),
        )
