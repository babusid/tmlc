"""Mutable builder for Compute IR programs."""

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
    """Accumulate blocks and tensors; `compute` returns outputs and `finish` emits a program."""

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
        output_axes: tuple[Axis, ...],
        body: ScalarExprBase,
        reduce_axes: tuple[Axis, ...] = (),
        combiner: Combiner | None = None,
        dtype: str = "float32",
        hint: str = "t",
    ) -> ComputeTensor:
        shape = tuple(1 if axis.kind is AxisKind.REDUCE else axis.extent for axis in output_axes)
        out = ComputeTensor(self._fresh(hint), shape, dtype)
        block = ComputeBlock(
            output=out,
            output_axes=output_axes,
            reduce_axes=reduce_axes,
            body=body,
            combiner=combiner,
        )
        # Import lazily to avoid the verify -> program -> builder cycle.
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
