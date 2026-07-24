"""
Tensors, blocks, and programs for the Compute IR.

A ComputeBlock has exactly ONE iteration domain, which is the index space of the block's OUTPUT.
Operands are read at affine index expressions over that domain (see `index.py`), and the body is a
scalar expression over those reads (see `scalar.py`).

    output.shape == tuple(a.extent for a in domain if a.kind is SPATIAL)

Spatial axes survive into the output in domain order (identity write map). Reduce axes are combined
away by the block's `combiner`; a block has at most ONE combiner shared by all reduce axes, so an op
needing two different reductions (e.g. logsumexp: max then sum) becomes multiple blocks.

A ComputeProgram is a flat, ordered list of blocks built by ComputeProgramBuilder. The builder is
the accumulation channel: `compute()` records the block and its output tensor internally and returns
only the output tensor, which is the handle a consumer needs. Multi-block ops leave every block in
the builder and return just their final tensor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from tmlc.compute.axis import Axis, AxisKind
from tmlc.compute.index import IndexExpr, as_index
from tmlc.util.types import StrictInt
from tmlc.compute.scalar import ScalarExprBase, ScalarOpKind

# A dense constant payload: a scalar, or a (possibly nested) tuple of them. Splat constants are the
# Fill op (a block).
type DenseConst = float | tuple[DenseConst, ...]


@dataclass(frozen=True, eq=False)
class ComputeTensor:
    """
    A named buffer. eq=False gives identity semantics, so tensors are safe dict keys and two
    same-shaped intermediates never alias. There is exactly one ComputeTensor per block output,
    created by the builder and handed back.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def rank(self) -> int:
        return len(self.shape)

    def __getitem__(self, index: IndexExpr | StrictInt | tuple[IndexExpr | StrictInt, ...]) -> Read:
        """
        Sugar for a Read of this tensor at `index`. A single index is wrapped to a 1-tuple, and
        the arity must match the tensor's rank. Bare ints are coerced to IntConst by Read.
        """
        coords = index if isinstance(index, tuple) else (index,)
        if len(coords) != self.rank:
            raise ValueError(f"index arity {len(coords)} does not match tensor rank {self.rank}")
        return Read(self, coords)


@dataclass(frozen=True, init=False)
class Read(ScalarExprBase):
    """
    The ScalarExprBase leaf: a read of `tensor` at an affine coordinate. `len(index)` must equal
    the tensor's rank. Bare ints in the coordinate are coerced to
    IntConst, so `Read(x, (0, AxisRef(j)))` is a broadcast read.
    """

    # TODO: enforce length == rank via verifier

    tensor: ComputeTensor
    index: tuple[IndexExpr, ...]

    def __init__(self, tensor: ComputeTensor, index: tuple[IndexExpr | StrictInt, ...]) -> None:
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "index", tuple(as_index(i) for i in index))


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
        # local import: verify depends on compute's types, so a top-level import would cycle
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
