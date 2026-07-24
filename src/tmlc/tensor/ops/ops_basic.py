from __future__ import annotations

from typing_extensions import override
from tmlc.tensor.literal import LiteralData, LiteralValue
from tmlc.tensor.tensor import Tensor, TensorOp
from tmlc.compute.compute import ComputeProgramBuilder, ComputeTensor, DenseConst

# The native structure of `ndarray.tolist()`: a scalar, or nested lists of them.
type _NestedScalars = float | int | list[_NestedScalars]


def _dense_const(value: _NestedScalars) -> DenseConst:
    if isinstance(value, list):
        return tuple(_dense_const(item) for item in value)
    return float(value)


class Constant(TensorOp):
    value: LiteralValue

    def __init__(self, value: LiteralData | LiteralValue):
        self.value = value if isinstance(value, LiteralValue) else LiteralValue(value)

    @override
    def __call__(
        self,
        inputs: tuple[Tensor, ...],
        label: str | None = None,
    ) -> Tensor:
        assert inputs is None or len(inputs) == 0, "Constant op cannot accept any input tensors"
        return Tensor(
            inputs=tuple(),
            shape=self.infer_shape(tuple()),
            op=self,
            label=f"{label}={self.value}" if label else f"const={self.value}",
        )

    @override
    def infer_shape(self, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        assert inputs is None or len(inputs) == 0, "Constant op cannot accept any input tensors"
        return self.value.shape

    @override
    def fold(self, inputs: list[LiteralValue]) -> LiteralValue:
        assert not inputs, "Constant op cannot accept any inputs"
        return self.value

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return []  # Constant nodes do not have gradients

    @override
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 0, "Constant lowering cannot accept input tensors"
        return (
            builder.declare_constant(
                self.value.shape, "float32", _dense_const(self.value.tolist()), hint="const"
            ),
        )


class Input(TensorOp):
    """
    Input denotes an input to the computational graph.

    It cannot accept any Tensors or gradients, and is just a leaf node placeholder.
    """

    shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    @override
    def __call__(
        self,
        inputs: tuple[Tensor, ...],
        label: str | None = None,
    ) -> Tensor:
        assert inputs is None or len(inputs) == 0, "Input op cannot accept any input tensors"
        return Tensor(
            inputs=tuple(),
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        assert inputs is None or len(inputs) == 0, "Input op cannot accept any input tensors"
        return self.shape

    @override
    def fold(self, inputs: list[LiteralValue]) -> LiteralValue:
        raise RuntimeError("Input nodes cannot be folded")

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return []  # Input nodes do not have gradients

    @override
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 0, "Input lowering cannot accept input tensors"
        return (builder.declare_input(self.shape, "float32", hint="input"),)


def constant(value: LiteralData | LiteralValue, label: str | None = None) -> Tensor:
    """
    Create a constant Tensor with the given value and an optional label.
    This is used to denote constant values in the computational graph that are not supplied at
    evaluation time, but rather determined at graph construction time.
    """
    return Constant(value)(inputs=tuple(), label=label)


def zeros(shape: tuple[int, ...], label: str | None = None) -> Tensor:
    """Create a constant tensor filled with zeros."""
    from tmlc.tensor.ops.ops_shape import Fill

    return Fill(shape=shape, value=0.0)(inputs=tuple(), label=label)


def ones(shape: tuple[int, ...], label: str | None = None) -> Tensor:
    """Create a constant tensor filled with ones."""
    from tmlc.tensor.ops.ops_shape import Fill

    return Fill(shape=shape, value=1.0)(inputs=tuple(), label=label)


def input(shape: tuple[int, ...], label: str | None = None) -> Tensor:
    """
    Create an input Tensor with an optional label.
    This is used to denote the inputs to the computational graph.
    All input nodes must be assigned a
    value at evaluation time.
    """
    return Input(shape=shape)(inputs=tuple(), label=label)
