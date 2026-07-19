from __future__ import annotations

import numpy as np
from tmlc.ndarray import ndarray
from typing_extensions import override
from tmlc.tensor.tensor import Tensor, TensorOp
from tmlc.tensor.ops.ops_shape import broadcast_to
from tmlc.tensor.ops.ops_logarithmic import log
from tmlc.tensor.traits import commutative


def _broadcast_shape(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> tuple[int, ...]:
    rank = max(len(shape1), len(shape2))
    padded_shape1 = (1,) * (rank - len(shape1)) + shape1
    padded_shape2 = (1,) * (rank - len(shape2)) + shape2

    output_shape: list[int] = []
    for dim1, dim2 in zip(padded_shape1, padded_shape2):
        assert dim1 == dim2 or dim1 == 1 or dim2 == 1, "Input shapes are not broadcastable"
        output_shape.append(max(dim1, dim2))
    return tuple(output_shape)


def _broadcast_pair(t1: Tensor, t2: Tensor) -> tuple[Tensor, Tensor]:
    shape = _broadcast_shape(shape1=t1.shape, shape2=t2.shape)
    if t1.shape != shape:
        t1 = broadcast_to(t1, shape=shape)
    if t2.shape != shape:
        t2 = broadcast_to(t2, shape=shape)
    return t1, t2


@commutative
class Add(TensorOp):
    @override
    def __call__(
        self,
        inputs: tuple[Tensor, ...],
        label: str | None = None,
    ) -> Tensor:
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        assert len(inputs) == 2, "Add op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, "Add op requires tensors to have the same shape"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 2, "Add op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, "Add op requires tensors to have the same shape"
        return np.asarray(inputs[0] + inputs[1])

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad, incoming_grad]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


@commutative
class Mul(TensorOp):
    @override
    def __call__(
        self,
        inputs: tuple[Tensor, ...],
        label: str | None = None,
    ) -> Tensor:
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        assert len(inputs) == 2, "Mul op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, "Mul op requires tensors to have the same shape"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 2, "Mul op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, "Mul op requires tensors to have the same shape"
        return np.asarray(inputs[0] * inputs[1])

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [tensor.inputs[1] * incoming_grad, tensor.inputs[0] * incoming_grad]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Div(TensorOp):
    @override
    def __call__(
        self,
        inputs: tuple[Tensor, ...],
        label: str | None = None,
    ) -> Tensor:
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        assert len(inputs) == 2, "Div op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, "Div op requires tensors to have the same shape"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 2, "Div op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, "Div op requires tensors to have the same shape"
        return np.asarray(inputs[0] / inputs[1])

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [
            incoming_grad / tensor.inputs[1],
            (incoming_grad * -1 * tensor.inputs[0]) / (tensor.inputs[1] * tensor.inputs[1]),
        ]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Matmul(TensorOp):
    @override
    def __call__(
        self,
        inputs: tuple[Tensor, ...],
        label: str | None = None,
    ) -> Tensor:
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        assert len(inputs) == 2, "Matmul op requires exactly 2 input tensors"
        assert len(inputs[0].shape) == 2 and len(inputs[1].shape) == 2, (
            "Matmul op requires 2D input tensors"
        )
        assert inputs[0].shape[1] == inputs[1].shape[0], "Matmul input shapes are incompatible"
        return (inputs[0].shape[0], inputs[1].shape[1])

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 2, "Matmul op requires exactly 2 input tensors"
        assert inputs[0].ndim == 2 and inputs[1].ndim == 2, "Matmul op requires 2D input tensors"
        return inputs[0] @ inputs[1]

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad @ tensor.inputs[1].T, tensor.inputs[0].T @ incoming_grad]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Negate(TensorOp):
    @override
    def __call__(
        self,
        inputs: tuple[Tensor, ...],
        label: str | None = None,
    ) -> Tensor:
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        assert len(inputs) == 1, "Negate op requires exactly 1 input tensor"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 1, "Negate op requires exactly 1 input tensor"
        return np.asarray(-inputs[0])

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [negate(incoming_grad)]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Pow(TensorOp):
    @override
    def __call__(
        self,
        inputs: tuple[Tensor, ...],
        label: str | None = None,
    ) -> Tensor:
        return Tensor(
            inputs=inputs,
            op=self,
            shape=self.infer_shape(inputs=inputs),
            label=label,
        )

    @override
    def infer_shape(self, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        assert len(inputs) == 2, "Power op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, (
            "Power op requires tensors to have the same shape"
        )
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 2, "Power op requires exactly 2 input tensors"
        assert inputs[0].shape == inputs[1].shape, (
            "Power op requires tensors to have the same shape"
        )
        return np.asarray(inputs[0] ** inputs[1])

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        lhs, rhs = tensor.inputs
        return [
            incoming_grad * rhs * power(lhs, rhs - 1),
            incoming_grad * tensor * log(lhs),
        ]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


def add(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    t1, t2 = _broadcast_pair(t1=t1, t2=t2)
    return Add()(inputs=(t1, t2), label=label)


def mul(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    t1, t2 = _broadcast_pair(t1=t1, t2=t2)
    return Mul()(inputs=(t1, t2), label=label)


def div(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    t1, t2 = _broadcast_pair(t1=t1, t2=t2)
    return Div()(inputs=(t1, t2), label=label)


def mm(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    return Matmul()(inputs=(t1, t2), label=label)


def power(t1: Tensor, t2: Tensor, label: str | None = None) -> Tensor:
    t1, t2 = _broadcast_pair(t1=t1, t2=t2)
    return Pow()(inputs=(t1, t2), label=label)


def negate(t: Tensor, label: str | None = None) -> Tensor:
    return Negate()(inputs=(t,), label=label)
