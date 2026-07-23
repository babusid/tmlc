from __future__ import annotations

import numpy as np
from tmlc.ndarray import ndarray
from typing_extensions import override
from tmlc.tensor.tensor import Tensor, TensorOp
from tmlc.compute.compute import Combiner, ComputeProgramBuilder, ComputeTensor
from tmlc.compute.index import AxisRef
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
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 2, "Add lowering requires exactly 2 input tensors"
        lhs, rhs = inputs
        assert lhs.shape == rhs.shape, "Add lowering requires equal input shapes"
        domain = tuple(builder.spatial(extent, "add") for extent in lhs.shape)
        index = tuple(AxisRef(axis) for axis in domain)
        body = lhs[index] + rhs[index]
        return (builder.compute(domain, body, dtype=lhs.dtype, hint="add"),)


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
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 2, "Mul lowering requires exactly 2 input tensors"
        lhs, rhs = inputs
        assert lhs.shape == rhs.shape, "Mul lowering requires equal input shapes"
        domain = tuple(builder.spatial(extent, "mul") for extent in lhs.shape)
        index = tuple(AxisRef(axis) for axis in domain)
        body = lhs[index] * rhs[index]
        return (builder.compute(domain, body, dtype=lhs.dtype, hint="mul"),)


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
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 2, "Div lowering requires exactly 2 input tensors"
        lhs, rhs = inputs
        assert lhs.shape == rhs.shape, "Div lowering requires equal input shapes"
        domain = tuple(builder.spatial(extent, "div") for extent in lhs.shape)
        index = tuple(AxisRef(axis) for axis in domain)
        body = lhs[index] / rhs[index]
        return (builder.compute(domain, body, dtype=lhs.dtype, hint="div"),)


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
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 2, "Matmul lowering requires exactly 2 input tensors"
        lhs, rhs = inputs
        assert len(lhs.shape) == 2 and len(rhs.shape) == 2, (
            "Matmul lowering requires 2D input tensors"
        )
        assert lhs.shape[1] == rhs.shape[0], "Matmul lowering input shapes are incompatible"
        i = builder.spatial(lhs.shape[0], "matmul_i")
        j = builder.spatial(rhs.shape[1], "matmul_j")
        k = builder.reduce(lhs.shape[1], "matmul_k")
        body = lhs[AxisRef(i), AxisRef(k)] * rhs[AxisRef(k), AxisRef(j)]
        return (
            builder.compute((i, j, k), body, combiner=Combiner.SUM, dtype=lhs.dtype, hint="matmul"),
        )


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
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 1, "Negate lowering requires exactly 1 input tensor"
        (source,) = inputs
        domain = tuple(builder.spatial(extent, "negate") for extent in source.shape)
        index = tuple(AxisRef(axis) for axis in domain)
        return (builder.compute(domain, -source[index], dtype=source.dtype, hint="negate"),)


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
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 2, "Pow lowering requires exactly 2 input tensors"
        lhs, rhs = inputs
        assert lhs.shape == rhs.shape, "Pow lowering requires equal input shapes"
        domain = tuple(builder.spatial(extent, "pow") for extent in lhs.shape)
        index = tuple(AxisRef(axis) for axis in domain)
        body = lhs[index] ** rhs[index]
        return (builder.compute(domain, body, dtype=lhs.dtype, hint="pow"),)


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
