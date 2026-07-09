from __future__ import annotations

import numpy as np
from numpy import ndarray
from typing_extensions import override
from tmlc.tensor.tensor import Tensor, TensorOp
from tmlc.tensor.ops.ops_shape import normalize_axes, broadcast_to, reshape


class Exp(TensorOp):
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
        assert len(inputs) == 1, "Exp op requires exactly 1 input tensor"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 1, "Exp op requires exactly 1 input tensor"
        return np.asarray(np.exp(inputs[0]))

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad * tensor]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Log(TensorOp):
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
        assert len(inputs) == 1, "Log op requires exactly 1 input tensor"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 1, "Log op requires exactly 1 input tensor"
        return np.asarray(np.log(inputs[0]))

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad / tensor.inputs[0]]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class Tanh(TensorOp):
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
        assert len(inputs) == 1, "Tanh op requires exactly 1 input tensor"
        return inputs[0].shape

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 1, "Tanh op requires exactly 1 input tensor"
        return np.asarray(np.tanh(inputs[0]))

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad * (1 - tensor * tensor)]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


class LogSumExp(TensorOp):
    axes: tuple[int, ...] | None

    def __init__(self, axes: tuple[int, ...] | int | None = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

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
        assert len(inputs) == 1, "LogSumExp op requires exactly 1 input tensor"
        axes = normalize_axes(axes=self.axes, shape=inputs[0].shape)
        if axes is None:
            return ()
        return tuple(dim for axis, dim in enumerate(inputs[0].shape) if axis not in axes)

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 1, "LogSumExp op requires exactly 1 input tensor"
        max_value = np.max(inputs[0], axis=self.axes, keepdims=True)
        shifted = inputs[0] - max_value
        sum_exp = np.sum(np.exp(shifted), axis=self.axes)
        return np.asarray(np.log(sum_exp) + np.reshape(max_value, sum_exp.shape))

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        input_tensor = tensor.inputs[0]
        axes = normalize_axes(axes=self.axes, shape=input_tensor.shape)
        if axes is None:
            reduced_shape = (1,) * len(input_tensor.shape)
        else:
            reduced_shape = tuple(
                1 if axis in axes else dim for axis, dim in enumerate(input_tensor.shape)
            )

        broadcast_grad = broadcast_to(
            reshape(incoming_grad, shape=reduced_shape), shape=input_tensor.shape
        )
        broadcast_output = broadcast_to(
            reshape(tensor, shape=reduced_shape), shape=input_tensor.shape
        )
        softmax = exp(input_tensor - broadcast_output)
        return [broadcast_grad * softmax]

    @override
    def emit_ir(self, inputs: list[str]) -> str:
        return ""


def exp(t: Tensor, label: str | None = None) -> Tensor:
    return Exp()(inputs=(t,), label=label)


def log(t: Tensor, label: str | None = None) -> Tensor:
    return Log()(inputs=(t,), label=label)


def tanh(t: Tensor, label: str | None = None) -> Tensor:
    return Tanh()(inputs=(t,), label=label)


def logsumexp(
    t: Tensor,
    axes: tuple[int, ...] | int | None = None,
    label: str | None = None,
) -> Tensor:
    return LogSumExp(axes=axes)(inputs=(t,), label=label)
