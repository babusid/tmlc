from __future__ import annotations

from collections.abc import Iterable
from math import exp as scalar_exp
from math import log as scalar_log
from math import tanh as scalar_tanh

from typing_extensions import override
from tmlc.tensor.literal import (
    LiteralScalar,
    LiteralValue,
)
from tmlc.tensor.tensor import Tensor, TensorOp
from tmlc.compute import Combiner, ComputeProgramBuilder, ComputeTensor
from tmlc.compute.index import AxisRef
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
    def fold(self, inputs: list[LiteralValue]) -> LiteralValue:
        assert len(inputs) == 1, "Exp op requires exactly 1 input"
        return inputs[0].map(scalar_exp)

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad * tensor]

    @override
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 1, "Exp lowering requires exactly 1 input tensor"
        (source,) = inputs
        output_axes = tuple(builder.spatial(extent, "exp") for extent in source.shape)
        index = tuple(AxisRef(axis) for axis in output_axes)
        return (
            builder.compute(
                output_axes=output_axes, body=source[index].exp(), dtype=source.dtype, hint="exp"
            ),
        )


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
    def fold(self, inputs: list[LiteralValue]) -> LiteralValue:
        assert len(inputs) == 1, "Log op requires exactly 1 input"
        return inputs[0].map(scalar_log)

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad / tensor.inputs[0]]

    @override
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 1, "Log lowering requires exactly 1 input tensor"
        (source,) = inputs
        output_axes = tuple(builder.spatial(extent, "log") for extent in source.shape)
        index = tuple(AxisRef(axis) for axis in output_axes)
        return (
            builder.compute(
                output_axes=output_axes, body=source[index].log(), dtype=source.dtype, hint="log"
            ),
        )


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
    def fold(self, inputs: list[LiteralValue]) -> LiteralValue:
        assert len(inputs) == 1, "Tanh op requires exactly 1 input"
        return inputs[0].map(scalar_tanh)

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [incoming_grad * (1 - tensor * tensor)]

    @override
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 1, "Tanh lowering requires exactly 1 input tensor"
        (source,) = inputs
        output_axes = tuple(builder.spatial(extent, "tanh") for extent in source.shape)
        index = tuple(AxisRef(axis) for axis in output_axes)
        return (
            builder.compute(
                output_axes=output_axes, body=source[index].tanh(), dtype=source.dtype, hint="tanh"
            ),
        )


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
    def fold(self, inputs: list[LiteralValue]) -> LiteralValue:
        assert len(inputs) == 1, "LogSumExp op requires exactly 1 input"

        def logsumexp(values: Iterable[LiteralScalar]) -> LiteralScalar:
            concrete_values = tuple(values)
            max_value = max(concrete_values)
            shifted_sum = sum(scalar_exp(value - max_value) for value in concrete_values)
            return scalar_log(shifted_sum) + max_value

        axes = normalize_axes(self.axes, shape=inputs[0].shape)
        return inputs[0].reduce(axes, logsumexp)

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
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 1, "LogSumExp lowering requires exactly 1 input tensor"
        source = inputs[0]
        normalized = normalize_axes(self.axes, source.shape)
        reduced_axes = set(range(len(source.shape))) if normalized is None else set(normalized)

        max_axes = tuple(
            builder.reduce(extent, "logsumexp_max_reduce")
            if dim in reduced_axes
            else builder.spatial(extent, "logsumexp_max_spatial")
            for dim, extent in enumerate(source.shape)
        )
        max_output_axes = tuple(
            axis for dim, axis in enumerate(max_axes) if dim not in reduced_axes
        )
        max_reduction_axes = tuple(axis for dim, axis in enumerate(max_axes) if dim in reduced_axes)
        max_refs = tuple(AxisRef(axis) for axis in max_axes)
        maximum = builder.compute(
            output_axes=max_output_axes,
            body=source[max_refs],
            reduce_axes=max_reduction_axes,
            combiner=Combiner.MAX if max_reduction_axes else None,
            dtype=source.dtype,
            hint="logsumexp_max",
        )

        sum_axes = tuple(
            builder.reduce(extent, "logsumexp_sum_reduce")
            if dim in reduced_axes
            else builder.spatial(extent, "logsumexp_sum_spatial")
            for dim, extent in enumerate(source.shape)
        )
        sum_output_axes = tuple(
            axis for dim, axis in enumerate(sum_axes) if dim not in reduced_axes
        )
        sum_reduction_axes = tuple(axis for dim, axis in enumerate(sum_axes) if dim in reduced_axes)
        sum_refs = tuple(AxisRef(axis) for axis in sum_axes)
        retained_refs = tuple(AxisRef(axis) for axis in sum_output_axes)
        exponentiated = (source[sum_refs] - maximum[retained_refs]).exp()
        summed = builder.compute(
            output_axes=sum_output_axes,
            body=exponentiated,
            reduce_axes=sum_reduction_axes,
            combiner=Combiner.SUM if sum_reduction_axes else None,
            dtype=source.dtype,
            hint="logsumexp_sum",
        )

        output_axes = tuple(builder.spatial(extent, "logsumexp") for extent in maximum.shape)
        final_refs = tuple(AxisRef(axis) for axis in output_axes)
        result = summed[final_refs].log() + maximum[final_refs]
        return (
            builder.compute(
                output_axes=output_axes, body=result, dtype=source.dtype, hint="logsumexp"
            ),
        )


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
