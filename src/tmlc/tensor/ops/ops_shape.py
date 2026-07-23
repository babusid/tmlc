from __future__ import annotations

import numpy as np
from math import prod
from tmlc.ndarray import ndarray
from typing_extensions import override

from tmlc.tensor.tensor import Tensor, TensorOp
from tmlc.compute.compute import Combiner, ComputeProgramBuilder, ComputeTensor
from tmlc.compute.index import AxisRef, IndexExpr, IntConst
from tmlc.compute.scalar import ScalarConst


def normalize_axes(
    axes: tuple[int, ...] | int | None, shape: tuple[int, ...]
) -> tuple[int, ...] | None:
    if axes is None:
        return None
    if isinstance(axes, int):
        axes = (axes,)

    normalized: list[int] = []
    for axis in axes:
        if axis < 0:
            axis += len(shape)
        assert 0 <= axis < len(shape), "Axis is out of bounds"
        normalized.append(axis)
    assert len(set(normalized)) == len(normalized), "Axes must be unique"
    return tuple(sorted(normalized))


def _assert_broadcastable(input_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> None:
    assert len(input_shape) <= len(target_shape), "Cannot broadcast to a lower-rank shape"
    padded_shape = (1,) * (len(target_shape) - len(input_shape)) + input_shape
    for input_dim, target_dim in zip(padded_shape, target_shape):
        assert input_dim == 1 or input_dim == target_dim, "Input shape is not broadcastable"


class Transpose(TensorOp):
    axes: tuple[int, int] | None

    def __init__(self, axes: tuple[int, int] | None = None):
        self.axes = axes

    def _permutation(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        assert len(shape) >= 2, "Transpose op requires at least 2 dimensions"
        axes = self.axes if self.axes is not None else (-2, -1)
        a1, a2 = axes
        ndim = len(shape)
        if a1 < 0:
            a1 += ndim
        if a2 < 0:
            a2 += ndim
        assert 0 <= a1 < ndim and 0 <= a2 < ndim, "Transpose axes are out of bounds"

        permutation = list(range(ndim))
        permutation[a1], permutation[a2] = permutation[a2], permutation[a1]
        return tuple(permutation)

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
        assert len(inputs) == 1, "Transpose op requires exactly 1 input tensor"
        permutation = self._permutation(shape=inputs[0].shape)
        return tuple(inputs[0].shape[axis] for axis in permutation)

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 1, "Transpose op requires exactly 1 input tensor"
        return np.transpose(inputs[0], axes=self._permutation(shape=inputs[0].shape))

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [transpose(incoming_grad, axes=self.axes)]

    @override
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 1, "Transpose lowering requires exactly 1 input tensor"
        source = inputs[0]
        permutation = self._permutation(source.shape)
        output_shape = tuple(source.shape[dim] for dim in permutation)
        domain = tuple(builder.spatial(extent, "transpose") for extent in output_shape)
        input_index = tuple(AxisRef(domain[permutation.index(dim)]) for dim in range(len(domain)))
        return (builder.compute(domain, source[input_index], dtype=source.dtype, hint="transpose"),)


class Summation(TensorOp):
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
        assert len(inputs) == 1, "Summation op requires exactly 1 input tensor"
        axes = normalize_axes(axes=self.axes, shape=inputs[0].shape)
        if axes is None:
            return ()
        return tuple(dim for axis, dim in enumerate(inputs[0].shape) if axis not in axes)

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 1, "Summation op requires exactly 1 input tensor"
        return np.asarray(np.sum(inputs[0], axis=self.axes))

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        input_shape = tensor.inputs[0].shape
        axes = normalize_axes(axes=self.axes, shape=input_shape)
        if axes is None:
            reshaped_grad = reshape(incoming_grad, shape=(1,) * len(input_shape))
        else:
            grad_shape = tuple(1 if axis in axes else dim for axis, dim in enumerate(input_shape))
            reshaped_grad = reshape(incoming_grad, shape=grad_shape)
        return [broadcast_to(reshaped_grad, shape=input_shape)]

    @override
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 1, "Summation lowering requires exactly 1 input tensor"
        source = inputs[0]
        normalized = normalize_axes(self.axes, source.shape)
        reduced_axes = set(range(len(source.shape))) if normalized is None else set(normalized)
        domain = tuple(
            builder.reduce(extent, "sum_reduce")
            if dim in reduced_axes
            else builder.spatial(extent, "sum_spatial")
            for dim, extent in enumerate(source.shape)
        )
        index = tuple(AxisRef(axis) for axis in domain)
        return (
            builder.compute(
                domain,
                source[index],
                combiner=Combiner.SUM if reduced_axes else None,
                dtype=source.dtype,
                hint="sum",
            ),
        )


class Fill(TensorOp):
    """
    Nullary op producing a constant array of `shape`/`dtype` filled with `value`.

    Unlike `Constant`, the array is not baked in at trace time: it carries no input edges and
    is materialized lazily in `compute()`. This keeps `zeros_like`/`ones_like` cheap to trace
    (no eager allocation, no spurious dependency on the tensor they mirror) and gives the
    eventual MLIR lowering a splat constant to target instead of a literal buffer.
    """

    shape: tuple[int, ...]
    value: float
    dtype: str

    def __init__(self, shape: tuple[int, ...], value: float, dtype: str = "float32"):
        self.shape = shape
        self.value = value
        self.dtype = dtype

    @override
    def __call__(
        self,
        inputs: tuple[Tensor, ...],
        label: str | None = None,
    ) -> Tensor:
        assert inputs is None or len(inputs) == 0, "Fill op cannot accept any input tensors"
        return Tensor(
            inputs=tuple(),
            op=self,
            shape=self.shape,
            dtype=self.dtype,
            label=label,
        )

    @override
    def infer_shape(self, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        assert inputs is None or len(inputs) == 0, "Fill op cannot accept any input tensors"
        return self.shape

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert inputs is None or len(inputs) == 0, "Fill op cannot accept any input tensors"
        return np.full(self.shape, self.value, dtype=self.dtype)

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return []  # Fill nodes are nullary: nothing to propagate to

    @override
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 0, "Fill lowering cannot accept input tensors"
        domain = tuple(builder.spatial(extent, "fill") for extent in self.shape)
        return (
            builder.compute(domain, ScalarConst(float(self.value)), dtype=self.dtype, hint="fill"),
        )


class Reshape(TensorOp):
    shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

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
        assert len(inputs) == 1, "Reshape op requires exactly 1 input tensor"
        assert np.prod(inputs[0].shape) == np.prod(self.shape), "Reshape cannot change tensor size"
        return self.shape

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 1, "Reshape op requires exactly 1 input tensor"
        assert np.prod(inputs[0].shape) == np.prod(self.shape), "Reshape cannot change tensor size"
        return np.reshape(inputs[0], self.shape)

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        return [reshape(incoming_grad, shape=tensor.inputs[0].shape)]

    @override
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 1, "Reshape lowering requires exactly 1 input tensor"
        source = inputs[0]
        assert prod(source.shape) == prod(self.shape), "Reshape lowering cannot change tensor size"
        domain = tuple(builder.spatial(extent, "reshape") for extent in self.shape)

        linear: IndexExpr = IntConst(0)
        for dim, axis in enumerate(domain):
            stride = prod(self.shape[dim + 1 :])
            term: IndexExpr = AxisRef(axis)
            if stride != 1:
                term = term * stride
            linear = linear + term

        input_index: list[IndexExpr] = []
        for dim, extent in enumerate(source.shape):
            if extent == 1:
                input_index.append(IntConst(0))
                continue
            stride = prod(source.shape[dim + 1 :])
            coordinate = linear if stride == 1 else linear // stride
            input_index.append(coordinate % extent)

        return (
            builder.compute(domain, source[tuple(input_index)], dtype=source.dtype, hint="reshape"),
        )


class BroadcastTo(TensorOp):
    shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

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
        assert len(inputs) == 1, "BroadcastTo op requires exactly 1 input tensor"
        _assert_broadcastable(input_shape=inputs[0].shape, target_shape=self.shape)
        return self.shape

    @override
    def compute(self, inputs: list[ndarray]) -> ndarray:
        assert len(inputs) == 1, "BroadcastTo op requires exactly 1 input tensor"
        _assert_broadcastable(input_shape=inputs[0].shape, target_shape=self.shape)
        return np.broadcast_to(inputs[0], self.shape)

    @override
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        input_shape = tensor.inputs[0].shape
        padded_shape = (1,) * (len(tensor.shape) - len(input_shape)) + input_shape
        axes = tuple(axis for axis, dim in enumerate(padded_shape) if dim == 1)
        grad = summation(incoming_grad, axes=axes) if axes else incoming_grad
        return [reshape(grad, shape=input_shape)]

    @override
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        assert len(inputs) == 1, "BroadcastTo lowering requires exactly 1 input tensor"
        source = inputs[0]
        _assert_broadcastable(source.shape, self.shape)
        domain = tuple(builder.spatial(extent, "broadcast") for extent in self.shape)
        offset = len(self.shape) - len(source.shape)
        input_index = tuple(
            IntConst(0) if extent == 1 else AxisRef(domain[offset + dim])
            for dim, extent in enumerate(source.shape)
        )
        return (builder.compute(domain, source[input_index], dtype=source.dtype, hint="broadcast"),)


def transpose(
    t: Tensor,
    axes: tuple[int, int] | None = None,
    label: str | None = None,
) -> Tensor:
    return Transpose(axes=axes)(inputs=(t,), label=label)


def reshape(t: Tensor, shape: tuple[int, ...], label: str | None = None) -> Tensor:
    return Reshape(shape=shape)(inputs=(t,), label=label)


def broadcast_to(t: Tensor, shape: tuple[int, ...], label: str | None = None) -> Tensor:
    return BroadcastTo(shape=shape)(inputs=(t,), label=label)


def summation(
    t: Tensor,
    axes: tuple[int, ...] | int | None = None,
    label: str | None = None,
) -> Tensor:
    return Summation(axes=axes)(inputs=(t,), label=label)


def zeros_like(t: Tensor, label: str | None = None) -> Tensor:
    return Fill(shape=t.shape, value=0.0, dtype=t.dtype)(inputs=tuple(), label=label)


def ones_like(t: Tensor, label: str | None = None) -> Tensor:
    return Fill(shape=t.shape, value=1.0, dtype=t.dtype)(inputs=tuple(), label=label)
