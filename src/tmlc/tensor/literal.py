from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import product

type LiteralScalar = float | int
type LiteralData = LiteralScalar | tuple


@dataclass(frozen=True)
class LiteralValue:
    data: LiteralData
    shape: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", self._infer_shape(self.data))

    @classmethod
    def _infer_shape(cls, data: LiteralData) -> tuple[int, ...]:
        if not isinstance(data, tuple):
            assert isinstance(data, (float, int))
            return ()
        if not data:
            return (0,)

        element_shape = cls._infer_shape(data[0])
        assert all(cls._infer_shape(element) == element_shape for element in data), (
            "Constant values must have a rectangular shape"
        )
        return (len(data), *element_shape)

    @classmethod
    def _build(
        cls,
        shape: tuple[int, ...],
        value_at: Callable[[tuple[int, ...]], LiteralScalar],
        prefix: tuple[int, ...] = (),
    ) -> LiteralData:
        if not shape:
            return value_at(prefix)
        return tuple(cls._build(shape[1:], value_at, (*prefix, index)) for index in range(shape[0]))

    def __getitem__(self, index: tuple[int, ...]) -> LiteralScalar:
        current = self.data
        for axis_index in index:
            assert isinstance(current, tuple)
            current = current[axis_index]
        assert isinstance(current, (float, int))
        return current

    def __str__(self) -> str:
        return str(self.data)

    def _binary(
        self,
        other: LiteralValue,
        function: Callable[[LiteralScalar, LiteralScalar], LiteralScalar],
    ) -> LiteralValue:
        assert self.shape == other.shape
        return LiteralValue(
            self._build(self.shape, lambda index: function(self[index], other[index]))
        )

    def __add__(self, other: LiteralValue) -> LiteralValue:
        return self._binary(other, lambda lhs, rhs: lhs + rhs)

    def __mul__(self, other: LiteralValue) -> LiteralValue:
        return self._binary(other, lambda lhs, rhs: lhs * rhs)

    def __truediv__(self, other: LiteralValue) -> LiteralValue:
        return self._binary(other, lambda lhs, rhs: lhs / rhs)

    def __pow__(self, other: LiteralValue) -> LiteralValue:
        return self._binary(other, lambda lhs, rhs: lhs**rhs)

    def __neg__(self) -> LiteralValue:
        return self.map(lambda value: -value)

    def __matmul__(self, other: LiteralValue) -> LiteralValue:
        assert len(self.shape) == 2 and len(other.shape) == 2
        assert self.shape[1] == other.shape[0]
        data = self._build(
            (self.shape[0], other.shape[1]),
            lambda index: sum(
                self[(index[0], inner)] * other[(inner, index[1])] for inner in range(self.shape[1])
            ),
        )
        return LiteralValue(data)

    def map(self, function: Callable[[LiteralScalar], LiteralScalar]) -> LiteralValue:
        return LiteralValue(self._build(self.shape, lambda index: function(self[index])))

    def _flatten(self) -> Iterable[LiteralScalar]:
        def flatten(data: LiteralData) -> Iterable[LiteralScalar]:
            if isinstance(data, tuple):
                for element in data:
                    yield from flatten(element)
            else:
                yield data

        return flatten(self.data)

    def tolist(self) -> LiteralScalar | list:
        def build(data: LiteralData) -> LiteralScalar | list:
            if isinstance(data, tuple):
                return [build(element) for element in data]
            return data

        return build(self.data)

    def reshape(self, shape: tuple[int, ...]) -> LiteralValue:
        flattened = iter(self._flatten())
        data = self._build(shape, lambda _: next(flattened))
        try:
            next(flattened)
        except StopIteration:
            return LiteralValue(data)
        raise AssertionError("Reshape cannot change tensor size")

    def broadcast_to(self, shape: tuple[int, ...]) -> LiteralValue:
        rank_offset = len(shape) - len(self.shape)
        assert rank_offset >= 0

        def value_at(index: tuple[int, ...]) -> LiteralScalar:
            input_index = tuple(
                0 if dimension == 1 else index[rank_offset + axis]
                for axis, dimension in enumerate(self.shape)
            )
            return self[input_index]

        return LiteralValue(self._build(shape, value_at))

    def transpose(self, permutation: tuple[int, ...]) -> LiteralValue:
        output_shape = tuple(self.shape[axis] for axis in permutation)

        def value_at(index: tuple[int, ...]) -> LiteralScalar:
            input_index = [0] * len(permutation)
            for output_axis, input_axis in enumerate(permutation):
                input_index[input_axis] = index[output_axis]
            return self[tuple(input_index)]

        return LiteralValue(self._build(output_shape, value_at))

    def reduce(
        self,
        axes: tuple[int, ...] | None,
        reducer: Callable[[Iterable[LiteralScalar]], LiteralScalar],
    ) -> LiteralValue:
        reduced_axes = tuple(range(len(self.shape))) if axes is None else axes
        reduced_axis_set = set(reduced_axes)
        retained_axes = tuple(
            axis for axis in range(len(self.shape)) if axis not in reduced_axis_set
        )
        output_shape = tuple(self.shape[axis] for axis in retained_axes)

        def value_at(output_index: tuple[int, ...]) -> LiteralScalar:
            input_index = [0] * len(self.shape)
            for axis, index in zip(retained_axes, output_index):
                input_index[axis] = index

            def values() -> Iterable[LiteralScalar]:
                ranges = (range(self.shape[axis]) for axis in reduced_axes)
                for reduced_index in product(*ranges):
                    for axis, index in zip(reduced_axes, reduced_index):
                        input_index[axis] = index
                    yield self[tuple(input_index)]

            return reducer(values())

        return LiteralValue(self._build(output_shape, value_at))
