from __future__ import annotations

from abc import ABC, abstractmethod
from tmlc.compute import ComputeProgramBuilder, ComputeTensor
from typing_extensions import override

from tmlc.tensor.literal import LiteralValue


def _ensure_tensor(value: Tensor | float | int) -> Tensor:
    """Wrap a Python scalar as a constant Tensor; pass a Tensor through unchanged."""
    if isinstance(value, (int, float)):
        from tmlc.tensor.ops.ops_basic import constant

        return constant(value)
    return value


class Tensor:
    """
    A Tensor is a node in a computational graph, representing a multi-dimensional array.
    Tensors are the inputs to tensor operations, which output new tensors. A sequence of Tensor
    Operations chained together produces a computational graph, which we can compile and optimize.
    Tensors do NOT actually hold data themseleves, but rather represent the flow of data through
    the graph. The actual data is held in buffers that are supplied at evaluation time.
    """

    inputs: tuple[Tensor, ...]
    op: TensorOp
    label: str
    shape: tuple[int, ...]
    dtype: str

    def __init__(
        self,
        inputs: tuple[Tensor, ...],
        op: TensorOp,
        shape: tuple[int, ...],
        label: str | None = None,
        dtype: str = "float32",
    ):
        self.inputs = inputs
        self.op = op
        if label is None:
            self.label = self.op.__class__.__name__
        else:
            self.label = label

        self.shape = shape

        # TODO: support more dtypes
        # TODO: inheirit dtypes from input tensors
        # TODO: dtype promotion logic for mismatched input tensor
        self.dtype = "float32"

    @override
    def __str__(self):
        inputs = ", ".join(str(tensor) for tensor in self.inputs)
        return (
            f"Tensor(inputs=[{inputs}],"
            + f"op={self.op},"
            + f"shape={self.shape},"
            + f"label={self.label},"
            + f"dtype={self.dtype}"
        )

    @override
    def __repr__(self):
        return self.__str__()

    def __add__(self, other: Tensor | float | int) -> Tensor:
        from tmlc.tensor.ops.ops_arithmetic import add

        return add(self, _ensure_tensor(other))

    def __radd__(self, other: Tensor | float | int) -> Tensor:
        from tmlc.tensor.ops.ops_arithmetic import add

        return add(self, _ensure_tensor(other))

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        from tmlc.tensor.ops.ops_arithmetic import mul

        return mul(self, _ensure_tensor(other))

    def __rmul__(self, other: Tensor | float | int) -> Tensor:
        from tmlc.tensor.ops.ops_arithmetic import mul

        return mul(self, _ensure_tensor(other))

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        from tmlc.tensor.ops.ops_arithmetic import div

        return div(self, _ensure_tensor(other))

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        from tmlc.tensor.ops.ops_arithmetic import add, negate

        return add(self, negate(_ensure_tensor(other)))

    def __rsub__(self, other: Tensor | float | int) -> Tensor:
        from tmlc.tensor.ops.ops_arithmetic import add, negate

        return add(_ensure_tensor(other), negate(self))

    def __neg__(self) -> Tensor:
        from tmlc.tensor.ops.ops_arithmetic import negate

        return negate(self)

    def __pow__(self, other: Tensor | float | int) -> Tensor:
        from tmlc.tensor.ops.ops_arithmetic import power

        return power(self, _ensure_tensor(other))

    def __matmul__(self, other: Tensor) -> Tensor:
        from tmlc.tensor.ops.ops_arithmetic import mm

        return mm(self, other)

    @property
    def T(self) -> Tensor:
        from tmlc.tensor.ops.ops_shape import transpose

        return transpose(self)


class TensorOp(ABC):
    """TensorOp interface represents an operation that can be performed on Tensors."""

    @override
    def __str__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def __call__(
        self,
        inputs: tuple[Tensor, ...],
        label: str | None = None,
    ) -> Tensor:
        """
        When a TensorOp is called, it should create a new Tensor that represents the output of
        this operation.
        """
        raise NotImplementedError("TensorOp subclasses must implement __call__")

    @abstractmethod
    def infer_shape(self, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        """
        Given the input tensors (which contain their shapes), infer the shape of the
        output tensor that this operation will produce.
        """
        raise NotImplementedError("TensorOp subclasses must implement infer_shape()")

    @abstractmethod
    def fold(self, inputs: list[LiteralValue]) -> LiteralValue:
        """
        Evaluate this operation over constant inputs during compile-time constant folding.

        This method defines how constants propagate through the operation. It is part of the
        TensorOp IR contract, not a runtime execution API: graph interpreters and compiled runtimes
        provide their own operation implementations.
        """
        raise NotImplementedError("TensorOp subclasses must implement fold()")

    @abstractmethod
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        """
        Given the output of the forward `call` method and the incoming gradient from the
        backwards pass, this method calculates the gradients to propagate to the inputs.

        The calculated gradients must be arranged in a list that corresponds to the original
        ordering of the input tensors.
        """
        raise NotImplementedError("TensorOp subclasses must implement gradients()")

    @abstractmethod
    def lower(
        self, builder: ComputeProgramBuilder, inputs: tuple[ComputeTensor, ...]
    ) -> tuple[ComputeTensor, ...]:
        """
        Lower this op into the Compute IR by appending its block(s) to `builder`, returning the
        output ComputeTensor(s) that downstream ops read as their inputs.

        `inputs` are the ComputeTensors this op's graph inputs lowered to. The return is a tuple so
        multi-output ops are expressible later; single-output ops return a 1-tuple.
        """
        raise NotImplementedError("TensorOp subclasses must implement lower()")
