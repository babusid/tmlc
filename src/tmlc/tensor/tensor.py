from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from numpy import ndarray
from typing_extensions import override


# Tensor's operator dunders (__add__, __mul__, .T, etc.) are intentionally NOT implemented here.
# Ops modules need Tensor/TensorOp as base classes, and operators need ops, which would make
# tensor.py and the ops modules import each other. To keep that a one-directional dependency
# (ops -> tensor only), the dunders are attached onto this class after the fact by
# tmlc/_operators.py, which is imported once from tmlc/__init__.py. The signatures below
# (TYPE_CHECKING-only) exist purely so static analysis and editors know the dunders exist.
class Tensor:
    """A Tensor is a node in a computational graph, representing a multi-dimensional array.
    Tensors are the inputs to tensor operations, which output new tensors. A sequence of Tensor
    Operations chained together produces a computational graph, which we can compile and optimize.
    Tensors do NOT actually hold data themseleves, but rather represent the flow of data through
    the graph. The actual data is held in buffers that are supplied at evaluation time.
    """

    inputs: list[Tensor]
    op: TensorOp
    label: str
    shape: tuple[int, ...]
    dtype: str

    def __init__(
        self,
        inputs: list[Tensor],
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
        self.dtype = dtype

    @override
    def __str__(self):
        inputs = ", ".join(str(tensor) for tensor in self.inputs)
        return f"Tensor(op={self.label}, inputs=[{inputs}])"

    @override
    def __repr__(self):
        return self.__str__()

    if TYPE_CHECKING:
        # Real implementations attached by tmlc/_operators.py — see comment above the class.
        def __add__(self, other: Tensor | float | int) -> Tensor: ...
        def __radd__(self, other: Tensor | float | int) -> Tensor: ...
        def __mul__(self, other: Tensor | float | int) -> Tensor: ...
        def __rmul__(self, other: Tensor | float | int) -> Tensor: ...
        def __truediv__(self, other: Tensor | float | int) -> Tensor: ...
        def __sub__(self, other: Tensor | float | int) -> Tensor: ...
        def __rsub__(self, other: Tensor | float | int) -> Tensor: ...
        def __neg__(self) -> Tensor: ...
        def __pow__(self, other: Tensor | float | int) -> Tensor: ...
        def __matmul__(self, other: Tensor) -> Tensor: ...
        @property
        def T(self) -> Tensor: ...


class ConstantTensor(Tensor):
    """
    Constant is a special type of Tensor that holds a buffer that is managed outside of the graph.
    It is operationally equivalent to an Input node, except that its value is determined at creation
    time, rather than supplied at evaluation time.
    It is a leaf node in the graph and does not have any input tensors.
    The value of a Constant tensor is stored in the `constval` field.
    """

    constval: ndarray

    def __init__(self, value: ndarray, op: TensorOp, label: str | None = None):
        super().__init__(inputs=[], op=op, label=label, shape=value.shape)
        self.constval = value


class TensorOp(ABC):
    """TensorOp interface represents an operation that can be performed on Tensors."""

    @abstractmethod
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        """When a TensorOp is called, it should create a new Tensor that represents the output of
        this operation."""
        raise NotImplementedError("TensorOp subclasses must implement __call__")

    @abstractmethod
    def infer_shape(self, inputs: list[Tensor]) -> tuple[int, ...]:
        """
        Given the input tensors (which contain their shapes), infer the shape of the
        output tensor that this operation will produce.
        """
        raise NotImplementedError("TensorOp subclasses must implement infer_shape()")

    @abstractmethod
    def compute(self, inputs: list[ndarray]) -> ndarray:
        """Given the input arrays, compute the output arrays of this operation.

        This is used by the evaluator to compute the values of the output tensors in the graph. This
        operates on concrete arrays to actually determine a concrete value, and is used for eager
        mode evaluation.
        """
        raise NotImplementedError("TensorOp subclasses must implement compute()")

    @abstractmethod
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        """Given the output of the forward `call` method and the incoming gradient from the
        backwards pass, this method calculates the gradients to propagate to the inputs.

        The calculated gradients must be arranged in a list that corresponds to the original
        ordering of the input tensors.
        """
        raise NotImplementedError("TensorOp subclasses must implement gradients()")

    @abstractmethod
    def emit_ir(self, inputs: list[str]) -> str:
        # TODO: may need to update function signature here. Do we need input tensor labels?
        """If compiling the graph, each TensorOp needs to emit IR that represents this operations
        computation.

        The compiler composes the graphs full IR to optimize and generate the final code.
        """
        raise NotImplementedError("TensorOp subclasses must implement emit_ir()")
