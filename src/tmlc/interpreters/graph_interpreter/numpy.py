from collections.abc import Mapping

import numpy as np

from tmlc import (
    Add,
    BroadcastTo,
    Constant,
    Div,
    Exp,
    Fill,
    Graph,
    Input,
    Log,
    LogSumExp,
    Matmul,
    Mul,
    Negate,
    Pow,
    Reshape,
    Summation,
    Tanh,
    Tensor,
    Transpose,
)

from .registry import OpRegistry


NUMPY_OPS = OpRegistry[np.ndarray]()


@NUMPY_OPS.register(Constant)
def _constant(op: Constant, inputs: list[np.ndarray]) -> np.ndarray:
    assert not inputs, "Constant op cannot accept any inputs"
    return np.asarray(op.value.data)


@NUMPY_OPS.register(Fill)
def _fill(op: Fill, inputs: list[np.ndarray]) -> np.ndarray:
    assert not inputs, "Fill op cannot accept any inputs"
    return np.full(op.shape, op.value, dtype=op.dtype)


@NUMPY_OPS.register(Add)
def _add(op: Add, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 2, "Add op requires exactly 2 inputs"
    assert inputs[0].shape == inputs[1].shape, "Add op requires values with the same shape"
    return np.asarray(inputs[0] + inputs[1])


@NUMPY_OPS.register(Mul)
def _mul(op: Mul, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 2, "Mul op requires exactly 2 inputs"
    assert inputs[0].shape == inputs[1].shape, "Mul op requires values with the same shape"
    return np.asarray(inputs[0] * inputs[1])


@NUMPY_OPS.register(Div)
def _div(op: Div, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 2, "Div op requires exactly 2 inputs"
    assert inputs[0].shape == inputs[1].shape, "Div op requires values with the same shape"
    return np.asarray(inputs[0] / inputs[1])


@NUMPY_OPS.register(Matmul)
def _matmul(op: Matmul, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 2, "Matmul op requires exactly 2 inputs"
    assert inputs[0].ndim == 2 and inputs[1].ndim == 2, "Matmul op requires 2D inputs"
    return inputs[0] @ inputs[1]


@NUMPY_OPS.register(Negate)
def _negate(op: Negate, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 1, "Negate op requires exactly 1 input"
    return np.asarray(-inputs[0])


@NUMPY_OPS.register(Pow)
def _pow(op: Pow, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 2, "Power op requires exactly 2 inputs"
    assert inputs[0].shape == inputs[1].shape, "Power op requires values with the same shape"
    return np.asarray(inputs[0] ** inputs[1])


@NUMPY_OPS.register(Transpose)
def _transpose(op: Transpose, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 1, "Transpose op requires exactly 1 input"
    return np.transpose(inputs[0], axes=op._permutation(shape=inputs[0].shape))


@NUMPY_OPS.register(Summation)
def _summation(op: Summation, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 1, "Summation op requires exactly 1 input"
    return np.asarray(np.sum(inputs[0], axis=op.axes))


@NUMPY_OPS.register(Reshape)
def _reshape(op: Reshape, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 1, "Reshape op requires exactly 1 input"
    return np.reshape(inputs[0], op.shape)


@NUMPY_OPS.register(BroadcastTo)
def _broadcast_to(op: BroadcastTo, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 1, "BroadcastTo op requires exactly 1 input"
    return np.broadcast_to(inputs[0], op.shape)


@NUMPY_OPS.register(Exp)
def _exp(op: Exp, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 1, "Exp op requires exactly 1 input"
    return np.asarray(np.exp(inputs[0]))


@NUMPY_OPS.register(Log)
def _log(op: Log, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 1, "Log op requires exactly 1 input"
    return np.asarray(np.log(inputs[0]))


@NUMPY_OPS.register(Tanh)
def _tanh(op: Tanh, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 1, "Tanh op requires exactly 1 input"
    return np.asarray(np.tanh(inputs[0]))


@NUMPY_OPS.register(LogSumExp)
def _logsumexp(op: LogSumExp, inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) == 1, "LogSumExp op requires exactly 1 input"
    max_value = np.max(inputs[0], axis=op.axes, keepdims=True)
    shifted = inputs[0] - max_value
    sum_exp = np.sum(np.exp(shifted), axis=op.axes)
    return np.asarray(np.log(sum_exp) + np.reshape(max_value, sum_exp.shape))


class GraphInterpreter:
    def __init__(self, registry: OpRegistry[np.ndarray] = NUMPY_OPS) -> None:
        self.registry = registry

    def run(
        self,
        graph: Graph,
        inputs: Mapping[Tensor, np.ndarray],
    ) -> list[np.ndarray]:
        intermediates: dict[Tensor, np.ndarray] = {}
        for node in graph.topo_sort:
            if node in inputs:
                intermediates[node] = np.asarray(inputs[node])
            elif isinstance(node.op, Input):
                raise RuntimeError(
                    f"Input node '{node.label}' (shape={node.shape}) was not provided a value"
                )
            else:
                input_values = [intermediates[input_node] for input_node in node.inputs]
                intermediates[node] = self.registry.evaluate(node.op, input_values)
        return [intermediates[output] for output in graph.outputs]
