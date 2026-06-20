from beartype.claw import beartype_this_package

beartype_this_package()  # must precede submodule imports below to hook them

from .ndarray import ndarray
from .tensor import ConstantTensor, Tensor, TensorOp
from . import _operators  # noqa: F401  (attaches Tensor's operator dunders, see tensor.py)
from .ops.ops_basic import Constant, Input, constant, zeros, ones, input
from .ops.ops_arithmetic import Add, Div, Matmul, Mul, Negate, Pow, add, div, mm, mul, negate, power
from .ops.ops_logarithmic import Exp, Log, LogSumExp, Tanh, exp, log, logsumexp, tanh
from .ops.ops_shape import (
    BroadcastTo,
    OnesLike,
    Reshape,
    Summation,
    Transpose,
    ZerosLike,
    broadcast_to,
    ones_like,
    reshape,
    summation,
    transpose,
    zeros_like,
)
from .evaluator import run, gradients

__all__ = [
    "ndarray",
    "Tensor",
    "ConstantTensor",
    "TensorOp",
    "run",
    "gradients",
    "constant",
    "zeros",
    "ones",
    "input",
    "add",
    "div",
    "mm",
    "mul",
    "negate",
    "power",
    "exp",
    "log",
    "logsumexp",
    "tanh",
    "broadcast_to",
    "ones_like",
    "reshape",
    "summation",
    "transpose",
    "zeros_like",
    "Constant",
    "Input",
    "Add",
    "Div",
    "Matmul",
    "Mul",
    "Negate",
    "Pow",
    "Exp",
    "Log",
    "LogSumExp",
    "Tanh",
    "BroadcastTo",
    "OnesLike",
    "Reshape",
    "Summation",
    "Transpose",
    "ZerosLike",
]
