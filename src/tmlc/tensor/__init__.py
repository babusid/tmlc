"""Tensor core: the Tensor/TensorOp class hierarchy, operator dunders, and built-in ops."""

from __future__ import annotations

from .tensor import ConstantTensor, Tensor, TensorOp
from . import _operators  # noqa: F401  (attaches Tensor's operator dunders, see tensor.py)
from .ops.ops_basic import Constant, Input, constant, zeros, ones, input
from .ops.ops_arithmetic import Add, Div, Matmul, Mul, Negate, Pow, add, div, mm, mul, negate, power
from .ops.ops_logarithmic import Exp, Log, LogSumExp, Tanh, exp, log, logsumexp, tanh
from .ops.ops_shape import (
    BroadcastTo,
    Fill,
    Reshape,
    Summation,
    Transpose,
    broadcast_to,
    ones_like,
    reshape,
    summation,
    transpose,
    zeros_like,
)

from .traits import CommutativeOp

__all__ = [
    "ConstantTensor",
    "Tensor",
    "TensorOp",
    "Constant",
    "Input",
    "constant",
    "zeros",
    "ones",
    "input",
    "Add",
    "Div",
    "Matmul",
    "Mul",
    "Negate",
    "Pow",
    "add",
    "div",
    "mm",
    "mul",
    "negate",
    "power",
    "Exp",
    "Log",
    "LogSumExp",
    "Tanh",
    "exp",
    "log",
    "logsumexp",
    "tanh",
    "BroadcastTo",
    "Fill",
    "Reshape",
    "Summation",
    "Transpose",
    "broadcast_to",
    "ones_like",
    "reshape",
    "summation",
    "transpose",
    "zeros_like",
    "CommutativeOp"
]
