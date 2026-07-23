"""
Attaches Tensor's operator dunders to their backing ops.

See the comment above the `Tensor` class definition in tensor.py for why this lives in its own
module: it breaks the circular dependency between tensor.py and the ops modules. This module
is imported once (for its side effect) from tmlc/__init__.py, before any Tensor is used.
"""

from __future__ import annotations

from tmlc.tensor.tensor import Tensor
from tmlc.tensor.ops.ops_arithmetic import add, div, mm, mul, negate, power
from tmlc.tensor.ops.ops_basic import constant
from tmlc.tensor.ops.ops_shape import transpose


def ensure_tensor(other: Tensor | float | int) -> Tensor:
    if isinstance(other, (int, float)):
        return constant(other)
    return other


def _add(self: Tensor, other: Tensor | float | int) -> Tensor:
    return add(self, ensure_tensor(other))


def _mul(self: Tensor, other: Tensor | float | int) -> Tensor:
    return mul(self, ensure_tensor(other))


def _truediv(self: Tensor, other: Tensor | float | int) -> Tensor:
    return div(self, ensure_tensor(other))


def _sub(self: Tensor, other: Tensor | float | int) -> Tensor:
    return add(self, negate(ensure_tensor(other)))


def _rsub(self: Tensor, other: Tensor | float | int) -> Tensor:
    return add(ensure_tensor(other), negate(self))


def _neg(self: Tensor) -> Tensor:
    return negate(self)


def _pow(self: Tensor, other: Tensor | float | int) -> Tensor:
    return power(self, ensure_tensor(other))


def _matmul(self: Tensor, other: Tensor) -> Tensor:
    return mm(self, other)


Tensor.__add__ = _add
Tensor.__radd__ = _add
Tensor.__mul__ = _mul
Tensor.__rmul__ = _mul
Tensor.__truediv__ = _truediv
Tensor.__sub__ = _sub
Tensor.__rsub__ = _rsub
Tensor.__neg__ = _neg
Tensor.__pow__ = _pow
Tensor.__matmul__ = _matmul
Tensor.T = property(transpose)  # pyright: ignore[reportAttributeAccessIssue]
