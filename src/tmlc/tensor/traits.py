from __future__ import annotations

from abc import ABC
from typing import TypeVar

from .tensor import TensorOp

_TensorOp = TypeVar("_TensorOp", bound=TensorOp)


class Commutative(ABC):
    """Runtime marker for operations whose inputs may be reordered."""


def commutative(op: type[_TensorOp]) -> type[_TensorOp]:
    """Mark a TensorOp class as commutative without replacing it."""
    if not issubclass(op, TensorOp):
        raise TypeError("@commutative can only decorate TensorOp subclasses")
    Commutative.register(op)
    return op
