from __future__ import annotations

from typing import Protocol, runtime_checkable

from .tensor import TensorOp


@runtime_checkable
class CommutativeOp(Protocol):
    """Marker protocol: this op's inputs may be matched in any order."""
    commutative: bool


def is_commutative(op: TensorOp) -> bool:
    """Guard against runtime_checkable's attribute-presence-only isinstance check."""
    return isinstance(op, CommutativeOp) and op.commutative
