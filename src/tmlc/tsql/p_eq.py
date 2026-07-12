from __future__ import annotations

from collections.abc import Iterator
from typing_extensions import override

import numpy as np

from tmlc import Tensor, Constant, Input

from .pattern import Env, Pattern


def _structurally_equal(a: Tensor, b: Tensor) -> bool:
    """Recursive structural equality on the subgraph rooted at a and b."""
    if a is b:
        return True
    if type(a.op) is not type(b.op):
        return False
    if isinstance(a.op, Input):
        return False
    if isinstance(a.op, Constant) and isinstance(b.op, Constant):
        if not np.array_equal(a.op.value, b.op.value):
            return False
    elif vars(a.op) != vars(b.op):
        return False
    if len(a.inputs) != len(b.inputs):
        return False
    return all(_structurally_equal(x, y) for x, y in zip(a.inputs, b.inputs))


class EqualTo(Pattern):
    """
    Structural back-reference: matches when the candidate node is structurally equal
    (same op, same op params, same input subgraph) to the node already bound to label.
    Use this to find duplicated computations — distinct objects computing the same thing.
    """

    def __init__(self, label: str) -> None:
        super().__init__(label=label)

    @override
    def _match(self, node: Tensor, env: Env) -> Iterator[Env]:
        if self.label in env and _structurally_equal(node, env[self.label]):
            yield env
