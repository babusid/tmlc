from __future__ import annotations

from collections.abc import Iterator
from typing_extensions import override

from tmlc import Tensor

from .pattern import Env, Pattern


class Ref(Pattern):
    """
    Identity back-reference: matches only when the candidate node IS (identity-equal to)
    the node already bound to label. Use this to express that a variable is reused in
    the graph, e.g. A*B + A*C where A is the same Tensor object.
    """

    def __init__(self, label: str) -> None:
        super().__init__(label=label)

    @override
    def _match(self, node: Tensor, env: Env) -> Iterator[Env]:
        if self.label in env and env[self.label] is node:
            yield env
