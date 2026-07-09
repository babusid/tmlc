from .pattern import Pattern, Env
from collections.abc import Iterator
from typing_extensions import override
from tmlc import Tensor, Constant


class Const(Pattern):
    """
    A Pattern that matches a constant Tensor in the graph.
    """

    def __init__(self, label: str) -> None:
        super().__init__(label, input_patterns=[])

    @override
    def _match(self, node: Tensor, env: Env) -> Iterator[Env]:
        if isinstance(node.op, Constant):
            yield {**env, self.label: node}
