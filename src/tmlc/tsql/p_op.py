from __future__ import annotations

from collections.abc import Iterator
from itertools import permutations
from typing import Callable
from typing_extensions import override

from tmlc import Tensor
from tmlc.tensor.traits import is_commutative

from .pattern import Env, Pattern


def _match_inputs(
    subpats: list[Pattern],
    nodes: list[Tensor],
    env: Env,
) -> Iterator[Env]:
    if not subpats:
        yield env
        return
    first, *rest = subpats
    fnode, *rnodes = nodes
    for env1 in first._match(fnode, env):
        yield from _match_inputs(rest, rnodes, env1)


class Op(Pattern):
    """
    Matches a node whose op is an instance of spec (a concrete class or protocol).
    Input sub-patterns are matched positionally — Op does not handle commutativity.
    Optionally binds the matched root node to label, and accepts a where predicate
    for additional constraints.
    """

    def __init__(
        self,
        spec: type,
        inputs: list[Pattern] | None = None,
        label: str | None = None,
        where: Callable[[Tensor], bool] | None = None,
    ) -> None:
        super().__init__(label or "", input_patterns=inputs or [])
        self.spec = spec
        self.where = where
        self._has_label = label is not None

    @override
    def _match(self, node: Tensor, env: Env) -> Iterator[Env]:
        if not isinstance(node.op, self.spec):
            return
        if self.where is not None and not self.where(node):
            return
        env_out = {**env, self.label: node} if self._has_label else env
        if not self.input_patterns:
            yield env_out
            return
        if len(node.inputs) != len(self.input_patterns):
            return
        orderings = (
            permutations(node.inputs) if is_commutative(node.op) else [node.inputs]
        )
        for ordering in orderings:
            yield from _match_inputs(self.input_patterns, list(ordering), env_out)
