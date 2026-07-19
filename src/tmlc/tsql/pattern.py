from __future__ import annotations

from collections.abc import Iterator
from itertools import permutations
from typing import Callable, TypeAlias, override
from tmlc import Tensor, Constant
from tmlc.tensor.traits import Commutative

Env: TypeAlias = dict[str, Tensor]


class Match:
    def __init__(self, node: Tensor, env: Env) -> None:
        self.anchor: Tensor = node
        self.env: Env = env

    def __getitem__(self, key: str) -> Tensor:
        return self.env[key]


class Pattern:
    """
    Universal pattern primitive for matching subgraphs in a computational graph.
    Matches a node whose op is an instance of spec, passes where (if given), and
    recursively matches input sub-patterns positionally (with commutativity support).
    Binds the matched node to label if one is given.

    Ref and EqualTo subclass this and override _match for env-based back-reference logic.
    """

    def __init__(
        self,
        spec: type = object,
        inputs: list[Pattern] | None = None,
        label: str | None = None,
        where: Callable[[Tensor], bool] | None = None,
    ) -> None:
        self.spec: type = spec
        self.label: str = label or ""
        self._has_label: bool = label is not None
        self.input_patterns: list[Pattern] = inputs or []
        self.where: Callable[[Tensor], bool] | None = where

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.spec.__name__})"

    @override
    def __repr__(self) -> str:
        return str(self)

    def match(self, node: Tensor, initial_env: Env | None = None) -> Match | None:
        env = next(self._match(node, initial_env or {}), None)
        return None if env is None else Match(node, env)

    def _match_inputs(
        self,
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
            yield from self._match_inputs(rest, rnodes, env1)

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
        orderings = permutations(node.inputs) if isinstance(node.op, Commutative) else [node.inputs]
        for ordering in orderings:
            yield from self._match_inputs(self.input_patterns, list(ordering), env_out)


def Var(label: str) -> Pattern:
    """Matches any node and binds it to label."""
    return Pattern(label=label)


def Const(label: str) -> Pattern:
    """Matches a Constant node and binds it to label."""
    return Pattern(Constant, label=label)
