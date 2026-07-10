from __future__ import annotations
from collections.abc import Iterator
from abc import ABC, abstractmethod
from typing import TypeAlias, override
from tmlc import Tensor

"""
Env is the binding environment for a match object.
A Pattern Match contains a mapping between the entities in the pattern,
and the actual nodes in the graph that were matched.
"""
Env: TypeAlias = dict[str, Tensor]

"""
Match is a container for the result of a successful pattern match.
"""


class Match:
    def __init__(self, node: Tensor, env: Env) -> None:
        self.anchor: Tensor = node
        self.env: Env = env

    def __getitem__(self, key: str) -> Tensor:
        """
        Returns the node bound to the given key in the match's environment.
        """
        return self.env[key]


class Pattern(ABC):
    """
    A Pattern is a template for matching subgraphs in a computational graph.
    Given a node in the graph, a Pattern can be used to determine if the subgraph rooted at that
    node matches the pattern. If it does, the Pattern returns a Match object containing the
    matched node and any variable bindings (environment) that were found during the match.
    """

    @override
    def __str__(self):
        return f"{self.__class__.__name__}()"

    @override
    def __repr__(self):
        return str(self)

    def __init__(self, label: str, input_patterns: list[Pattern]) -> None:
        self.label: str = label
        self.input_patterns: list[Pattern] = input_patterns

    def match(self, node: Tensor) -> Match | None:
        """
        Public-facing API just requires the Tensor itself.
        All binding environment mechanics are handled internally.
        """
        env = next(self._match(node, {}), None)
        return None if env is None else Match(node, env)

    @abstractmethod
    def _match(self, node: Tensor, env: Env) -> Iterator[Env]:
        """
        Internal method that performs the actual matching logic.
        Subclasses should implement the recursive matching logic here,
        yielding binding environments for each successful match.
        """
        raise NotImplementedError("Subclasses must implement _match method")
