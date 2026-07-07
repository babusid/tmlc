from __future__ import annotations
from collections.abc import Iterator
from abc import ABC, abstractmethod
from typing import TypeAlias
from tmlc import Tensor
from .pattern import Pattern, Match, Env
from ..graph import Graph


def match_pattern(graph: Graph, pattern: Pattern) -> Iterator[Match]:
    """
    Given a graph and a pattern, yield all matches of the pattern in the graph.
    """
    for node in graph.topo_sort:
        match = pattern.match(node)
        if match is not None:
            yield match
