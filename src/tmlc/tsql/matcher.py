from __future__ import annotations
from collections.abc import Iterator
from .pattern import Pattern, Match, Env
from ..graph import Graph


def match_pattern(
    graph: Graph,
    pattern: Pattern,
    initial_env: Env | None = None,
) -> Iterator[Match]:
    """
    Given a graph and a pattern, yield all matches of the pattern in the graph.
    """
    for node in graph.topo_sort:
        match = pattern.match(node, initial_env)
        if match is not None:
            yield match
