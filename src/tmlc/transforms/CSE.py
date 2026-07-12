from tmlc.graph import Graph, GraphTransform
from tmlc.tensor import Tensor
from tmlc.tsql import EqualTo, match_pattern
from typing_extensions import override


class CSE(GraphTransform):
    """
    Common Subexpression Elimination.

    Walks nodes in topological order and replaces later nodes that are
    structurally identical to an earlier one with the earlier node, eliminating
    redundant computation. Uses EqualTo pattern seeded with each candidate
    canonical node to find all structural duplicates.
    """

    @override
    def __call__(self, graph: Graph) -> Graph:
        replacements: dict[Tensor, Tensor] = {}
        nodes = list(graph.topo_sort)
        reflabel = "canonical"
        refpattern = EqualTo(reflabel)
        for canonical in nodes:
            if canonical in replacements:
                continue
            for match in match_pattern(graph, refpattern, initial_env={reflabel: canonical}):
                if match.anchor is not canonical:
                    replacements[match.anchor] = canonical
        return graph.replace(replacements)
