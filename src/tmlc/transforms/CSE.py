from tmlc.graph import Graph, GraphTransform
from tmlc.tensor import Tensor
from tmlc.tsql.p_eq import structurally_equal
from typing_extensions import override


class CSE(GraphTransform):
    """
    Common Subexpression Elimination.

    Walks nodes in topological order and replaces later nodes that are
    structurally identical to an earlier one with the earlier node, eliminating
    redundant computation. Uses _structurally_equal from tmlc.tsql.p_eq for
    the equality check.
    """

    @override
    def __call__(self, graph: Graph) -> Graph:
        replacements:dict[Tensor,Tensor] = {}
        nodes = list(graph.topo_sort)
        for i, first_occurrence in enumerate(nodes):
            if first_occurrence in replacements:
                continue
            # first_occurrence hasn't been replaced, meaning it is unique
            for possible_duplicate in nodes[i + 1 :]: #scan everything beyond it
                if structurally_equal(first_occurrence, possible_duplicate):
                    replacements[possible_duplicate] = first_occurrence # record the duplication
        return graph.replace(replacements)
