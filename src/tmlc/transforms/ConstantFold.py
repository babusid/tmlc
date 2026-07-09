import numpy as np
import tmlc
from tmlc.tsql import Pattern, Const, Op, match_pattern
from tmlc.tensor import CommutativeOp, ConstantTensor, Add, Mul, BroadcastTo
from tmlc.graph import Graph, GraphTransform
from typing_extensions import override


class ConstantFold(GraphTransform):
    @override
    def __call__(self, graph: Graph) -> Graph:
        fold_pattern: Pattern = Op(CommutativeOp, [Const("c1"), Const("c2")], label="fold")
        # contains root nodes that are ops with inputs that are leaves
        matches = list(match_pattern(graph, fold_pattern))
        ret_graph: Graph = graph
        while matches:
            for match in matches:
                op_node = match.anchor  # the root node
                c1_node = match.get_binding("c1", ConstantTensor)  # the two constant inputs
                c2_node = match.get_binding("c2", ConstantTensor)
                # compute the result of the operation on the constant values
                result_value = op_node.op.compute([c1_node.value, c2_node.value])
                # create a new constant node with the computed value
                new_const_node = tmlc.constant(result_value, label=f"folded_{op_node.label}")
                # replace the op node in the graph with the new constant node
                ret_graph = ret_graph.replace({op_node: new_const_node})
            # re-match after replacements
            matches = list(match_pattern(ret_graph, fold_pattern))

        return ret_graph
