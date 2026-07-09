import numpy as np
import tmlc
from tmlc.tsql import Pattern, Const, Op, match_pattern
from tmlc.tensor import CommutativeOp, ConstantTensor, Negate, TensorOp
from tmlc.graph import Graph, GraphTransform
from typing_extensions import override


class ConstantFold(GraphTransform):
    @override
    @override
    def __call__(self, graph: Graph) -> Graph:
        # TODO: kinda abusing the generic matcher, Any/Var construct should
        # probably accept the same sort of where lambda to filter. Should all patterns?
        fold_pattern: Pattern = Op(
            TensorOp,
            where=(
                lambda x: all(isinstance(inp, ConstantTensor) for inp in x.inputs)
                and len(x.inputs) > 0
            ),
            label="ConstantFold",
        )
        # contains root nodes that are ops with inputs that are leaves
        matches = list(match_pattern(graph, fold_pattern))
        ret_graph: Graph = graph
        while matches:
            for match in matches:
                op_node = match.anchor  # root node
                # result of the operation on the constant values
                result_value = op_node.op.compute([inp.value for inp in op_node.inputs])
                # create a new constant node with the computed value
                new_const_node = tmlc.constant(result_value, label=f"folded_{op_node.label}")
                # replace the op node in the graph with the new constant node
                ret_graph = ret_graph.replace({op_node: new_const_node})
            # re-match after replacements
            matches = list(match_pattern(ret_graph, fold_pattern))

        return ret_graph
