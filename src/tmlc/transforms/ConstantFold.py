import tmlc
from typing import cast
from tmlc.tsql import Pattern, match_pattern
from tmlc.tensor import TensorOp, Constant
from tmlc.graph import Graph, GraphTransform
from typing_extensions import override


class ConstantFold(GraphTransform):
    @override
    def __call__(self, graph: Graph) -> Graph:
        fold_pattern: Pattern = Pattern(
            TensorOp,
            where=lambda x: (
                len(x.inputs) > 0 and all(isinstance(inp.op, Constant) for inp in x.inputs)
            ),
            label="ConstantFold",
        )
        matches = list(match_pattern(graph, fold_pattern))
        ret_graph: Graph = graph
        while matches:
            for match in matches:
                op_node = match.anchor
                result_value = op_node.op.compute(
                    [cast(Constant, inp.op).value for inp in op_node.inputs]
                )
                new_const_node = tmlc.constant(result_value, label=f"folded_{op_node.label}")
                ret_graph = ret_graph.replace({op_node: new_const_node})
            matches = list(match_pattern(ret_graph, fold_pattern))

        return ret_graph
