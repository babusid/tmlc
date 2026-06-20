"""Evaluator for TMLC computational graphs."""

from collections import defaultdict
import warnings
from tmlc.ndarray import ndarray
from tmlc.tensor import Tensor
from tmlc.ops.ops_basic import Input, Constant
from tmlc.ops.ops_shape import ones_like


def run(inputs: dict[Tensor, ndarray], outputs: list[Tensor]) -> list[list[ndarray]]:
    # eager mode evaluation
    # 1. reverse topo sort graph from outputs to inputs and
    #    build a traversal / target set of nodes that we have to compute
    #    compute them in order until outputs have been computed
    visited: set[Tensor] = set()
    topo_sort: list[Tensor] = []
    for node in outputs:
        _dfs_helper_topo_sort(node, visited, topo_sort)

    if any(input not in visited for input in inputs.keys()):
        warnings.warn(
            "Some provided input tensors not used in computation graph for requested outputs."
        )
    intermediates: dict[Tensor, list[ndarray]] = {}
    for node in topo_sort:
        if node in inputs:
            intermediates[node] = [inputs[node]]
        else:
            input_values = [intermediates[input][0] for input in node.inputs]
            assert len(input_values) == len(node.inputs), (
                "Mismatch in number of input values and node inputs"
            )
            intermediates[node] = node.op.compute(input_values)
    output: list[list[ndarray]] = []
    for out in outputs:
        output.append(intermediates[out])

    return output


def gradients(output_node: Tensor, target_nodes: list[Tensor]) -> list[Tensor]:
    # extend graph with gradients for desired node grads
    # topo sort, then reverse so output is at the front
    # then construct a oneslike node for the output node, then call
    # gradient repeatedly. After each node gradient, store that computed gradient
    # to the corresponding input
    # in a dictionary mapping nodes :: incoming_gradients
    # store the total gradient for this node (sum up all gradients) if needed
    # finally after finishing, return a list of gradients corresponding to the nodes we wanted
    # OPTIM: prune the set of nodes we compute gradients for by isolating
    # to only the ones on the path to the target nodes

    visited: set[Tensor] = set()
    rev_topo_sort: list[Tensor] = []
    _dfs_helper_topo_sort(output_node, visited, rev_topo_sort)
    rev_topo_sort.reverse()

    output_grad = ones_like(output_node, "output_grad")

    # track which nodes in the graph we actually have to
    # compute gradients for. This includes the targets,
    # the output gradient, and everything on the path between them
    target_set = set(target_nodes + [output_node])
    visited = set()

    def generate_target_set(tensor: Tensor) -> bool:
        # if we've already explored this node, just return whether it ended up a target
        if tensor in visited:
            return tensor in target_set
        visited.add(tensor)

        # inputs and constants are leaves: they're only targets if explicitly requested
        if isinstance(tensor.op, Input) or isinstance(tensor.op, Constant):
            return tensor in target_set

        # always recurse, even if `tensor` is already an explicit target, since ancestors
        # further up the graph still need to be connected through to it
        reached_target = tensor in target_set
        for input in tensor.inputs:
            if generate_target_set(input):
                reached_target = True

        if reached_target:
            target_set.add(tensor)
        return reached_target

    _ = generate_target_set(output_node)

    # now we have all nodes we have to compute gradients for in target_set
    # for each node, we have to track what is coming in backwards
    node_grad_incoming: dict[Tensor, list[Tensor]] = defaultdict(list)
    # output node just gets the all one output gradient
    node_grad_incoming[output_node] = [output_grad]
    # map tensor to the aggregate of its input gradients
    node_grad: dict[Tensor, Tensor] = {}
    for node in rev_topo_sort:
        if node not in target_set:
            continue
        # get all incoming partial gradients, and aggregate them
        incoming_grad = node_grad_incoming[node]
        sum_grad = incoming_grad[0]
        for grad in incoming_grad[1:]:
            sum_grad += grad
        node_grad[node] = sum_grad
        # pass in aggregate gradient and calculate the gradients
        # wrt this nodes inputs
        input_grads = node.op.gradients(node, sum_grad)
        for input, input_grad in zip(node.inputs, input_grads):
            # pass in the appropriate gradients
            node_grad_incoming[input].append(input_grad)

    return [node_grad[target] for target in target_nodes]


def compile() -> None:
    return


def run_compiled() -> None:
    return


def _dfs_helper_topo_sort(node: Tensor, visited: set[Tensor], topo_sort: list[Tensor]) -> None:
    """Helper function for topological sort using post-order DFS traversal.

    This ensures all nodes are processed after their children (dependencies).

    Parameters
    ----------
    node: Node
        The current node to process
    visited: Set[Node]
        Set of already visited nodes
    topo_sort: List[Node]
        List to append nodes in topological order
    """
    if node in visited:
        return
    visited.add(node)
    # Process children FIRST
    for input_node in node.inputs:
        _dfs_helper_topo_sort(input_node, visited, topo_sort)
    # THEN add this node
    topo_sort.append(node)
