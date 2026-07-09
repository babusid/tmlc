from __future__ import annotations

from tmlc.tensor.tensor import Tensor


def dfs_helper_topo_sort(node: Tensor, visited: set[Tensor], topo_sort: list[Tensor]) -> None:
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
        dfs_helper_topo_sort(input_node, visited, topo_sort)
    # THEN add this node
    topo_sort.append(node)
