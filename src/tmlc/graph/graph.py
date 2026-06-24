from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from functools import reduce
from tmlc.tensor.tensor import Tensor
from tmlc.tensor.ops.ops_shape import ones_like, zeros_like
from tmlc.util.topo_sort import dfs_helper_topo_sort
from tmlc.ndarray import ndarray
from abc import ABC, abstractmethod

class GraphTransform(ABC):
    '''
    Base class for graph transformations. Subclasses must implement the __call__ method.
    We implement a class for GraphTransform rather than allowing a raw Callable in order to avoid
    variadic arguments in the __call__ method, which would make it difficult to type check the
    transform functions. Instead, the intended pattern is to define custom constructors for
    each transform class that emit custom GraphTransform objects with the appropriate parameters,
    thereby allowing the actual __call__ signature to be uniform across all GraphTransform
    subclasses.
    '''
    @abstractmethod
    def __call__(self, graph: Graph) -> Graph:
        raise NotImplementedError("GraphTransform subclasses must implement __call__")

class Graph:
    """
    Explicit graph object that traces a computation graph built from Tensor objects.
    Must be built in order to actually run the computation graph, as well as to apply
    graph optimizations for increased performance. Building a Graph is the first part
    of the compilation process.

    Immutable by design: `inputs`/`outputs`/`topo_sort` are read-only tuples. A GraphTransform
    must never mutate the graph it's given — it must build and return a new Graph instead.
    """
    _inputs: tuple[Tensor, ...]
    _outputs: tuple[Tensor, ...]
    _topo_sort: tuple[Tensor, ...]

    def __init__(self, inputs: Sequence[Tensor], outputs: Sequence[Tensor]) -> None:
        self._inputs = tuple(inputs)
        self._outputs = tuple(outputs)
        self._topo_sort = tuple(self._build_topo_sort())

    @property
    def inputs(self) -> tuple[Tensor, ...]:
        return self._inputs

    @property
    def outputs(self) -> tuple[Tensor, ...]:
        return self._outputs

    @property
    def topo_sort(self) -> tuple[Tensor, ...]:
        return self._topo_sort

    def _build_topo_sort(self) -> list[Tensor]:
        """Return a topological sort of the graph's nodes."""
        visited: set[Tensor] = set()
        _topo: list[Tensor] = []
        for node in self._outputs:
            dfs_helper_topo_sort(node, visited, _topo)
        return _topo

    def apply_transforms(self, transform_fns: list[GraphTransform]) -> Graph:
        """Apply a transform pipeline to the graph"""
        return reduce(lambda graph, fn: fn(graph), transform_fns, self)

    def run(self, inputs: dict[Tensor, ndarray]) -> list[ndarray]:
        outputs = self.outputs
        topo_sort = self.topo_sort
        intermediates: dict[Tensor, ndarray] = {}
        for node in topo_sort:
            if node in inputs:
                intermediates[node] = inputs[node]
            else:
                input_values = [intermediates[input] for input in node.inputs]
                assert len(input_values) == len(node.inputs), (
                    "Mismatch in number of input values and node inputs"
                )
                intermediates[node] = node.op.compute(input_values)
        output: list[ndarray] = []
        for out in outputs:
            output.append(intermediates[out])

        return output

    def compile(self) -> None:
        return

    def run_compiled(self) -> None:
        return

def differentiate(graph: Graph, output_node: Tensor, target_nodes: list[Tensor]) -> Graph:
    rev_topo_sort: list[Tensor] = []
    rev_topo_sort = [t for t in graph.topo_sort]
    rev_topo_sort.reverse()
    # TODO: some online resources seem to indicate that ones_like isn't always the
    # right seed for the gradient? It IS correct for scalar losses,
    # but not for vector-valued outputs. For now, we assume the output
    # is a scalar loss. This will still work with vector-valued output,
    # but it'll be the gradient wrt the sum of the outputs
    output_grad = ones_like(output_node, "output_grad")

    # now we have all nodes we have to compute gradients for in target_set
    # for each node, we have to track what is coming in backwards
    node_grad_incoming: dict[Tensor, list[Tensor]] = defaultdict(list)
    # output node just gets the all one output gradient
    node_grad_incoming[output_node] = [output_grad]
    # map tensor to the aggregate of its input gradients
    node_grad: dict[Tensor, Tensor] = {}
    for node in rev_topo_sort:
        # get all incoming partial gradients, and aggregate them
        incoming_grad = node_grad_incoming[node]
        if not incoming_grad:
            # skips nodes in the
            # reverse that weren't populated by any of their children,
            # e.g. inputs that don't affect the output
            continue

        sum_grad: Tensor = reduce(lambda a, b: a + b, incoming_grad[1:], incoming_grad[0])
        node_grad[node] = sum_grad
        # pass in aggregate gradient and calculate the gradients
        # wrt this nodes inputs
        input_grads = node.op.gradients(node, sum_grad)
        for input, input_grad in zip(node.inputs, input_grads):
            # pass in the appropriate gradients
            node_grad_incoming[input].append(input_grad)

    bwd_outputs = [
        node_grad.get(target, zeros_like(target))
        for target in target_nodes
    ]
    return Graph(graph.inputs, bwd_outputs)


