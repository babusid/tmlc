"""
Numerical-equivalence check for the Compute IR interpreter.

Builds the logistic-regression graph (matmul + broadcast bias + softmax loss via logsumexp, plus its
autodiff gradients), evaluates it two ways -- through the graph interpreter, and by lowering it to a
ComputeProgram and running that on the compute interpreter -- and asserts the results agree. This
exercises matmul (reduce), broadcast/reshape index arithmetic, elementwise scalar ops, and MAX/SUM
reductions across both the forward and backward graphs.
"""

import time

import numpy as np

import tmlc
from tmlc import Input
from tmlc.compute import ComputeProgram, ComputeTensor
from tmlc.interpreters.compute_interpreter import ComputeInterpreter
from tmlc.interpreters.graph_interpreter import GraphInterpreter

NUM_CLASSES = 3
IN_FEATURES = 4
BATCH_SIZE = 120


def build_graphs() -> tuple[
    tmlc.Tensor, tmlc.Tensor, tmlc.Tensor, tmlc.Tensor, tmlc.Graph, tmlc.Graph
]:
    x = tmlc.input(shape=(BATCH_SIZE, IN_FEATURES), label="x")
    W = tmlc.input(shape=(IN_FEATURES, NUM_CLASSES), label="W")
    b = tmlc.input(shape=(NUM_CLASSES,), label="b")
    y_one_hot = tmlc.input(shape=(BATCH_SIZE, NUM_CLASSES), label="y_one_hot")

    logits = x @ W + b
    log_partition = tmlc.logsumexp(logits, axes=(1,))
    correct_logit = tmlc.summation(logits * y_one_hot, axes=(1,))
    loss = tmlc.summation(log_partition - correct_logit, axes=(0,)) / BATCH_SIZE

    loss_graph = tmlc.Graph([loss])
    grad_graph, _ = tmlc.differentiate(graph=loss_graph, output_node=loss, target_nodes=[W, b])
    train_graph = tmlc.Graph([loss, *grad_graph.outputs])
    eval_graph = tmlc.Graph([logits])
    return x, W, b, y_one_hot, train_graph, eval_graph


def input_bindings(
    graph: tmlc.Graph,
    program: ComputeProgram,
    values: dict[tmlc.Tensor, np.ndarray],
) -> dict[ComputeTensor, np.ndarray]:
    """
    Map each program input tensor to its value.

    `lower` declares one input per Input node in graph topological order, so the graph's Input nodes
    in that same order line up one-to-one with `program.inputs`.
    """
    graph_inputs = [node for node in graph.topo_sort if isinstance(node.op, Input)]
    assert len(graph_inputs) == len(program.inputs), (
        "input count mismatch between graph and program"
    )
    return {tensor: values[node] for node, tensor in zip(graph_inputs, program.inputs)}


def check_equivalence(graph: tmlc.Graph, values: dict[tmlc.Tensor, np.ndarray], label: str) -> None:
    graph_time = time.time()
    graph_outputs = GraphInterpreter().run(graph, inputs=values)
    graph_time = time.time() - graph_time

    program = graph.lower()
    bindings = input_bindings(graph, program, values)

    start = time.time()
    compute_outputs = ComputeInterpreter().run(program, inputs=bindings)
    elapsed = time.time() - start

    assert len(graph_outputs) == len(compute_outputs), "output count mismatch"
    for index, (expected, actual) in enumerate(zip(graph_outputs, compute_outputs)):
        assert expected.shape == actual.shape, (
            f"{label} output {index}: shape {expected.shape} != {actual.shape}"
        )
        assert np.allclose(expected, actual, rtol=1e-5, atol=1e-6), (
            f"{label} output {index}: values disagree\n{expected}\n!=\n{actual}"
        )

    print(
        f"{label}: {len(program.blocks)} blocks, {len(graph_outputs)} outputs match "
        f"(compute interp {elapsed * 1e3:.1f} ms)"
        f" vs (graph interp {graph_time * 1e3:.1f} ms)"
    )


def main() -> None:
    rng = np.random.default_rng(0)
    x, W, b, y_one_hot, train_graph, eval_graph = build_graphs()

    labels = rng.integers(0, NUM_CLASSES, size=BATCH_SIZE)
    values: dict[tmlc.Tensor, np.ndarray] = {
        x: rng.standard_normal((BATCH_SIZE, IN_FEATURES)),
        W: 0.1 * rng.standard_normal((IN_FEATURES, NUM_CLASSES)),
        b: rng.standard_normal(NUM_CLASSES),
        y_one_hot: np.eye(NUM_CLASSES)[labels],
    }

    check_equivalence(eval_graph, {k: values[k] for k in (x, W, b)}, "logits")
    check_equivalence(train_graph, values, "loss + grads")
    print("compute interpreter matches graph interpreter")


if __name__ == "__main__":
    main()
