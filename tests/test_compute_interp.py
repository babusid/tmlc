"""
Numerical equivalence between the graph interpreter and the Compute IR interpreter.

Builds the logistic-regression graph (matmul + broadcast bias + softmax loss via logsumexp, plus its
autodiff gradients), evaluates it through the graph interpreter and by lowering to a ComputeProgram
and running that, and asserts the results agree. This exercises matmul (reduce), broadcast/reshape
index arithmetic, elementwise scalar ops, and MAX/SUM reductions across forward and backward graphs.
"""

import numpy as np
import pytest

import tmlc
from tmlc.interpreters.compute_interpreter import ComputeInterpreter
from tmlc.interpreters.graph_interpreter import GraphInterpreter

from support import graph_input_bindings

NUM_CLASSES = 3
IN_FEATURES = 4
BATCH_SIZE = 120


def _build_graphs():
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
    return {"x": x, "W": W, "b": b, "y": y_one_hot, "train": train_graph, "eval": eval_graph}


def _values(rng):
    graphs = _build_graphs()
    labels = rng.integers(0, NUM_CLASSES, size=BATCH_SIZE)
    values = {
        graphs["x"]: rng.standard_normal((BATCH_SIZE, IN_FEATURES)),
        graphs["W"]: 0.1 * rng.standard_normal((IN_FEATURES, NUM_CLASSES)),
        graphs["b"]: rng.standard_normal(NUM_CLASSES),
        graphs["y"]: np.eye(NUM_CLASSES)[labels],
    }
    return graphs, values


@pytest.mark.parametrize("which", ["eval", "train"])
def test_compute_interp_matches_graph_interp(rng, which):
    graphs, values = _values(rng)
    graph = graphs[which]
    if which == "eval":
        values = {k: values[k] for k in (graphs["x"], graphs["W"], graphs["b"])}

    graph_outputs = GraphInterpreter().run(graph, inputs=values)

    program = graph.lower()
    bindings = graph_input_bindings(graph, program, values)
    compute_outputs = ComputeInterpreter().run(program, inputs=bindings)

    assert len(graph_outputs) == len(compute_outputs)
    for expected, actual in zip(graph_outputs, compute_outputs):
        assert expected.shape == actual.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
