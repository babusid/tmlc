"""
Logistic regression, three ways.

A copy of tests/logreg.py, extended to run the forward+backward train graph through three backends
and time them:

1. the GraphInterpreter (eager, over Tensor ops);
2. the ComputeInterpreter (over the fused Compute IR); and
3. native code from the generic C backend (Graph -> Compute -> Loop -> C -> shared lib).

All three must agree on the loss and gradients before timing. The emitted C is saved to
tests/logreg.c for inspection.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import numpy as np

import tmlc
from tmlc import Input
from tmlc.backends.c import CProgram, emit_c
from tmlc.compute.transforms import InlineReductionFree
from tmlc.interpreters.compute_interpreter import ComputeInterpreter
from tmlc.interpreters.graph_interpreter import GraphInterpreter
from tmlc.loop import lower_to_loops

NUM_CLASSES = 3
IN_FEATURES = 4
SAMPLES_PER_CLASS = 40
BATCH_SIZE = NUM_CLASSES * SAMPLES_PER_CLASS


def make_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A handful of well-separated Gaussian blobs, one per class."""
    centers = rng.uniform(-6, 6, size=(NUM_CLASSES, IN_FEATURES))
    X = np.concatenate(
        [center + 0.5 * rng.standard_normal((SAMPLES_PER_CLASS, IN_FEATURES)) for center in centers]
    )
    labels = np.repeat(np.arange(NUM_CLASSES), SAMPLES_PER_CLASS)
    return X, np.eye(NUM_CLASSES)[labels]


def build_graph():
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
    return x, W, b, y_one_hot, train_graph


def median_ms(fn: Callable[[], object], warmups: int = 5, reps: int = 100) -> float:
    for _ in range(warmups):
        fn()
    samples: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)) * 1e3


def main() -> None:
    rng = np.random.default_rng(0)
    X, y_one_hot = make_dataset(rng)
    x, W, b, y_node, train_graph = build_graph()

    W_val = (0.01 * rng.standard_normal((IN_FEATURES, NUM_CLASSES))).astype(np.float32)
    b_val = np.zeros(NUM_CLASSES, dtype=np.float32)
    X = X.astype(np.float32)
    y_one_hot = y_one_hot.astype(np.float32)

    # lower once: Graph -> fused Compute IR -> Loop IR -> C.
    compute_program = InlineReductionFree()(train_graph.lower())
    loop_program = lower_to_loops(compute_program)
    compiled = CProgram(loop_program)

    source = emit_c(loop_program)
    out_path = os.path.join(os.path.dirname(__file__), "logreg.c")
    with open(out_path, "w") as handle:
        handle.write(source)

    # Compute and Loop IR own distinct input objects, but both preserve graph-input order.
    graph_inputs = [node for node in train_graph.topo_sort if isinstance(node.op, Input)]

    def bindings(W_val: np.ndarray, b_val: np.ndarray):
        values = {x: X, W: W_val, b: b_val, y_node: y_one_hot}
        ordered = [values[g] for g in graph_inputs]
        compute_inputs = {ct: arr for ct, arr in zip(compute_program.inputs, ordered)}
        loop_inputs = {buffer: arr for buffer, arr in zip(loop_program.inputs, ordered)}
        return values, compute_inputs, loop_inputs

    graph_interp = GraphInterpreter()
    compute_interp = ComputeInterpreter()

    def run_graph(W_val, b_val):
        values, _, _ = bindings(W_val, b_val)
        return graph_interp.run(train_graph, values)

    def run_compute(W_val, b_val):
        _, ins, _ = bindings(W_val, b_val)
        return compute_interp.run(compute_program, ins)

    def run_c(W_val, b_val):
        _, _, ins = bindings(W_val, b_val)
        return compiled.run(ins)

    # correctness: all three backends agree on loss and gradients.
    g_loss, g_gW, g_gb = run_graph(W_val, b_val)
    c_loss, c_gW, c_gb = run_compute(W_val, b_val)
    n_loss, n_gW, n_gb = run_c(W_val, b_val)
    for a, b_ in ((c_loss, g_loss), (c_gW, g_gW), (c_gb, g_gb)):
        np.testing.assert_allclose(a, b_, rtol=1e-4, atol=1e-4)
    for a, b_ in ((n_loss, g_loss), (n_gW, g_gW), (n_gb, g_gb)):
        np.testing.assert_allclose(a, b_, rtol=1e-4, atol=1e-4)
    print("all three backends agree on loss + gradients\n")

    # timing: one forward+backward step.
    graph_ms = median_ms(lambda: run_graph(W_val, b_val))
    compute_ms = median_ms(lambda: run_compute(W_val, b_val))
    c_ms = median_ms(lambda: run_c(W_val, b_val))
    print(f"forward+backward, {BATCH_SIZE} samples, median of 100 reps:")
    print(f"  GraphInterpreter:    {graph_ms:8.4f} ms  (1.00x)")
    print(f"  ComputeInterpreter:  {compute_ms:8.4f} ms  ({graph_ms / compute_ms:.2f}x)")
    print(f"  compiled C:          {c_ms:8.4f} ms  ({graph_ms / c_ms:.2f}x)")
    print(f"\nemitted C saved to {out_path} ({len(source.splitlines())} lines)")

    # the compiled path still trains: plain SGD should reduce the loss and fit the blobs.
    initial_loss = float(g_loss)
    lr = 0.5
    for _ in range(200):
        _, grad_W_val, grad_b_val = run_c(W_val, b_val)
        W_val = (W_val - lr * grad_W_val).astype(np.float32)
        b_val = (b_val - lr * grad_b_val).astype(np.float32)
    final_loss = float(run_c(W_val, b_val)[0])
    print(f"\ntrained via compiled C: loss {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < initial_loss, "training should reduce the loss"


if __name__ == "__main__":
    main()
