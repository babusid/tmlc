"""Logistic regression test case for tmlc.

Builds a small multi-class logistic regression graph (matmul + broadcast bias + softmax loss
via logsumexp), checks tmlc's autodiff gradients against a finite-difference estimate, then
trains the model with plain SGD on a synthetic, linearly-separable dataset and checks that it
actually learns.
"""

import numpy as np

import tmlc

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

    # bias broadcasts (NUM_CLASSES,) -> (BATCH_SIZE, NUM_CLASSES) through the `+` overload.
    logits = x @ W + b

    # per-sample negative log likelihood: logsumexp(logits) - logits[correct_class]
    log_partition = tmlc.logsumexp(logits, axes=(1,))
    correct_logit = tmlc.summation(logits * y_one_hot, axes=(1,))
    loss = tmlc.summation(log_partition - correct_logit, axes=(0,)) / BATCH_SIZE

    loss_graph = tmlc.Graph([loss])
    grad_graph, _ = tmlc.differentiate(graph=loss_graph, output_node=loss, target_nodes=[W, b])
    train_graph = tmlc.Graph([loss, *grad_graph.outputs])
    eval_graph = tmlc.Graph([logits])
    return x, W, b, y_one_hot, train_graph, eval_graph


def main():
    rng = np.random.default_rng(0)
    X, y_one_hot = make_dataset(rng)
    x, W, b, y_one_hot_node, train_graph, eval_graph = build_graph()

    def forward_backward(W_val: np.ndarray, b_val: np.ndarray):
        loss_val, grad_W_val, grad_b_val = train_graph.run(
            inputs={x: X, y_one_hot_node: y_one_hot, W: W_val, b: b_val},
        )
        return loss_val, grad_W_val, grad_b_val

    W_val = 0.01 * rng.standard_normal((IN_FEATURES, NUM_CLASSES))
    b_val = np.zeros(NUM_CLASSES)

    # sanity-check tmlc's autodiff against a central-difference estimate before training.
    eps = 1e-4
    i, j = 0, 1
    base_loss, base_grad_W, _ = forward_backward(W_val, b_val)
    W_plus, W_minus = W_val.copy(), W_val.copy()
    W_plus[i, j] += eps
    W_minus[i, j] -= eps
    loss_plus, _, _ = forward_backward(W_plus, b_val)
    loss_minus, _, _ = forward_backward(W_minus, b_val)
    numerical_grad = (loss_plus - loss_minus) / (2 * eps)
    assert np.isclose(
        base_grad_W[i, j], numerical_grad, rtol=1e-3, atol=1e-3
    ), f"autodiff grad {base_grad_W[i, j]} != numerical grad {numerical_grad}"

    initial_loss = float(base_loss)
    lr = 0.5
    for _ in range(200):
        _, grad_W_val, grad_b_val = forward_backward(W_val, b_val)
        W_val = W_val - lr * grad_W_val
        b_val = b_val - lr * grad_b_val

    final_loss, _, _ = forward_backward(W_val, b_val)
    predicted_logits = eval_graph.run(inputs={x: X, W: W_val, b: b_val})[0]
    accuracy = np.mean(np.argmax(predicted_logits, axis=1) == np.argmax(y_one_hot, axis=1))

    print(f"initial loss: {initial_loss:.4f}")
    print(f"final loss:   {float(final_loss):.4f}")
    print(f"accuracy:     {accuracy:.2f}")

    assert final_loss < initial_loss, "training should reduce the loss"
    assert accuracy > 0.95, "model should fit this linearly-separable synthetic dataset"


if __name__ == "__main__":
    main()
