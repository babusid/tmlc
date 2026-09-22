"""
Forward evaluation and reverse-mode autodiff over a small elementwise graph.

Builds a = x*y, b = a*z, c = b*a, out = c + a, evaluates the forward tensors, then differentiates
`out` with respect to the intermediates and prints the gradients.
"""

import numpy as np

import tmlc
from tmlc.interpreters.graph_interpreter import GraphInterpreter


def main() -> None:
    interpreter = GraphInterpreter()

    x = tmlc.input(shape=(2, 2), label="x")
    y = tmlc.input(shape=(2, 2), label="y")
    z = tmlc.input(shape=(2, 2), label="z")

    a = x * y
    b = a * z
    c = b * a
    out = c + a

    inputs = {
        x: np.array([[1, 2], [3, 4]]),
        y: np.array([[5, 6], [7, 8]]),
        z: np.array([[1, 1], [1, 1]]),
    }

    forward = interpreter.run(tmlc.Graph([a, b, c]), inputs=inputs)
    print("forward [a, b, c]:", forward)

    grad_graph, _ = tmlc.differentiate(
        graph=tmlc.Graph([out]), output_node=out, target_nodes=[a, b, c]
    )
    grads = interpreter.run(grad_graph, inputs=inputs)
    print("d(out)/d[a, b, c]:", grads)


if __name__ == "__main__":
    main()
