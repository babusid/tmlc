import numpy as np
import tmlc
from tmlc.interpreters.graph_interpreter import GraphInterpreter

interpreter = GraphInterpreter()

x = tmlc.input(shape=(2, 2), label="x")
y = tmlc.input(shape=(2, 2), label="y")
z = tmlc.input(shape=(2, 2), label="z")

a = x * y
b = a * z
c = b * a
out = c + a

forward_graph = tmlc.Graph([a, b, c])

output = interpreter.run(
    forward_graph,
    inputs={
        x: np.array([[1, 2], [3, 4]]),
        y: np.array([[5, 6], [7, 8]]),
        z: np.array([[1, 1], [1, 1]]),
    },
)

print(output)

diff_graph = tmlc.Graph([out])
grad_graph, _ = tmlc.differentiate(graph=diff_graph, output_node=out, target_nodes=[a, b, c])

output = interpreter.run(
    grad_graph,
    inputs={
        x: np.array([[1, 2], [3, 4]]),
        y: np.array([[5, 6], [7, 8]]),
        z: np.array([[1, 1], [1, 1]]),
    },
)

print(output)
