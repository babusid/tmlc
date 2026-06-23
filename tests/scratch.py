import numpy as np
import tmlc

x = tmlc.input(shape=(2, 2), label="x")
y = tmlc.input(shape=(2, 2), label="y")
z = tmlc.input(shape=(2, 2), label="z")

a = x * y
b = a * z
c = b * a
out = c + a

forward_graph = tmlc.Graph(inputs=[x, y, z], outputs=[a, b, c])

output = forward_graph.run(
    inputs={
        x: np.array([[1, 2], [3, 4]]),
        y: np.array([[5, 6], [7, 8]]),
        z: np.array([[1, 1], [1, 1]]),
    },
)

print(output)

diff_graph = tmlc.Graph(inputs=[x, y, z], outputs=[out])
grad_graph = tmlc.differentiate(graph=diff_graph, output_node=out, target_nodes=[a, b, c])

output = grad_graph.run(
    inputs={
        x: np.array([[1, 2], [3, 4]]),
        y: np.array([[5, 6], [7, 8]]),
        z: np.array([[1, 1], [1, 1]]),
    },
)

print(output)
