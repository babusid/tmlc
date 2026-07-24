import numpy as np

import tmlc
from tmlc.interpreters.graph_interpreter import GraphInterpreter
from tmlc.transforms import ConstantFold


interpreter = GraphInterpreter()

matrix = tmlc.constant(((1.0, 2.0), (3.0, 4.0)))
positive = tmlc.constant(((2.0, 3.0), (4.0, 5.0)))
row = tmlc.constant((2.0, 3.0))

outputs = [
    matrix + positive,
    matrix * positive,
    matrix / positive,
    -matrix,
    matrix ** tmlc.constant(2.0),
    matrix @ positive,
    matrix.T,
    tmlc.summation(matrix, axes=0),
    tmlc.reshape(matrix, (4,)),
    matrix + row,
    tmlc.exp(matrix),
    tmlc.log(positive),
    tmlc.tanh(matrix),
    tmlc.logsumexp(matrix, axes=1),
    tmlc.ones((2, 2)) + matrix,
]

graph = tmlc.Graph(outputs)
expected = interpreter.run(graph, inputs={})
folded_graph = graph.apply_transforms([ConstantFold()])
actual = interpreter.run(folded_graph, inputs={})

assert all(isinstance(output.op, tmlc.Constant) for output in folded_graph.outputs)
for expected_output, actual_output in zip(expected, actual):
    assert np.allclose(expected_output, actual_output)

print("all folding assertions passed")
