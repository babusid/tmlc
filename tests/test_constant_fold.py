"""ConstantFold: evaluate constant subgraphs at compile time without changing results."""

import numpy as np

import tmlc
from tmlc.interpreters.graph_interpreter import GraphInterpreter
from tmlc.transforms import ConstantFold


def test_folds_constant_only_graph_to_constants():
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
    expected = GraphInterpreter().run(graph, inputs={})
    folded = graph.apply_transforms([ConstantFold()])
    actual = GraphInterpreter().run(folded, inputs={})

    assert all(isinstance(out.op, tmlc.Constant) for out in folded.outputs)
    for expected_output, actual_output in zip(expected, actual):
        assert np.allclose(expected_output, actual_output)


def test_folds_constant_subtree_under_input_and_preserves_value():
    interpreter = GraphInterpreter()
    x = tmlc.input(shape=(2,), label="x")
    a = tmlc.constant(2, label="a")
    b = tmlc.constant(4, label="b")
    c = tmlc.constant(3, label="c")
    out = a + b
    out -= c
    out *= 5
    out **= 3
    out -= 1
    out = out * x
    graph = tmlc.Graph([out])

    x_val = np.array([1, 2])
    before = interpreter.run(graph, inputs={x: x_val})[0]

    folded = graph.apply_transforms([ConstantFold()])
    after = interpreter.run(folded, inputs={x: x_val})[0]

    np.testing.assert_allclose(before, after)
    assert len(folded.topo_sort) < len(graph.topo_sort), "the constant subtree should collapse"
