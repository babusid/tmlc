"""CSE deduplicates structurally identical subgraphs, alone and alongside ConstantFold."""

import numpy as np

import tmlc
from tmlc.interpreters.graph_interpreter import GraphInterpreter
from tmlc.transforms import CSE, ConstantFold


def test_cse_collapses_repeated_subexpression():
    x = tmlc.input(shape=(512,), label="x")
    # x*x computed three times as independent Tensor objects: same subgraph, distinct nodes. CSE
    # collapses the duplicates, reducing the graph from 6 nodes to 4.
    x2_a, x2_b, x2_c = x * x, x * x, x * x
    graph = tmlc.Graph([x2_a + x2_b + x2_c])

    x_val = np.arange(512, dtype=float)
    before = GraphInterpreter().run(graph, inputs={x: x_val})[0]

    folded = graph.apply_transforms([CSE()])
    after = GraphInterpreter().run(folded, inputs={x: x_val})[0]

    np.testing.assert_allclose(before, after)
    assert len(graph.topo_sort) == 6
    assert len(folded.topo_sort) == 4


def test_cf_and_cse_compose_without_changing_result():
    x = tmlc.input(shape=(512,), label="x")
    a = tmlc.constant(2.0, label="a")
    b = tmlc.constant(3.0, label="b")

    # c1, c2 are equal constant subexpressions (ConstantFold target); s1, s2 are equal
    # input-dependent subexpressions (CSE target). p1 = c1*s1 and p2 = c2*s2 become equal only once
    # both passes have run.
    c1, c2 = a + b, a + b
    s1, s2 = x * x, x * x
    graph = tmlc.Graph([c1 * s1 + c2 * s2 + c1 * s1 + c2 * s2])

    x_val = np.arange(512, dtype=float)
    interpreter = GraphInterpreter()
    ref = interpreter.run(graph, inputs={x: x_val})[0]

    variants = {
        "raw": [],
        "CF": [ConstantFold()],
        "CSE": [CSE()],
        "CF then CSE": [ConstantFold(), CSE()],
        "CSE then CF": [CSE(), ConstantFold()],
    }
    node_counts = {}
    for name, passes in variants.items():
        transformed = graph.apply_transforms(passes)
        result = interpreter.run(transformed, inputs={x: x_val})[0]
        np.testing.assert_allclose(ref, result, err_msg=name)
        node_counts[name] = len(transformed.topo_sort)

    # Each pass can only shrink or preserve the graph; running both shrinks it the most.
    assert node_counts["CF then CSE"] < node_counts["raw"]
    assert node_counts["CF then CSE"] <= min(node_counts["CF"], node_counts["CSE"])
