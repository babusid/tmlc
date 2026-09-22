"""Shared helpers for the tmlc test suite."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import tmlc
from tmlc import Input
from tmlc.compute import ComputeProgram, ComputeTensor


def graph_input_bindings(
    graph: tmlc.Graph,
    program: ComputeProgram,
    values: Mapping[tmlc.Tensor, np.ndarray],
) -> dict[ComputeTensor, np.ndarray]:
    """
    Map each program input to its ndarray value.

    Lowering declares one input per Input node in graph topological order, so the graph's Input
    nodes in that order line up one-to-one with `program.inputs`.
    """
    graph_inputs = [node for node in graph.topo_sort if isinstance(node.op, Input)]
    assert len(graph_inputs) == len(program.inputs)
    return {tensor: values[node] for node, tensor in zip(graph_inputs, program.inputs)}
