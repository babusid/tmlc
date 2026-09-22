"""Explicit output/reduction axes and singleton reduction dimensions in Compute IR."""

import numpy as np
import pytest

from tmlc.compute import AxisRef, Combiner, ComputeProgramBuilder, VerifyError
from tmlc.interpreters.compute_interpreter import ComputeInterpreter

VALUES = np.arange(24, dtype=np.float32).reshape(2, 3, 4)


def _builder_with_axes():
    builder = ComputeProgramBuilder()
    a = builder.spatial(2, "a")
    b = builder.reduce(3, "b")
    c = builder.spatial(4, "c")
    inp = builder.declare_input((2, 3, 4), "float32", hint="input")
    return builder, a, b, c, inp


def test_sum_keeps_reduced_dimension():
    builder, a, b, c, inp = _builder_with_axes()
    out = builder.compute(
        output_axes=(a, b, c),
        reduce_axes=(b,),
        body=inp[AxisRef(a), AxisRef(b), AxisRef(c)],
        combiner=Combiner.SUM,
        hint="sum_keepdims",
    )
    assert out.shape == (2, 1, 4)

    program = builder.finish((out,))
    (actual,) = ComputeInterpreter().run(program, {inp: VALUES})
    np.testing.assert_array_equal(actual, VALUES.sum(axis=1, keepdims=True))


def test_reduce_axis_absent_from_body_repeats_scalar():
    # A reduce axis missing from the body contributes the same scalar once per reduction coordinate.
    builder, a, b, c, inp = _builder_with_axes()
    repeated = builder.compute(
        output_axes=(a,),
        reduce_axes=(b,),
        body=inp[AxisRef(a), 0, 0],
        combiner=Combiner.SUM,
        hint="repeated",
    )
    program = builder.finish((repeated,))
    (actual,) = ComputeInterpreter().run(program, {inp: VALUES})
    np.testing.assert_array_equal(actual, VALUES[:, 0, 0] * 3)


def test_unquantified_reduce_output_axis_rejected():
    builder, a, b, c, inp = _builder_with_axes()
    with pytest.raises(VerifyError):
        builder.compute(output_axes=(b,), body=inp[0, AxisRef(b), 0], hint="unquantified_reduce")


def test_spatial_output_axis_cannot_be_reduced():
    builder, a, b, c, inp = _builder_with_axes()
    with pytest.raises(VerifyError):
        builder.compute(
            output_axes=(a,),
            reduce_axes=(a,),
            body=inp[AxisRef(a), 0, 0],
            combiner=Combiner.SUM,
            hint="reduced_spatial_output",
        )
