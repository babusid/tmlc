"""Exercise explicit output/reduction axes and singleton reduction dimensions."""

import numpy as np

from tmlc.compute import AxisRef, Combiner, ComputeProgramBuilder, VerifyError
from tmlc.interpreters.compute_interpreter import ComputeInterpreter


def main() -> None:
    builder = ComputeProgramBuilder()
    a = builder.spatial(2, "a")
    b = builder.reduce(3, "b")
    c = builder.spatial(4, "c")
    inp = builder.declare_input((2, 3, 4), "float32", hint="input")

    out = builder.compute(
        output_axes=(a, b, c),
        reduce_axes=(b,),
        body=inp[AxisRef(a), AxisRef(b), AxisRef(c)],
        combiner=Combiner.SUM,
        hint="sum_keepdims",
    )
    assert out.shape == (2, 1, 4)

    program = builder.finish((out,))
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    (actual,) = ComputeInterpreter().run(program, {inp: values})
    np.testing.assert_array_equal(actual, values.sum(axis=1, keepdims=True))

    # A reduce axis absent from the body contributes the same scalar once per reduction coordinate.
    repeated = builder.compute(
        output_axes=(a,),
        reduce_axes=(b,),
        body=inp[AxisRef(a), 0, 0],
        combiner=Combiner.SUM,
        hint="repeated",
    )
    repeated_program = builder.finish((repeated,))
    (repeated_actual,) = ComputeInterpreter().run(repeated_program, {inp: values})
    np.testing.assert_array_equal(repeated_actual, values[:, 0, 0] * 3)

    try:
        builder.compute(output_axes=(b,), body=inp[0, AxisRef(b), 0], hint="unquantified_reduce")
    except VerifyError:
        pass
    else:
        raise AssertionError("a reduced output axis must be explicitly quantified")

    try:
        builder.compute(
            output_axes=(a,),
            reduce_axes=(a,),
            body=inp[AxisRef(a), 0, 0],
            combiner=Combiner.SUM,
            hint="reduced_spatial_output",
        )
    except VerifyError:
        pass
    else:
        raise AssertionError("a spatial output axis cannot be reduced")


if __name__ == "__main__":
    main()
