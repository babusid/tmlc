"""
Hand-build a Compute IR program and run it on the NumPy ComputeInterpreter.

The program sums inp[a, b, c] * inp2[a] over the two reduction axes (b, c), leaving a single spatial
axis a -- i.e. a weighted double reduction down to a vector of length `a_extent`.
"""

import numpy as np

from tmlc.compute import Axis, AxisRef, Combiner, ComputeProgramBuilder, ComputeTensor
from tmlc.interpreters.compute_interpreter import ComputeInterpreter

A_EXTENT = 128
B_EXTENT = 256
C_EXTENT = 512


def main() -> None:
    builder: ComputeProgramBuilder = ComputeProgramBuilder()
    a: Axis = builder.spatial(A_EXTENT, "a")
    b: Axis = builder.reduce(B_EXTENT, "b")
    c: Axis = builder.reduce(C_EXTENT, "c")
    inp: ComputeTensor = builder.declare_input(
        shape=(A_EXTENT, B_EXTENT, C_EXTENT), dtype="float32", hint="input"
    )
    inp2: ComputeTensor = builder.declare_input(shape=(A_EXTENT,), dtype="float32", hint="input")
    out: ComputeTensor = builder.compute(
        output_axes=(a,),
        body=inp[AxisRef(a), AxisRef(b), AxisRef(c)] * inp2[AxisRef(a)],
        reduce_axes=(b, c),
        hint="testing",
        combiner=Combiner.SUM,
    )
    program = builder.finish((out,))

    inp_val = np.broadcast_to(
        np.arange(B_EXTENT)[np.newaxis, :, np.newaxis], (A_EXTENT, B_EXTENT, C_EXTENT)
    )
    inp2_val = np.arange(A_EXTENT)
    (result,) = ComputeInterpreter().run(program, {inp: inp_val, inp2: inp2_val})
    print(result)
    print(result.shape)


if __name__ == "__main__":
    main()
