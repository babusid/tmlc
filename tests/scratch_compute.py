import numpy as np
from tmlc.interpreters.compute_interpreter import ComputeInterpreter
from tmlc.compute import (
    ComputeProgramBuilder,
    ComputeTensor,
    Axis,
    AxisRef,
    Combiner,
)

# tensor sizes
a_extent = 128
b_extent = 256
c_extent = 512

# build a program
builder: ComputeProgramBuilder = ComputeProgramBuilder()
a: Axis = builder.spatial(a_extent, "a")
b: Axis = builder.reduce(b_extent, "b")
c: Axis = builder.reduce(c_extent, "c")
inp: ComputeTensor = builder.declare_input(
    shape=(a_extent, b_extent, c_extent), dtype="float32", hint="input"
)
inp2: ComputeTensor = builder.declare_input(shape=(a_extent,), dtype="float32", hint="input")
out: ComputeTensor = builder.compute(
    output_axes=(a,),
    body=inp[AxisRef(a), AxisRef(b), AxisRef(c)] * inp2[AxisRef(a)],
    reduce_axes=(b, c),
    hint="testing",
    combiner=Combiner.SUM,
)
prog = builder.finish((out,))

# run the program
interp = ComputeInterpreter()
inp_val = np.broadcast_to(
    (np.arange(b_extent)[np.newaxis, :, np.newaxis]), (a_extent, b_extent, c_extent)
)
inp2_val = np.arange(a_extent)
inputs = {inp: inp_val, inp2: inp2_val}
results = interp.run(prog, inputs)
print(results)
print(results[0].shape)
