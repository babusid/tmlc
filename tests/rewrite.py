import numpy as np
import tmlc
from tmlc.tsql import Var, Const, Op, Ref, EqualTo, match_pattern
from tmlc.tensor.ops.ops_arithmetic import Add, Mul


x = tmlc.input(shape=(2,), label="x")
out = x + 2 * 2
graph = tmlc.Graph(inputs=[x], outputs=[out])

out_val_pre_opt = graph.run(
    inputs={x: np.array([1,2])}
)
print(out_val_pre_opt)


