import numpy as np
import tmlc
from tmlc.tsql import Var, Const, Op, Ref, EqualTo, match_pattern
from tmlc.tensor.ops.ops_arithmetic import Add, Mul
from tmlc.transforms.ConstantFold import ConstantFold

x = tmlc.input(shape=(2,), label="x")
a = tmlc.constant(2, label="a")
b = tmlc.constant(4, label="b")
c = tmlc.constant(3, label="c")
out = a + b - c * 5 + 3 - 1
out = out * x
graph = tmlc.Graph([out])
x_val = np.array([1, 2])
out_val_pre_opt = graph.run(inputs={x: x_val})
print(f"Graph return value before opt:{out_val_pre_opt}")
print(f"Graph before optimization: {len(graph.topo_sort)}")
print([node.label for node in graph.topo_sort])

graph = graph.apply_transforms([ConstantFold()])
out_val_post_opt = graph.run(inputs={x: x_val})
print(f"Graph return value after opt:{out_val_post_opt}")
print(f"Graph after ConstantFold optimization: {len(graph.topo_sort)}")
print([node.label for node in graph.topo_sort])
