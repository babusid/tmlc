import time
import numpy as np
import tmlc
from tmlc.tsql import Var, Const, Op, Ref, EqualTo, match_pattern
from tmlc.tensor.ops.ops_arithmetic import Add, Mul
from tmlc.transforms.ConstantFold import ConstantFold

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
start = time.time()
out_val_pre_opt = graph.run(inputs={x: x_val})
end = time.time() - start
print(f"Graph return value before opt:{out_val_pre_opt}")
print(f"Graph before optimization: {len(graph.topo_sort)}")
print(f"Graph execution time before optimization: {end:.6f} seconds")
print([node.label for node in graph.topo_sort])
print()

comp_start = time.time()
graph = graph.apply_transforms([ConstantFold()])
comp_end = time.time() - comp_start
start = time.time()
out_val_post_opt = graph.run(inputs={x: x_val})
end = time.time() - start
print(f"Graph return value after opt:{out_val_post_opt}")
print(f"Graph after ConstantFold optimization: {len(graph.topo_sort)}")
print(f"Graph execution time after optimization: {end:.6f} seconds")
print(f"Graph optimization time: {comp_end:.6f} seconds")
print([node.label for node in graph.topo_sort])
