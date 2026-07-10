import time
import numpy as np
import tmlc
from tmlc.transforms import ConstantFold

N = 1000

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

out_val_pre = graph.run(inputs={x: x_val})  # warm up allocator and ufunc dispatch cache
start = time.time()
for _ in range(N):
    graph.run(inputs={x: x_val})
pre_time = (time.time() - start) / N

print(f"Graph value before opt: {out_val_pre[0]}")
print(f"Graph before optimization: {len(graph.topo_sort)} nodes")
print(f"Graph execution time before optimization: {pre_time * 1e6:.2f} µs  (mean of {N})")
print([node.label for node in graph.topo_sort])
print()

comp_start = time.time()
graph = graph.apply_transforms([ConstantFold()])
comp_time = time.time() - comp_start

out_val_post = graph.run(inputs={x: x_val})  # warm up after transform
start = time.time()
for _ in range(N):
    graph.run(inputs={x: x_val})
post_time = (time.time() - start) / N

print(f"Graph value after opt: {out_val_post[0]}")
print(f"Graph after ConstantFold optimization: {len(graph.topo_sort)} nodes")
print(f"Graph execution time after optimization: {post_time * 1e6:.2f} µs  (mean of {N})")
print(f"Graph optimization time: {comp_time * 1e6:.2f} µs")
print([node.label for node in graph.topo_sort])

assert np.allclose(out_val_pre[0], out_val_post[0]), "ConstantFold changed the output value"
