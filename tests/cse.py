import time
import numpy as np
import tmlc
from tmlc.transforms import CSE

N = 1000

x = tmlc.input(shape=(512,), label="x")

# x*x computed three times as independent Tensor objects — same subgraph, distinct nodes.
# CSE should collapse x2_b and x2_c into x2_a, reducing the graph from 6 nodes to 4.
x2_a = x * x
x2_b = x * x
x2_c = x * x
out = x2_a + x2_b + x2_c
graph = tmlc.Graph([out])

x_val = np.arange(512, dtype=float)

out_val_pre = graph.run(inputs={x: x_val})  # warm up allocator and ufunc dispatch cache
start = time.time()
for _ in range(N):
    _ = graph.run(inputs={x: x_val})
pre_time = (time.time() - start) / N

print(f"Graph value before opt (first 4): {out_val_pre[0][:4]}")
print(f"Graph before CSE: {len(graph.topo_sort)} nodes")
print(f"Graph execution time before optimization: {pre_time * 1e6:.2f} µs  (mean of {N})")
print([node.label for node in graph.topo_sort])
print()

comp_start = time.time()
graph = graph.apply_transforms([CSE()])
comp_time = time.time() - comp_start

out_val_post = graph.run(inputs={x: x_val})  # warm up after transform
start = time.time()
for _ in range(N):
    _ = graph.run(inputs={x: x_val})
post_time = (time.time() - start) / N

print(f"Graph value after opt (first 4): {out_val_post[0][:4]}")
print(f"Graph after CSE: {len(graph.topo_sort)} nodes")
print(f"Graph execution time after optimization: {post_time * 1e6:.2f} µs  (mean of {N})")
print(f"Graph optimization time: {comp_time * 1e6:.2f} µs")
print([node.label for node in graph.topo_sort])

assert np.allclose(out_val_pre[0], out_val_post[0]), "CSE changed the output value"
