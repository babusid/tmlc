import time
import numpy as np
import tmlc
from tmlc.graph import GraphTransform
from tmlc.transforms import ConstantFold, CSE

N = 1000

x = tmlc.input(shape=(512,), label="x")
a = tmlc.constant(2.0, label="a")
b = tmlc.constant(3.0, label="b")

# c1, c2: structurally equal constant subexpressions — Constant folding should fold these into
# two constants, and CSE should then remove one of the resulting two constant nodes
c1 = a + b
c2 = a + b

# s1, s2: structurally equal input-dependent subexpressions — CSE target
s1 = x * x
s2 = x * x

# p1 = c1*s1 and p2 = c2*s2 are equal only after both passes run
# (CF sees c1≡c2 as foldable; CSE sees s1≡s2 and then p1≡p2)
out = c1 * s1 + c2 * s2 + c1 * s1 + c2 * s2
base_graph = tmlc.Graph([out])

variants: dict[str, list[GraphTransform]] = {
    "raw": [],
    "CF": [ConstantFold()],
    "CSE": [CSE()],
    "CF → CSE": [ConstantFold(), CSE()],
    "CSE → CF": [CSE(), ConstantFold()],
}

graphs = {name: base_graph.apply_transforms(passes) for name, passes in variants.items()}

x_val = np.arange(512, dtype=float)

# Correctness check before timing
ref = graphs["raw"].run(inputs={x: x_val})[0]
for name, graph in graphs.items():
    result = graph.run(inputs={x: x_val})[0]
    assert np.allclose(ref, result), f"{name} produced wrong output"

# Timing: warm up then N iterations
results = {}
for name, graph in graphs.items():
    graph.run(inputs={x: x_val})
    start = time.time()
    for _ in range(N):
        graph.run(inputs={x: x_val})
    results[name] = {
        "time_us": (time.time() - start) / N * 1e6,
        "nodes": len(graph.topo_sort),
        "labels": [node.label for node in graph.topo_sort],
    }

# Print
raw_time = results["raw"]["time_us"]
name_w = max(len(n) for n in results) + 2
print(f"{'Variant':<{name_w}} {'Nodes':>6}  {'µs/run':>8}  {'Speedup':>8}")
print("-" * (name_w + 30))
for name, r in results.items():
    speedup = raw_time / r["time_us"]
    print(f"{name:<{name_w}} {r['nodes']:>6}  {r['time_us']:>8.2f}  {speedup:>7.2f}x")
print()
for name, r in results.items():
    labels = [label.split("=")[0] + ("=…" if "=" in label else "") for label in r["labels"]]
    print(f"{name}: {labels}")
