# tmlc — a tiny machine learning compiler

`tmlc` is a small, from-scratch ML compiler/autograd system. The goal is learning, not
performance or production use: build up a tensor compiler the way real ones work — IR, eager
execution, reverse-mode autodiff, graph optimization passes, and (eventually) lowering to
MLIR — one understandable layer at a time, without hiding the mechanics behind a library.

If you're modifying this repo, also read `TODO.md` for what's planned but not built yet.

## Quickstart

```python
import numpy as np
import tmlc

x = tmlc.input(shape=(2, 2), label="x")
y = tmlc.input(shape=(2, 2), label="y")
z = x * y + 1

output = tmlc.run(inputs={x: np.array([[1, 2], [3, 4]]), y: np.array([[5, 6], [7, 8]])}, outputs=[z])

grad_x, = tmlc.gradients(output_node=z, target_nodes=[x])
```

`import tmlc` is the only import you need — it re-exports the op access functions (`add`,
`mul`, `input`, `transpose`, ...), the `TensorOp` subclasses themselves (`Add`, `Mul`, `Input`,
...) for type hints / `isinstance` checks, and `run`/`gradients` from the evaluator. The package
also runs under `beartype` (`beartype_this_package()` in `__init__.py`), so type hints
throughout `tmlc` are enforced at runtime, not just checked statically.

## How it's put together

- **`tensor.py`** — the IR. `Tensor` is a graph node (op + inputs + shape, no data). `TensorOp`
  is the abstract interface every operation implements: `__call__` (build the node),
  `infer_shape`, `compute` (eager numpy execution), `gradients` (local backward rule),
  `emit_ir` (MLIR lowering — currently a stub everywhere, see `TODO.md`).
- **`ops/ops_*.py`** — one file per category (`ops_basic`, `ops_arithmetic`, `ops_logarithmic`,
  `ops_shape`). Each op is a `TensorOp` subclass plus a lowercase access function at the bottom
  (e.g. `Mul` → `mul()`). The access function is the real public API — `TensorOp`s aren't meant
  to be constructed by hand.
- **`_operators.py`** — attaches `Tensor`'s dunders (`__add__`, `__matmul__`, `.T`, ...) onto the
  class after the fact. This exists to break a real circular dependency: ops need `Tensor` as a
  base class, and operators need ops. `tensor.py` stays ops-agnostic; this module wires the two
  together once, at import time. See the comment above the `Tensor` class for the long version.
- **`evaluator.py`** — `run()` does eager evaluation: topo-sort from the requested outputs,
  execute each node's `compute()`. `gradients()` does reverse-mode autodiff: it prunes the graph
  to just the nodes on the path between the output and the requested targets, then walks it in
  reverse, accumulating and pushing gradients via each node's `op.gradients()`. `compile()` /
  `run_compiled()` are unimplemented stubs — this is where graph-optimization passes and MLIR
  codegen will eventually live.
- **`ndarray.py`** — just a numpy alias today; the seam where a pluggable array backend
  (numpy/cupy/etc.) would go.

## What works right now

- Building a computational graph with broadcasting-aware arithmetic, matmul, exp/log/tanh,
  logsumexp, reshape/transpose/summation, and constants.
- Eager evaluation of any subset of the graph (`tmlc.run`), only computing the nodes actually
  needed for the requested outputs.
- Reverse-mode autodiff (`tmlc.gradients`) with the same "only touch what's needed" pruning.
- A logistic-regression test case (`tests/logreg.py`) that trains end-to-end and checks
  gradients against finite differences — the best example of a non-trivial graph working.

## What's not built yet

MLIR emission, an actual `compile()`/`run_compiled()` path, graph optimization passes (dead
code elimination, constant folding, CSE, op fusion), a real test suite (pytest isn't wired up —
`tests/*.py` are runnable scripts, not pytest files), and a pluggable array backend. See
`TODO.md` for the live list.
