# Examples

Runnable programs that demonstrate tmlc end to end. Each is a standalone script with a `main()`:

```bash
uv run python examples/logreg.py
```

| Example | What it shows |
| --- | --- |
| `autodiff.py` | Forward evaluation and reverse-mode autodiff over a small graph |
| `compute_reduction.py` | Hand-building a Compute IR program and running it on the interpreter |
| `fusion_benchmark.py` | Elementwise fusion and shape/index collapsing in Compute IR (with timings) |
| `logreg.py` | Multi-class logistic regression: autodiff-vs-finite-difference check, then SGD training |
| `logreg_compiled.py` | The same model through three backends (graph, compute, compiled C) with timings |
| `rmsnorm_projection_activation.py` | Handwritten RMSNorm -> projection -> tanh Compute IR, fusion, and saved C artifacts |

These double as regression tests: `tests/test_examples.py` runs each `main()`, and the programs
self-check their own numerics. The compiled-C example is marked `c_backend`.
