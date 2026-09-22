"""
Hand-build and compile an RMSNorm -> projection -> activation pipeline.

The example is deliberately written as small model-like helpers rather than one large scalar
expression.  ``InlineReductionFree`` removes the temporary square, normalized, and biased tensors,
then the C backend saves its inspectable build products under ``examples/artifacts/``.

This also documents a current boundary of the pass: reductions are never inlined and an output is
always materialized.  Consequently the exact pipeline has three surviving Compute blocks (RMS
reduction, matmul reduction, and tanh output), not one.  Reaching one block requires Loop IR fusion
with a post-reduction expression, which the current IR does not yet represent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import time

from tmlc.backends.c import CProgram
from tmlc.compute import AxisRef, Combiner, ComputeProgram, ComputeProgramBuilder, ComputeTensor
from tmlc.compute.transforms import InlineReductionFree
from tmlc.loop import lower_to_loops

BATCH = 4
HIDDEN = 8
OUTPUT = 6
EPSILON = 1e-5
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "rmsnorm_projection_activation"


@dataclass(frozen=True)
class ModelInputs:
    activations: ComputeTensor
    norm_weight: ComputeTensor
    projection_weight: ComputeTensor
    bias: ComputeTensor


def declare_inputs(builder: ComputeProgramBuilder) -> ModelInputs:
    """Declare the buffers a handwritten fused model layer would receive."""
    return ModelInputs(
        activations=builder.declare_input((BATCH, HIDDEN), "float32", hint="activations"),
        norm_weight=builder.declare_input((HIDDEN,), "float32", hint="norm_weight"),
        projection_weight=builder.declare_input(
            (HIDDEN, OUTPUT), "float32", hint="projection_weight"
        ),
        bias=builder.declare_input((OUTPUT,), "float32", hint="bias"),
    )


def rmsnorm(
    builder: ComputeProgramBuilder, activations: ComputeTensor, weight: ComputeTensor
) -> ComputeTensor:
    """RMS-normalize each row and apply the learned per-channel weight."""
    row = builder.spatial(BATCH, "rms_row")
    channel = builder.spatial(HIDDEN, "rms_channel")
    square = builder.compute(
        output_axes=(row, channel),
        body=activations[AxisRef(row), AxisRef(channel)] ** 2.0,
        hint="square",
    )

    mean_row = builder.spatial(BATCH, "mean_row")
    reduced_channel = builder.reduce(HIDDEN, "mean_channel")
    sum_of_squares = builder.compute(
        output_axes=(mean_row,),
        body=square[AxisRef(mean_row), AxisRef(reduced_channel)],
        reduce_axes=(reduced_channel,),
        combiner=Combiner.SUM,
        hint="sum_of_squares",
    )

    normalized_row = builder.spatial(BATCH, "normalized_row")
    normalized_channel = builder.spatial(HIDDEN, "normalized_channel")
    row_ref = AxisRef(normalized_row)
    channel_ref = AxisRef(normalized_channel)
    inverse_rms = ((sum_of_squares[row_ref] / HIDDEN) + EPSILON) ** -0.5
    return builder.compute(
        output_axes=(normalized_row, normalized_channel),
        body=activations[row_ref, channel_ref] * inverse_rms * weight[channel_ref],
        hint="normalized",
    )


def project(
    builder: ComputeProgramBuilder, source: ComputeTensor, weight: ComputeTensor
) -> ComputeTensor:
    """Apply a dense projection over the hidden dimension."""
    row = builder.spatial(BATCH, "projection_row")
    column = builder.spatial(OUTPUT, "projection_column")
    channel = builder.reduce(HIDDEN, "projection_channel")
    return builder.compute(
        output_axes=(row, column),
        body=source[AxisRef(row), AxisRef(channel)] * weight[AxisRef(channel), AxisRef(column)],
        reduce_axes=(channel,),
        combiner=Combiner.SUM,
        hint="projection",
    )


def activation_epilogue(
    builder: ComputeProgramBuilder, projection: ComputeTensor, bias: ComputeTensor
) -> ComputeTensor:
    """Add bias and apply tanh as two ordinary reduction-free epilogue operations."""
    bias_row = builder.spatial(BATCH, "bias_row")
    bias_column = builder.spatial(OUTPUT, "bias_column")
    biased = builder.compute(
        output_axes=(bias_row, bias_column),
        body=projection[AxisRef(bias_row), AxisRef(bias_column)] + bias[AxisRef(bias_column)],
        hint="biased",
    )

    output_row = builder.spatial(BATCH, "output_row")
    output_column = builder.spatial(OUTPUT, "output_column")
    return builder.compute(
        output_axes=(output_row, output_column),
        body=biased[AxisRef(output_row), AxisRef(output_column)].tanh(),
        hint="activated",
    )


def build_program() -> tuple[ComputeProgram, ModelInputs]:
    """Assemble the layer while keeping its three logical pieces independent."""
    builder = ComputeProgramBuilder()
    inputs = declare_inputs(builder)
    normalized = rmsnorm(builder, inputs.activations, inputs.norm_weight)
    projected = project(builder, normalized, inputs.projection_weight)
    output = activation_epilogue(builder, projected, inputs.bias)
    return builder.finish((output,)), inputs


def main(save_dir: Path = ARTIFACT_DIR) -> None:
    unfused, _ = build_program()
    fused = InlineReductionFree()(unfused)
    loop_program = lower_to_loops(fused)

    assert len(unfused.blocks) == 6
    assert len(fused.blocks) == 3
    assert len(loop_program.body) == 3

    compiled = CProgram(loop_program, save_dir=save_dir)
    kernel_path = save_dir / "kernel.c"
    assert kernel_path.read_text() == compiled.source
    # Three materialized results plus initialization of the two explicit reduction buffers.
    assert compiled.source.count("/* store ") == 5

    rng = np.random.default_rng(0)
    activations = rng.standard_normal((BATCH, HIDDEN), dtype=np.float32)
    norm_weight = rng.standard_normal(HIDDEN, dtype=np.float32)
    projection_weight = rng.standard_normal((HIDDEN, OUTPUT), dtype=np.float32)
    bias = rng.standard_normal(OUTPUT, dtype=np.float32)

    compiled_time = time.perf_counter()
    loop_inputs = dict(
        zip(
            loop_program.inputs,
            (activations, norm_weight, projection_weight, bias),
        )
    )
    (actual,) = compiled.run(loop_inputs)
    compiled_time = time.perf_counter() - compiled_time
    numpy_time = time.perf_counter()
    inverse_rms = np.power(np.mean(activations * activations, axis=1) + EPSILON, -0.5)
    normalized = activations * inverse_rms[:, None] * norm_weight
    expected = np.tanh(normalized @ projection_weight + bias)
    numpy_time = time.perf_counter() - numpy_time
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    print(f"Compute blocks: {len(unfused.blocks)} before inlining, {len(fused.blocks)} after")
    print(f"C stores:       {compiled.source.count('/* store ')}")
    print(f"emitted C:      {kernel_path}")
    print(f"compiled time:  {compiled_time:.6f} s")
    print(f"numpy time:     {numpy_time:.6f} s")
    print(f"speedup:        {numpy_time / compiled_time:.2f}x")


if __name__ == "__main__":
    main()
