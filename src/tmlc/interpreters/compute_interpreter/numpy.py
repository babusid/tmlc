"""
NumPy reference interpreter for the Compute IR.

Executes a ComputeProgram block by block, honoring each block's single iteration domain, the affine
access map of every operand read, and the scalar body. Storage is dense ndarrays over the block's
LOGICAL shapes -- nothing is linearized and no particular layout is assumed -- matching the
abstraction level of this IR.

The executor is deliberately generic: it never pattern-matches a block back to a high-level op (no
np.matmul shortcut). A block is run by materializing its full iteration grid (spatial + reduce
axes), gathering each operand at its affine index arrays via advanced indexing, evaluating the body
over that grid, and folding the reduce axes away with the combiner. Cost is therefore dominated by
per-block grid materialization, so eliminating a block via fusion (pure reshape+broadcast chains
folded into index arithmetic, or an epilogue fused into a matmul) removes a real pass and shows up
as a speedup when an optimized ComputeProgram is run on this same interpreter.

It is a reference interpreter, not a scalar VM: within one block the scalar body is evaluated with
vectorized numpy, so it models materialization passes and flop/byte counts, not instruction-level
scheduling.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from tmlc.compute import (
    Axis,
    AxisKind,
    AxisRef,
    Combiner,
    CompareOp,
    ComputeBlock,
    ComputeProgram,
    ComputeTensor,
    IndexAdd,
    IndexCompare,
    IndexExpr,
    IndexFloorDiv,
    IndexMod,
    IndexMul,
    IntConst,
    Read,
    ScalarConst,
    ScalarExpr,
    ScalarExprBase,
    ScalarOpKind,
    Select,
)

# Scalar body ops. Keyed by ScalarOpKind so the evaluator looks up rather than branches; each entry
# takes the already-evaluated argument arrays (arity matches the op) and returns the result. numpy
# collapses 0-d array arithmetic to np.generic scalars, so rank-0 values flow through as those.
SCALAR_OPS: dict[ScalarOpKind, Callable[..., np.ndarray | np.generic]] = {
    ScalarOpKind.ADD: lambda a, b: a + b,
    ScalarOpKind.SUB: lambda a, b: a - b,
    ScalarOpKind.MUL: lambda a, b: a * b,
    ScalarOpKind.DIV: lambda a, b: a / b,
    ScalarOpKind.POW: lambda a, b: a**b,
    ScalarOpKind.MAX: np.maximum,
    ScalarOpKind.NEG: lambda a: -a,
    ScalarOpKind.EXP: np.exp,
    ScalarOpKind.LOG: np.log,
    ScalarOpKind.TANH: np.tanh,
}

# Comparison ops for IndexCompare. Each returns a boolean array; the caller casts to intp so the
# 0/1 result is an ordinary integer index like every other IndexExpr.
COMPARE_OPS: dict[CompareOp, Callable[..., np.ndarray | np.generic]] = {
    CompareOp.LT: np.less,
    CompareOp.LE: np.less_equal,
    CompareOp.GT: np.greater,
    CompareOp.GE: np.greater_equal,
    CompareOp.EQ: np.equal,
    CompareOp.NE: np.not_equal,
}

# Reduction ufunc per combiner. `ufunc.reduce` folds the reduce axes; every extent is >= 1 so the
# combiner identity is never observed here (it matters only for empty reductions the IR can't emit).
COMBINER_REDUCE: dict[Combiner, np.ufunc] = {
    Combiner.SUM: np.add,
    Combiner.PROD: np.multiply,
    Combiner.MAX: np.maximum,
}


def _eval_index(expr: IndexExpr, coords: dict[Axis, np.ndarray]) -> np.ndarray:
    """
    Evaluate an affine index expression to an integer ndarray broadcasting over the iteration grid.

    `coords` maps each domain axis to its coordinate vector, reshaped so the axis occupies its own
    grid dimension; the affine arithmetic then broadcasts to exactly the dimensions it references.
    Results are wrapped with asarray since numpy collapses 0-d array arithmetic to scalars.
    """
    if isinstance(expr, AxisRef):
        return coords[expr.axis]
    if isinstance(expr, IntConst):
        return np.asarray(expr.value, dtype=np.intp)
    if isinstance(expr, IndexAdd):
        return np.asarray(_eval_index(expr.lhs, coords) + _eval_index(expr.rhs, coords))
    if isinstance(expr, IndexMul):
        return np.asarray(_eval_index(expr.lhs, coords) * _eval_index(expr.rhs, coords))
    if isinstance(expr, IndexFloorDiv):
        return np.asarray(_eval_index(expr.lhs, coords) // expr.divisor)
    if isinstance(expr, IndexMod):
        return np.asarray(_eval_index(expr.lhs, coords) % expr.modulus)
    if isinstance(expr, IndexCompare):
        lhs = _eval_index(expr.lhs, coords)
        rhs = _eval_index(expr.rhs, coords)
        return np.asarray(COMPARE_OPS[expr.op](lhs, rhs), dtype=np.intp)
    raise TypeError(f"unknown IndexExpr: {type(expr).__name__}")


def _eval_body(
    expr: ScalarExprBase,
    coords: dict[Axis, np.ndarray],
    storage: dict[ComputeTensor, np.ndarray],
) -> np.ndarray:
    """
    Evaluate a scalar body expression over the iteration grid.

    A Read gathers its operand through the operand's affine index arrays (pure advanced indexing, so
    broadcast reads and transposes fall out for free); a ScalarExpr applies its op to its args; a
    Select evaluates both branches over the grid and picks per position by its index condition.
    Results are wrapped with asarray because numpy collapses 0-d array arithmetic to scalars.
    """
    if isinstance(expr, ScalarConst):
        return np.asarray(expr.value)
    if isinstance(expr, Read):
        array = storage[expr.tensor]
        index = tuple(_eval_index(coordinate, coords) for coordinate in expr.index)
        return np.asarray(array[index])
    if isinstance(expr, Select):
        cond = _eval_index(expr.cond, coords)
        if_true = _eval_body(expr.if_true, coords, storage)
        if_false = _eval_body(expr.if_false, coords, storage)
        return np.asarray(np.where(cond != 0, if_true, if_false))
    if isinstance(expr, ScalarExpr):
        args = tuple(_eval_body(arg, coords, storage) for arg in expr.args)
        return np.asarray(SCALAR_OPS[expr.kind](*args))
    raise TypeError(f"unknown ScalarExprBase: {type(expr).__name__}")


def _run_block(block: ComputeBlock, storage: dict[ComputeTensor, np.ndarray]) -> np.ndarray:
    """
    Execute one block: build its iteration grid, evaluate the body, then fold reduce axes away.

    The body is broadcast to the full domain extents before reduction so a combiner over an axis the
    body does not depend on still folds the correct number of terms. Spatial axes survive in domain
    order, which is the block output's identity write map.
    """
    domain = block.domain
    extents = tuple(axis.extent for axis in domain)
    coords: dict[Axis, np.ndarray] = {}
    for position, axis in enumerate(domain):
        shape = [1] * len(domain)
        shape[position] = axis.extent
        coords[axis] = np.arange(axis.extent, dtype=np.intp).reshape(shape)

    body = np.broadcast_to(_eval_body(block.body, coords, storage), extents)

    reduce_positions = tuple(
        position for position, axis in enumerate(domain) if axis.kind is AxisKind.REDUCE
    )
    if reduce_positions:
        if block.combiner is None:
            raise ValueError(f"block '{block.output.name}' has reduce axes but no combiner")
        # reduce over every reduce axis at once; a full reduction yields a numpy scalar, so asarray
        # below restores a (0-d) ndarray.
        result = COMBINER_REDUCE[block.combiner].reduce(body, axis=reduce_positions)
    else:
        result = np.array(body)  # own the buffer; broadcast_to returns a read-only view

    return np.asarray(result, dtype=block.output.dtype)


class ComputeInterpreter:
    """
    Runs a ComputeProgram on dense ndarray storage, one block at a time in dependency order.
    """

    def run(
        self,
        program: ComputeProgram,
        inputs: Mapping[ComputeTensor, np.ndarray],
    ) -> list[np.ndarray]:
        storage: dict[ComputeTensor, np.ndarray] = {}

        for tensor, value in program.constants:
            storage[tensor] = np.asarray(value, dtype=tensor.dtype).reshape(tensor.shape)

        for tensor in program.inputs:
            if tensor not in inputs:
                raise RuntimeError(f"input '{tensor.name}' (shape={tensor.shape}) was not provided")
            provided = np.asarray(inputs[tensor], dtype=tensor.dtype)
            if provided.shape != tensor.shape:
                raise ValueError(
                    f"input '{tensor.name}' expected shape {tensor.shape}, got {provided.shape}"
                )
            storage[tensor] = provided

        for block in program.blocks:
            storage[block.output] = _run_block(block, storage)

        return [storage[tensor] for tensor in program.outputs]
