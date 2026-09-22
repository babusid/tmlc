"""
Substitution utilities over Compute IR index and scalar-expression trees.

These are the mechanical building blocks that every structural rewrite of the Compute IR needs:

* ``subst_index`` rewrites the axes referenced by an affine index expression.
* ``subst_body`` rewrites the axes referenced anywhere in a scalar body (reads and select conds).
* ``replace_reads`` swaps whole ``Read`` leaves for arbitrary sub-expressions.

They are deliberately pure (tree in, new tree out) and hold no policy: fusion decides *what* to
substitute, these decide *how*. Keeping them here means the fusion pass -- and, later, the Loop IR
lowering -- share one correct implementation of index composition instead of each reinventing it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from tmlc.compute.axis import Axis
from tmlc.compute.index import (
    AxisRef,
    IndexAdd,
    IndexCompare,
    IndexExpr,
    IndexFloorDiv,
    IndexMod,
    IndexMul,
    IntConst,
)
from tmlc.compute.scalar import (
    Read,
    ScalarConst,
    ScalarExpr,
    ScalarExprBase,
    Select,
)
from tmlc.compute.scalar.base import _scalar_expr


def subst_index(expr: IndexExpr, mapping: Mapping[Axis, IndexExpr]) -> IndexExpr:
    """
    Return ``expr`` with each ``AxisRef(a)`` replaced by ``mapping[a]`` for ``a`` in ``mapping``.

    This is affine-map composition: if ``expr`` addresses a producer's element in terms of the
    producer's axes, and ``mapping`` sends those axes to the consumer's coordinate expressions, the
    result addresses the same element in terms of the consumer's axes. Axes absent from ``mapping``
    are left untouched.
    """
    if isinstance(expr, AxisRef):
        return mapping.get(expr.axis, expr)
    if isinstance(expr, IntConst):
        return expr
    if isinstance(expr, IndexAdd):
        return IndexAdd(subst_index(expr.lhs, mapping), subst_index(expr.rhs, mapping))
    if isinstance(expr, IndexMul):
        return IndexMul(subst_index(expr.lhs, mapping), subst_index(expr.rhs, mapping))
    if isinstance(expr, IndexFloorDiv):
        return IndexFloorDiv(subst_index(expr.lhs, mapping), expr.divisor)
    if isinstance(expr, IndexMod):
        return IndexMod(subst_index(expr.lhs, mapping), expr.modulus)
    if isinstance(expr, IndexCompare):
        return IndexCompare(expr.op, subst_index(expr.lhs, mapping), subst_index(expr.rhs, mapping))
    raise TypeError(f"unknown IndexExpr: {type(expr).__name__}")


def subst_body(body: ScalarExprBase, mapping: Mapping[Axis, IndexExpr]) -> ScalarExprBase:
    """Return ``body`` with ``subst_index(_, mapping)`` applied to every index it contains."""
    if isinstance(body, ScalarConst):
        return body
    if isinstance(body, Read):
        return Read(body.tensor, tuple(subst_index(c, mapping) for c in body.index))
    if isinstance(body, Select):
        return Select(
            subst_index(body.cond, mapping),
            subst_body(body.if_true, mapping),
            subst_body(body.if_false, mapping),
        )
    if isinstance(body, ScalarExpr):
        return _scalar_expr(body.kind, tuple(subst_body(arg, mapping) for arg in body.args))
    raise TypeError(f"unknown ScalarExprBase: {type(body).__name__}")


def replace_reads(
    body: ScalarExprBase,
    replace: Callable[[Read], ScalarExprBase | None],
) -> ScalarExprBase:
    """
    Rebuild ``body``, swapping each ``Read`` for ``replace(read)`` when that returns non-None.

    The replacement sub-expression is inserted verbatim and is *not* re-walked, so a substitution
    that itself contains reads (e.g. an inlined producer body) will not be recursively rewritten --
    callers are expected to hand in already-final sub-expressions.
    """
    if isinstance(body, ScalarConst):
        return body
    if isinstance(body, Read):
        replacement = replace(body)
        return body if replacement is None else replacement
    if isinstance(body, Select):
        return Select(
            body.cond,
            replace_reads(body.if_true, replace),
            replace_reads(body.if_false, replace),
        )
    if isinstance(body, ScalarExpr):
        return _scalar_expr(body.kind, tuple(replace_reads(arg, replace) for arg in body.args))
    raise TypeError(f"unknown ScalarExprBase: {type(body).__name__}")


def iter_reads(body: ScalarExprBase) -> list[Read]:
    """Collect every ``Read`` leaf in ``body`` (depth-first, left-to-right, with duplicates)."""
    if isinstance(body, ScalarConst):
        return []
    if isinstance(body, Read):
        return [body]
    if isinstance(body, Select):
        return iter_reads(body.if_true) + iter_reads(body.if_false)
    if isinstance(body, ScalarExpr):
        reads: list[Read] = []
        for arg in body.args:
            reads.extend(iter_reads(arg))
        return reads
    raise TypeError(f"unknown ScalarExprBase: {type(body).__name__}")
