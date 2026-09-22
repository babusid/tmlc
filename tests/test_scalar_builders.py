"""Typed ScalarExpr builders and their operator-sugar call paths."""

import pytest

from tmlc.compute import (
    ScalarAdd,
    ScalarConst,
    ScalarDiv,
    ScalarExp,
    ScalarExpr,
    ScalarLog,
    ScalarMax,
    ScalarMul,
    ScalarNeg,
    ScalarOpKind,
    ScalarPow,
    ScalarSub,
    ScalarTanh,
)

LHS = ScalarConst(2.0)
RHS = ScalarConst(3.0)


@pytest.mark.parametrize(
    "builder,kind",
    [
        (ScalarNeg, ScalarOpKind.NEG),
        (ScalarExp, ScalarOpKind.EXP),
        (ScalarLog, ScalarOpKind.LOG),
        (ScalarTanh, ScalarOpKind.TANH),
    ],
)
def test_unary_builders(builder, kind):
    expr = builder((LHS,))
    assert isinstance(expr, ScalarExpr)
    assert expr.kind is kind
    assert expr.args == (LHS,)


@pytest.mark.parametrize(
    "builder,kind",
    [
        (ScalarAdd, ScalarOpKind.ADD),
        (ScalarSub, ScalarOpKind.SUB),
        (ScalarMul, ScalarOpKind.MUL),
        (ScalarDiv, ScalarOpKind.DIV),
        (ScalarMax, ScalarOpKind.MAX),
        (ScalarPow, ScalarOpKind.POW),
    ],
)
def test_binary_builders(builder, kind):
    expr = builder((LHS, RHS))
    assert isinstance(expr, ScalarExpr)
    assert expr.kind is kind
    assert expr.args == (LHS, RHS)


def test_operator_sugar():
    assert LHS + RHS == ScalarAdd((LHS, RHS))
    assert LHS - RHS == ScalarSub((LHS, RHS))
    assert LHS * RHS == ScalarMul((LHS, RHS))
    assert LHS / RHS == ScalarDiv((LHS, RHS))
    assert LHS**RHS == ScalarPow((LHS, RHS))
    assert -LHS == ScalarNeg((LHS,))
    assert LHS.exp() == ScalarExp((LHS,))
    assert LHS.log() == ScalarLog((LHS,))
    assert LHS.tanh() == ScalarTanh((LHS,))
