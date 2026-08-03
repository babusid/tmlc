"""Checks for typed ScalarExpr builders and their operator-sugar call paths."""

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


def main() -> None:
    lhs = ScalarConst(2.0)
    rhs = ScalarConst(3.0)

    unary_builders = (
        (ScalarNeg, ScalarOpKind.NEG),
        (ScalarExp, ScalarOpKind.EXP),
        (ScalarLog, ScalarOpKind.LOG),
        (ScalarTanh, ScalarOpKind.TANH),
    )
    for builder, kind in unary_builders:
        expr = builder((lhs,))
        assert isinstance(expr, ScalarExpr)
        assert expr.kind is kind
        assert expr.args == (lhs,)

    binary_builders = (
        (ScalarAdd, ScalarOpKind.ADD),
        (ScalarSub, ScalarOpKind.SUB),
        (ScalarMul, ScalarOpKind.MUL),
        (ScalarDiv, ScalarOpKind.DIV),
        (ScalarMax, ScalarOpKind.MAX),
        (ScalarPow, ScalarOpKind.POW),
    )
    for builder, kind in binary_builders:
        expr = builder((lhs, rhs))
        assert isinstance(expr, ScalarExpr)
        assert expr.kind is kind
        assert expr.args == (lhs, rhs)

    assert lhs + rhs == ScalarAdd((lhs, rhs))
    assert lhs - rhs == ScalarSub((lhs, rhs))
    assert lhs * rhs == ScalarMul((lhs, rhs))
    assert lhs / rhs == ScalarDiv((lhs, rhs))
    assert lhs**rhs == ScalarPow((lhs, rhs))
    assert -lhs == ScalarNeg((lhs,))
    assert lhs.exp() == ScalarExp((lhs,))
    assert lhs.log() == ScalarLog((lhs,))
    assert lhs.tanh() == ScalarTanh((lhs,))


if __name__ == "__main__":
    main()
