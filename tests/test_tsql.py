"""Pattern matching over the tensor graph via tmlc.tsql."""

import tmlc
from tmlc.tensor.ops.ops_arithmetic import Add, Matmul, Mul
from tmlc.tensor.traits import Commutative, commutative
from tmlc.tsql import Const, EqualTo, Pattern, Ref, Var, match_pattern


def _abc_graph() -> tmlc.Graph:
    # x + a + b  ->  Add(Add(x, a), b)
    x = tmlc.input(shape=(2,), label="x")
    a = tmlc.constant((1.0, 1.0), label="a")
    b = tmlc.constant((2.0, 2.0), label="b")
    return tmlc.Graph([x + a + b])


def test_const_matches_only_constants():
    graph = _abc_graph()
    matches = list(match_pattern(graph, Const("c")))
    assert len(matches) == 2
    assert all(isinstance(m.anchor.op, tmlc.Constant) for m in matches)


def test_var_matches_every_node():
    graph = _abc_graph()
    matches = list(match_pattern(graph, Var("v")))
    assert len(matches) == len(graph.topo_sort)


def test_pattern_matches_by_op_type():
    graph = _abc_graph()
    assert len(list(match_pattern(graph, Pattern(Add)))) == 2
    assert len(list(match_pattern(graph, Pattern(Mul)))) == 0


def test_pattern_with_positional_inputs():
    graph = _abc_graph()
    inner_add = Pattern(Add, [Var("x"), Var("a")])
    assert len(list(match_pattern(graph, inner_add))) == 2


def test_ref_is_identity_back_reference():
    a = tmlc.constant((1.0, 1.0), label="a")
    b = tmlc.constant((2.0, 2.0), label="b")

    ref = list(match_pattern(tmlc.Graph([a * a]), Pattern(Mul, [Var("v"), Ref("v")])))
    assert len(ref) == 1
    assert ref[0].env["v"] is a

    no_ref = list(match_pattern(tmlc.Graph([a * b]), Pattern(Mul, [Var("v"), Ref("v")])))
    assert len(no_ref) == 0


def test_equalto_is_structural_equality():
    c1 = tmlc.constant((1.0, 1.0))
    c2 = tmlc.constant((1.0, 1.0))  # distinct object, same value
    c3 = tmlc.constant((0.0, 0.0))
    assert c1 is not c2

    eq = list(match_pattern(tmlc.Graph([c1 + c2]), Pattern(Add, [Var("s"), EqualTo("s")])))
    assert len(eq) == 1

    neq = list(match_pattern(tmlc.Graph([c1 + c3]), Pattern(Add, [Var("s"), EqualTo("s")])))
    assert len(neq) == 0


def test_commutative_add_matches_either_operand_order():
    x = tmlc.input(shape=(2,), label="x")
    a = tmlc.constant((1.0, 1.0), label="a")
    # x + a builds Add(x, broadcast_to(a)); the const is the second operand but the commutative
    # permutation still finds Pattern(Add, [Const, Var]).
    matches = list(match_pattern(tmlc.Graph([x + a]), Pattern(Add, [Const("c"), Var("v")])))
    assert len(matches) == 1
    assert isinstance(matches[0].env["c"].op, tmlc.Constant)


def test_commutativity_traits():
    assert isinstance(Add(), Commutative)
    assert isinstance(Mul(), Commutative)
    assert not isinstance(Matmul(), Commutative)
    assert commutative(Add) is Add


def test_matmul_is_not_commutative():
    x = tmlc.input(shape=(2,), label="x")
    mm = tmlc.mm(tmlc.reshape(x, (1, 2)), tmlc.reshape(x, (2, 1)))
    matches = list(match_pattern(tmlc.Graph([mm]), Pattern(Matmul, [Var("a"), Var("b")])))
    assert len(matches) == 1
    assert matches[0].env["a"].shape == (1, 2)
    assert matches[0].env["b"].shape == (2, 1)
