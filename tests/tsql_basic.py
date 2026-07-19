import numpy as np
import tmlc
from tmlc.tsql import Var, Const, Pattern, Ref, EqualTo, match_pattern
from tmlc.tensor.ops.ops_arithmetic import Add, Mul
from tmlc.tensor.ops.ops_arithmetic import Matmul
from tmlc.tensor.traits import Commutative, commutative

x = tmlc.input(shape=(2,), label="x")
a = tmlc.constant(np.ones((2,)), label="a")
b = tmlc.constant(np.ones((2,)) * 2, label="b")

# x + a + b  ->  Add(Add(x, a), b)
out = x + a + b
graph = tmlc.Graph([out])

# Const matches constant nodes only
const_matches = list(match_pattern(graph, Const("c")))
assert len(const_matches) == 2, f"expected 2 constants, got {len(const_matches)}"
assert all(isinstance(m.anchor.op, tmlc.Constant) for m in const_matches)

# Var matches every node in topo order
var_matches = list(match_pattern(graph, Var("v")))
assert len(var_matches) == len(graph.topo_sort)

# Pattern(Add) matches both Add nodes, no input constraints
add_matches = list(match_pattern(graph, Pattern(Add)))
assert len(add_matches) == 2, f"expected 2 Add nodes, got {len(add_matches)}"

# Pattern(Add) with inputs: positional, inner Add first
inner_add = Pattern(Add, [Var("x"), Var("a")])
inner_matches = list(match_pattern(graph, inner_add))
assert len(inner_matches) == 2

# Op does NOT match on wrong op type
mul_matches = list(match_pattern(graph, Pattern(Mul)))
assert len(mul_matches) == 0

# Ref: identity back-reference
# Build a graph where the same node is used twice: a * a
sq = a * a
sq_graph = tmlc.Graph([sq])
ref_matches = list(match_pattern(sq_graph, Pattern(Mul, [Var("v"), Ref("v")])))
assert len(ref_matches) == 1
assert ref_matches[0].env["v"] is a

# Ref does NOT match when the two inputs are different nodes
diff = a * b
diff_graph = tmlc.Graph([diff])
ref_no_match = list(match_pattern(diff_graph, Pattern(Mul, [Var("v"), Ref("v")])))
assert len(ref_no_match) == 0

# EqualTo: structural equality on distinct objects
c1 = tmlc.constant(np.ones((2,)))
c2 = tmlc.constant(np.ones((2,)))  # distinct object, same value
assert c1 is not c2
eq_sum = c1 + c2
eq_graph = tmlc.Graph([eq_sum])
eq_matches = list(match_pattern(eq_graph, Pattern(Add, [Var("s"), EqualTo("s")])))
assert len(eq_matches) == 1

# EqualTo does NOT match when values differ
c3 = tmlc.constant(np.zeros((2,)))
neq_sum = c1 + c3
neq_graph = tmlc.Graph([neq_sum])
neq_matches = list(match_pattern(neq_graph, Pattern(Add, [Var("s"), EqualTo("s")])))
assert len(neq_matches) == 0

# Commutativity: Pattern(Add, [Const, Var]) matches both x+a and a+x
comm_graph = tmlc.Graph([x + a])
comm_pattern = Pattern(Add, [Const("c"), Var("v")])
comm_matches = list(match_pattern(comm_graph, comm_pattern))
# x + a is built as Add(x, broadcast_to(a)) — const is second operand,
# but the commutative permutation should still find it
assert len(comm_matches) == 1
assert isinstance(comm_matches[0].env["c"].op, tmlc.Constant)
assert isinstance(Add(), Commutative)
assert isinstance(Mul(), Commutative)
assert not isinstance(Matmul(), Commutative)
assert commutative(Add) is Add

# Matmul is NOT commutative — wrong operand order should not match
mm_result = tmlc.mm(tmlc.reshape(x, (1, 2)), tmlc.reshape(x, (2, 1)))
mm_graph = tmlc.Graph([mm_result])
mm_pattern = Pattern(Matmul, [Var("a"), Var("b")])
mm_matches = list(match_pattern(mm_graph, mm_pattern))
assert len(mm_matches) == 1
assert mm_matches[0].env["a"].shape == (1, 2)
assert mm_matches[0].env["b"].shape == (2, 1)

print("all assertions passed")
