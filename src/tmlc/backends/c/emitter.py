"""A direct portable-C emitter for the self-contained Loop IR dialect."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tmlc.loop.axis import LoopAxis
from tmlc.loop.buffer import Buffer, BufferConstant, ConstantValue, LoopScope, ProgramScope
from tmlc.loop.expr import (
    BufferLoad,
    FloatConst,
    IndexAdd,
    IndexCompare,
    IndexConst,
    IndexFloorDiv,
    IndexMod,
    IndexMul,
    LoopCompareOp,
    LoopIndex,
    LoopIndexExpr,
    LoopValue,
    LoopValueOp,
    ValueExpr,
    ValueSelect,
)
from tmlc.loop.program import (
    Accumulate,
    ExternalCall,
    Loop,
    LoopCombiner,
    LoopProgram,
    LoopStatement,
    Store,
)
from tmlc.loop.verify import verify_loop_program

_COMPARE = {
    LoopCompareOp.LT: "<",
    LoopCompareOp.LE: "<=",
    LoopCompareOp.GT: ">",
    LoopCompareOp.GE: ">=",
    LoopCompareOp.EQ: "==",
    LoopCompareOp.NE: "!=",
}

_BINARY = {
    LoopValueOp.ADD: "+",
    LoopValueOp.SUB: "-",
    LoopValueOp.MUL: "*",
    LoopValueOp.DIV: "/",
}

_UNARY_FUNCTION = {
    LoopValueOp.EXP: "expf",
    LoopValueOp.LOG: "logf",
    LoopValueOp.TANH: "tanhf",
}


def _ident(name: str) -> str:
    out = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    if not out or out[0].isdigit():
        out = "_" + out
    return out


def _c_float(value: float) -> str:
    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    return f"{value!r}f"


def _flatten_constant(value: ConstantValue) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    flattened: list[float] = []
    for item in value:
        flattened.extend(_flatten_constant(item))
    return flattened


@dataclass
class _Emitter:
    program: LoopProgram
    function_name: str
    lines: list[str] = field(default_factory=list)
    depth: int = 0
    local_buffers: dict[LoopAxis, tuple[Buffer, ...]] = field(default_factory=dict)
    external_buffers: set[Buffer] = field(default_factory=set)

    def _line(self, text: str) -> None:
        self.lines.append("    " * self.depth + text if text else "")

    def _index(self, expr: LoopIndexExpr) -> str:
        if isinstance(expr, LoopIndex):
            return _ident(expr.axis.name)
        if isinstance(expr, IndexConst):
            return str(expr.value)
        if isinstance(expr, IndexAdd):
            return f"({self._index(expr.lhs)} + {self._index(expr.rhs)})"
        if isinstance(expr, IndexMul):
            return f"({self._index(expr.lhs)} * {self._index(expr.rhs)})"
        if isinstance(expr, IndexFloorDiv):
            return f"({self._index(expr.lhs)} / {expr.divisor})"
        if isinstance(expr, IndexMod):
            return f"({self._index(expr.lhs)} % {expr.modulus})"
        if isinstance(expr, IndexCompare):
            return f"({self._index(expr.lhs)} {_COMPARE[expr.op]} {self._index(expr.rhs)})"
        raise TypeError(f"unknown LoopIndexExpr: {type(expr).__name__}")

    def _linear_index(self, buffer: Buffer, coordinates: tuple[LoopIndexExpr, ...]) -> str:
        terms: list[str] = []
        for coordinate, stride in zip(coordinates, buffer.layout.strides):
            rendered = self._index(coordinate)
            terms.append(rendered if stride == 1 else f"({rendered}) * {stride}")
        expression = " + ".join(terms) if terms else "0"
        if buffer.layout.offset:
            expression = f"{buffer.layout.offset} + {expression}"
        return expression

    def _access(self, buffer: Buffer, coordinates: tuple[LoopIndexExpr, ...]) -> str:
        name = _ident(buffer.name)
        if not buffer.shape:
            return f"{name}[0]" if buffer in self.external_buffers else name
        return f"{name}[{self._linear_index(buffer, coordinates)}]"

    def _value(self, value: LoopValue) -> str:
        if isinstance(value, FloatConst):
            return _c_float(value.value)
        if isinstance(value, BufferLoad):
            return self._access(value.buffer, value.index)
        if isinstance(value, ValueSelect):
            condition = self._index(value.condition)
            return (
                f"(({condition}) != 0 ? {self._value(value.if_true)} : "
                f"{self._value(value.if_false)})"
            )
        if isinstance(value, ValueExpr):
            if value.op is LoopValueOp.NEG:
                return f"(-{self._value(value.args[0])})"
            if value.op in _UNARY_FUNCTION:
                return f"{_UNARY_FUNCTION[value.op]}({self._value(value.args[0])})"
            if value.op is LoopValueOp.POW:
                return f"powf({self._value(value.args[0])}, {self._value(value.args[1])})"
            if value.op is LoopValueOp.MAX:
                return f"fmaxf({self._value(value.args[0])}, {self._value(value.args[1])})"
            operator = _BINARY[value.op]
            return f"({self._value(value.args[0])} {operator} {self._value(value.args[1])})"
        raise TypeError(f"unknown LoopValue: {type(value).__name__}")

    def _combine(self, combiner: LoopCombiner, target: str, value: str) -> str:
        if combiner is LoopCombiner.SUM:
            return f"{target} + {value}"
        if combiner is LoopCombiner.PROD:
            return f"{target} * {value}"
        return f"fmaxf({target}, {value})"

    def _declare_local(self, buffer: Buffer) -> None:
        name = _ident(buffer.name)
        if buffer.shape:
            self._line(f"float {name}[{buffer.numel}];")
        else:
            self._line(f"float {name};")

    def _statement(self, statement: LoopStatement) -> None:
        if isinstance(statement, Loop):
            variable = _ident(statement.axis.name)
            self._line(
                f"for (long {variable} = 0; {variable} < {statement.axis.extent}; {variable}++) {{"
            )
            self.depth += 1
            for buffer in self.local_buffers.get(statement.axis, ()):
                self._declare_local(buffer)
            for child in statement.body:
                self._statement(child)
            self.depth -= 1
            self._line("}")
            return
        if isinstance(statement, Store):
            self._line(f"/* store {statement.buffer.name} */")
            self._line(
                f"{self._access(statement.buffer, statement.index)} = "
                f"{self._value(statement.value)};"
            )
            return
        if isinstance(statement, Accumulate):
            target = self._access(statement.buffer, statement.index)
            combined = self._combine(statement.combiner, target, self._value(statement.value))
            self._line(f"{target} = {combined};")
            return
        if isinstance(statement, ExternalCall):
            raise NotImplementedError("the C backend does not implement ExternalCall yet")
        raise TypeError(f"unknown LoopStatement: {type(statement).__name__}")

    def _constant(self, constant: BufferConstant) -> None:
        buffer = constant.buffer
        values = _flatten_constant(constant.value)
        if not buffer.shape:
            self._line(f"static const float {_ident(buffer.name)} = {_c_float(values[0])};")
            return
        initializer = ", ".join(_c_float(value) for value in values)
        declaration = f"static const float {_ident(buffer.name)}[{len(values)}]"
        self._line(f"{declaration} = {{ {initializer} }};")

    def emit(self) -> str:
        input_set = set(self.program.inputs)
        output_set = set(self.program.outputs)
        self.external_buffers = input_set | output_set
        constant_set = {constant.buffer for constant in self.program.constants}
        internal_globals = tuple(
            buffer
            for buffer in self.program.buffers
            if isinstance(buffer.scope, ProgramScope)
            and buffer not in input_set
            and buffer not in output_set
            and buffer not in constant_set
        )
        self.local_buffers = {
            axis: tuple(
                buffer
                for buffer in self.program.buffers
                if isinstance(buffer.scope, LoopScope) and buffer.scope.axis is axis
            )
            for axis in {
                buffer.scope.axis
                for buffer in self.program.buffers
                if isinstance(buffer.scope, LoopScope)
            }
        }

        parameters = [f"const float* {_ident(buffer.name)}" for buffer in self.program.inputs]
        parameters += [f"float* {_ident(buffer.name)}" for buffer in self.program.outputs]
        signature = ", ".join(parameters) if parameters else "void"

        self._line("#include <math.h>")
        self._line("#include <stdlib.h>")
        self._line("")
        self._line(f"void {self.function_name}({signature}) {{")
        self.depth += 1

        for constant in self.program.constants:
            self._constant(constant)
        for buffer in internal_globals:
            name = _ident(buffer.name)
            if buffer.shape:
                self._line(f"float* {name} = (float*)malloc({buffer.numel} * sizeof(float));")
            else:
                self._line(f"float {name};")

        for statement in self.program.body:
            self._statement(statement)

        for buffer in internal_globals:
            if buffer.shape:
                self._line(f"free({_ident(buffer.name)});")

        self.depth -= 1
        self._line("}")
        return "\n".join(self.lines) + "\n"


def emit_c(program: LoopProgram, function_name: str = "tmlc_program") -> str:
    """Verify and emit a portable C99 translation unit for ``program``."""
    verify_loop_program(program)
    return _Emitter(program, function_name).emit()
