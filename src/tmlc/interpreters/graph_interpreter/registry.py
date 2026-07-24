from collections.abc import Callable
from typing import Any

from tmlc import TensorOp

type OpImplementation[Value] = Callable[[Any, list[Value]], Value]


class OpRegistry[Value]:
    """Maps graph operation types to concrete backend implementations."""

    def __init__(self) -> None:
        self._implementations: dict[type[TensorOp], OpImplementation[Value]] = {}

    def register(
        self, op_type: type[TensorOp]
    ) -> Callable[[OpImplementation[Value]], OpImplementation[Value]]:
        def decorator(implementation: OpImplementation[Value]) -> OpImplementation[Value]:
            if op_type in self._implementations:
                raise ValueError(f"An implementation for {op_type.__name__} is already registered")
            self._implementations[op_type] = implementation
            return implementation

        return decorator

    def evaluate(self, op: TensorOp, inputs: list[Value]) -> Value:
        try:
            implementation = self._implementations[type(op)]
        except KeyError as error:
            raise NotImplementedError(
                f"No interpreter implementation registered for {type(op).__name__}"
            ) from error
        return implementation(op, inputs)
