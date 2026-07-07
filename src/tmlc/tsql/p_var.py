from __future__ import annotations

from collections.abc import Iterator
from typing_extensions import override

from tmlc import Tensor

from .pattern import Env, Pattern


class Var(Pattern):
    """Matches any node and binds it to label."""

    def __init__(self, label: str) -> None:
        super().__init__(label, input_patterns=[])

    @override
    def _match(self, node: Tensor, env: Env) -> Iterator[Env]:
        yield {**env, self.label: node}
