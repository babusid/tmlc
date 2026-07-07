from __future__ import annotations

from collections.abc import Iterator
from typing_extensions import override

from tmlc import Tensor

from .pattern import Env, Pattern


class Any(Pattern):
    """Matches any node. Binds it to label if one is given."""

    def __init__(self, label: str | None = None) -> None:
        super().__init__(label or "", input_patterns=[])
        self._has_label = label is not None

    @override
    def _match(self, node: Tensor, env: Env) -> Iterator[Env]:
        yield {**env, self.label: node} if self._has_label else env
