"""Identity-valued induction variables for explicit Loop IR loops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class LoopAxis:
    """One induction variable with a fixed, positive extent."""

    extent: int
    name: str
