"""Graph transform passes (DCE, CSE, etc.) implementing tmlc.graph.graph.GraphTransform."""

from __future__ import annotations

from .ConstantFold import ConstantFold
from .CSE import CSE

__all__ = ["ConstantFold", "CSE"]
