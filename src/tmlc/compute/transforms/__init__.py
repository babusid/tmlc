"""Compute IR passes (ComputeProgram -> ComputeProgram)."""

from __future__ import annotations

from .fuse import (
    InlineCandidate,
    InlineReductionFree,
    ShouldInline,
    default_should_inline,
)

__all__ = [
    "InlineReductionFree",
    "InlineCandidate",
    "ShouldInline",
    "default_should_inline",
]
