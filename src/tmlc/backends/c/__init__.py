"""Generic C99 backend for the Loop IR (portable, single-precision, no threads or intrinsics)."""

from __future__ import annotations

from .emitter import emit_c
from .runner import CProgram

__all__ = ["emit_c", "CProgram"]
