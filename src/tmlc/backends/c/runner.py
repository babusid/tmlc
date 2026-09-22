"""
Compile an emitted C program and call it from Python.

``CProgram`` is the dispatch layer. It emits C for a ``LoopProgram``, compiles it to a shared
library once, and loads it with ctypes. ``run`` then binds NumPy arrays to the function arguments
and returns the outputs. The library is compiled once and reused, so a training loop pays the
compile cost a single time and calls native code on every step.

Arguments follow the emitter's contract: inputs first as ``const float*``, then outputs as
``float*``. ``run`` takes inputs keyed by Loop IR ``Buffer`` objects, allocates the output buffers,
and returns them reshaped to each output's shape.
"""

from __future__ import annotations

import ctypes
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from tmlc.backends.c.emitter import emit_c
from tmlc.loop.buffer import Buffer
from tmlc.loop.program import LoopProgram

_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


class CProgram:
    """A compiled LoopProgram, callable from Python."""

    def __init__(
        self,
        program: LoopProgram,
        function_name: str = "tmlc_program",
        cc: str = "cc",
        opt: str = "-O2",
        save_dir: str | Path | None = None,
    ) -> None:
        self.program = program
        self.source = emit_c(program, function_name)

        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        if save_dir is not None:
            workdir = Path(save_dir)
            workdir.mkdir(parents=True, exist_ok=True)
        else:
            self._temporary_directory = tempfile.TemporaryDirectory(prefix="tmlc-")
            workdir = Path(self._temporary_directory.name)

        csrc = workdir / "kernel.c"
        lib = workdir / "kernel.so"
        csrc.write_text(self.source)
        subprocess.run([cc, opt, "-std=c99", "-shared", "-fPIC", "-o", lib, csrc], check=True)
        self._fn = getattr(ctypes.CDLL(lib), function_name)
        self._fn.restype = None
        self._fn.argtypes = [_FLOAT_PTR] * (len(program.inputs) + len(program.outputs))

    def run(self, inputs: Mapping[Buffer, np.ndarray]) -> list[np.ndarray]:
        """Call the compiled program, returning one array per output in ``outputs`` order."""
        in_bufs: list[np.ndarray] = []
        for buffer in self.program.inputs:
            buf = np.ascontiguousarray(inputs[buffer], dtype=np.float32).ravel()
            if buf.size != buffer.numel:
                raise ValueError(
                    f"input {buffer.name!r} expected {buffer.numel} elements, got {buf.size}"
                )
            in_bufs.append(buf)
        out_bufs = [np.zeros(buffer.numel, dtype=np.float32) for buffer in self.program.outputs]
        args = [buf.ctypes.data_as(_FLOAT_PTR) for buf in in_bufs + out_bufs]
        self._fn(*args)
        return [buf.reshape(buffer.shape) for buf, buffer in zip(out_bufs, self.program.outputs)]
