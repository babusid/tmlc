from __future__ import annotations

from .types import PositiveInt as PositiveInt, StrictInt as StrictInt

# NOTE: do not eagerly import `topo_sort` here — it depends on `tmlc.tensor`, and `util` is now a
# low-level dependency of `tmlc.compute`, so importing it would create a cycle. Import it directly
# from `tmlc.util.topo_sort` where needed.
__all__ = ["StrictInt", "PositiveInt"]
