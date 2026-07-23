"""Shared beartype-validated type aliases."""

from __future__ import annotations

from typing import Annotated

from beartype.vale import Is, IsInstance

# int, excluding bool (an int subclass that slips through a bare `int` annotation).
StrictInt = Annotated[int, ~IsInstance[bool]]


# a StrictInt further constrained to be positive.
def _is_positive(n: StrictInt) -> bool:
    return n > 0


PositiveInt = Annotated[StrictInt, Is[_is_positive]]
