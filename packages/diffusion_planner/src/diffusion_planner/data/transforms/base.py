"""Common frame-transform types."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from numpy.typing import NDArray

Frame = dict[str, NDArray[Any]]
FrameLike = Mapping[str, NDArray[Any]]


class Transform(Protocol):
    """Transform one unbatched NumPy planner frame without mutating its input."""

    def __call__(self, frame: FrameLike) -> Frame: ...
