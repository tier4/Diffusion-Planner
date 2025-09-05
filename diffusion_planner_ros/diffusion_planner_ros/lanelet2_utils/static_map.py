from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from attr import define

from .polyline import Polyline

if TYPE_CHECKING:
    from .typing import NDArrayF32

__all__ = ("AWMLStaticMap", "LaneSegment")


@dataclass(frozen=True)
class AWMLStaticMap:
    """Represents a static map information.

    Attributes
    ----------
        id (str): Unique ID associated with this map.
        lane_segments (dict[int, LaneSegment]): Container of lanes stored by its id.
    """

    id: str
    lane_segments: dict[int, LaneSegment]

    def __post_init__(self) -> None:
        assert all(isinstance(item, LaneSegment) for _, item in self.lane_segments.items()), (
            "Expected all items are LaneSegments."
        )


@define
class LaneSegment:
    id: int
    turn_direction: int
    polyline: Polyline
    left_boundary: Polyline
    right_boundary: Polyline
    speed_limit_mph: float | None
    center: NDArrayF32
    traffic_lights: list
