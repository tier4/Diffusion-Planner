from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
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
        assert all(
            isinstance(item, LaneSegment) for _, item in self.lane_segments.items()
        ), "Expected all items are LaneSegments."


class LineType(IntEnum):
    LINE_THIN = 0
    LINE_THICK = 1
    VIRTUAL = 2
    ROAD_BORDER = 3
    ROAD_SHOULDER = 4
    GUARD_RAIL = 5
    CURBSTONE = 6
    NUM = 7

    @classmethod
    def from_str(cls, type_str: str) -> LineType:
        return cls._line_type_mapping[type_str]


# クラス定義の後にマッピングを定義
LineType._line_type_mapping = {
    "line_thin": LineType.LINE_THIN,
    "line_thick": LineType.LINE_THICK,
    "virtual": LineType.VIRTUAL,
    "road_border": LineType.ROAD_BORDER,
    "road_shoulder": LineType.ROAD_SHOULDER,
    "guard_rail": LineType.GUARD_RAIL,
    "curbstone": LineType.CURBSTONE,
}


@define
class LaneSegment:
    id: int
    turn_direction: int
    polyline: Polyline
    left_boundary: Polyline
    left_line_type: LineType
    right_boundary: Polyline
    right_line_type: LineType
    speed_limit_mph: float | None
    center: NDArrayF32
    traffic_lights: list

    TENSOR_DIM = 27
