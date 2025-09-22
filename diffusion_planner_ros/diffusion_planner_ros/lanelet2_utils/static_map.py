from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from attr import define

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


class LineType(IntEnum):
    CROSSWALK = 0
    CURBSTONE = 1
    GUARD_RAIL = 2
    LINE_THICK = 3
    LINE_THIN = 4
    PEDESTRIAN_MARKING = 5
    ROAD_BORDER = 6
    ROAD_SHOULDER = 7
    VIRTUAL = 8
    ZEBRA_MARKING = 9
    NUM = 10

    @classmethod
    def from_str(cls, type_str: str) -> LineType:
        return cls._line_type_mapping[type_str]


# クラス定義の後にマッピングを定義
LineType._line_type_mapping = {
    "crosswalk": LineType.CROSSWALK,
    "curbstone": LineType.CURBSTONE,
    "guard_rail": LineType.GUARD_RAIL,
    "line_thick": LineType.LINE_THICK,
    "line_thin": LineType.LINE_THIN,
    "pedestrian_marking": LineType.PEDESTRIAN_MARKING,
    "road_border": LineType.ROAD_BORDER,
    "road_shoulder": LineType.ROAD_SHOULDER,
    "virtual": LineType.VIRTUAL,
    "zebra_marking": LineType.ZEBRA_MARKING,
}


@define
class LaneSegment:
    id: int
    turn_direction: int
    polyline: NDArrayF32
    left_boundary: NDArrayF32
    left_line_type: LineType
    right_boundary: NDArrayF32
    right_line_type: LineType
    speed_limit_mph: float | None
    center: NDArrayF32
    traffic_lights: list

    TENSOR_DIM = 13 + 2 * LineType.NUM.value
