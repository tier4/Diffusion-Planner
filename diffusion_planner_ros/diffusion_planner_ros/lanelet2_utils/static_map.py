from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from attr import define, field
from typing_extensions import Self

from .map import MapType
from .polyline import Polyline

if TYPE_CHECKING:
    from .typing import NDArrayF32

__all__ = ("AWMLStaticMap", "LaneSegment", "BoundarySegment")


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
    """Represents a lane segment.

    Attributes
    ----------
        id (int): Unique ID associated with this lane.
        polyline (Polyline): `Polyline` instance.
        left_boundaries (list[BoundarySegment]): List of `BoundarySegment` instances.
        right_boundaries (list[BoundarySegment]): List of `BoundarySegment` instances.
        speed_limit_mph (float | None, optional): Lane speed limit in [miles/h].

    """

    id: int
    turn_direction: int
    polyline: Polyline
    left_boundary: BoundarySegment
    right_boundary: BoundarySegment
    speed_limit_mph: float | None
    center: NDArrayF32
    traffic_lights: list

    @property
    def lane_type(self) -> MapType:
        """Return the type of the lane.

        Returns
        -------
            MapType: Lane type.

        """
        return self.polyline.polyline_type

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct a instance from a dict data.

        Args:
        ----
            data (dict[str, Any]): Dict data of `LaneSegment`.

        Returns:
        -------
            LaneSegment: Constructed instance.

        """
        return cls(**data)

    def is_drivable(self) -> bool:
        """Whether the lane is allowed to drive by car like vehicle.

        Returns
        -------
            bool: Return True if the lane is allowed to drive.

        """
        return self.lane_type.is_drivable()


@dataclass
class BoundarySegment:
    """Represents a boundary segment which is RoadLine or RoadEdge.

    Attributes
    ----------
        id (int): Unique ID associated with this boundary.
        boundary_type (BoundaryType): `BoundaryType` instance.
        polyline (Polyline): `Polyline` instance.

    """

    id: int
    polyline: Polyline

    def __post_init__(self) -> None:
        assert isinstance(self.polyline, Polyline), "Expected Polyline."

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundarySegment:
        """Construct a instance from a dict data.

        Args:
        ----
            data (dict[str, Any]): Dict data of `BoundarySegment`.

        Returns:
        -------
            BoundarySegment: Constructed instance.

        """
        return cls(**data)

    def as_dict(self) -> dict:
        """Convert the instance to a dict.

        Returns
        -------
            dict: Converted data.

        """
        return asdict(self)

    def is_crossable(self) -> bool:
        """Indicate whether the boundary is allowed to cross or not.

        Return value depends on the `BoundaryType` definition.

        Returns
        -------
            bool: Return True if the boundary is allowed to cross.

        """
        return self.boundary_type.is_crossable()

    def is_virtual(self) -> bool:
        """Indicate whether the boundary is virtual(or Unknown) or not.

        Returns
        -------
            bool: Return True if the boundary is virtual.

        """
        return self.boundary_type.is_virtual()
