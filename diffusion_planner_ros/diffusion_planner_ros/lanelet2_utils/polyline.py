from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

# from dataclasses import field
from attr import define, field
from typing_extensions import Self

from .map import MapType

if TYPE_CHECKING:
    from typing import NDArray, NDArrayF32

__all__ = ["Polyline"]


def to_np_f32(x):
    """Convert an array like object to a numpy float32 array."""
    return np.array(x, dtype=np.float32)


@define
class Polyline:
    """A dataclass of Polyline.

    Attributes
    ----------
        polyline_type (MapType): Type of polyline.
        waypoints (NDArrayF32): Waypoints of polyline.

    """

    polyline_type: MapType = field()
    waypoints: NDArrayF32 = field(converter=to_np_f32)

    # NOTE: For the 1DArray indices must be a list.
    XYZ_IDX: ClassVar[list[int]] = [0, 1, 2]
    XY_IDX: ClassVar[list[int]] = [0, 1]
    FULL_DIM3D: ClassVar[int] = 7
    FULL_DIM2D: ClassVar[int] = 5

    @polyline_type.validator
    def _check_type(self, attr, value) -> None:
        if not isinstance(value, MapType):
            raise TypeError(f"Unexpected type of {attr.name}: {type(value)}")

    @waypoints.validator
    def _check_dim(self, attribute, value) -> None:
        if value.ndim < 1 or value.shape[1] != 3:
            raise ValueError(f"Unexpected {attribute.name} dimensions.")

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Construct an instance from dict data.

        Args:
        ----
            data (dict): Dict data of `Polyline`.

        Returns:
        -------
            Polyline: Constructed instance.

        """
        return cls(**data)

    @property
    def xyz(self) -> NDArray:
        """Return 3D positions.

        Returns
        -------
            NDArray: (x, y, z) positions.

        """
        return self.waypoints[..., self.XYZ_IDX]

    @xyz.setter
    def xyz(self, xyz: NDArray) -> None:
        self.waypoints[..., self.XYZ_IDX] = xyz

    @property
    def xy(self) -> NDArray:
        """Return 2D positions.

        Returns
        -------
            NDArray: (x, y) positions.

        """
        return self.waypoints[..., self.XY_IDX]

    @xy.setter
    def xy(self, xy: NDArray) -> None:
        self.waypoints[..., self.XY_IDX] = xy

    @property
    def dxyz(self) -> NDArray:
        """Return 3D normalized directions. The first element always becomes (0, 0, 0).

        Returns
        -------
            NDArray: (dx, dy, dz) positions.

        """
        if self.is_empty():
            return np.empty((0, 3), dtype=np.float32)
        diff = np.diff(self.xyz, axis=0, prepend=self.xyz[0].reshape(-1, 3))
        norm = np.clip(np.linalg.norm(diff, axis=-1, keepdims=True), a_min=1e-6, a_max=1e9)
        return np.divide(diff, norm)

    @property
    def dxy(self) -> NDArray:
        """Return 2D normalized directions. The first element always becomes (0, 0).

        Returns
        -------
            NDArray: (dx, dy) positions.

        """
        if self.is_empty():
            return np.empty((0, 2), dtype=np.float32)
        diff = np.diff(self.xy, axis=0, prepend=self.xy[0].reshape(-1, 2))
        norm = np.clip(np.linalg.norm(diff, axis=-1, keepdims=True), a_min=1e-6, a_max=1e9)
        return np.divide(diff, norm)

    def __len__(self) -> int:
        return len(self.waypoints)

    def is_empty(self) -> bool:
        """Indicate whether waypoints is empty array.

        Returns
        -------
            bool: Return `True` if the number of points is 0.

        """
        return len(self.waypoints) == 0

    def as_array(self, *, full: bool = False, as_3d: bool = True) -> NDArrayF32:
        """Return the polyline as `NDArray`.

        Args:
        ----
            full (bool, optional): Indicates whether to return `(x, y, z, dx, dy, dz, type_id)`.
                If `False`, returns `(x, y, z)`. Defaults to False.
            as_3d (bool, optional): If `True` returns array containing 3D coordinates.
                Otherwise, 2D coordinates. Defaults to True.

        Returns:
        -------
            NDArrayF32: Polyline array.

        """
        if full:
            if self.is_empty():
                return (
                    np.empty((0, self.FULL_DIM3D), dtype=np.float32)
                    if as_3d
                    else np.empty((0, self.FULL_DIM2D), dtype=np.float32)
                )

            shape = self.waypoints.shape[:-1]
            type_id = np.full((*shape, 1), self.polyline_type.value)
            return (
                np.concatenate([self.xyz, self.dxyz, type_id], axis=1, dtype=np.float32)
                if as_3d
                else np.concatenate([self.xy, self.dxy, type_id], axis=1, dtype=np.float32)
            )
        else:
            return self.xyz if as_3d else self.xy
