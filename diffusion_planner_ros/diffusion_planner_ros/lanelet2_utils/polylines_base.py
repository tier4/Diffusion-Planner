from abc import abstractmethod

from .base import LabelBaseType

__all__ = "BoundaryType"


class BoundaryType(LabelBaseType):
    """A base enum of RoadLine and RoadEdge."""

    def is_dynamic(self) -> bool:
        """Indicate whether the lane is drivable.

        Returns
        -------
            bool: Return always False.

        """
        return False

    @abstractmethod
    def is_virtual(self) -> bool:
        """Whether the boundary is virtual or not.

        Returns
        -------
            bool: Return `True` if the boundary is virtual.

        """

    @abstractmethod
    def is_crossable(self) -> bool:
        """Whether the boundary is allowed to cross or not.

        Returns
        -------
            bool: Return `True` if the boundary is allowed to cross.

        """
