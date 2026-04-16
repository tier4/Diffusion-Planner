"""Route specification for closed-loop replay.

A ``Route`` captures an ego vehicle's driving intent on a Lanelet2 map:

* A start pose (picked in the GUI by selecting Start mode and dragging)
* A goal pose (picked by selecting Goal mode and dragging)
* Zero or more intermediate waypoints (picked by selecting Waypoint mode and
  dragging), each forcing the route to pass through a specific lanelet even
  when it is off the shortest path from start to goal.
* The resolved lanelet sequence connecting start → waypoints → goal, computed
  via ``lanelet2.routing.RoutingGraph.shortestPathWithVia``.

The Route is intentionally lightweight — no SceneContext, no neighbors.
Neighbors are synthesized on the fly at replay time by ``replay.py`` so that
saved routes stay reusable under different traffic densities and seeds.

Coordinate frame: all poses are in MGRS local Cartesian (the world frame used
throughout ``scenario_generation/``), matching the lanelet2 map origin
``MGRSProjector(Origin(0.0, 0.0))``.

Persistence uses pickle to match the existing ``.map_snippets/*.pkl`` format
(see ``batch_generate._load_snippet``).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Route:
    """A reusable route spec for closed-loop replay.

    Attributes:
        map_path: Absolute path to the ``lanelet2_map.osm`` file used when the
            route was authored. The replay loader rebuilds the ``LaneletSceneBuilder``
            from this path, so the map must exist at replay time.
        start_pose: ``(3,)`` float32 ``[x, y, heading_rad]`` in world frame.
        goal_pose: ``(3,)`` float32 ``[x, y, heading_rad]`` in world frame.
        start_lanelet_id: Snapped lanelet id at save time; can be ``None`` when
            the start could not be snapped to a drivable lanelet (in which case
            the route should not have been saved).
        goal_lanelet_id: Snapped lanelet id for the goal. Same ``None`` caveat.
        waypoint_poses: Ordered list of ``(3,)`` float32 ``[x, y, heading_rad]``
            world-frame poses, one per waypoint. Empty when no waypoints were
            dropped.
        waypoint_lanelet_ids: Snapped lanelet ids, parallel to
            ``waypoint_poses``. Same length.
        route_lanelet_ids: Full resolved lanelet path ``[start, ..., goal]``.
            Set at save time via ``shortestPathWithVia``. ``None`` indicates
            the author explicitly skipped resolution (e.g. the map changed);
            replay will fall back to ``builder.find_route`` with a warning.
    """

    map_path: str
    start_pose: np.ndarray
    goal_pose: np.ndarray
    start_lanelet_id: int | None
    goal_lanelet_id: int | None
    waypoint_poses: list[np.ndarray] = field(default_factory=list)
    waypoint_lanelet_ids: list[int] = field(default_factory=list)
    route_lanelet_ids: list[int] | None = None

    def __post_init__(self) -> None:
        if self.start_pose.shape != (3,):
            raise ValueError(f"start_pose must be shape (3,), got {self.start_pose.shape}")
        if self.goal_pose.shape != (3,):
            raise ValueError(f"goal_pose must be shape (3,), got {self.goal_pose.shape}")
        if len(self.waypoint_poses) != len(self.waypoint_lanelet_ids):
            raise ValueError(
                f"waypoint_poses ({len(self.waypoint_poses)}) and "
                f"waypoint_lanelet_ids ({len(self.waypoint_lanelet_ids)}) "
                "must have the same length"
            )
        for i, wp in enumerate(self.waypoint_poses):
            if wp.shape != (3,):
                raise ValueError(f"waypoint_poses[{i}] must be shape (3,), got {wp.shape}")

    def save(self, path: str | Path) -> None:
        """Pickle the Route to disk. Parent directories are created as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "Route":
        """Load a Route previously written with :meth:`save`."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected pickled Route, got {type(obj).__name__}")
        return obj

    def num_waypoints(self) -> int:
        """Convenience accessor for the number of intermediate waypoints."""
        return len(self.waypoint_lanelet_ids)

    def is_resolved(self) -> bool:
        """Whether ``route_lanelet_ids`` has been computed."""
        return self.route_lanelet_ids is not None and len(self.route_lanelet_ids) > 0
