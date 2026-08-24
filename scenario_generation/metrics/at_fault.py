"""Per-step at-fault classification of ego/neighbor OBB collisions (closed-loop eval).

The collision *set* is exactly the one ``score_object_step`` reports (same ego/neighbor
OBB geometry, ``batch_signed_distance_rect < 0``).  Each colliding neighbor is then
classified with the nuPlan/navsim ``get_collision_type`` port in
``planner_metrics.pdms_navsim`` and the step is *at fault* when any collision is an
``ACTIVE_FRONT_COLLISION`` or a ``STOPPED_TRACK_COLLISION``.  ``STOPPED_EGO_COLLISION``
(a stopped ego being hit) and ``ACTIVE_REAR_COLLISION`` (hit from behind) are never at
fault, so ghost contacts from the non-reactive log replay -- a recorded follower driving
into an ego that stopped earlier than the human did -- do not count against the planner.
``ACTIVE_LATERAL_COLLISION`` keeps navsim's map-less lenient branch (not at fault).
"""

from __future__ import annotations

import numpy as np
from diffusion_planner.model.guidance.collision import batch_signed_distance_rect

from planner_metrics.pdms_navsim import CollisionType, get_collision_type
from scenario_generation.metrics.object import _ego_neighbor_obb

AT_FAULT_TYPES = frozenset(
    {int(CollisionType.ACTIVE_FRONT_COLLISION), int(CollisionType.STOPPED_TRACK_COLLISION)}
)
# ``_build_ego_bbox_corners`` emits [FL, FR, RR, RL]; nuplan's front-bumper test reads the
# polygon's exterior ``coords[0] -> coords[3]`` as FL -> FR, i.e. corner order [FL, RL, RR, FR].
_NUPLAN_CORNER_ORDER = (0, 3, 2, 1)


def classify_collision_step(
    neighbors_live: np.ndarray,
    ego_shape: np.ndarray,
    ego_speed: float,
    device: str,
) -> tuple[bool, list[int]]:
    """Return ``(at_fault, collision_types)`` for one step in the live-ego frame.

    ``neighbors_live`` is the (N, 11) ``[x, y, cos, sin, vx, vy, w, l, type...]`` block
    ``score_object_step`` scores; the ego sits at the origin heading +x.  ``collision_types``
    lists the ``CollisionType`` value of every overlapping neighbor (empty = no collision).
    """
    from shapely.geometry import Polygon

    ego_b, npc_corners, M = _ego_neighbor_obb(neighbors_live, ego_shape, device)
    if M == 0:
        return False, []
    signed = batch_signed_distance_rect(ego_b, npc_corners)
    hit = (signed < 0).cpu().numpy().reshape(-1)
    if not hit.any():
        return False, []
    valid = np.abs(neighbors_live[:, :6]).sum(axis=1) > 0
    nb = neighbors_live[valid]
    ego_corners = ego_b[0].cpu().numpy()[list(_NUPLAN_CORNER_ORDER)]
    ego_poly = Polygon(ego_corners)
    ego_xyh = (0.0, 0.0, 0.0)
    npc_np = npc_corners.cpu().numpy()
    types: list[int] = []
    for j in np.flatnonzero(hit):
        row = nb[j]
        heading = float(np.arctan2(row[3], row[2]))
        # navsim box layout: [x, y, z, l, w, h, heading, vx, vy]
        box = np.array(
            [row[0], row[1], 0.0, row[7], row[6], 0.0, heading, row[4], row[5]],
            dtype=np.float64,
        )
        kind = get_collision_type(ego_xyh, float(ego_speed), ego_poly, box, Polygon(npc_np[j]))
        types.append(int(kind))
    return any(t in AT_FAULT_TYPES for t in types), types


def at_fault_block(at_fault_mask: np.ndarray, types_per_step: list[list[int]], event_count) -> dict:
    """Metrics block: at-fault steps / debounced events + per-type step tallies."""
    mask = np.asarray(at_fault_mask, dtype=bool)
    by_type = {name: 0 for name in CollisionType.__members__}
    for types in types_per_step:
        for t in set(types):
            by_type[CollisionType(t).name] += 1
    return {
        "steps": int(mask.sum()),
        "count": int(event_count(mask)),
        "by_type_steps": by_type,
    }
