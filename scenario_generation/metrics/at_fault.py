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
) -> tuple[bool, list[int], list[int]]:
    """Return ``(at_fault, collision_types, slots)`` for one step in the live-ego frame.

    ``neighbors_live`` is the (N, 11) ``[x, y, cos, sin, vx, vy, w, l, type...]`` block
    ``score_object_step`` scores; the ego sits at the origin heading +x.  ``collision_types``
    lists the ``CollisionType`` value of every overlapping neighbor (empty = no collision) and
    ``slots`` the matching row index into ``neighbors_live`` (for UUID lookup).
    """
    from shapely.geometry import Polygon

    ego_b, npc_corners, M = _ego_neighbor_obb(neighbors_live, ego_shape, device)
    if M == 0:
        return False, [], []
    signed = batch_signed_distance_rect(ego_b, npc_corners)
    hit = (signed < 0).cpu().numpy().reshape(-1)
    if not hit.any():
        return False, [], []
    valid = np.abs(neighbors_live[:, :6]).sum(axis=1) > 0
    valid_slots = np.flatnonzero(valid)
    nb = neighbors_live[valid]
    ego_corners = ego_b[0].cpu().numpy()[list(_NUPLAN_CORNER_ORDER)]
    ego_poly = Polygon(ego_corners)
    ego_xyh = (0.0, 0.0, 0.0)
    npc_np = npc_corners.cpu().numpy()
    types: list[int] = []
    slots: list[int] = []
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
        slots.append(int(valid_slots[j]))
    return any(t in AT_FAULT_TYPES for t in types), types, slots


def classify_new_collisions(
    neighbors_live: np.ndarray,
    ego_shape: np.ndarray,
    ego_speed: float,
    device: str,
    *,
    slot_uuids: list | None,
    collided: dict,
) -> tuple[bool, list[int], list[str]]:
    """Per-step classification with navsim's already-collided dedup.

    A track (by sidecar UUID; ``slot<i>`` when no UUID list is available) is classified once,
    at the first tick it overlaps the ego.  Later overlaps of the same track -- the vehicles
    sliding along each other, a recorded follower that pushed through the ego and now sits
    ahead of it -- are not re-classified, so a rear-end by a replayed follower cannot turn
    into an "at-fault front collision" a few ticks later.  ``collided`` (uuid -> first
    ``CollisionType``) is updated in place.  Returns ``(at_fault, new_types, new_uuids)`` for
    the tracks first hit this tick only.
    """
    _fault, types, slots = classify_collision_step(neighbors_live, ego_shape, ego_speed, device)
    new_types: list[int] = []
    new_uuids: list[str] = []
    for kind, slot in zip(types, slots):
        uuid = None
        if slot_uuids is not None and slot < len(slot_uuids) and slot_uuids[slot]:
            uuid = str(slot_uuids[slot]).strip()
        uuid = uuid or f"slot{slot}"
        if uuid in collided:
            continue
        collided[uuid] = int(kind)
        new_types.append(int(kind))
        new_uuids.append(uuid)
    return any(t in AT_FAULT_TYPES for t in new_types), new_types, new_uuids


def at_fault_block(
    at_fault_mask: np.ndarray,
    types_per_step: list[list[int]],
    event_count,
    *,
    collided: dict | None = None,
    rear_under_hard_brake: int = 0,
) -> dict:
    """Metrics block: at-fault steps / debounced events + per-type NEW-collision tallies.

    ``by_type_steps`` counts ticks on which a track was first hit with that type (dedup'd
    per track).  ``rear_under_hard_brake_tracks`` counts rear-end hits that started while the
    ego was braking hard -- reported, not penalised, so the "ego brake-checked the follower"
    case can be inspected.
    """
    mask = np.asarray(at_fault_mask, dtype=bool)
    by_type = {name: 0 for name in CollisionType.__members__}
    for types in types_per_step:
        for t in set(types):
            by_type[CollisionType(t).name] += 1
    return {
        "steps": int(mask.sum()),
        "count": int(event_count(mask)),
        "by_type_steps": by_type,
        "collided_tracks": int(len(collided)) if collided is not None else None,
        "rear_under_hard_brake_tracks": int(rear_under_hard_brake),
    }
