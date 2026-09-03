"""Per-step object (neighbor OBB) clearance / collision for closed-loop eval."""

from __future__ import annotations

import numpy as np
import torch
from diffusion_planner.model.guidance.collision import (
    batch_signed_distance_rect,
    center_rect_to_points,
)

from planner_metrics.geometry import (
    _build_ego_bbox_corners,
    _closest_points_between_rects,
)


def _ego_neighbor_obb(neighbors_live: np.ndarray, ego_shape: np.ndarray, device: str):
    """Build ego corners (at origin) + valid-neighbor corners; return (ego_b, npc_corners, M).

    Canonical OBB geometry (``_build_ego_bbox_corners`` + ``center_rect_to_points``).
    Returns (None, None, 0) if there are no valid neighbors.
    """
    valid = np.abs(neighbors_live[:, :6]).sum(axis=1) > 0
    if not valid.any():
        return None, None, 0
    nb = neighbors_live[valid]
    M = nb.shape[0]
    et = torch.zeros((1, 1, 4), dtype=torch.float32, device=device)
    et[0, 0, 2] = 1.0
    ego_c = _build_ego_bbox_corners(
        et, torch.tensor(ego_shape[:3], dtype=torch.float32, device=device)
    )[:, :1].reshape(1, 4, 2)
    rects = torch.tensor(
        np.stack([nb[:, 0], nb[:, 1], nb[:, 2], nb[:, 3], nb[:, 7], nb[:, 6]], axis=-1),
        dtype=torch.float32,
        device=device,
    )  # x, y, cos, sin, length, width
    return ego_c.expand(M, 4, 2), center_rect_to_points(rects), M


def _rear_edge_x(ego_shape: np.ndarray) -> float:
    """Local-frame x of the ego box's rear face (box center is ``0.5*wheelbase`` ahead of
    the rear-axle reference point at the local origin; the rear face sits ``length/2``
    behind that center)."""
    return 0.5 * float(ego_shape[0]) - float(ego_shape[1]) / 2.0


# Half-thickness (m) of the "rear edge" sliver used to test contact against the ego's
# rear face. batch_signed_distance_rect normalizes each rect's edge vectors, so a truly
# zero-area rectangle (an exact line segment) divides by zero -- this keeps the sliver a
# hair's-width rectangle instead, thin enough to still mean "touches the rear edge," not
# "touches the rear half of the car."
_REAR_SLAB_HALF_LEN_M = 0.01


def _ego_rear_slab_corners(ego_shape: np.ndarray, device: str) -> torch.Tensor:
    """The ego box's rear FACE as a thin axis-aligned sliver straddling the rear edge
    (full ego width, ``2 * _REAR_SLAB_HALF_LEN_M`` deep in x). Ego heading is fixed at
    local +x in this frame (see ``_ego_neighbor_obb``), so the corners are built directly
    rather than via ``_build_ego_bbox_corners`` (which centers on the box's own center,
    not its rear face).

    Returns (4, 2) corners.
    """
    rear_x = _rear_edge_x(ego_shape)
    half_width = float(ego_shape[2]) / 2.0
    x0, x1 = rear_x - _REAR_SLAB_HALF_LEN_M, rear_x + _REAR_SLAB_HALF_LEN_M
    return torch.tensor(
        [[x1, half_width], [x1, -half_width], [x0, -half_width], [x0, half_width]],
        dtype=torch.float32,
        device=device,
    )


def score_object_step(
    neighbors_live: np.ndarray,
    ego_shape: np.ndarray,
    device: str,
) -> tuple[float, bool, bool, int]:
    """Min ego-neighbor clearance (m), collision flag, rear-collision flag, and #valid neighbors.

    RAW oriented-bounding-box check against EVERY valid neighbor — moving and
    static alike, with NO direction/rear-end or ego-speed gating. Collision =
    the ego box overlaps any neighbor box (canonical ``batch_signed_distance_rect``
    < 0); clearance = exact closest-point distance to the nearest neighbor
    (``_closest_points_between_rects``). This deliberately differs from the
    avoidance reward's ``compute_static_collision_penalty``, which only scores
    *stopped* neighbors and filters out rear-end hits — for mining we want to
    catch collisions with moving neighbors AND the ego being struck from behind.

    ``rear_collision``: True when a neighbor's box overlaps the ego box's rear-edge
    sliver (see ``_ego_rear_slab_corners``) — i.e. the contact region actually touches
    the ego box's REAR edge, as opposed to a front/side hit whose overlap never reaches
    that far back. Lets callers separate "ego rear-ended by a following NPC" (e.g. after
    the ego stops/slows while diverging from the recorded GT trajectory) from
    ego-at-fault collisions (``collision_count - rear_collision_count``). The sliver is a
    subset of the ego box, so a rear-edge overlap always implies ``collided`` too — no
    separate AND needed.

    neighbors_live: (320, 11) in live-ego frame [x,y,cos,sin,vx,vy,w,l,type...].
    """
    ego_b, npc_corners, M = _ego_neighbor_obb(neighbors_live, ego_shape, device)
    if M == 0:
        return float("inf"), False, False, 0
    p1, p2 = _closest_points_between_rects(ego_b, npc_corners)
    clr = (p1 - p2).norm(dim=-1)  # exact closest-point distance per neighbor
    signed = batch_signed_distance_rect(ego_b, npc_corners)  # < 0 => overlap
    collided = signed < 0
    ego_rear = _ego_rear_slab_corners(ego_shape, device).unsqueeze(0).expand(M, 4, 2)
    rear_collided = batch_signed_distance_rect(ego_rear, npc_corners) < 0
    return float(clr.min()), bool(collided.any()), bool(rear_collided.any()), M


def score_object_step_batched(
    neighbors_list: list[np.ndarray],
    ego_shapes: list[np.ndarray],
    device: str,
) -> list[tuple[float, bool, bool, int, int]]:
    """``score_object_step`` for many segments at once: ONE batched OBB pass.

    The OBB primitives (``_closest_points_between_rects`` / ``batch_signed_distance_rect``)
    are per-pair independent, so we concatenate every segment's (ego, neighbor) box
    pairs into one big batch, run the geometry once, then slice the result back per
    segment. The (min_clearance, collision, rear_collision, n_valid) values are
    bit-identical to calling ``score_object_step`` per segment (no cross-segment
    interaction), but this collapses N tiny GPU launches per tick into one — including
    the box construction: ONE host->device transfer + ONE ``center_rect_to_points`` for
    every neighbor across all segments (ego corners built once when shapes match, else
    per segment and repeat-interleaved).
    Returns a list aligned to the inputs:
    (min_clearance, collision, rear_collision, n_valid_neighbors, collider_slot), where
    ``collider_slot`` is the ORIGINAL neighbor index (same order as the input
    ``neighbors_list`` rows, i.e. the build()/slot_uuids order) achieving
    ``min_clearance`` — or -1 when no valid neighbor. Callers use it to identify the
    actually-colliding agent (the OBB-closest one, which can differ from the
    centroid-nearest slot 0). ``rear_collision`` follows ``score_object_step``'s
    definition (a neighbor's box overlapping the ego's rear-edge sliver).
    """
    valids = [np.abs(nb[:, :6]).sum(axis=1) > 0 for nb in neighbors_list]
    counts = [int(v.sum()) for v in valids]
    if sum(counts) == 0:
        return [(float("inf"), False, False, 0, -1) for _ in neighbors_list]

    # All valid neighbors across all segments -> one transfer -> one corner build.
    nb_all = np.concatenate(
        [nb[v] for nb, v in zip(neighbors_list, valids) if v.any()], axis=0
    )  # (K, 11)
    rects = torch.tensor(
        np.stack(
            [nb_all[:, 0], nb_all[:, 1], nb_all[:, 2], nb_all[:, 3], nb_all[:, 7], nb_all[:, 6]],
            axis=-1,
        ),
        dtype=torch.float32,
        device=device,
    )  # (K, 6) x, y, cos, sin, length, width
    npc_all = center_rect_to_points(rects)  # (K, 4, 2)

    # Ego box at origin (heading +x), one per segment, repeated per its neighbors.
    # K from the CPU list (avoids a per-tick GPU->CPU sync that int(counts_t.sum()) forces).
    K = sum(counts)
    et = torch.zeros((len(neighbors_list), 1, 4), dtype=torch.float32, device=device)
    et[:, 0, 2] = 1.0
    if all(np.array_equal(ego_shapes[0], sh) for sh in ego_shapes):
        ego1 = _build_ego_bbox_corners(
            et[:1], torch.tensor(ego_shapes[0][:3], dtype=torch.float32, device=device)
        ).reshape(1, 4, 2)
        ego_all = ego1.expand(K, 4, 2)
        rear1 = _ego_rear_slab_corners(ego_shapes[0], device).unsqueeze(0)
        rear_all = rear1.expand(K, 4, 2)
    else:
        ego_each = torch.stack(
            [
                _build_ego_bbox_corners(
                    et[i : i + 1], torch.tensor(sh[:3], dtype=torch.float32, device=device)
                ).reshape(4, 2)
                for i, sh in enumerate(ego_shapes)
            ],
            dim=0,
        )  # (B, 4, 2)
        rear_each = torch.stack(
            [_ego_rear_slab_corners(sh, device) for sh in ego_shapes], dim=0
        )  # (B, 4, 2)
        counts_t = torch.tensor(counts, device=device)
        ego_all = ego_each.repeat_interleave(counts_t, dim=0)  # (K, 4, 2)
        rear_all = rear_each.repeat_interleave(counts_t, dim=0)  # (K, 4, 2)

    p1, p2 = _closest_points_between_rects(ego_all, npc_all)
    # Move both result vectors to host ONCE, then slice/reduce per segment in numpy — avoids
    # the per-segment GPU->CPU syncs that min()/any()/argmin() would each force. min/argmin are
    # pure selection, so the per-segment values stay bit-identical to the on-device reduction.
    clr_all = (p1 - p2).norm(dim=-1).cpu().numpy()  # (K,)
    signed_neg = (batch_signed_distance_rect(ego_all, npc_all) < 0).cpu().numpy()  # (K,)
    rear_neg = (batch_signed_distance_rect(rear_all, npc_all) < 0).cpu().numpy()  # (K,)
    out: list[tuple[float, bool, bool, int, int]] = []
    off = 0
    for seg_i, m in enumerate(counts):
        if m == 0:
            out.append((float("inf"), False, False, 0, -1))
        else:
            seg_clr = clr_all[off : off + m]
            amin = int(seg_clr.argmin())  # index within this segment's VALID subset
            # Map the valid-subset argmin back to the original neighbor slot (build() order).
            collider_slot = int(np.flatnonzero(valids[seg_i])[amin])
            seg_collided = signed_neg[off : off + m]
            seg_rear = rear_neg[off : off + m]
            out.append(
                (
                    float(seg_clr.min()),
                    bool(seg_collided.any()),
                    bool(seg_rear.any()),
                    m,
                    collider_slot,
                )
            )
            off += m
    return out
