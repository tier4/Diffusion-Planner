"""Single canonical vehicle-OBB collision-detection entry point.

Every DETECTION / labeling / GUI / viz site in the repo should build oriented
bounding boxes and test overlap/clearance through THIS module. It is a thin
numpy/torch shim over the canonical torch primitives in
:mod:`planner_metrics.collision_geometry` and :mod:`planner_metrics.geometry`,
so the whole codebase shares ONE collision algorithm.

Nothing here re-implements SAT or closest-point math — it only adapts
representation (numpy<->torch, radians<->cos/sin, single<->batched). The frozen
reward/SFT/eval path (:mod:`planner_metrics.subscores` / :mod:`rlvr.reward`)
calls the same primitives directly. Two deliberate differences from that path:
this shim promotes inputs to float64 (``_WRAPPER_DTYPE``) for stable detection
geometry, and ``overlap`` treats exact boundary touch (distance ``== 0``) as a
hit (``<= 0``) whereas the reward penalty uses a strict ``< 0``. Results agree to
float precision away from the exact-touch boundary; do not assume bit-identical
scores against the reward path.

Notes:
- ``wheelbase is not None`` selects the ego (rear-axle) convention, matching
  ``geometry._build_ego_bbox_corners`` (box centre = ``0.5 * wheelbase`` forward
  of the rear-axle reference point). ``wheelbase is None`` selects the centroid
  convention (perception / tracked-object boxes), matching
  ``collision_geometry.center_rect_to_points``.
- Corner order is a valid rectangle ring but NOT a fixed semantic order; callers
  that only need a polygon (drawing) or an order-agnostic overlap test are
  unaffected. Do not rely on a specific corner being "front-left".
- ``overlap`` semantics: two convex OBBs overlap iff their SAT signed distance is
  ``<= 0``. Exact boundary touch (distance ``== 0``) counts as overlap, matching
  both the legacy hand-rolled SAT and shapely ``Polygon.intersects`` for convex
  rectangles.
"""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics.collision_geometry import batch_signed_distance_rect, center_rect_to_points
from planner_metrics.geometry import _build_ego_bbox_corners, _closest_points_between_rects

_WRAPPER_DTYPE = torch.float64


def obb_corners(
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    wheelbase: float | None = None,
) -> np.ndarray:
    """Return ``(4, 2)`` OBB corners (numpy) for a single box.

    Args:
        x, y: reference point. Rear-axle midpoint when ``wheelbase`` is given,
            otherwise the box centroid.
        heading: yaw in radians.
        length, width: box extents in metres.
        wheelbase: ego wheel base (rear-axle convention) or ``None`` (centroid).

    Delegates to the canonical torch corner builders — no local corner math.
    """
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    if wheelbase is None:
        rect = torch.tensor([[x, y, cos_h, sin_h, length, width]], dtype=_WRAPPER_DTYPE)  # (1, 6)
        corners = center_rect_to_points(rect)[0]  # (4, 2)
    else:
        ego_traj = torch.tensor([[[x, y, cos_h, sin_h]]], dtype=_WRAPPER_DTYPE)  # (1, 1, 4)
        ego_shape = torch.tensor([wheelbase, length, width], dtype=_WRAPPER_DTYPE)
        corners = _build_ego_bbox_corners(ego_traj, ego_shape)[0, 0]  # (4, 2)
    return corners.cpu().numpy()


def _as_corner_tensor(corners) -> torch.Tensor:
    """Coerce a single ``(4, 2)`` corner set to a ``(1, 4, 2)`` torch batch."""
    if isinstance(corners, torch.Tensor):
        t = corners.detach().to(_WRAPPER_DTYPE)
    else:
        t = torch.as_tensor(np.asarray(corners, dtype=np.float64), dtype=_WRAPPER_DTYPE)
    if t.shape[-2:] != (4, 2):
        raise ValueError(f"expected corners shaped (4, 2), got {tuple(t.shape)}")
    return t.reshape(1, 4, 2)


def obb_signed_distance(corners_a, corners_b) -> float:
    """Signed SAT distance between two OBBs (``< 0`` = penetration/overlap)."""
    da = batch_signed_distance_rect(_as_corner_tensor(corners_a), _as_corner_tensor(corners_b))
    return float(da[0].item())


def obb_overlap(corners_a, corners_b) -> bool:
    """True iff the two OBBs overlap (SAT signed distance ``<= 0``)."""
    return obb_signed_distance(corners_a, corners_b) <= 0.0


def obb_closest_points(corners_a, corners_b) -> tuple[np.ndarray, np.ndarray]:
    """Closest-point pair between two OBBs (numpy ``(2,)`` each)."""
    pt1, pt2 = _closest_points_between_rects(
        _as_corner_tensor(corners_a), _as_corner_tensor(corners_b)
    )
    return pt1[0].cpu().numpy(), pt2[0].cpu().numpy()
