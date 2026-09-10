"""Per-step road-border (curb) clearance for closed-loop eval."""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics.config import RewardConfig
from planner_metrics.geometry import (
    point_inside_any_lane_polygon,
)
from planner_metrics.subscores import compute_road_border_penalty


def _as_float_tensor(value, device: str) -> torch.Tensor:
    """numpy array OR torch tensor -> float32 tensor on ``device``, no copy
    when it already matches (lets the rollout pass GPU-resident batch slices
    instead of forcing a host->device upload every step)."""
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.tensor(np.asarray(value), dtype=torch.float32, device=device)


def _ego_inside_lane(np_dict: dict, device: str) -> bool | None:
    """Is the ego origin inside any lane polygon? ``None`` when ``lanes`` is absent
    (sign cannot be determined, e.g. older callers/tests that only pass line_strings)."""
    if "lanes" not in np_dict:
        return None
    lanes_t = _as_float_tensor(np_dict["lanes"], device)
    if lanes_t.dim() == 4:
        lanes_t = lanes_t[0]
    origin = torch.zeros(2, dtype=torch.float32, device=device)
    return point_inside_any_lane_polygon(origin, lanes_t)


def _ego_intersects_border_segments(
    seg_p1: torch.Tensor,
    seg_p2: torch.Tensor,
    ego_shape: torch.Tensor,
    ego_pose: torch.Tensor | None = None,
) -> bool:
    """Whether any border segment intersects the current ego OBB.

    In the ego frame the OBB is an axis-aligned rectangle.  ``ego_pose`` is
    ``[x, y, cos_yaw, sin_yaw]`` in the segment frame; it is omitted when the
    segments are already in the current ego frame.  This exact test avoids
    missing an intersection between the 80 sampled perimeter points.
    """
    if seg_p1.shape[0] == 0:
        return False

    if ego_pose is not None:
        offset = torch.stack([ego_pose[0], ego_pose[1]])
        cos_yaw, sin_yaw = ego_pose[2], ego_pose[3]
        rot_inv = torch.stack([torch.stack([cos_yaw, sin_yaw]), torch.stack([-sin_yaw, cos_yaw])])
        seg_p1 = (seg_p1 - offset) @ rot_inv.T
        seg_p2 = (seg_p2 - offset) @ rot_inv.T

    length = ego_shape[1]
    width = ego_shape[2]
    wheelbase = ego_shape[0]
    rear_overhang = (length - wheelbase) / 2.0
    xmin = -rear_overhang
    xmax = length - rear_overhang
    ymin = -width / 2.0
    ymax = width / 2.0

    def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def point_in_rect(p: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        return (
            (p[:, 0] >= xmin - eps)
            & (p[:, 0] <= xmax + eps)
            & (p[:, 1] >= ymin - eps)
            & (p[:, 1] <= ymax + eps)
        )

    # An endpoint inside is an intersection.
    if point_in_rect(seg_p1).any() or point_in_rect(seg_p2).any():
        return True

    rect = torch.stack(
        [
            torch.stack([xmin, ymin]),
            torch.stack([xmax, ymin]),
            torch.stack([xmax, ymax]),
            torch.stack([xmin, ymax]),
        ]
    )
    rect_next = torch.roll(rect, shifts=-1, dims=0)

    # Inclusive segment intersection, vectorized over border segments and
    # rectangle edges.  The epsilon also treats touching as intersection.
    a = seg_p1[:, None, :]
    b = seg_p2[:, None, :]
    c = rect[None, :, :]
    d = rect_next[None, :, :]
    ab = b - a
    cd = d - c
    ac = c - a
    ad = d - a
    ca = a - c
    cb = b - c
    o1 = cross(ab, ac)
    o2 = cross(ab, ad)
    o3 = cross(cd, ca)
    o4 = cross(cd, cb)
    eps = 1e-6
    bbox_overlap = (
        (torch.minimum(a[..., 0], b[..., 0]) <= torch.maximum(c[..., 0], d[..., 0]) + eps)
        & (torch.maximum(a[..., 0], b[..., 0]) + eps >= torch.minimum(c[..., 0], d[..., 0]))
        & (torch.minimum(a[..., 1], b[..., 1]) <= torch.maximum(c[..., 1], d[..., 1]) + eps)
        & (torch.maximum(a[..., 1], b[..., 1]) + eps >= torch.minimum(c[..., 1], d[..., 1]))
    )
    proper = (o1 * o2 <= eps) & (o3 * o4 <= eps) & bbox_overlap
    return bool(proper.any().item())


@torch.no_grad()
def _ego_inside_road_border_corridor(
    line_strings: torch.Tensor,
    *,
    max_longitudinal_m: float = 50.0,
    max_side_pair_gap_m: float = 10.0,
) -> bool:
    """Return whether ego is inside a polygon formed by two border segments."""
    if line_strings.dim() == 4:
        line_strings = line_strings[0]
    xy = line_strings[..., :2]
    valid = (line_strings[..., 3] > 0.5) & (xy.norm(dim=-1) > 1e-3)
    pair_valid = valid[:, :-1] & valid[:, 1:]
    if not bool(pair_valid.any().item()):
        return False
    p1, p2 = xy[:, :-1][pair_valid], xy[:, 1:][pair_valid]
    seg = p2 - p1
    length2 = (seg * seg).sum(dim=-1).clamp_min(1e-10)
    t = ((-p1 * seg).sum(dim=-1) / length2).clamp(0.0, 1.0)
    closest = p1 + t[:, None] * seg
    usable = closest[:, 0].abs() <= max_longitudinal_m
    upper_mask = usable & (closest[:, 1] > 1e-4)
    lower_mask = usable & (closest[:, 1] < -1e-4)
    upper_points = closest[upper_mask]
    lower_points = closest[lower_mask]
    if not upper_points.numel() or not lower_points.numel():
        return False

    # Build every plausible upper/lower quadrilateral at once.  The previous
    # implementation called point-in-polygon from a Python double loop, which
    # was expensive because this function runs once per simulation step.
    upper_indices = torch.where(upper_mask)[0]
    lower_indices = torch.where(lower_mask)[0]
    upper = torch.stack((p1[upper_indices], p2[upper_indices]), dim=1)
    lower = torch.stack((p1[lower_indices], p2[lower_indices]), dim=1)
    upper_order = torch.argsort(upper[..., 0], dim=1)
    lower_order = torch.argsort(lower[..., 0], dim=1)
    upper = upper.gather(1, upper_order[..., None].expand(-1, -1, 2))
    lower = lower.gather(1, lower_order[..., None].expand(-1, -1, 2))

    pair_valid = (
        closest[upper_indices, 0][:, None] - closest[lower_indices, 0][None, :]
    ).abs() <= max_side_pair_gap_m
    polygons = torch.stack(
        (
            upper[:, None, 0, :].expand(-1, lower.shape[0], -1),
            upper[:, None, 1, :].expand(-1, lower.shape[0], -1),
            lower[None, :, 1, :].expand(upper.shape[0], -1, -1),
            lower[None, :, 0, :].expand(upper.shape[0], -1, -1),
        ),
        dim=2,
    )

    # Vectorized ray casting for the origin against all quadrilaterals.
    v1 = polygons
    v2 = torch.roll(polygons, shifts=-1, dims=2)
    y1, y2 = v1[..., 1], v2[..., 1]
    x1, x2 = v1[..., 0], v2[..., 0]
    cond_y = (y1 > 0) != (y2 > 0)
    dy = y2 - y1
    safe_dy = torch.where(dy.abs() < 1e-10, torch.ones_like(dy), dy)
    crossing_x = x1 - y1 * (x2 - x1) / safe_dy
    inside = (cond_y & (0 < crossing_x)).sum(dim=2) % 2 == 1
    return bool((inside & pair_valid).any().item())


def score_road_border_step(np_dict: dict, *, device: str) -> dict:
    """Ego-to-road-border distance at the current step.

    Uses ``line_strings`` channel 3 as road border. Collision / miss masks are
    derived later in ``_finalize`` from the distance series. Current pose is the
    ego-frame origin; history is not needed.

    Signed by lane containment when ``lanes`` is available: positive while the ego
    origin is inside a lane polygon (normal clearance), negative once it has crossed
    outside (magnitude = how far past the border) -- this is the opposite convention
    from ``_point_to_segments_signed_min_dist`` (lane departure: +outside/-inside),
    chosen to match how ``rb_dist_m`` is consumed downstream: smaller/more negative
    already reads as "worse" (collision thresholds, ``clearance_min_m``/``p5``), so
    inside=positive keeps that ordering intact once crossings go negative.
    """
    rb_dist_m = float("inf")
    if "line_strings" in np_dict and "ego_shape" in np_dict:
        ego_shape_t = _as_float_tensor(np_dict["ego_shape"], device).reshape(-1)[:3]
        traj = torch.zeros(1, 1, 4, dtype=torch.float32, device=device)
        traj[..., 2] = 1.0  # origin, heading +x
        ls = _as_float_tensor(np_dict["line_strings"], device)
        if ls.dim() == 4:
            ls = ls[0]
        data = {"line_strings": ls}
        _gate, _near, _wide, _steps, _cont, per_ts_min = compute_road_border_penalty(
            traj, ego_shape_t, data, config=RewardConfig()
        )
        rb_dist_m = float(per_ts_min[0, 0].item())
        if np.isfinite(rb_dist_m):
            inside = _ego_inside_lane(np_dict, device)
            # A border crossing/contact takes precedence over the signed
            # clearance: the requested value at intersection is exactly 0.
            if "line_strings" in np_dict:
                ls_for_intersection = ls
                border_xy = ls_for_intersection[..., :2]
                valid = (ls_for_intersection[..., 3] > 0.5) & (border_xy.norm(dim=-1) > 1e-3)
                valid_pair = valid[:, :-1] & valid[:, 1:]
                idx = torch.where(valid_pair.reshape(-1))[0]
                seg_p1 = border_xy[:, :-1].reshape(-1, 2)[idx]
                seg_p2 = border_xy[:, 1:].reshape(-1, 2)[idx]
                if _ego_intersects_border_segments(seg_p1, seg_p2, ego_shape_t):
                    rb_dist_m = 0.0
            border_inside = False
            if rb_dist_m != 0.0 and inside is False:
                # Lane polygons are the primary containment test.  At turns
                # the lane representation can be incomplete, so only use the
                # road-border corridor as a fallback when no lane contains
                # the ego origin.
                border_inside = _ego_inside_road_border_corridor(ls_for_intersection)
                if not border_inside:
                    rb_dist_m = -rb_dist_m
    return {"rb_dist_m": rb_dist_m}
