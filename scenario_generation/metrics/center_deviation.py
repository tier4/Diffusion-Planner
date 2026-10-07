"""Per-step signed lateral offset from the centerline of the lane the ego is driving in."""

from __future__ import annotations

import numpy as np

_SEGMENT_MIN_LENGTH = 1e-6
_BOUNDARY_MIN_WIDTH = 1e-6


def _boundary_offset_columns(dim: int) -> tuple[slice, slice]:
    """(left, right) centerline->boundary offset columns: the legacy 33-column layout puts them
    at 4:6 / 6:8, the native h5 layout (xy, left, right) at 2:4 / 4:6."""
    return (slice(4, 6), slice(6, 8)) if dim >= 33 else (slice(2, 4), slice(4, 6))


def score_center_deviation_step(np_dict: dict) -> dict:
    """Signed lateral offset (m) of the live ego from its driving lane's centerline.

    The driving lane is the ``route_lanes`` centerline segment nearest the ego among those
    pointing within 90 degrees of the ego heading, so an opposing or crossing lane never
    wins. The ego frame has the ego at the origin heading +x. Positive = ego is left of the
    centerline. The offset is measured against the segment's supporting line, so running past
    the segment end does not leak into it. Unlike ``score_route_deviation_step`` this has no
    ``lanes`` fallback: a neighbouring lane must not be taken for the driving lane.

    The ego must also be *inside* that lane: the lane is its centerline buffered by the
    half-width to the boundary on the ego's side (interpolated along the segment), so the
    distance to the centerline -- lateral and any overshoot past the segment end together --
    must not exceed it. Returns ``nan`` (unmeasured) when no segment qualifies, the ego is
    outside the lane, or the boundary on the ego's side is missing (zero offset).
    """
    out = {"center_dev_m": float("nan")}
    if "route_lanes" not in np_dict:
        return out
    arr = np.asarray(np_dict["route_lanes"], dtype=np.float64)
    lanes = arr.reshape(-1, *arr.shape[-2:])  # (S, P, D)
    left_cols, right_cols = _boundary_offset_columns(lanes.shape[-1])
    xy = lanes[..., :2]
    point_ok = np.abs(lanes[..., :4]).sum(axis=-1) > 1e-6
    p1, p2 = xy[:, :-1], xy[:, 1:]
    seg = p2 - p1
    length = np.maximum(np.linalg.norm(seg, axis=-1), _SEGMENT_MIN_LENGTH)
    direction = seg / length[..., None]
    ok = point_ok[:, :-1] & point_ok[:, 1:] & (np.linalg.norm(seg, axis=-1) > _SEGMENT_MIN_LENGTH)
    ok &= direction[..., 0] >= 0.0  # heading is +x: cos(angle) >= 0 <=> within 90 degrees
    if not ok.any():
        return out

    t = np.clip(-(p1 * seg).sum(axis=-1) / length**2, 0.0, 1.0)
    dist = np.linalg.norm(p1 + t[..., None] * seg, axis=-1)
    s, j = np.unravel_index(np.argmin(np.where(ok, dist, np.inf)), dist.shape)
    d, a, tt = direction[s, j], p1[s, j], t[s, j]
    center_dev = float(d[1] * a[0] - d[0] * a[1])  # cross(d, ego - a), ego = origin

    normal = np.array([-d[1], d[0]])  # left normal
    side_cols, sign = (left_cols, 1.0) if center_dev >= 0.0 else (right_cols, -1.0)
    offsets = lanes[s, j : j + 2, side_cols]
    half_width = sign * float(((1.0 - tt) * offsets[0] + tt * offsets[1]) @ normal)
    if half_width <= _BOUNDARY_MIN_WIDTH or dist[s, j] > half_width:
        return out
    out["center_dev_m"] = center_dev
    return out
