"""Shared geometry + plotting helpers for the route-heatmap tools.

Used by both:
  - heatmap_route_deviation.py   (per-step realised ego pose)
  - heatmap_prediction_deviation.py (per-frame 80-step prediction trajectory)

Both tools share the same building blocks:
  * concatenated route centerline polyline + arc length
  * point-onto-polyline projection -> (arc, signed_lateral)
  * arc-binned aggregation (mean/max |lateral|)
  * per-bin coloured route plot
The difference is only WHICH (x,y) points get projected.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def load_route(route_path: Path):
    with open(route_path, "rb") as f:
        return pickle.load(f)


def build_route_polyline(route) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate route lanelets into one polyline + cumulative arc length.

    Returns (pts (N,2), s (N,)). Skips zero-length segments and drops
    duplicated joint points between consecutive lanelets.
    """
    from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder

    builder = LaneletSceneBuilder(str(route.map_path))
    pts_list = []
    for ll_id in route.route_lanelet_ids:
        if not builder.has_lanelet_id(int(ll_id)):
            print(f"  [WARN] lanelet {ll_id} missing from map; skipping")
            continue
        cl = builder.raw_centerline(int(ll_id)).astype(np.float32)
        if len(cl) == 0:
            continue
        if pts_list:
            prev_tail = pts_list[-1][-1]
            if np.allclose(cl[0], prev_tail, atol=1e-4):
                cl = cl[1:]
            if len(cl) == 0:
                continue
        pts_list.append(cl)
    if not pts_list:
        raise SystemExit("Route produced no centerline points — bad route or map mismatch.")
    pts = np.concatenate(pts_list, axis=0)
    seg = np.diff(pts, axis=0)
    seg_len = np.sqrt((seg * seg).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    return pts, s


def project_to_polyline(
    xy: np.ndarray, pts: np.ndarray, s: np.ndarray
) -> tuple[float, float, float]:
    """Project one (x, y) onto polyline. Returns (s_arc, lateral_signed, |lat|)."""
    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    ap = xy[None, :] - a
    seg_len2 = (ab * ab).sum(axis=1)
    seg_len2 = np.maximum(seg_len2, 1e-9)
    t = (ap * ab).sum(axis=1) / seg_len2
    t_clamped = np.clip(t, 0.0, 1.0)
    proj = a + t_clamped[:, None] * ab
    d = np.linalg.norm(xy[None, :] - proj, axis=1)
    i = int(np.argmin(d))
    cross = ab[i, 0] * (xy[1] - a[i, 1]) - ab[i, 1] * (xy[0] - a[i, 0])
    seg_l = math.sqrt(float(seg_len2[i]))
    sign = 1.0 if cross >= 0 else -1.0
    s_arc = float(s[i] + t_clamped[i] * seg_l)
    return s_arc, sign * float(d[i]), float(d[i])


def project_points_to_polyline(xys: np.ndarray, pts: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Batch projection via per-point loop. xys: (M, 2). Returns (M, 3) = (arc, signed, |lat|).

    O(M*N); fine up to a few hundred thousand points.
    """
    out = np.zeros((len(xys), 3), dtype=np.float64)
    for k, p in enumerate(xys):
        out[k] = project_to_polyline(p, pts, s)
    return out


def recover_ego_world_pose_from_goal(goal_pose: np.ndarray, route) -> tuple[float, float, float]:
    """Recover (ex, ey, eyaw) in world frame from NPZ goal_pose + Route goal.

    goal_pose is the route goal expressed in the current ego frame:
      - [x, y, yaw_rad]                 (parse_rosbag.py / cpp converter)
      - [x, y, cos(yaw), sin(yaw)]      (tensor_converter._build_goal_pose)
    Both are supported.
    """
    gx_w = float(route.goal_pose[0])
    gy_w = float(route.goal_pose[1])
    gyaw_w = float(route.goal_pose[2])
    dx, dy = float(goal_pose[0]), float(goal_pose[1])
    if goal_pose.shape[0] >= 4:
        dyaw = math.atan2(float(goal_pose[3]), float(goal_pose[2]))
    else:
        dyaw = float(goal_pose[2])
    eyaw = gyaw_w - dyaw
    eyaw = math.atan2(math.sin(eyaw), math.cos(eyaw))
    c, s_ = math.cos(eyaw), math.sin(eyaw)
    ex = gx_w - (c * dx - s_ * dy)
    ey = gy_w - (s_ * dx + c * dy)
    return ex, ey, eyaw


def deviation_series(xys: np.ndarray, pts: np.ndarray, s: np.ndarray) -> np.ndarray:
    """For each (x, y) in xys, return (arc, signed_lateral, |lat|). Shape (M, 3)."""
    return project_points_to_polyline(xys, pts, s)


def bin_by_arc(
    dev: np.ndarray, s_max: float, bin_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and max |lateral| per arc-length bin."""
    n_bins = max(1, int(math.ceil(s_max / bin_m)))
    bin_s_mid = (np.arange(n_bins) + 0.5) * bin_m
    mean_abs = np.full(n_bins, np.nan, dtype=np.float64)
    max_abs = np.full(n_bins, np.nan, dtype=np.float64)
    idx = np.clip((dev[:, 0] // bin_m).astype(int), 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            mean_abs[b] = np.mean(dev[mask, 2])
            max_abs[b] = np.max(dev[mask, 2])
    return bin_s_mid, mean_abs, max_abs


def bin_scalar_by_arc(
    arc_s: np.ndarray,
    values: np.ndarray,
    s_max: float,
    bin_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin an arbitrary per-point scalar by arc-length.

    Args:
        arc_s:  (M,) arc-length position of each sample.
        values: (M,) scalar value at each sample (e.g. min clearance).
        s_max:  total route arc-length.
        bin_m:  bin width in metres.

    Returns (bin_s_mid, mean_val, min_val) — each (n_bins,), NaN for empty.
    """
    n_bins = max(1, int(math.ceil(s_max / bin_m)))
    bin_s_mid = (np.arange(n_bins) + 0.5) * bin_m
    mean_val = np.full(n_bins, np.nan, dtype=np.float64)
    min_val = np.full(n_bins, np.nan, dtype=np.float64)
    idx = np.clip((arc_s // bin_m).astype(int), 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            mean_val[b] = np.mean(values[mask])
            min_val[b] = np.min(values[mask])
    return bin_s_mid, mean_val, min_val


def segments_from_polyline(pts: np.ndarray, s: np.ndarray, bin_m: float, n_bins: int) -> list:
    """For each arc-length bin, collect the sub-polyline that falls in it."""
    segs = [[] for _ in range(n_bins)]
    for i in range(len(pts) - 1):
        s0, s1 = s[i], s[i + 1]
        b0 = int(s0 // bin_m)
        b1 = int(s1 // bin_m)
        if b0 == b1 and 0 <= b0 < n_bins:
            segs[b0].append(pts[i : i + 2])
        else:
            curr_s = s0
            curr_pt = pts[i]
            for b in range(b0, b1 + 1):
                b_end_s = (b + 1) * bin_m
                next_s = min(b_end_s, s1)
                if next_s <= curr_s:
                    continue
                t = (next_s - s0) / max(s1 - s0, 1e-9)
                next_pt = pts[i] + t * (pts[i + 1] - pts[i])
                if 0 <= b < n_bins:
                    segs[b].append(np.stack([curr_pt, next_pt], axis=0))
                curr_s = next_s
                curr_pt = next_pt
    out = []
    for bs in segs:
        if bs:
            out.append(np.concatenate(bs, axis=0) if len(bs) > 1 else bs[0])
        else:
            out.append(None)
    return out


def plot_route_heatmap(ax, pts, bin_segments, bin_values, title, vmin, vmax, cmap):
    ax.set_aspect("equal")
    ax.plot(pts[:, 0], pts[:, 1], color="#cccccc", lw=1.0, zorder=1)
    valid_segs, valid_vals = [], []
    for segs, v in zip(bin_segments, bin_values):
        if segs is None or np.isnan(v):
            continue
        valid_segs.append(segs)
        valid_vals.append(v)
    if valid_vals:
        lc_lines, lc_colors = [], []
        for segs, v in zip(valid_segs, valid_vals):
            for i in range(len(segs) - 1):
                lc_lines.append([segs[i], segs[i + 1]])
                lc_colors.append(v)
        lc = LineCollection(
            lc_lines,
            array=np.array(lc_colors),
            cmap=cmap,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
            linewidths=4.0,
        )
        ax.add_collection(lc)
    ax.set_title(title, fontsize=10)
