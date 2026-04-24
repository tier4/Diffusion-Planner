#!/usr/bin/env python3
"""Route-centerline deviation heatmap for two replay runs.

Compares two closed-loop sim dumps on the SAME ``scenario_generation.Route``
(e.g. perfect-tracker vs MPC+delay=3) by projecting every dumped ego pose
onto the route's concatenated lanelet centerlines and colouring the route
geometry by the resulting lateral deviation.

Output: a single PNG with three stacked route maps (run A, run B, A−B)
plus a deviation-vs-route-arc-length line chart.

Ego world pose is recovered from each dumped NPZ without touching replay.py:
the NPZ's ``goal_pose`` is the Route goal expressed in the current ego
frame, and the Route pickle carries the goal's world pose, so

    ego_yaw_w = goal_yaw_w - dyaw
    ego_xy_w  = goal_xy_w - R(ego_yaw_w) @ (dx, dy)

Usage:
    python -m scenario_generation.tools.heatmap_route_deviation \\
        --route   /path/to/route.pkl \\
        --run_a   /path/to/perfect_d0 --label_a perfect_d0 \\
        --run_b   /path/to/mpc_d3     --label_b mpc_d3 \\
        --output  /path/to/heatmap.png
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def _load_route(route_path: Path):
    with open(route_path, "rb") as f:
        return pickle.load(f)


def _build_route_polyline(route) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate route lanelets into one polyline + cumulative arc length.

    Returns (pts (N,2), s (N,)). Skips zero-length segments and drops
    duplicated joint points between consecutive lanelets.
    """
    from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder

    builder = LaneletSceneBuilder(str(route.map_path))
    pts_list = []
    for i, ll_id in enumerate(route.route_lanelet_ids):
        cache = builder._cache.get(int(ll_id))
        if cache is None:
            print(f"  [WARN] lanelet {ll_id} missing from map; skipping")
            continue
        cl = cache.raw_centerline.astype(np.float32)
        if len(cl) == 0:
            continue
        if pts_list:
            # drop leading point if it equals previous tail (lanelet joints
            # usually share a point exactly).
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


def _project_to_polyline(xy: np.ndarray, pts: np.ndarray, s: np.ndarray):
    """Project one point onto polyline. Returns (s_arc, lateral_signed, dist).

    Iterates all segments — fine for ~a few thousand points × ~a few hundred
    steps, but O(N*M). Keep polyline resampled if runs get long.
    """
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
    # Signed side (left = +, right = −) from the cross product
    cross = ab[i, 0] * (xy[1] - a[i, 1]) - ab[i, 1] * (xy[0] - a[i, 0])
    seg_l = math.sqrt(float(seg_len2[i]))
    sign = 1.0 if cross >= 0 else -1.0
    s_arc = float(s[i] + t_clamped[i] * seg_l)
    return s_arc, sign * float(d[i]), float(d[i])


def _recover_ego_world_series(run_dir: Path, route) -> np.ndarray:
    """Recover (T,3) ego world poses [x, y, yaw_rad] from dumped NPZs."""
    npz_dir = run_dir / "npz"
    files = sorted(npz_dir.glob("replay_step_*.npz"))
    if not files:
        raise SystemExit(f"No replay_step_*.npz under {npz_dir}")
    gx_w, gy_w, gyaw_w = float(route.goal_pose[0]), float(route.goal_pose[1]), float(route.goal_pose[2])
    poses = np.zeros((len(files), 3), dtype=np.float64)
    for k, fp in enumerate(files):
        with np.load(fp, allow_pickle=True) as d:
            gp = d["goal_pose"]
        dx, dy, dyaw = float(gp[0]), float(gp[1]), float(gp[2])
        eyaw = gyaw_w - dyaw
        # wrap to [-pi, pi]
        eyaw = math.atan2(math.sin(eyaw), math.cos(eyaw))
        c, s = math.cos(eyaw), math.sin(eyaw)
        ex = gx_w - (c * dx - s * dy)
        ey = gy_w - (s * dx + c * dy)
        poses[k] = (ex, ey, eyaw)
    return poses


def _deviation_series(poses: np.ndarray, pts: np.ndarray, s: np.ndarray):
    """For each ego pose, return (arc, signed_lateral, abs_lateral)."""
    out = np.zeros((len(poses), 3), dtype=np.float64)
    for i, (x, y, _) in enumerate(poses):
        s_arc, lat_signed, lat_abs = _project_to_polyline(np.array([x, y]), pts, s)
        out[i] = (s_arc, lat_signed, lat_abs)
    return out


def _bin_by_arc(dev: np.ndarray, s_max: float, bin_m: float):
    """Mean |lateral| per arc-length bin."""
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


def _segments_from_polyline(pts: np.ndarray, s: np.ndarray, bin_m: float, n_bins: int):
    """For each arc-length bin, collect the sub-polyline that falls in it."""
    segs = [[] for _ in range(n_bins)]
    for i in range(len(pts) - 1):
        s0, s1 = s[i], s[i + 1]
        b0 = int(s0 // bin_m)
        b1 = int(s1 // bin_m)
        if b0 == b1 and 0 <= b0 < n_bins:
            segs[b0].append(pts[i:i + 2])
        else:
            # cut the segment at each bin boundary
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
    # stack each bin
    out = []
    for bs in segs:
        if bs:
            out.append(np.concatenate(bs, axis=0) if len(bs) > 1 else bs[0])
        else:
            out.append(None)
    return out


def _plot_heatmap(ax, pts, bin_segments, bin_values, title, vmin, vmax, cmap):
    ax.set_aspect("equal")
    # light grey base route
    ax.plot(pts[:, 0], pts[:, 1], color="#cccccc", lw=1.0, zorder=1)

    # per-bin coloured overlay
    colors = plt.get_cmap(cmap)
    valid_segs = []
    valid_vals = []
    for segs, v in zip(bin_segments, bin_values):
        if segs is None or np.isnan(v):
            continue
        pairs = segs.reshape(-1, 2, 2) if segs.ndim == 2 and segs.shape[0] % 2 == 0 else None
        # Just draw as a polyline; sequential segments already stitched.
        valid_segs.append(segs)
        valid_vals.append(v)
    if valid_vals:
        lc_lines = []
        lc_colors = []
        for segs, v in zip(valid_segs, valid_vals):
            for i in range(len(segs) - 1):
                lc_lines.append([segs[i], segs[i + 1]])
                lc_colors.append(v)
        lc = LineCollection(lc_lines, array=np.array(lc_colors), cmap=cmap,
                            norm=plt.Normalize(vmin=vmin, vmax=vmax), linewidths=4.0)
        ax.add_collection(lc)
    ax.set_title(title, fontsize=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--route", type=Path, required=True)
    p.add_argument("--run_a", type=Path, required=True)
    p.add_argument("--run_b", type=Path, required=True)
    p.add_argument("--label_a", default="A")
    p.add_argument("--label_b", default="B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bin_m", type=float, default=5.0)
    p.add_argument("--clip_max_m", type=float, default=None,
                   help="Clamp the color scale to this many metres of |lateral|. "
                        "Default: 95th percentile across both runs.")
    p.add_argument("--n_steps", type=int, default=None,
                   help="Truncate both runs to the first N steps. Default: "
                        "min(len(run_a), len(run_b)) — so the comparison "
                        "covers the same temporal window.")
    args = p.parse_args()

    print(f"Loading route {args.route}")
    route = _load_route(args.route)
    print(f"  segments: {len(route.route_lanelet_ids)}  "
          f"start=({route.start_pose[0]:.1f},{route.start_pose[1]:.1f}) "
          f"goal=({route.goal_pose[0]:.1f},{route.goal_pose[1]:.1f})")

    pts, s = _build_route_polyline(route)
    s_max = float(s[-1])
    print(f"Route polyline: {len(pts)} pts, arc length {s_max:.1f} m")

    print(f"[{args.label_a}] recovering ego world poses from {args.run_a}")
    poses_a = _recover_ego_world_series(args.run_a, route)
    print(f"  {len(poses_a)} steps. ego start=({poses_a[0,0]:.1f},{poses_a[0,1]:.1f}) "
          f"end=({poses_a[-1,0]:.1f},{poses_a[-1,1]:.1f})")

    print(f"[{args.label_b}] recovering ego world poses from {args.run_b}")
    poses_b = _recover_ego_world_series(args.run_b, route)
    print(f"  {len(poses_b)} steps. ego start=({poses_b[0,0]:.1f},{poses_b[0,1]:.1f}) "
          f"end=({poses_b[-1,0]:.1f},{poses_b[-1,1]:.1f})")

    n_steps = args.n_steps if args.n_steps is not None else min(len(poses_a), len(poses_b))
    if n_steps < min(len(poses_a), len(poses_b)):
        print(f"Truncating both runs to first {n_steps} steps (explicit --n_steps)")
    else:
        print(f"Comparing first {n_steps} steps of each run (min of both)")
    poses_a = poses_a[:n_steps]
    poses_b = poses_b[:n_steps]
    dev_a = _deviation_series(poses_a, pts, s)
    dev_b = _deviation_series(poses_b, pts, s)

    bin_m = float(args.bin_m)
    bs_mid, mean_a, max_a = _bin_by_arc(dev_a, s_max, bin_m)
    bs_mid_b, mean_b, max_b = _bin_by_arc(dev_b, s_max, bin_m)
    n_bins = len(bs_mid)
    bin_segments = _segments_from_polyline(pts, s, bin_m, n_bins)

    # color scale
    if args.clip_max_m is not None:
        vmax = float(args.clip_max_m)
    else:
        concat = np.concatenate([
            mean_a[~np.isnan(mean_a)], mean_b[~np.isnan(mean_b)]
        ])
        vmax = float(np.percentile(concat, 95)) if len(concat) else 1.0
        vmax = max(vmax, 0.25)
    print(f"Color scale: 0 → {vmax:.2f} m")

    # diff: A − B, symmetric color
    diff = mean_a - mean_b
    diff_abs = np.nanmax(np.abs(diff)) if np.any(~np.isnan(diff)) else 1.0
    diff_abs = max(float(diff_abs), 0.1)

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.6, 1.6, 1.6, 1.0])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a, sharey=ax_a)
    ax_d = fig.add_subplot(gs[2], sharex=ax_a, sharey=ax_a)
    ax_line = fig.add_subplot(gs[3])

    _plot_heatmap(ax_a, pts, bin_segments, mean_a,
                  f"{args.label_a}: |route deviation| per {bin_m:.0f} m bin",
                  0.0, vmax, "viridis")
    _plot_heatmap(ax_b, pts, bin_segments, mean_b,
                  f"{args.label_b}: |route deviation| per {bin_m:.0f} m bin",
                  0.0, vmax, "viridis")
    _plot_heatmap(ax_d, pts, bin_segments, diff,
                  f"{args.label_a} − {args.label_b}  (positive = A worse, blue = B worse)",
                  -diff_abs, diff_abs, "RdBu_r")

    # colorbars
    for ax, vmin_, vmax_, cm in [
        (ax_a, 0.0, vmax, "viridis"),
        (ax_b, 0.0, vmax, "viridis"),
        (ax_d, -diff_abs, diff_abs, "RdBu_r"),
    ]:
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin_, vmax=vmax_))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("m", fontsize=8)

    ax_line.plot(bs_mid, mean_a, label=f"{args.label_a} mean |lat|", color="C0", lw=1.4)
    ax_line.plot(bs_mid, mean_b, label=f"{args.label_b} mean |lat|", color="C1", lw=1.4)
    ax_line.plot(bs_mid, max_a, label=f"{args.label_a} max |lat|", color="C0", lw=0.8, alpha=0.5)
    ax_line.plot(bs_mid, max_b, label=f"{args.label_b} max |lat|", color="C1", lw=0.8, alpha=0.5)
    ax_line.axhline(0.25, color="#888", lw=0.5, ls="--")
    ax_line.set_xlabel("Route arc length (m)")
    ax_line.set_ylabel("Deviation from route centerline (m)")
    ax_line.legend(fontsize=8, ncol=2)
    ax_line.grid(alpha=0.3)

    fig.suptitle(
        f"Route centerline deviation: {args.label_a} vs {args.label_b}  "
        f"({len(poses_a)} / {len(poses_b)} steps, route {s_max:.0f} m, "
        f"{n_bins} bins × {bin_m:.0f} m)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"saved {args.output}")

    # also dump the numeric series for downstream analysis
    np.savez(args.output.with_suffix(".npz"),
             route_pts=pts, route_s=s, bin_m=bin_m, bin_s_mid=bs_mid,
             mean_abs_a=mean_a, mean_abs_b=mean_b, max_abs_a=max_a, max_abs_b=max_b,
             dev_series_a=dev_a, dev_series_b=dev_b,
             poses_a=poses_a, poses_b=poses_b)
    print(f"saved {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
