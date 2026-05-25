#!/usr/bin/env python3
"""Route-centerline deviation heatmap for two replay runs.

Compares two closed-loop sim dumps on the SAME ``scenario_generation.Route``
(e.g. perfect-tracker vs MPC+delay=3) by projecting every dumped ego pose
onto the route's concatenated lanelet centerlines and colouring the route
geometry by the resulting lateral deviation.

Output: a single PNG with three stacked route maps (run A, run B, A−B)
plus a deviation-vs-route-arc-length line chart.

Two input modes:

1. **NPZ mode** (``--run_a`` / ``--run_b``): recovers ego world pose from
   each dumped NPZ's ``goal_pose`` field + Route pickle's goal.

2. **Rosbag mode** (``--bag_a`` / ``--bag_b``): extracts ego world pose
   directly from ``/localization/kinematic_state`` in a psim ``.db3`` bag.
   More robust for cpp-converted NPZs where goal_pose recovery may have
   coordinate mismatches.

Usage:
    # NPZ mode
    python -m scenario_generation.tools.heatmap_route_deviation \\
        --route /path/to/route.pkl \\
        --run_a /path/to/npz_dir --run_b /path/to/npz_dir \\
        --output /path/to/heatmap.png

    # Rosbag mode
    python -m scenario_generation.tools.heatmap_route_deviation \\
        --route /path/to/route.pkl \\
        --bag_a /path/to/run_a.db3 --bag_b /path/to/run_b.db3 \\
        --output /path/to/heatmap.png
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scenario_generation.tools._heatmap_common import (
    bin_by_arc,
    build_route_polyline,
    deviation_series,
    load_route,
    plot_route_heatmap,
    recover_ego_world_pose_from_goal,
    segments_from_polyline,
)


def _recover_ego_world_series(run_dir: Path, route) -> np.ndarray:
    """Recover (T,4) array [x, y, yaw_rad, speed_mps] from dumped NPZs.

    Speed comes from ego_current_state[4:6] (vx, vy in base_link).
    """
    npz_dir = run_dir / "npz"
    files = sorted(npz_dir.glob("replay_step_*.npz"))
    if not files:
        files = sorted(f for f in npz_dir.glob("*.npz")
                        if "summary" not in f.name and "heatmap" not in f.name)
    if not files:
        raise SystemExit(f"No step NPZs under {npz_dir}")
    poses = np.zeros((len(files), 4), dtype=np.float64)
    for k, fp in enumerate(files):
        with np.load(fp, allow_pickle=True) as d:
            gp = d["goal_pose"]
            ecs = d["ego_current_state"]
        poses[k, :3] = recover_ego_world_pose_from_goal(gp, route)
        poses[k, 3] = float(np.linalg.norm(ecs[4:6]))
    return poses


def _extract_poses_from_bag(bag_path: Path) -> np.ndarray:
    """Extract (T,4) [x, y, yaw, speed] from a psim rosbag .db3 file."""
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry

    db3 = bag_path
    if bag_path.is_dir():
        candidates = sorted(bag_path.glob("*.db3"))
        if not candidates:
            raise SystemExit(f"No .db3 files in {bag_path}")
        db3 = candidates[0]

    db = sqlite3.connect(str(db3))
    rows = db.execute(
        "SELECT m.data FROM messages m JOIN topics t ON m.topic_id=t.id "
        "WHERE t.name='/localization/kinematic_state' ORDER BY m.timestamp"
    ).fetchall()
    db.close()
    if not rows:
        raise SystemExit(f"No /localization/kinematic_state in {db3}")

    poses = np.zeros((len(rows), 4), dtype=np.float64)
    for i, (data,) in enumerate(rows):
        msg = deserialize_message(data, Odometry)
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y ** 2 + q.z ** 2))
        v = msg.twist.twist.linear
        poses[i] = [p.position.x, p.position.y, yaw,
                    math.sqrt(v.x ** 2 + v.y ** 2)]
    return poses


def _print_stats(dev_a: np.ndarray, dev_b: np.ndarray,
                 label_a: str, label_b: str,
                 arc_ranges: list[tuple[float, float]] | None = None) -> None:
    """Print overall and per-arc deviation statistics."""
    lat_a, lat_b = dev_a[:, 2], dev_b[:, 2]

    print(f"\n{'='*60}")
    print(f"  Overall statistics")
    print(f"{'='*60}")
    print(f"  {'Metric':>8s}  {label_a:>12s}  {label_b:>12s}  {'Delta':>8s}")
    for name, fa, fb in [
        ("Mean", np.mean(lat_a), np.mean(lat_b)),
        ("Median", np.median(lat_a), np.median(lat_b)),
        ("p95", np.percentile(lat_a, 95), np.percentile(lat_b, 95)),
        ("Max", np.max(lat_a), np.max(lat_b)),
    ]:
        pct = (fb - fa) / fa * 100 if fa > 1e-9 else 0.0
        print(f"  {name:>8s}  {fa:12.3f}m  {fb:12.3f}m  {pct:+7.1f}%")

    if arc_ranges:
        print(f"\n{'='*60}")
        print(f"  Per-arc breakdown (mean |lateral|)")
        print(f"{'='*60}")
        print(f"  {'Arc':>14s}  {label_a:>10s}  {label_b:>10s}  {'Delta':>8s}  {'Abs Δ':>8s}")
        for lo, hi in arc_ranges:
            ma = lat_a[(dev_a[:, 0] >= lo) & (dev_a[:, 0] <= hi)]
            mb = lat_b[(dev_b[:, 0] >= lo) & (dev_b[:, 0] <= hi)]
            if len(ma) and len(mb):
                mean_a_arc = np.mean(ma)
                mean_b_arc = np.mean(mb)
                pct = (mean_b_arc - mean_a_arc) / mean_a_arc * 100 if mean_a_arc > 1e-9 else 0.0
                abs_d = mean_b_arc - mean_a_arc
                print(f"  {lo:>6.0f}-{hi:<6.0f}  {mean_a_arc:10.3f}m  {mean_b_arc:10.3f}m  {pct:+7.1f}%  {abs_d:+7.3f}m")

        print(f"\n  Per-arc breakdown (max |lateral|)")
        print(f"  {'Arc':>14s}  {label_a:>10s}  {label_b:>10s}  {'Delta':>8s}  {'Abs Δ':>8s}")
        for lo, hi in arc_ranges:
            ma = lat_a[(dev_a[:, 0] >= lo) & (dev_a[:, 0] <= hi)]
            mb = lat_b[(dev_b[:, 0] >= lo) & (dev_b[:, 0] <= hi)]
            if len(ma) and len(mb):
                max_a_arc = np.max(ma)
                max_b_arc = np.max(mb)
                pct = (max_b_arc - max_a_arc) / max_a_arc * 100 if max_a_arc > 1e-9 else 0.0
                abs_d = max_b_arc - max_a_arc
                print(f"  {lo:>6.0f}-{hi:<6.0f}  {max_a_arc:10.3f}m  {max_b_arc:10.3f}m  {pct:+7.1f}%  {abs_d:+7.3f}m")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--route", type=Path, required=True)
    p.add_argument("--run_a", type=Path, default=None,
                   help="NPZ dir for run A (goal_pose recovery mode)")
    p.add_argument("--run_b", type=Path, default=None,
                   help="NPZ dir for run B (goal_pose recovery mode)")
    p.add_argument("--bag_a", type=Path, default=None,
                   help="Rosbag .db3 or dir for run A (direct extraction)")
    p.add_argument("--bag_b", type=Path, default=None,
                   help="Rosbag .db3 or dir for run B (direct extraction)")
    p.add_argument("--label_a", default="A")
    p.add_argument("--label_b", default="B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arc_ranges", type=str, default=None,
                   help="Comma-separated arc ranges for per-arc stats, "
                        "e.g. '450,510;728,768;1400,1442'")
    p.add_argument("--bin_m", type=float, default=5.0)
    p.add_argument("--clip_max_m", type=float, default=None,
                   help="Clamp the color scale to this many metres of |lateral|. "
                        "Default: 95th percentile across both runs.")
    p.add_argument("--n_steps", type=int, default=None,
                   help="Truncate both runs to the first N steps. Default: "
                        "min(len(run_a), len(run_b)) — so the comparison "
                        "covers the same temporal window.")
    p.add_argument("--min_arc_m", type=float, default=None,
                   help="Exclude route before this arc-length (metres). "
                        "Useful to trim the initial convergence from a "
                        "misplaced starting pose.")
    p.add_argument("--max_arc_m", type=float, default=None,
                   help="Exclude route beyond this arc-length (metres). "
                        "Useful to trim the last N metres where the goal is "
                        "far from the centerline.")
    p.add_argument("--min_speed_mps", type=float, default=0.5,
                   help="Exclude steps where ego speed is below this (m/s). "
                        "Filters out stopped / crawling frames so they don't "
                        "pollute the deviation signal. Default: 0.5 m/s.")
    args = p.parse_args()

    print(f"Loading route {args.route}")
    route = load_route(args.route)
    print(f"  segments: {len(route.route_lanelet_ids)}  "
          f"start=({route.start_pose[0]:.1f},{route.start_pose[1]:.1f}) "
          f"goal=({route.goal_pose[0]:.1f},{route.goal_pose[1]:.1f})")

    pts, s = build_route_polyline(route)
    s_max = float(s[-1])
    print(f"Route polyline: {len(pts)} pts, arc length {s_max:.1f} m")

    src_a = args.bag_a or args.run_a
    src_b = args.bag_b or args.run_b
    if src_a is None or src_b is None:
        raise SystemExit("Provide either --run_a/--run_b (NPZ) or --bag_a/--bag_b (rosbag)")

    if args.bag_a:
        print(f"[{args.label_a}] extracting poses from rosbag {args.bag_a}")
        poses_a = _extract_poses_from_bag(args.bag_a)
    else:
        print(f"[{args.label_a}] recovering ego world poses from {args.run_a}")
        poses_a = _recover_ego_world_series(args.run_a, route)
    print(f"  {len(poses_a)} steps. ego start=({poses_a[0,0]:.1f},{poses_a[0,1]:.1f}) "
          f"end=({poses_a[-1,0]:.1f},{poses_a[-1,1]:.1f})")

    if args.bag_b:
        print(f"[{args.label_b}] extracting poses from rosbag {args.bag_b}")
        poses_b = _extract_poses_from_bag(args.bag_b)
    else:
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

    if args.min_speed_mps > 0:
        mask_a = poses_a[:, 3] >= args.min_speed_mps
        mask_b = poses_b[:, 3] >= args.min_speed_mps
        n_drop_a = int((~mask_a).sum())
        n_drop_b = int((~mask_b).sum())
        poses_a = poses_a[mask_a]
        poses_b = poses_b[mask_b]
        print(f"Speed filter >= {args.min_speed_mps:.1f} m/s: "
              f"dropped {n_drop_a}/{n_steps} A, {n_drop_b}/{n_steps} B")

    dev_a = deviation_series(poses_a[:, :2], pts, s)
    dev_b = deviation_series(poses_b[:, :2], pts, s)

    if args.min_arc_m is not None:
        dev_a = dev_a[dev_a[:, 0] >= args.min_arc_m]
        dev_b = dev_b[dev_b[:, 0] >= args.min_arc_m]
        print(f"Trimmed to arc >= {args.min_arc_m:.1f} m  "
              f"(A: {len(dev_a)} pts, B: {len(dev_b)} pts)")

    if args.max_arc_m is not None:
        dev_a = dev_a[dev_a[:, 0] <= args.max_arc_m]
        dev_b = dev_b[dev_b[:, 0] <= args.max_arc_m]
        s_max = min(s_max, args.max_arc_m)
        print(f"Trimmed to arc <= {args.max_arc_m:.1f} m  "
              f"(A: {len(dev_a)} pts, B: {len(dev_b)} pts)")

    arc_ranges = None
    if args.arc_ranges:
        arc_ranges = []
        for pair in args.arc_ranges.split(";"):
            lo, hi = pair.strip().split(",")
            arc_ranges.append((float(lo), float(hi)))

    _print_stats(dev_a, dev_b, args.label_a, args.label_b, arc_ranges)

    bin_m = float(args.bin_m)
    bs_mid, mean_a, max_a = bin_by_arc(dev_a, s_max, bin_m)
    bs_mid_b, mean_b, max_b = bin_by_arc(dev_b, s_max, bin_m)
    n_bins = len(bs_mid)
    bin_segments = segments_from_polyline(pts, s, bin_m, n_bins)

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

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.8, 1.0],
                          width_ratios=[1, 1, 1], hspace=0.25, wspace=0.15)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharex=ax_a, sharey=ax_a)
    ax_d = fig.add_subplot(gs[0, 2], sharex=ax_a, sharey=ax_a)
    ax_line = fig.add_subplot(gs[1, :])

    plot_route_heatmap(ax_a, pts, bin_segments, mean_a,
                  f"{args.label_a}: |deviation| per {bin_m:.0f} m bin",
                  0.0, vmax, "viridis")
    plot_route_heatmap(ax_b, pts, bin_segments, mean_b,
                  f"{args.label_b}: |deviation| per {bin_m:.0f} m bin",
                  0.0, vmax, "viridis")
    plot_route_heatmap(ax_d, pts, bin_segments, diff,
                  f"A − B  (red = A worse, blue = B worse)",
                  -diff_abs, diff_abs, "RdBu_r")

    for ax, vmin_, vmax_, cm in [
        (ax_a, 0.0, vmax, "viridis"),
        (ax_b, 0.0, vmax, "viridis"),
        (ax_d, -diff_abs, diff_abs, "RdBu_r"),
    ]:
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin_, vmax=vmax_))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.04, aspect=30)
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
