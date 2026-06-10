"""Realized closed-loop COMFORT (lateral accel + jerk) from a psim bag.

Scores the ACTUAL realized ego motion for ride comfort — NOT the model's open-loop
prediction. Companion to psim_realized_rb.py (same bag + route + arc-binning).

Comfort signals come from /localization/acceleration (geometry_msgs/AccelWithCovarianceStamped),
the EKF-filtered ego acceleration in the base_link frame:
  - lateral accel  = |accel.y|                        (already lateral; no derivation)
  - long accel     = |accel.x|
  - jerk           = |d(accel.x, accel.y)/dt|         (ONE derivative of a FILTERED signal)

This is deliberate: jerk via 3rd-derivative of localization POSITION is hopelessly noisy
(triple-differentiating position noise → spurious 100+ m/s³ spikes). Differentiating the
filtered acceleration once is the correct, low-noise jerk.

Pose (for route-arc projection + speed gating) comes from /localization/kinematic_state;
accel is aligned onto the pose timestamps by interpolation.

Run under the ROS env (lanelet2 + rosbag deserialization), ROS_DOMAIN_ID 20-29.
"""
import argparse
import math
import sqlite3
from pathlib import Path

import numpy as np

from scenario_generation.tools._heatmap_common import (
    build_route_polyline, load_route, project_to_polyline,
    segments_from_polyline, plot_route_heatmap,
)
from rlvr.reward import RewardConfig


def _db3(bag_path: Path) -> Path:
    if bag_path.is_dir():
        cands = sorted(bag_path.glob("*.db3"))
        if not cands:
            raise SystemExit(f"No .db3 files in {bag_path}")
        return cands[0]
    return bag_path


def _extract_poses_times(bag_path: Path):
    """(poses (T,4) [x,y,yaw,speed], times sec) from /localization/kinematic_state."""
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry
    db = sqlite3.connect(str(_db3(bag_path)))
    rows = db.execute(
        "SELECT m.timestamp, m.data FROM messages m JOIN topics t ON m.topic_id=t.id "
        "WHERE t.name='/localization/kinematic_state' ORDER BY m.timestamp").fetchall()
    db.close()
    if not rows:
        raise SystemExit(f"No /localization/kinematic_state in {bag_path}")
    poses = np.zeros((len(rows), 4)); times = np.zeros(len(rows))
    for i, (ts, data) in enumerate(rows):
        msg = deserialize_message(data, Odometry)
        p = msg.pose.pose; q = p.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        v = msg.twist.twist.linear
        poses[i] = [p.position.x, p.position.y, yaw, math.sqrt(v.x ** 2 + v.y ** 2)]
        times[i] = ts * 1e-9
    return poses, times


def _extract_accel_times(bag_path: Path):
    """(accel (T,2) [ax_long, ay_lat] base_link, times sec) from /localization/acceleration."""
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import AccelWithCovarianceStamped
    db = sqlite3.connect(str(_db3(bag_path)))
    rows = db.execute(
        "SELECT m.timestamp, m.data FROM messages m JOIN topics t ON m.topic_id=t.id "
        "WHERE t.name='/localization/acceleration' ORDER BY m.timestamp").fetchall()
    db.close()
    if not rows:
        raise SystemExit(f"No /localization/acceleration in {bag_path}")
    acc = np.zeros((len(rows), 2)); times = np.zeros(len(rows))
    for i, (ts, data) in enumerate(rows):
        msg = deserialize_message(data, AccelWithCovarianceStamped)
        a = msg.accel.accel.linear
        acc[i] = [a.x, a.y]; times[i] = ts * 1e-9
    good = np.isfinite(acc).all(axis=1)  # drop uninitialized (first msg can be NaN)
    return acc[good], times[good]


def _comfort_for_bag(bag, route, pts, arc, args):
    poses, ptimes = _extract_poses_times(Path(bag))
    acc, atimes = _extract_accel_times(Path(bag))
    if args.stride > 1:
        poses, ptimes = poses[::args.stride], ptimes[::args.stride]

    # jerk = d(accel)/dt at native accel rate, then |.|; align onto pose times.
    dt_a = np.median(np.diff(atimes))
    jerk_a = np.linalg.norm(np.gradient(acc, axis=0) / dt_a, axis=1)  # (Ta,)
    lat_a = np.abs(acc[:, 1])                                          # |ay| lateral
    long_a = np.abs(acc[:, 0])
    # interpolate accel-derived signals onto pose timestamps
    lat = np.interp(ptimes, atimes, lat_a)
    lon = np.interp(ptimes, atimes, long_a)
    jerk = np.interp(ptimes, atimes, jerk_a)

    x, y, yaw, speed = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
    arc_max = float(arc.max())
    pose_arc = np.array([float(project_to_polyline(np.array([px, py]), pts, arc)[0])
                         for px, py in zip(x, y)])
    keep = (pose_arc >= args.front_cut) & (pose_arc <= arc_max - args.tail_cut) & (speed > 0.5)
    if keep.sum() == 0:
        raise SystemExit(f"no in-bounds moving poses for {bag}")
    return {"n_in": int(keep.sum()), "keep": keep, "pose_arc": pose_arc,
            "lat_accel": lat, "long_accel": lon, "jerk": jerk, "arc_max": arc_max,
            "dt": float(np.median(np.diff(ptimes)))}


def _bin_max(arc_s, values, s_max, bin_m):
    n_bins = max(1, int(math.ceil(s_max / bin_m)))
    mid = (np.arange(n_bins) + 0.5) * bin_m
    out = np.full(n_bins, np.nan)
    idx = np.clip((arc_s // bin_m).astype(int), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out[b] = np.nanmax(values[m])
    return mid, out


def _print_stats(label, c, max_lat):
    k = c["keep"]; la, lo, jk = c["lat_accel"][k], c["long_accel"][k], c["jerk"][k]
    n_viol = int((la > max_lat).sum())
    q = lambda a, p: float(np.percentile(a, p))
    print(f"{label}: {c['n_in']} steps in-bounds (dt={c['dt']*1000:.0f}ms)")
    print(f"  lat_accel |m/s²| (|accel.y|): mean={la.mean():.2f} p50={q(la,50):.2f} p95={q(la,95):.2f} "
          f"max={la.max():.2f} | >{max_lat:.1f}: {n_viol}/{c['n_in']}")
    print(f"  long_accel|m/s²| (|accel.x|): mean={lo.mean():.2f} p95={q(lo,95):.2f} max={lo.max():.2f}")
    print(f"  jerk |m/s³| (d accel/dt): mean={jk.mean():.2f} p95={q(jk,95):.2f} max={jk.max():.2f}")
    return {"lat_max": la.max(), "lat_p95": q(la, 95), "jerk_mean": jk.mean(),
            "jerk_p95": q(jk, 95), "jerk_max": jk.max(), "n_viol": n_viol}


def _print_per_arc(label, c, bin_m):
    mid, lat_max = _bin_max(c["pose_arc"][c["keep"]], c["lat_accel"][c["keep"]], c["arc_max"], bin_m)
    _, jerk_max = _bin_max(c["pose_arc"][c["keep"]], c["jerk"][c["keep"]], c["arc_max"], bin_m)
    print(f"  [{label}] worst per {bin_m:.0f}m arc bin (lat_accel | jerk):")
    for m, lv, jv in zip(mid, lat_max, jerk_max):
        if not np.isnan(lv):
            print(f"    {m-bin_m/2:.0f}-{m+bin_m/2:.0f}m: lat {lv:.2f}  jerk {jv:.1f}")
    return mid, lat_max, jerk_max


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--baseline_bag", default=None)
    ap.add_argument("--label", default="model")
    ap.add_argument("--baseline_label", default="baseline")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--front_cut", type=float, default=50.0)
    ap.add_argument("--tail_cut", type=float, default=25.0)
    ap.add_argument("--bin_m", type=float, default=100.0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    max_lat = RewardConfig().max_lat_accel
    route = load_route(Path(args.route)); pts, arc = build_route_polyline(route)
    cm = _comfort_for_bag(args.bag, route, pts, arc, args)
    print("=" * 70); sm = _print_stats(args.label, cm, max_lat); _print_per_arc(args.label, cm, args.bin_m)
    if args.baseline_bag:
        cb = _comfort_for_bag(args.baseline_bag, route, pts, arc, args)
        print("-" * 70); sb = _print_stats(args.baseline_label, cb, max_lat); _print_per_arc(args.baseline_label, cb, args.bin_m)
        print("-" * 70)
        print(f"Δ ({args.label} − {args.baseline_label}): lat_max {sm['lat_max']-sb['lat_max']:+.2f} "
              f"lat_p95 {sm['lat_p95']-sb['lat_p95']:+.2f} jerk_mean {sm['jerk_mean']-sb['jerk_mean']:+.2f} "
              f"jerk_p95 {sm['jerk_p95']-sb['jerk_p95']:+.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
