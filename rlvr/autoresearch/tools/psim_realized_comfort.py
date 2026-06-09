"""Realized closed-loop COMFORT (lateral accel + jerk) from a psim bag.

Scores the ACTUAL realized ego motion (from /localization/kinematic_state in the
psim .db3 bag) for ride comfort — NOT the model's open-loop prediction. This is
the metric the real planning stack / vehicle actually produces, which our
reported metrics (RB / CL / collision / L2) never captured.

Companion to psim_realized_rb.py (same bag + route + arc-binning machinery), but
measures dynamics instead of road-border distance.

Reuses (no hand-rolled math):
- world poses + timestamps: local Odometry read (mirrors heatmap_route_deviation._extract_poses_from_bag, adds stamps for correct dt)
- SG differentiation kernels: rlvr.reward._build_sg_diff_kernel (same as the
  lat-accel / jerk the reward computes on the model plan)
- comfort threshold: RewardConfig.max_lat_accel
- route polyline + arc projection + per-arc binning + plot: _heatmap_common

Lateral accel: a_lat = speed * yaw_rate, with speed measured and yaw_rate from an
  SG deriv=1 of unwrapped heading (robust to localization xy noise).
Jerk: |d³(x,y)/dt³| via SG deriv=3 on position (matches reward smoothness notion).
Longitudinal accel: SG deriv=1 of measured speed.

Run under the ROS env (lanelet2 + rosbag deserialization):
  source /opt/ros/humble/setup.bash && \
  ROS_DOMAIN_ID=2X .venv/bin/python -m rlvr.autoresearch.tools.psim_realized_comfort \
    --route <pkl> --bag <model_bag> [--baseline_bag <b>] --out_dir <dir>
"""
import argparse
import math
import sqlite3
from pathlib import Path

import numpy as np
import torch

from scenario_generation.tools._heatmap_common import (
    build_route_polyline, load_route, project_to_polyline,
    segments_from_polyline, plot_route_heatmap,
)
from rlvr.reward import _build_sg_diff_kernel, RewardConfig


def _extract_poses_times(bag_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (poses (T,4) [x,y,yaw,speed], times (T,) seconds) from a psim bag.

    Mirrors heatmap_route_deviation._extract_poses_from_bag but also returns the
    message timestamps so derivatives use the true sample spacing.
    """
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry

    db3 = bag_path
    if bag_path.is_dir():
        cands = sorted(bag_path.glob("*.db3"))
        if not cands:
            raise SystemExit(f"No .db3 files in {bag_path}")
        db3 = cands[0]
    db = sqlite3.connect(str(db3))
    rows = db.execute(
        "SELECT m.timestamp, m.data FROM messages m JOIN topics t ON m.topic_id=t.id "
        "WHERE t.name='/localization/kinematic_state' ORDER BY m.timestamp"
    ).fetchall()
    db.close()
    if not rows:
        raise SystemExit(f"No /localization/kinematic_state in {db3}")
    poses = np.zeros((len(rows), 4), dtype=np.float64)
    times = np.zeros(len(rows), dtype=np.float64)
    for i, (ts, data) in enumerate(rows):
        msg = deserialize_message(data, Odometry)
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        v = msg.twist.twist.linear
        poses[i] = [p.position.x, p.position.y, yaw, math.sqrt(v.x ** 2 + v.y ** 2)]
        times[i] = ts * 1e-9  # bag stamps are nanoseconds
    return poses, times


def _sg(values: np.ndarray, window: int, deriv: int, dt: float) -> np.ndarray:
    """Savitzky-Golay derivative of a 1-D signal via reward's conv1d kernel."""
    if values.shape[0] < window:
        window = values.shape[0] - (1 - values.shape[0] % 2)  # largest odd <= len
    if window < 5:
        raise SystemExit(f"too few samples ({values.shape[0]}) for SG derivative")
    kern = _build_sg_diff_kernel(window=window, poly=3, deriv=deriv, delta=dt)
    pad = window // 2
    x = torch.from_numpy(values).float().view(1, 1, -1)
    x = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
    out = torch.nn.functional.conv1d(x, kern.view(1, 1, -1))
    return out.view(-1).numpy()


def _bin_max(arc_s: np.ndarray, values: np.ndarray, s_max: float, bin_m: float):
    """Worst (max) |value| per arc bin. Returns (bin_mid, max_val) with NaN for empty bins."""
    n_bins = max(1, int(math.ceil(s_max / bin_m)))
    mid = (np.arange(n_bins) + 0.5) * bin_m
    out = np.full(n_bins, np.nan)
    idx = np.clip((arc_s // bin_m).astype(int), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out[b] = np.nanmax(values[m])
    return mid, out


def _comfort_for_bag(bag, route, pts, arc, args):
    """Compute realized lat-accel/jerk along the route for one bag."""
    poses, times = _extract_poses_times(Path(bag))
    if args.stride > 1:
        poses, times = poses[::args.stride], times[::args.stride]
    dt = float(np.median(np.diff(times)))
    if not (dt > 0):
        raise SystemExit(f"non-positive median dt={dt} from bag {bag}")

    x, y, yaw, speed = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
    yaw_u = np.unwrap(yaw)
    yaw_rate = _sg(yaw_u, args.sg_window, deriv=1, dt=dt)        # rad/s
    lat_accel = np.abs(speed * yaw_rate)                         # m/s^2
    long_accel = np.abs(_sg(speed, args.sg_window, deriv=1, dt=dt))  # m/s^2
    jx = _sg(x, args.sg_window, deriv=3, dt=dt)
    jy = _sg(y, args.sg_window, deriv=3, dt=dt)
    jerk = np.sqrt(jx ** 2 + jy ** 2)                            # m/s^3

    arc_max = float(arc.max())
    pose_arc = np.array([float(project_to_polyline(np.array([px, py]), pts, arc)[0])
                         for px, py in zip(x, y)])
    keep = (pose_arc >= args.front_cut) & (pose_arc <= arc_max - args.tail_cut) & (speed > 0.5)
    if keep.sum() == 0:
        raise SystemExit(f"no in-bounds moving poses for {bag}")
    return {
        "dt": dt, "n_in": int(keep.sum()), "keep": keep, "pose_arc": pose_arc,
        "lat_accel": lat_accel, "long_accel": long_accel, "jerk": jerk,
        "arc_max": arc_max,
    }


def _print_stats(label, c, max_lat):
    k = c["keep"]
    la, lo, jk = c["lat_accel"][k], c["long_accel"][k], c["jerk"][k]
    n_viol = int((la > max_lat).sum())
    def q(a, p): return float(np.percentile(a, p))
    print(f"{label}: {c['n_in']} steps in-bounds (dt={c['dt']*1000:.0f}ms)")
    print(f"  lat_accel |m/s²|: mean={la.mean():.2f} p50={q(la,50):.2f} p95={q(la,95):.2f} "
          f"max={la.max():.2f} | >{max_lat:.1f}: {n_viol}/{c['n_in']} ({100*n_viol/c['n_in']:.1f}%)")
    print(f"  long_accel|m/s²|: mean={lo.mean():.2f} p95={q(lo,95):.2f} max={lo.max():.2f}")
    print(f"  jerk   |m/s³|: mean={jk.mean():.2f} p95={q(jk,95):.2f} max={jk.max():.2f}")
    return {"lat_max": la.max(), "lat_p95": q(la, 95), "jerk_p95": q(jk, 95), "n_viol": n_viol}


def _print_per_arc(label, c, bin_m):
    mid, lat_max = _bin_max(c["pose_arc"][c["keep"]], c["lat_accel"][c["keep"]], c["arc_max"], bin_m)
    _, jerk_max = _bin_max(c["pose_arc"][c["keep"]], c["jerk"][c["keep"]], c["arc_max"], bin_m)
    print(f"  [{label}] worst per {bin_m:.0f}m arc bin (lat_accel | jerk):")
    for m, lv, jv in zip(mid, lat_max, jerk_max):
        if not np.isnan(lv):
            print(f"    {m-bin_m/2:.0f}-{m+bin_m/2:.0f}m: lat {lv:.2f}  jerk {jv:.1f}")
    return mid, lat_max, jerk_max


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", required=True)
    ap.add_argument("--bag", required=True, help="model psim bag dir or .db3")
    ap.add_argument("--baseline_bag", default=None, help="optional baseline bag for A/B + overlay")
    ap.add_argument("--label", default="model")
    ap.add_argument("--baseline_label", default="baseline")
    ap.add_argument("--stride", type=int, default=1,
                    help="subsample localization (default 1: keep native rate for clean derivatives)")
    ap.add_argument("--sg_window", type=int, default=21,
                    help="Savitzky-Golay window in samples (~0.2s at 100Hz localization)")
    ap.add_argument("--front_cut", type=float, default=50.0)
    ap.add_argument("--tail_cut", type=float, default=25.0)
    ap.add_argument("--bin_m", type=float, default=100.0)
    ap.add_argument("--out_dir", default=None, help="if set, write per-arc lat-accel heatmap PNG")
    args = ap.parse_args()

    max_lat = RewardConfig().max_lat_accel
    route = load_route(Path(args.route))
    pts, arc = build_route_polyline(route)

    cm = _comfort_for_bag(args.bag, route, pts, arc, args)
    print("=" * 70)
    sm = _print_stats(args.label, cm, max_lat)
    mid_m, latm_m, jerkm_m = _print_per_arc(args.label, cm, args.bin_m)

    cb = None
    if args.baseline_bag:
        cb = _comfort_for_bag(args.baseline_bag, route, pts, arc, args)
        print("-" * 70)
        sb = _print_stats(args.baseline_label, cb, max_lat)
        _print_per_arc(args.baseline_label, cb, args.bin_m)
        print("-" * 70)
        print(f"Δ ({args.label} − {args.baseline_label}): "
              f"lat_max {sm['lat_max']-sb['lat_max']:+.2f}  lat_p95 {sm['lat_p95']-sb['lat_p95']:+.2f}  "
              f"jerk_p95 {sm['jerk_p95']-sb['jerk_p95']:+.2f}  viol {sm['n_viol']-sb['n_viol']:+d}")
    print("=" * 70)

    if args.out_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_bins = len(mid_m)
        seg = segments_from_polyline(pts, arc, args.bin_m, n_bins)
        rows = 2 if cb is not None else 1
        vmax = max(np.nanmax(latm_m), max_lat * 1.5)
        fig, axes = plt.subplots(rows, 1, figsize=(11, 5 * rows), squeeze=False)
        plot_route_heatmap(axes[0][0], pts, seg, latm_m,
                           f"{args.label}: worst lat_accel /{args.bin_m:.0f}m (thr {max_lat:.1f})",
                           0.0, vmax, "RdYlGn_r")
        if cb is not None:
            _, latm_b = _bin_max(cb["pose_arc"][cb["keep"]], cb["lat_accel"][cb["keep"]],
                                 cb["arc_max"], args.bin_m)
            plot_route_heatmap(axes[1][0], pts, seg, latm_b,
                               f"{args.baseline_label}: worst lat_accel /{args.bin_m:.0f}m",
                               0.0, vmax, "RdYlGn_r")
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        png = out / f"realized_comfort_{args.label}.png"
        fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
