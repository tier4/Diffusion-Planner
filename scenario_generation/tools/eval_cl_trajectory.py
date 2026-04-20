"""Evaluate closed-loop replay trajectories against road borders.

Reads ``trajectory_log.json`` from a replay output directory and computes
per-step and aggregate metrics: road-border distance/crossing, speed
profile, path length, stopped fraction, and progress toward goal.

Usage:
    python -m scenario_generation.tools.eval_cl_trajectory \
        --run_dirs cl_baseline cl_ep5 cl_ep9 \
        --map_path /path/to/lanelet2_map.osm \
        --ego_length 4.5 --ego_width 1.9 --ego_wheelbase 2.925
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_trajectory(run_dir: Path) -> list[dict]:
    log_path = run_dir / "trajectory_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"No trajectory_log.json in {run_dir}")
    with open(log_path) as f:
        return json.load(f)


def _compute_ego_corners(
    x: float, y: float, heading: float,
    half_length: float, half_width: float, wheelbase: float,
) -> list[tuple[float, float]]:
    """Compute 4 corners of the ego bounding box in world frame.

    Matches the repo-wide convention (e.g. ``gui.lanelet_scene_builder._obb_corners``
    and ``visualize.draw_agent_box``) where ``(x, y)`` is the rear-axle
    position. The longitudinal footprint spans from ``-rear_overhang`` behind
    the rear axle to ``wheelbase + rear_overhang`` in front of it.
    """
    length = 2 * half_length
    rear_overhang = (length - wheelbase) / 2
    front_offset = wheelbase + rear_overhang
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    corners = []
    for dx, dy in [
        (front_offset, half_width),
        (front_offset, -half_width),
        (-rear_overhang, half_width),
        (-rear_overhang, -half_width),
    ]:
        cx = x + dx * cos_h - dy * sin_h
        cy = y + dx * sin_h + dy * cos_h
        corners.append((cx, cy))
    return corners


def _flatten_segments(border_segments: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten list of polylines into (M, 2) segment start + (M, 2) segment end arrays."""
    starts = []
    ends = []
    for seg in border_segments:
        if len(seg) < 2:
            continue
        starts.append(seg[:-1])
        ends.append(seg[1:])
    if not starts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(starts, axis=0), np.concatenate(ends, axis=0)


def _min_dist_vectorized(
    corners: np.ndarray,  # (4, 2)
    seg_starts: np.ndarray,  # (M, 2)
    seg_ends: np.ndarray,  # (M, 2)
) -> float:
    """Vectorized min distance from any corner to any line segment."""
    # ab: (M, 2), ab_len2: (M,)
    ab = seg_ends - seg_starts
    ab_len2 = (ab * ab).sum(axis=1)
    ab_len2_safe = np.where(ab_len2 < 1e-12, 1.0, ab_len2)

    # For each corner, compute distance to all segments in one pass.
    # ap: (4, M, 2)
    ap = corners[:, None, :] - seg_starts[None, :, :]
    # dot: (4, M)
    dot = (ap * ab[None, :, :]).sum(axis=2)
    t = np.clip(dot / ab_len2_safe[None, :], 0.0, 1.0)
    # proj: (4, M, 2)
    proj = seg_starts[None, :, :] + t[:, :, None] * ab[None, :, :]
    # dist: (4, M)
    delta = corners[:, None, :] - proj
    dist = np.sqrt((delta * delta).sum(axis=2))
    # Handle degenerate (zero-length) segments
    dist_deg = np.linalg.norm(corners[:, None, :] - seg_starts[None, :, :], axis=2)
    dist = np.where(ab_len2[None, :] < 1e-12, dist_deg, dist)
    return float(dist.min())


def evaluate_trajectory(
    traj: list[dict],
    border_segments: list[np.ndarray],
    ego_length: float,
    ego_width: float,
    ego_wheelbase: float,
    rb_cross_thresh: float = 0.20,
) -> dict:
    """Compute metrics for a single CL trajectory."""
    half_l = ego_length / 2
    half_w = ego_width / 2

    seg_starts, seg_ends = _flatten_segments(border_segments)
    has_borders = seg_starts.shape[0] > 0

    rb_dists = []
    speeds = []
    positions = []

    for entry in traj:
        x, y, h = entry["x"], entry["y"], entry["heading"]
        speed = entry["speed"]
        positions.append((x, y))
        speeds.append(speed)

        if has_borders:
            corners = np.array(
                _compute_ego_corners(x, y, h, half_l, half_w, ego_wheelbase),
                dtype=np.float64,
            )
            rb_dists.append(_min_dist_vectorized(corners, seg_starts, seg_ends))

    rb_dists = np.array(rb_dists)
    speeds = np.array(speeds)
    positions = np.array(positions)

    # Path length
    if len(positions) > 1:
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    else:
        path_length = 0.0

    # RB metrics (skipped if the map has no road-border polylines).
    if len(rb_dists) > 0:
        rb_crossings = int((rb_dists < rb_cross_thresh).sum())
        first_rb_cross = int(np.argmax(rb_dists < rb_cross_thresh)) if rb_crossings > 0 else -1
    else:
        rb_crossings = 0
        first_rb_cross = -1

    # Progress
    start_goal_d = traj[0]["goal_d"] if traj else 0
    end_goal_d = traj[-1]["goal_d"] if traj else 0
    progress = start_goal_d - end_goal_d

    # Duration
    duration_s = len(traj) * 0.1

    # When the map has no road-border polylines we return NaN for
    # distance-valued metrics so they are distinguishable from a real
    # zero-distance crossing in downstream summaries/plots.
    rb_has_data = len(rb_dists) > 0
    return {
        "n_steps": len(traj),
        "duration_s": duration_s,
        "path_length_m": path_length,
        "progress_m": progress,
        "start_goal_d": start_goal_d,
        "end_goal_d": end_goal_d,
        "mean_speed_mps": float(speeds.mean()) if len(speeds) > 0 else 0,
        "max_speed_mps": float(speeds.max()) if len(speeds) > 0 else 0,
        "rb_has_data": rb_has_data,
        "rb_dist_min": float(rb_dists.min()) if rb_has_data else float("nan"),
        "rb_dist_p5": float(np.percentile(rb_dists, 5)) if rb_has_data else float("nan"),
        "rb_dist_p25": float(np.percentile(rb_dists, 25)) if rb_has_data else float("nan"),
        "rb_dist_med": float(np.median(rb_dists)) if rb_has_data else float("nan"),
        "rb_cross_steps": rb_crossings,
        "rb_cross_frac": rb_crossings / max(len(traj), 1) if rb_has_data else float("nan"),
        "first_rb_cross_step": first_rb_cross,
        "stopped_steps": int((speeds < 0.1).sum()),
        "stopped_frac": float((speeds < 0.1).mean()) if len(speeds) > 0 else 0,
    }


def load_border_segments(map_path: str) -> list[np.ndarray]:
    """Load road border polylines from a lanelet2 map."""
    import lanelet2
    from autoware_lanelet2_extension_python.projection import MGRSProjector

    projector = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    ll_map = lanelet2.io.load(map_path, projector)

    segments = []
    for ls in ll_map.lineStringLayer:
        attrs = ls.attributes
        ls_type = attrs["type"] if "type" in attrs else ""
        ls_subtype = attrs["subtype"] if "subtype" in attrs else ""
        if ls_type == "road_border" or ls_subtype == "road_border":
            pts = np.array([[p.x, p.y] for p in ls], dtype=np.float64)
            if len(pts) >= 2:
                segments.append(pts)
    print(f"Loaded {len(segments)} road border segments from map")
    return segments


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from e
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"value must be > 0, got {parsed}")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Evaluate CL replay trajectories")
    parser.add_argument("--run_dirs", nargs="+", required=True,
                        help="Replay output directories (each must contain trajectory_log.json)")
    parser.add_argument("--map_path", required=True, help="Lanelet2 map OSM file")
    parser.add_argument("--ego_length", type=_positive_float, required=True,
                        help="Ego length (m) — must match the vehicle used for replay")
    parser.add_argument("--ego_width", type=_positive_float, required=True,
                        help="Ego width (m) — must match the vehicle used for replay")
    parser.add_argument("--ego_wheelbase", type=_positive_float, required=True,
                        help="Ego wheelbase (m) — must match the vehicle used for replay")
    parser.add_argument("--rb_cross_thresh", type=float, default=0.20)
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    args = parser.parse_args()

    if args.ego_wheelbase > args.ego_length:
        parser.error(
            f"--ego_wheelbase ({args.ego_wheelbase}) must be <= "
            f"--ego_length ({args.ego_length})"
        )

    border_segments = load_border_segments(args.map_path)

    results = {}
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        name = run_dir.name
        print(f"\n=== {name} ===")
        try:
            traj = _load_trajectory(run_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        metrics = evaluate_trajectory(
            traj, border_segments,
            args.ego_length, args.ego_width, args.ego_wheelbase,
            args.rb_cross_thresh,
        )
        results[name] = metrics

        print(f"  Steps: {metrics['n_steps']}, Duration: {metrics['duration_s']:.1f}s")
        print(f"  Path: {metrics['path_length_m']:.1f}m, Progress: {metrics['progress_m']:.1f}m")
        print(f"  Speed: mean={metrics['mean_speed_mps']:.2f} max={metrics['max_speed_mps']:.2f} m/s")
        print(f"  RB dist: min={metrics['rb_dist_min']:.3f} p5={metrics['rb_dist_p5']:.3f} "
              f"p25={metrics['rb_dist_p25']:.3f} med={metrics['rb_dist_med']:.3f}")
        print(f"  RB crossings: {metrics['rb_cross_steps']} steps ({metrics['rb_cross_frac']:.1%})")
        if metrics['first_rb_cross_step'] >= 0:
            print(f"    First crossing at step {metrics['first_rb_cross_step']} "
                  f"({metrics['first_rb_cross_step'] * 0.1:.1f}s)")
        print(f"  Stopped: {metrics['stopped_steps']} steps ({metrics['stopped_frac']:.1%})")
        print(f"  Goal: {metrics['start_goal_d']:.0f}m -> {metrics['end_goal_d']:.0f}m")

    if len(results) > 1:
        print("\n=== COMPARISON ===")
        header = f"{'Metric':<20s}"
        for name in results:
            header += f" {name:>15s}"
        print(header)
        for key in ["path_length_m", "mean_speed_mps", "rb_dist_min", "rb_dist_med",
                     "rb_cross_steps", "stopped_frac", "progress_m"]:
            row = f"{key:<20s}"
            for name in results:
                v = results[name][key]
                row += f" {v:>15.3f}" if isinstance(v, float) else f" {v:>15d}"
            print(row)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
