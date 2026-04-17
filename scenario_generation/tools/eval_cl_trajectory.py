"""Evaluate closed-loop replay trajectories against road borders and lanes.

Reads ``trajectory_log.json`` from a replay output directory and computes
per-step and aggregate metrics: road border distance, lane departure,
speed profile, path length, and progress toward goal.

Usage:
    python -m scenario_generation.tools.eval_cl_trajectory \
        --run_dirs cl_baseline cl_ep5 cl_ep9 \
        --map_path /path/to/lanelet2_map.osm \
        --ego_length 7.2369 --ego_width 2.29156
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
    """Compute 4 corners of the ego bounding box in world frame."""
    rear_offset = (2 * half_length - wheelbase) / 2
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    corners = []
    for dx, dy in [
        (wheelbase / 2 + rear_offset, half_width),
        (wheelbase / 2 + rear_offset, -half_width),
        (-rear_offset, half_width),
        (-rear_offset, -half_width),
    ]:
        cx = x + dx * cos_h - dy * sin_h
        cy = y + dx * sin_h + dy * cos_h
        corners.append((cx, cy))
    return corners


def _min_dist_to_border_segments(
    corners: list[tuple[float, float]],
    border_segments: list[np.ndarray],
) -> float:
    """Minimum distance from any ego corner to any border line segment."""
    min_d = float("inf")
    for cx, cy in corners:
        p = np.array([cx, cy])
        for seg in border_segments:
            # seg: (N, 2) polyline
            for i in range(len(seg) - 1):
                a, b = seg[i], seg[i + 1]
                ab = b - a
                ab_len2 = float(np.dot(ab, ab))
                if ab_len2 < 1e-12:
                    d = float(np.linalg.norm(p - a))
                else:
                    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / ab_len2))
                    proj = a + t * ab
                    d = float(np.linalg.norm(p - proj))
                min_d = min(min_d, d)
    return min_d


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

    rb_dists = []
    speeds = []
    positions = []

    for entry in traj:
        x, y, h = entry["x"], entry["y"], entry["heading"]
        speed = entry["speed"]
        positions.append((x, y))
        speeds.append(speed)

        corners = _compute_ego_corners(x, y, h, half_l, half_w, ego_wheelbase)
        rb_d = _min_dist_to_border_segments(corners, border_segments)
        rb_dists.append(rb_d)

    rb_dists = np.array(rb_dists)
    speeds = np.array(speeds)
    positions = np.array(positions)

    # Path length
    if len(positions) > 1:
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    else:
        path_length = 0.0

    # RB metrics
    rb_crossings = int((rb_dists < rb_cross_thresh).sum())
    first_rb_cross = int(np.argmax(rb_dists < rb_cross_thresh)) if rb_crossings > 0 else -1

    # Progress
    start_goal_d = traj[0]["goal_d"] if traj else 0
    end_goal_d = traj[-1]["goal_d"] if traj else 0
    progress = start_goal_d - end_goal_d

    # Duration
    duration_s = len(traj) * 0.1

    return {
        "n_steps": len(traj),
        "duration_s": duration_s,
        "path_length_m": path_length,
        "progress_m": progress,
        "start_goal_d": start_goal_d,
        "end_goal_d": end_goal_d,
        "mean_speed_mps": float(speeds.mean()) if len(speeds) > 0 else 0,
        "max_speed_mps": float(speeds.max()) if len(speeds) > 0 else 0,
        "rb_dist_min": float(rb_dists.min()) if len(rb_dists) > 0 else 0,
        "rb_dist_p5": float(np.percentile(rb_dists, 5)) if len(rb_dists) > 0 else 0,
        "rb_dist_p25": float(np.percentile(rb_dists, 25)) if len(rb_dists) > 0 else 0,
        "rb_dist_med": float(np.median(rb_dists)) if len(rb_dists) > 0 else 0,
        "rb_cross_steps": rb_crossings,
        "rb_cross_frac": rb_crossings / max(len(traj), 1),
        "first_rb_cross_step": first_rb_cross,
        "stopped_steps": int((speeds < 0.1).sum()),
        "stopped_frac": float((speeds < 0.1).mean()) if len(speeds) > 0 else 0,
    }


def load_border_segments(map_path: str) -> list[np.ndarray]:
    """Load road border polylines from a lanelet2 map."""
    import lanelet2
    from lanelet2.io import Origin, load
    from lanelet2.projection import MGRSProjector

    projector = MGRSProjector(Origin(0.0, 0.0))
    ll_map = load(map_path, projector)

    segments = []
    for ls in ll_map.lineStringLayer:
        attrs = ls.attributes
        ls_type = attrs.get("type", "")
        ls_subtype = attrs.get("subtype", "")
        if ls_type == "road_border" or ls_subtype == "road_border":
            pts = np.array([[p.x, p.y] for p in ls], dtype=np.float64)
            if len(pts) >= 2:
                segments.append(pts)
    print(f"Loaded {len(segments)} road border segments from map")
    return segments


def main():
    parser = argparse.ArgumentParser(description="Evaluate CL replay trajectories")
    parser.add_argument("--run_dirs", nargs="+", required=True,
                        help="Replay output directories (each must contain trajectory_log.json)")
    parser.add_argument("--map_path", required=True, help="Lanelet2 map OSM file")
    parser.add_argument("--ego_length", type=float, default=7.2369)
    parser.add_argument("--ego_width", type=float, default=2.29156)
    parser.add_argument("--ego_wheelbase", type=float, default=4.76012)
    parser.add_argument("--rb_cross_thresh", type=float, default=0.20)
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    args = parser.parse_args()

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
