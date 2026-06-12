#!/usr/bin/env python3
"""Build a scenario_generation Route.pkl from an NPZ training/val scene.

Takes the ego's current position as start and the NPZ's ``goal_pose`` (the
scene's recorded planning goal) as end, snaps both to drivable lanelets,
resolves the full lanelet path, and saves a pickled ``Route`` usable by
``scenario_generation.replay``.

Usage:
    python -m scenario_generation.tools.make_route_from_npz \
        --npz /path/to/scene.npz \
        --map /path/to/lanelet2_map.osm \
        --output /path/to/route.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.route import Route


def _quat_to_heading(qx, qy, qz, qw):
    # yaw around Z
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # NPZ is ego-centric; the sidecar JSON next to it carries the world MGRS pose.
    sidecar = args.npz.with_suffix(".json")
    if not sidecar.exists():
        raise SystemExit(f"Sidecar JSON with world pose missing: {sidecar}")
    with open(sidecar) as f:
        pose = json.load(f)
    start_xy = np.array([pose["x"], pose["y"]], dtype=np.float32)
    start_h = _quat_to_heading(pose["qx"], pose["qy"], pose["qz"], pose["qw"])

    # Use the NPZ's route goal_pose (the scene's actual planning goal), NOT
    # the 8-second GT endpoint. goal_pose is [x, y, heading] ego-centric.
    with np.load(args.npz, allow_pickle=True) as raw:
        gp = raw["goal_pose"].copy()
    if np.linalg.norm(gp[:2]) < 1e-3:
        raise ValueError("goal_pose in NPZ is zero — scene has no route goal.")
    gx, gy, gh = float(gp[0]), float(gp[1]), float(gp[2])
    c, s = np.cos(start_h), np.sin(start_h)
    goal_xy = np.array(
        [
            start_xy[0] + gx * c - gy * s,
            start_xy[1] + gx * s + gy * c,
        ],
        dtype=np.float32,
    )
    goal_h = float(start_h + gh)

    print(f"start: ({start_xy[0]:.2f}, {start_xy[1]:.2f}) hdg={np.degrees(start_h):+.1f}°")
    print(f"goal:  ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) hdg={np.degrees(goal_h):+.1f}°")
    print(f"distance start→goal: {np.linalg.norm(goal_xy - start_xy):.1f} m")

    print(f"loading map {args.map}")
    builder = LaneletSceneBuilder(str(args.map))

    start_id = builder.snap_to_nearest_ll(start_xy, heading_rad=start_h)
    if start_id is None:
        raise SystemExit("Could not snap start pose to a drivable lanelet")
    goal_id = builder.snap_to_nearest_ll(
        goal_xy,
        reachable_from=start_id,
        heading_rad=goal_h,
    )
    if goal_id is None:
        raise SystemExit(f"Could not snap goal (reachable from start {start_id})")
    print(f"start_ll={start_id}  goal_ll={goal_id}")

    route_ids = builder.route_with_waypoints(start_id, [], goal_id)
    if route_ids is None:
        raise SystemExit(f"Routing failed start={start_id} → goal={goal_id}")
    print(f"resolved {len(route_ids)} lanelets: {route_ids}")

    route = Route(
        map_path=str(args.map),
        start_pose=np.array([start_xy[0], start_xy[1], start_h], dtype=np.float32),
        goal_pose=np.array([goal_xy[0], goal_xy[1], goal_h], dtype=np.float32),
        start_lanelet_id=start_id,
        goal_lanelet_id=goal_id,
        waypoint_poses=[],
        waypoint_lanelet_ids=[],
        route_lanelet_ids=route_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    route.save(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
