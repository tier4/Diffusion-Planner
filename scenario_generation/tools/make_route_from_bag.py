#!/usr/bin/env python3
"""Extract a ``scenario_generation.Route`` pickle from an Autoware rosbag.

Straight copy out of ``/planning/route``
(``autoware_planning_msgs/msg/LaneletRoute``): no re-routing, no snapping,
no shortest-path approximation. The bag already carries the authoritative
``start_pose``, ``goal_pose``, and the full ordered
``segments[].preferred_primitive.id`` list — we simply pickle those into a
``Route``.

We do NOT open ``metadata.yaml`` for the bag (the files we got don't have
one): sqlite3 reads the ``.db3`` directly and ROS2 deserialises the
message blob.

Usage:
    python -m scenario_generation.tools.make_route_from_bag \\
        --bag /path/to/bag_dir_or.db3 \\
        --map /path/to/lanelet2_map.osm \\
        --output /path/to/route.pkl
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import numpy as np


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return float(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def _read_route_msg(db_path: Path):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    row = cur.execute(
        "SELECT id, type FROM topics WHERE name='/planning/route'"
    ).fetchone()
    if row is None:
        raise SystemExit(f"/planning/route topic not found in {db_path}")
    tid, typ = row
    msg_cls = get_message(typ)
    raw = cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (tid,),
    ).fetchall()
    con.close()
    if not raw:
        raise SystemExit(f"No /planning/route messages in {db_path}")
    # The route is published once per mission; take the last one (in case
    # the mission was re-sent mid-recording).
    _, data = raw[-1]
    return deserialize_message(data, msg_cls), len(raw)


def _resolve_db3(bag_arg: Path) -> Path:
    if bag_arg.is_file() and bag_arg.suffix == ".db3":
        return bag_arg
    if bag_arg.is_dir():
        cands = sorted(bag_arg.glob("*.db3"))
        if len(cands) == 1:
            return cands[0]
        if len(cands) == 0:
            raise SystemExit(f"No .db3 in {bag_arg}")
        raise SystemExit(
            f"Multiple .db3 in {bag_arg}; pass one explicitly: {cands}"
        )
    raise SystemExit(f"--bag must be a .db3 or a directory containing one, got {bag_arg}")


def _assert_ids_in_map(lanelet_ids, map_path: Path) -> None:
    """Refuse to save a route referencing lanelet ids the map can't resolve."""
    from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder

    print(f"Loading map {map_path} to validate segment ids")
    builder = LaneletSceneBuilder(str(map_path))
    known = set(builder._ll_by_id.keys())
    missing = [i for i in lanelet_ids if i not in known]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(lanelet_ids)} bag lanelet ids not present in "
            f"{map_path}. The bag must have been recorded on a different map "
            f"version. First few missing: {missing[:10]}"
        )
    print(f"  all {len(lanelet_ids)} bag segment ids present in map")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bag", type=Path, required=True,
                   help="Path to rosbag .db3 (or directory containing one)")
    p.add_argument("--map", type=Path, required=True,
                   help="Path to lanelet2_map.osm the Route will reference at replay time")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--skip_map_check", action="store_true",
                   help="Don't validate that every bag segment id exists in --map")
    args = p.parse_args()

    db_path = _resolve_db3(args.bag)
    print(f"Reading /planning/route from {db_path}")
    msg, n_msgs = _read_route_msg(db_path)
    print(f"  found {n_msgs} route message(s); using the last one")

    sp = msg.start_pose
    gp = msg.goal_pose
    sxy = np.array([sp.position.x, sp.position.y], dtype=np.float32)
    gxy = np.array([gp.position.x, gp.position.y], dtype=np.float32)
    syaw = _quat_to_yaw(sp.orientation.x, sp.orientation.y, sp.orientation.z, sp.orientation.w)
    gyaw = _quat_to_yaw(gp.orientation.x, gp.orientation.y, gp.orientation.z, gp.orientation.w)
    bag_segment_ids = [int(s.preferred_primitive.id) for s in msg.segments]
    if len(bag_segment_ids) < 2:
        raise SystemExit(f"Bag route has < 2 segments ({bag_segment_ids})")
    print(f"  start=({sxy[0]:.2f},{sxy[1]:.2f}) yaw={math.degrees(syaw):+.1f}°")
    print(f"  goal =({gxy[0]:.2f},{gxy[1]:.2f}) yaw={math.degrees(gyaw):+.1f}°")
    print(f"  segments (bag, used as-is): {len(bag_segment_ids)}")
    print(f"    head: {bag_segment_ids[:5]}")
    print(f"    tail: {bag_segment_ids[-5:]}")

    if not args.skip_map_check:
        _assert_ids_in_map(bag_segment_ids, args.map)

    from scenario_generation.route import Route

    route = Route(
        map_path=str(args.map),
        start_pose=np.array([sxy[0], sxy[1], syaw], dtype=np.float32),
        goal_pose=np.array([gxy[0], gxy[1], gyaw], dtype=np.float32),
        start_lanelet_id=bag_segment_ids[0],
        goal_lanelet_id=bag_segment_ids[-1],
        waypoint_poses=[],
        waypoint_lanelet_ids=[],
        route_lanelet_ids=bag_segment_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    route.save(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
