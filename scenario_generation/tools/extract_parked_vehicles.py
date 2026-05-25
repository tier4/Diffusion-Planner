#!/usr/bin/env python3
"""Extract truly-parked vehicles from multiple rosbag splits.

Unlike the single-bag script, this processes ALL bags for a route together
so that track UUIDs are merged across splits. A vehicle is only considered
parked if its max speed across ALL observations is below the threshold.
This filters out cars temporarily stopped at traffic lights — they will
eventually move in a later split.

Requires: ROS2 Humble + pilot-auto sourced (for rclpy deserialization).

Usage:
    bash -c "source /opt/ros/humble/setup.bash && \\
             source ~/pilot-auto.x2/install/setup.bash && \\
             python3 -m scenario_generation.tools.extract_parked_vehicles \\
                 --bags /path/to/bag1.db3 /path/to/bag2.db3 \\
                 --osm_path /path/to/lanelet2_map.osm \\
                 --output /path/to/parked.yaml"
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path.home() / "at-team-tools" / "sakoda"))
from extract_parked_vehicles_from_rosbag import (
    VEHICLE_LABELS,
    collect_tracks_and_ego,
    merge_overlapping_vehicles,
    summarize_track,
    visualize,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bags", type=Path, nargs="+", required=True,
                   help="All .db3 bag files for this route (order doesn't matter)")
    p.add_argument("--osm_path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--velocity_threshold", type=float, default=0.5)
    p.add_argument("--min_frames", type=int, default=10)
    p.add_argument("--ego_overlap_threshold", type=float, default=2.0)
    p.add_argument("--topic", default="/perception/object_recognition/tracking/objects")
    p.add_argument("--kinematic_topic", default="/localization/kinematic_state")
    p.add_argument("--no_viz", action="store_true")
    args = p.parse_args()

    merged_tracks: dict[str, dict] = defaultdict(
        lambda: {
            "positions": [], "yaws": [], "dims_x": [], "dims_y": [],
            "dims_z": [], "speeds": [], "labels": [], "shape_types": [],
        }
    )
    all_ego: list[np.ndarray] = []

    for bag in args.bags:
        print(f"Reading {bag} ...")
        tracks, ego_pos = collect_tracks_and_ego(bag, args.topic, args.kinematic_topic)
        print(f"  {len(tracks)} tracks, {len(ego_pos)} ego poses")
        all_ego.append(ego_pos)
        for key, entry in tracks.items():
            dst = merged_tracks[key]
            for field in entry:
                dst[field].extend(entry[field])

    ego_positions = np.concatenate(all_ego, axis=0)
    print(f"\nMerged: {len(merged_tracks)} unique track UUIDs, "
          f"{len(ego_positions)} total ego poses")

    parked = []
    excluded_speed = 0
    excluded_ego = 0
    excluded_label = 0
    excluded_frames = 0
    for key, entry in merged_tracks.items():
        if len(entry["speeds"]) < args.min_frames:
            excluded_frames += 1
            continue
        max_speed = max(entry["speeds"])
        if max_speed >= args.velocity_threshold:
            excluded_speed += 1
            continue
        dominant_label = Counter(entry["labels"]).most_common(1)[0][0]
        if dominant_label not in VEHICLE_LABELS:
            excluded_label += 1
            continue
        summary = summarize_track(entry)

        vx = summary["pose"]["position"]["x"]
        vy = summary["pose"]["position"]["y"]
        diffs = ego_positions - np.array([vx, vy])
        ego_dist = float(np.hypot(diffs[:, 0], diffs[:, 1]).min())

        if ego_dist < args.ego_overlap_threshold:
            excluded_ego += 1
            continue
        summary["max_speed"] = float(max_speed)
        summary["num_frames"] = int(len(entry["speeds"]))
        summary["min_ego_distance"] = ego_dist
        parked.append(summary)

    print(f"\nFiltering results:")
    print(f"  Excluded by speed >= {args.velocity_threshold}: {excluded_speed}")
    print(f"  Excluded by < {args.min_frames} frames: {excluded_frames}")
    print(f"  Excluded by non-vehicle label: {excluded_label}")
    print(f"  Excluded by ego overlap < {args.ego_overlap_threshold}m: {excluded_ego}")
    print(f"  Parked vehicles (before merge): {len(parked)}")

    merged = merge_overlapping_vehicles(parked)
    print(f"  Parked vehicles (after merge): {len(merged)}")

    out_data = {"parked_vehicles": merged}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(out_data, f, default_flow_style=False, sort_keys=False)
    print(f"\nWritten to {args.output}")

    if not args.no_viz:
        visualize(args.osm_path, merged, ego_positions,
                  args.output.with_suffix(".png"), args.bags[0])


if __name__ == "__main__":
    main()
