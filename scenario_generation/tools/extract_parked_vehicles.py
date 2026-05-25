#!/usr/bin/env python3
"""Extract truly-parked vehicles from multiple rosbag splits.

Processes ALL bags for a route together so that track UUIDs are merged
across splits. A vehicle is only considered parked if its max speed
across ALL observations is below the threshold. This filters out cars
temporarily stopped at traffic lights — they will eventually move in a
later split.

Requires: ROS2 Humble + Autoware message types available (source the
relevant workspace before running).

Usage:
    python3 -m scenario_generation.tools.extract_parked_vehicles \\
        --bags /path/to/bag1.db3 /path/to/bag2.db3 \\
        --osm_path /path/to/lanelet2_map.osm \\
        --output /path/to/parked.yaml
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_CAR = 1
LABEL_TRUCK = 2
LABEL_BUS = 3
LABEL_TRAILER = 4

VEHICLE_LABELS = {LABEL_CAR, LABEL_TRUCK, LABEL_BUS, LABEL_TRAILER}
LABEL_TO_NAME = {
    LABEL_CAR: "car",
    LABEL_TRUCK: "truck",
    LABEL_BUS: "bus",
    LABEL_TRAILER: "trailer",
}
NAME_TO_COLOR = {
    "car": "tab:blue",
    "truck": "tab:orange",
    "bus": "tab:red",
    "trailer": "tab:purple",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _uuid_to_key(uuid_msg) -> str:
    return bytes(uuid_msg.uuid).hex()


def _primary_label(classification_list) -> int:
    if len(classification_list) == 0:
        return -1
    best = max(classification_list, key=lambda c: c.probability)
    return best.label


def _quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def _circular_mean(angles: list[float]) -> float:
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    return math.atan2(sin_sum, cos_sum)


def _obb_corners(
    cx: float, cy: float, yaw: float, dx: float, dy: float,
) -> list[tuple[float, float]]:
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    hx, hy = dx * 0.5, dy * 0.5
    corners_local = [(+hx, +hy), (+hx, -hy), (-hx, -hy), (-hx, +hy)]
    return [
        (cx + lx * cos_y - ly * sin_y, cy + lx * sin_y + ly * cos_y)
        for lx, ly in corners_local
    ]


# ---------------------------------------------------------------------------
# Rosbag reading
# ---------------------------------------------------------------------------
def collect_tracks_and_ego(
    rosbag_path: Path, track_topic: str, kinematic_topic: str,
) -> tuple[dict, np.ndarray]:
    """Read a rosbag and return (tracks_by_uuid, ego_positions)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    ext = rosbag_path.suffix.lower()
    storage_id = "mcap" if ext == ".mcap" else "sqlite3"

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(rosbag_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if track_topic not in type_map:
        raise RuntimeError(f"Topic '{track_topic}' not found in rosbag.")
    if kinematic_topic not in type_map:
        raise RuntimeError(f"Topic '{kinematic_topic}' not found in rosbag.")

    track_msg_type = get_message(type_map[track_topic])
    kinematic_msg_type = get_message(type_map[kinematic_topic])

    tracks: dict[str, dict] = defaultdict(
        lambda: {
            "positions": [], "yaws": [], "dims_x": [], "dims_y": [],
            "dims_z": [], "speeds": [], "labels": [], "shape_types": [],
        }
    )
    ego_positions: list[tuple[float, float]] = []

    while reader.has_next():
        topic_name, data, _ = reader.read_next()
        if topic_name == track_topic:
            msg = deserialize_message(data, track_msg_type)
            for obj in msg.objects:
                key = _uuid_to_key(obj.object_id)
                pose = obj.kinematics.pose_with_covariance.pose
                twist = obj.kinematics.twist_with_covariance.twist
                speed = math.hypot(twist.linear.x, twist.linear.y)
                label = _primary_label(obj.classification)
                entry = tracks[key]
                entry["positions"].append(
                    (pose.position.x, pose.position.y, pose.position.z)
                )
                entry["yaws"].append(_quaternion_to_yaw(pose.orientation))
                entry["dims_x"].append(obj.shape.dimensions.x)
                entry["dims_y"].append(obj.shape.dimensions.y)
                entry["dims_z"].append(obj.shape.dimensions.z)
                entry["speeds"].append(speed)
                entry["labels"].append(label)
                entry["shape_types"].append(obj.shape.type)
        elif topic_name == kinematic_topic:
            msg = deserialize_message(data, kinematic_msg_type)
            pos = msg.pose.pose.position
            ego_positions.append((pos.x, pos.y))

    return tracks, np.array(ego_positions)


# ---------------------------------------------------------------------------
# Track summarisation
# ---------------------------------------------------------------------------
def _summarize_track(entry: dict) -> dict:
    positions = np.array(entry["positions"])
    label_counter = Counter(entry["labels"])
    dominant_label, _ = label_counter.most_common(1)[0]
    shape_counter = Counter(entry["shape_types"])
    dominant_shape, _ = shape_counter.most_common(1)[0]

    pos_x = float(np.median(positions[:, 0]))
    pos_y = float(np.median(positions[:, 1]))
    pos_z = float(np.median(positions[:, 2]))

    yaw = _circular_mean(entry["yaws"])
    qx, qy, qz, qw = _yaw_to_quaternion(yaw)

    dim_x = float(np.median(entry["dims_x"]))
    dim_y = float(np.median(entry["dims_y"]))
    dim_z = float(np.median(entry["dims_z"]))

    return {
        "classification": LABEL_TO_NAME[dominant_label],
        "shape_type": int(dominant_shape),
        "dimensions": {"x": dim_x, "y": dim_y, "z": dim_z},
        "pose": {
            "position": {"x": pos_x, "y": pos_y, "z": pos_z},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        },
    }


# ---------------------------------------------------------------------------
# OBB overlap merge (union-find)
# ---------------------------------------------------------------------------
def _merge_overlapping_vehicles(parked_vehicles: list) -> list:
    from shapely.geometry import Polygon

    n = len(parked_vehicles)
    if n == 0:
        return []

    polygons, yaws = [], []
    for v in parked_vehicles:
        cx = v["pose"]["position"]["x"]
        cy = v["pose"]["position"]["y"]
        qz = v["pose"]["orientation"]["z"]
        qw = v["pose"]["orientation"]["w"]
        yaw = 2.0 * math.atan2(qz, qw)
        polygons.append(
            Polygon(_obb_corners(cx, cy, yaw, v["dimensions"]["x"], v["dimensions"]["y"]))
        )
        yaws.append(yaw)

    parent = list(range(n))

    def _root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i, j):
        ri, rj = _root(i), _root(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]):
                _union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[_root(i)].append(i)

    merged = []
    for indices in groups.values():
        if len(indices) == 1:
            merged.append(parked_vehicles[indices[0]])
            continue

        group = [parked_vehicles[i] for i in indices]
        mean_yaw = _circular_mean([yaws[i] for i in indices])
        cos_my, sin_my = math.cos(-mean_yaw), math.sin(-mean_yaw)

        all_local_x, all_local_y, all_z = [], [], []
        for v in group:
            cx = v["pose"]["position"]["x"]
            cy = v["pose"]["position"]["y"]
            yaw = 2.0 * math.atan2(v["pose"]["orientation"]["z"],
                                    v["pose"]["orientation"]["w"])
            for wx, wy in _obb_corners(cx, cy, yaw,
                                        v["dimensions"]["x"], v["dimensions"]["y"]):
                all_local_x.append(wx * cos_my - wy * sin_my)
                all_local_y.append(wx * sin_my + wy * cos_my)
            all_z.append(v["pose"]["position"]["z"])

        x_min, x_max = min(all_local_x), max(all_local_x)
        y_min, y_max = min(all_local_y), max(all_local_y)
        cx_l = (x_min + x_max) * 0.5
        cy_l = (y_min + y_max) * 0.5
        cos_yp, sin_yp = math.cos(mean_yaw), math.sin(mean_yaw)
        new_cx = cx_l * cos_yp - cy_l * sin_yp
        new_cy = cx_l * sin_yp + cy_l * cos_yp
        qx, qy, qz, qw = _yaw_to_quaternion(mean_yaw)

        label_votes: Counter = Counter()
        for v in group:
            label_votes[v["classification"]] += v["num_frames"]
        dominant_class = label_votes.most_common(1)[0][0]

        merged.append({
            "classification": dominant_class,
            "shape_type": group[0]["shape_type"],
            "dimensions": {"x": x_max - x_min, "y": y_max - y_min,
                           "z": max(v["dimensions"]["z"] for v in group)},
            "pose": {
                "position": {"x": new_cx, "y": new_cy,
                             "z": float(np.mean(all_z))},
                "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
            },
            "max_speed": max(v["max_speed"] for v in group),
            "num_frames": sum(v["num_frames"] for v in group),
            "min_ego_distance": min(v["min_ego_distance"] for v in group),
            "merged_count": len(group),
        })

    return merged


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _visualize(
    osm_path: Path,
    parked_vehicles: list,
    ego_positions: np.ndarray,
    viz_path: Path,
) -> None:
    import lanelet2
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from autoware_lanelet2_extension_python.projection import MGRSProjector

    if not parked_vehicles:
        print("No parked vehicles to visualize.")
        return

    projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    lanelet_map = lanelet2.io.load(str(osm_path), projection)

    xs = np.array([v["pose"]["position"]["x"] for v in parked_vehicles])
    ys = np.array([v["pose"]["position"]["y"] for v in parked_vehicles])
    pad = 30.0
    x_min = min(xs.min(), ego_positions[:, 0].min()) - pad
    x_max = max(xs.max(), ego_positions[:, 0].max()) + pad
    y_min = min(ys.min(), ego_positions[:, 1].min()) - pad
    y_max = max(ys.max(), ego_positions[:, 1].max()) + pad

    fig, ax = plt.subplots(figsize=(20, 20))
    for ll in lanelet_map.laneletLayer:
        for bound in (ll.leftBound, ll.rightBound):
            bxs = [p.x for p in bound]
            bys = [p.y for p in bound]
            if any(x_min <= x <= x_max and y_min <= y <= y_max
                   for x, y in zip(bxs, bys)):
                ax.plot(bxs, bys, color="gray", linewidth=0.5, alpha=0.7)

    ax.plot(ego_positions[:, 0], ego_positions[:, 1],
            color="tab:green", linewidth=1.0, alpha=0.6, label="ego trajectory")

    seen_labels: set[str] = set()
    for v in parked_vehicles:
        name = v["classification"]
        color = NAME_TO_COLOR[name]
        cx = v["pose"]["position"]["x"]
        cy = v["pose"]["position"]["y"]
        yaw = 2.0 * math.atan2(v["pose"]["orientation"]["z"],
                                v["pose"]["orientation"]["w"])
        corners = _obb_corners(cx, cy, yaw, v["dimensions"]["x"],
                               v["dimensions"]["y"])
        label = name if name not in seen_labels else None
        seen_labels.add(name)
        edge_color = "red" if "merged_count" in v else "black"
        lw = 1.2 if "merged_count" in v else 0.5
        polygon = mpatches.Polygon(
            corners, closed=True, facecolor=color, edgecolor=edge_color,
            linewidth=lw, alpha=0.7, label=label,
        )
        ax.add_patch(polygon)
        head_x = cx + v["dimensions"]["x"] * 0.5 * math.cos(yaw)
        head_y = cy + v["dimensions"]["x"] * 0.5 * math.sin(yaw)
        ax.plot([cx, head_x], [cy, head_y], color="black", linewidth=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    ax.set_title(f"Parked vehicles ({len(parked_vehicles)})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(viz_path, dpi=150)
    plt.close(fig)
    print(f"Visualization saved to {viz_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
        summary = _summarize_track(entry)

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

    merged = _merge_overlapping_vehicles(parked)
    print(f"  Parked vehicles (after merge): {len(merged)}")

    out_data = {"parked_vehicles": merged}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(out_data, f, default_flow_style=False, sort_keys=False)
    print(f"\nWritten to {args.output}")

    if not args.no_viz:
        _visualize(args.osm_path, merged, ego_positions,
                   args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
