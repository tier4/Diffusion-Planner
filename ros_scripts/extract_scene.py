#!/usr/bin/env python3
"""Stage 1 of the Perception Reproducer: read a rosbag + map and serialize a plain "scene".

This is the ONLY ROS-dependent stage (rosbag2_py + lanelet2 -> Python 3.10 / ROS env). It writes
a pickle with no ROS / lanelet2 objects so the closed-loop stage (perception_reproducer.py) can
run in a plain venv. Run it like parse_rosbag.py (source ROS + cpp_tools/install + PYTHONPATH).

Example:
    python3 ros_scripts/extract_scene.py \
        /mnt/nvme/rosbags_from_label/.../11-06-10 --out /mnt/nvme/test/scene_11-06-10.pkl

The scene dict:
  map     : LaneletMap (Lanelet/Polygon/LineString dataclasses; traffic_lights -> [ns(id=...)])
  frames  : list of per-tick dicts (ego pose+vel+accel, tracked objects in map frame, traffic
            snapshot {group_id:[(color,shape)]}, turn_indicator) for the longest route sequence
  route   : {lanelet_ids:[...], goal_pos:[3], goal_quat:[4]}
  meta    : {map_name, vector_map_path, dt}
"""

import argparse
import json
import pickle

# parse_rosbag holds the (parity-matched) bag reader + frame assembly.
import sys
from pathlib import Path
from types import SimpleNamespace

import attr
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from diffusion_planner_ros.lanelet2_utils.lanelet_map import LaneletMap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ego_shapes import lookup_ego_shape, read_vehicle_id  # noqa: E402
from parse_rosbag import build_sequences_from_rosbag  # noqa: E402

DT = 0.1


def resolve_vector_map_path(bag_path: Path) -> Path:
    info_path = bag_path / "log_file_info.json"
    date = bag_path.parent.name
    bag_time = bag_path.name
    map_version_id = None
    if info_path.is_file():
        map_version_id = json.loads(info_path.read_text(encoding="utf-8")).get(
            "area_map_version_id"
        )
    candidates = []
    for i in range(1, min(len(bag_path.parents), 6)):
        map_dir = bag_path.parents[i] / "map"
        if not map_dir.is_dir():
            continue
        if map_version_id:
            candidates.append(map_dir / map_version_id / "lanelet2_map.osm")
        candidates += [
            map_dir / date / bag_time / "lanelet2_map.osm",
            map_dir / date / "lanelet2_map.osm",
            map_dir / bag_time / "lanelet2_map.osm",
            map_dir / "lanelet2_map.osm",
        ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"lanelet2_map.osm not found for bag: {bag_path}")


def serialize_map(vector_map):
    """Strip lanelet2 objects from the LaneletMap so it pickles cleanly (only .id is read later)."""
    lanelets = {}
    for lid, lanelet in vector_map.lanelets.items():
        traffic_lights = [SimpleNamespace(id=tl.id) for tl in lanelet.traffic_lights]
        # Lanelet is an attrs class (attr.define); evolve keeps every other field.
        lanelets[lid] = attr.evolve(lanelet, traffic_lights=traffic_lights)
    # LaneletMap is a stdlib frozen dataclass; rebuild it with the cleaned lanelets.
    return LaneletMap(
        lanelets=lanelets, polygons=vector_map.polygons, line_strings=vector_map.line_strings
    )


def _pose_dict(pose):
    return {
        "pos": [pose.position.x, pose.position.y, pose.position.z],
        "quat": [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
    }


def serialize_frame(fd) -> dict:
    ks = fd.kinematic_state
    acc = fd.acceleration
    ego = {
        **_pose_dict(ks.pose.pose),
        "vx": ks.twist.twist.linear.x,
        "vy": ks.twist.twist.linear.y,
        "yaw_rate": ks.twist.twist.angular.z,
        "ax": acc.accel.accel.linear.x,
        "ay": acc.accel.accel.linear.y,
    }
    objects = []
    for obj in fd.tracked_objects.objects:
        objects.append(
            {
                "uuid": list(obj.object_id.uuid),
                "cls": [(c.label, c.probability) for c in obj.classification],
                **_pose_dict(obj.kinematics.pose_with_covariance.pose),
                "vx": obj.kinematics.twist_with_covariance.twist.linear.x,
                "vy": obj.kinematics.twist_with_covariance.twist.linear.y,
                "dim_x": obj.shape.dimensions.x,
                "dim_y": obj.shape.dimensions.y,
            }
        )
    # traffic_signals is the persistent recognition snapshot {group_id: [elements]}.
    traffic = {
        gid: [(el.color, el.shape) for el in elements]
        for gid, elements in fd.traffic_signals.items()
    }
    return {
        "ego": ego,
        "objects": objects,
        "traffic": traffic,
        "turn_indicator": int(fd.turn_indicator.report),
    }


def serialize_route(route) -> dict:
    return {
        "lanelet_ids": [seg.preferred_primitive.id for seg in route.segments],
        "goal": _pose_dict(route.goal_pose),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rosbag_dir", type=Path)
    parser.add_argument("--vector_map_path", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True, help="output scene .pkl path")
    parser.add_argument("--limit", type=int, default=-1, help="limit rosbag msgs read (debug)")
    return parser.parse_args()


def build_scene_pkl(rosbag_dir, out, vector_map_path, limit) -> None:
    """Read a rosbag + map and serialize the longest route sequence to a plain scene pickle."""
    bag_path = rosbag_dir.resolve()
    if vector_map_path is not None:
        resolved_map_path = vector_map_path.resolve()
    else:
        resolved_map_path = resolve_vector_map_path(bag_path)
    vehicle_id = read_vehicle_id(bag_path)
    wheel_base, ego_length, ego_width = lookup_ego_shape(vehicle_id)
    print(f"rosbag     : {bag_path}")
    print(f"vector map : {resolved_map_path}")
    print(
        f"vehicle_id : {vehicle_id}  ego_shape (wb,l,w)=({wheel_base}, {ego_length}, {ego_width})"
    )

    sequences = build_sequences_from_rosbag(bag_path, limit=limit, search_nearest_route=1)
    if not sequences:
        raise RuntimeError("No sequences built from rosbag")
    sequence = max(sequences, key=lambda s: len(s.data_list))
    print(f"sequence frames: {len(sequence.data_list)} (of {len(sequences)} sequences)")

    vector_map = convert_lanelet(str(resolved_map_path))

    scene = {
        "map": serialize_map(vector_map),
        "frames": [serialize_frame(fd) for fd in sequence.data_list],
        "route": serialize_route(sequence.route),
        "meta": {
            "map_name": bag_path.stem,
            "vector_map_path": str(resolved_map_path),
            "dt": DT,
            "vehicle_id": vehicle_id,
            "ego_shape": {
                "wheel_base": wheel_base,
                "ego_length": ego_length,
                "ego_width": ego_width,
            },
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(scene, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out.stat().st_size / 1e6
    print(f"Saved scene ({len(scene['frames'])} frames, {size_mb:.1f} MB) to {out}")


def main() -> None:
    args = parse_args()
    build_scene_pkl(args.rosbag_dir, args.out, args.vector_map_path, args.limit)


if __name__ == "__main__":
    main()
