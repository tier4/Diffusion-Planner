import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rosbag2_py
import yaml
from diffusion_planner_ros.utils import parse_timestamp
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from scenario_generation.canvas import Figure
from scenario_generation.colormaps import qualitative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_root_dir", type=Path)
    return parser.parse_args()


def parse_pose(pose):
    position = pose.position
    orientation = pose.orientation

    return {
        "px": position.x,
        "py": position.y,
        "pz": position.z,
        "qx": orientation.x,
        "qy": orientation.y,
        "qz": orientation.z,
        "qw": orientation.w,
    }


if __name__ == "__main__":
    args = parse_args()
    bag_root_dir = args.bag_root_dir

    date_str = bag_root_dir.stem

    rosbag_path_list = sorted(bag_root_dir.glob("*"))

    data_dict = {}

    all_pos_x = []
    all_pos_y = []

    for rosbag_path in rosbag_path_list:
        print(rosbag_path)
        time_Str = rosbag_path.stem
        data_dict[time_Str] = {}
        # parse rosbag
        serialization_format = "cdr"
        metadata_yaml_path = rosbag_path / "metadata.yaml"
        metadata_yaml = yaml.safe_load(metadata_yaml_path.read_text(encoding="utf-8"))
        storage_id = metadata_yaml["rosbag2_bagfile_information"]["storage_identifier"]
        storage_options = rosbag2_py.StorageOptions(uri=str(rosbag_path), storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format,
        )

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

        target_topic_list = [
            "/localization/kinematic_state",
            "/planning/mission_planning/route",
        ]

        storage_filter = rosbag2_py.StorageFilter(topics=target_topic_list)
        reader.set_filter(storage_filter)

        topic_name_to_data = defaultdict(list)
        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            if topic in target_topic_list:
                topic_name_to_data[topic].append(msg)

        for key, value in topic_name_to_data.items():
            print(f"{key}: {len(value)} msgs")

        kinematic_state_msgs = topic_name_to_data["/localization/kinematic_state"]
        route_msgs = topic_name_to_data["/planning/mission_planning/route"]

        data_dict[time_Str][f"kinematic_state_time"] = []
        data_dict[time_Str][f"pos_x"] = []
        data_dict[time_Str][f"pos_y"] = []
        for i, kinematic_state_msg in enumerate(kinematic_state_msgs[::50]):
            pose = kinematic_state_msg.pose.pose
            time = parse_timestamp(kinematic_state_msg.header.stamp)
            pose = parse_pose(pose)
            data_dict[time_Str][f"kinematic_state_time"].append(time)
            data_dict[time_Str][f"pos_x"].append(pose["px"])
            data_dict[time_Str][f"pos_y"].append(pose["py"])
            all_pos_x.append(pose["px"])
            all_pos_y.append(pose["py"])

        for i, route_msg in enumerate(route_msgs):
            data_dict[time_Str][f"route{i}"] = {}
            print(route_msg.header)
            start_pose = route_msg.start_pose
            goal_pose = route_msg.goal_pose

            print(f"{start_pose=}")
            print(f"{goal_pose=}")

            data_dict[time_Str][f"route{i}"]["time"] = parse_timestamp(route_msg.header.stamp)
            data_dict[time_Str][f"route{i}"]["start_pose"] = parse_pose(start_pose)
            data_dict[time_Str][f"route{i}"]["goal_pose"] = parse_pose(goal_pose)

    # save data_dict to json file
    output_dir = bag_root_dir.parent.parent / "route_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{date_str}.json"
    print(f"Saving to {output_path}")
    with open(output_path, "w") as f:
        json.dump(data_dict, f, indent=4)

    # plot
    output_dir = bag_root_dir.parent.parent / "route_visualization" / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    for time_str, data in data_dict.items():
        ax = Figure((8, 6), dpi=100).panel((0.1, 0.1, 0.62, 0.84), equal=True)
        ax.markers(
            np.stack([all_pos_x, all_pos_y], 1), "o", size=np.sqrt(0.1), color="#808080", z=1
        )
        if len(data[f"kinematic_state_time"]) == 0:
            continue
        kinematic_state_time = data[f"kinematic_state_time"][0]
        first_time = min(kinematic_state_time, data[f"route0"]["time"])

        kinematic_state_time = np.array(data[f"kinematic_state_time"])
        kinematic_state_time = (kinematic_state_time - first_time) / 1e9
        # kinematic_state_timeで色を変える
        ax.markers(
            np.stack([data[f"pos_x"], data[f"pos_y"]], 1),
            "o",
            size=6,
            values=kinematic_state_time,
            cmap="viridis",
            z=1,
        )
        ax.colorbar("viridis", kinematic_state_time.min(), kinematic_state_time.max(), "time [sec]")
        ax.title(f"{time_str}")
        for i in range(100000):
            if f"route{i}" not in data:
                break
            route = data[f"route{i}"]
            start_pose = route["start_pose"]
            goal_pose = route["goal_pose"]
            start_x = start_pose["px"]
            start_y = start_pose["py"]
            goal_x = goal_pose["px"]
            goal_y = goal_pose["py"]
            time = route["time"]
            diff_time = (time - first_time) / 1e9
            ax.markers(
                np.array([start_x, start_y]),
                "x",
                size=6,
                color=qualitative("tab10", 2 * i),
                z=1,
                label=f"{diff_time:.1f}sec: start pose{i}",
            )
            ax.markers(
                np.array([goal_x, goal_y]),
                "x",
                size=6,
                color=qualitative("tab10", 2 * i + 1),
                z=1,
                label=f"{diff_time:.1f}sec: goal pose{i}",
            )

        ax.xlabel("x")
        ax.ylabel("y")
        ax.legend("upper left")
        ax.grid(1.0)
        save_path = output_dir / f"{time_str}.png"
        print(f"Saving to {save_path}")
        ax.save(save_path)
