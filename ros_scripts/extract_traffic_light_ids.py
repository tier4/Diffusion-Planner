import argparse
from collections import defaultdict
from pathlib import Path

import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("rosbag_path", type=Path)
    return parser.parse_args()


def main(rosbag_path: Path):
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
        "/perception/traffic_light_recognition/traffic_signals",
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

    data_list = topic_name_to_data["/perception/traffic_light_recognition/traffic_signals"]
    id_list = []
    for i, data in enumerate(tqdm(data_list)):
        for group in data.traffic_light_groups:
            id_list.append(group.traffic_light_group_id)

    id_set = sorted(set(id_list))
    print(f"Existing traffic light IDs: {id_set}")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
