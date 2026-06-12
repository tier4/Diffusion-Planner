"""Read /planning/mission_planning/route from a session-0 db3 chunk and pickle it.

Session recordings are split into 1-minute chunks. The LaneletRoute msg is
latched once at engage time, so only chunk 0 of each session carries it.
For per-session processing we extract the msg once per session and hand it
to parse_rosbag.py via --external_route_pickle, then drop the session-0 db3
to reclaim disk before pulling the rest of the session.
"""

import argparse
import pickle
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


ROUTE_TOPIC = "/planning/mission_planning/route"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("rosbag_path", type=Path, help="Session-0 chunk dir (containing metadata.yaml).")
    parser.add_argument("output_pickle", type=Path, help="Where to write the pickled LaneletRoute msg.")
    parser.add_argument("--index", type=int, default=0, help="Which route msg to pick if multiple.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bag = args.rosbag_path
    if not (bag / "metadata.yaml").exists():
        raise SystemExit(f"No metadata.yaml in {bag} — run `ros2 bag reindex -s sqlite3 .` first.")

    storage_options = rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if ROUTE_TOPIC not in type_map:
        raise SystemExit(f"{ROUTE_TOPIC} not present in {bag}.")

    reader.set_filter(rosbag2_py.StorageFilter(topics=[ROUTE_TOPIC]))
    msg_cls = get_message(type_map[ROUTE_TOPIC])

    msgs = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == ROUTE_TOPIC:
            msgs.append(deserialize_message(data, msg_cls))

    if not msgs:
        raise SystemExit(f"{bag} has zero msgs on {ROUTE_TOPIC} — wrong chunk?")
    if args.index >= len(msgs):
        raise SystemExit(f"--index {args.index} but only {len(msgs)} msgs available.")

    selected = msgs[args.index]
    args.output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pickle, "wb") as f:
        pickle.dump(selected, f)

    print(f"Wrote {args.output_pickle} (segments={len(selected.segments)}, msgs_in_bag={len(msgs)}).")


if __name__ == "__main__":
    main()
