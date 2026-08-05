#!/usr/bin/env python3
"""Extract lanelet2 map from a rosbag as .bin, preserving native coordinates.

Unlike extract_map_from_bag.py (which shifts coordinates for MGRS and saves
.osm), this script saves the raw binary map bytes directly. The .bin file's
coordinates are in the bag's native map frame -- the same frame as
/localization/kinematic_state -- so lane lookups by ego position work
correctly.

The raw bytes are saved without a load/rebuild/write roundtrip because
``lanelet2.io.write()`` for .bin uses Boost serialization which fails on
expired weak pointers from dangling cached lanelet members, producing a
corrupt output file. The dangling members are benign during
``lanelet2.io.load()`` -- they only cause problems during ``write()``.
Saving raw bytes avoids both the corruption and the silent lanelet-dropping
that ``write()`` can cause.

After saving, the script loads the .bin back and prints a summary (lanelet
count, point count, coordinate ranges) as a sanity check.

Usage:
    source external/pilot-auto.x2/install/setup.bash
    python3 ros_scripts/extract_map_bin_from_bag.py <bag_path> --output map.bin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lanelet2
import rosbag2_py
from autoware_lanelet2_extension_python.projection import MGRSProjector
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_map_binary(bag_path: Path) -> bytes:
    """Read raw /map/vector_map binary data from a bag."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=""),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/map/vector_map"]))

    if reader.has_next():
        _, data, _ = reader.read_next()
        msg_type = get_message(type_map["/map/vector_map"])
        msg = deserialize_message(data, msg_type)
        return bytes(msg.data)

    raise RuntimeError(f"No /map/vector_map in {bag_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bag_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("lanelet2_map.bin"))
    args = parser.parse_args()

    raw_bytes = read_map_binary(args.bag_path)
    print(f"Read {len(raw_bytes)} bytes from /map/vector_map")

    # Save raw bytes directly -- no load/rebuild/write roundtrip.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw_bytes)

    # Sanity-check: load back and print summary.
    proj = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    lmap = lanelet2.io.load(str(args.output), proj)
    n_lanelets = len(list(lmap.laneletLayer))
    n_points = len(list(lmap.pointLayer))
    xs = [p.x for p in lmap.pointLayer]
    ys = [p.y for p in lmap.pointLayer]
    print(f"Loaded: {n_lanelets} lanelets, {n_points} points")
    print(f"X range: [{min(xs):.1f}, {max(xs):.1f}]")
    print(f"Y range: [{min(ys):.1f}, {max(ys):.1f}]")

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
