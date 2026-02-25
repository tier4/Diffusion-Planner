import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir.resolve()
    print(f"{root_dir=}")

    # search "metadata.yaml"
    metadata_list = sorted(root_dir.glob("**/metadata.yaml"))

    total_duration_sec = 0.0

    for metadata_path in metadata_list:
        metadata = yaml.safe_load(metadata_path.open("r"))
        nanoseconds = metadata["rosbag2_bagfile_information"]["duration"]["nanoseconds"]
        seconds = nanoseconds / 1e9
        print(f"{metadata_path.parent}: {seconds:.3f} sec")
        total_duration_sec += seconds

    total_sec = int(total_duration_sec)
    total_hou = total_sec // 3600
    total_sec -= total_hou * 3600
    total_min = total_sec // 60
    total_sec -= total_min * 60

    print(
        f"Total duration: {total_hou:02d} h {total_min:02d} min {total_sec:02d} sec ({total_duration_sec / 3600:.1f} h)"
    )
