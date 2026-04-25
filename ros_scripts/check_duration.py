import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--depth", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir.resolve()
    print(f"{root_dir=}")

    # root_dirからdepth階層のsubdirを検索
    subdirs = sorted(root_dir.glob("*/" * args.depth))
    subdirs = [subdir for subdir in subdirs if subdir.is_dir()]

    for subdir in subdirs:
        # search "metadata.yaml"
        metadata_list = sorted(subdir.glob("**/metadata.yaml"))
        last_date = metadata_list[-1].parent.parent.name

        total_duration_sec = 0.0

        for metadata_path in metadata_list:
            metadata = yaml.safe_load(metadata_path.open("r"))
            nanoseconds = metadata["rosbag2_bagfile_information"]["duration"]["nanoseconds"]
            seconds = nanoseconds / 1e9
            total_duration_sec += seconds

        total_sec = int(total_duration_sec)
        total_hou = total_sec // 3600
        total_sec -= total_hou * 3600
        total_min = total_sec // 60
        total_sec -= total_min * 60

        print(
            f"{subdir.name}\tLast date: {last_date}\tTotal duration: {total_hou:03d} h {total_min:02d} min {total_sec:02d} sec ( {total_duration_sec / 3600:.1f} h)"
        )
