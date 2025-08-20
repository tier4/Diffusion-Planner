import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cpp_binary_path", type=Path)
    parser.add_argument("rosbag_path", type=Path)
    parser.add_argument("vector_map_path", type=Path)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--min_frames", type=int, default=1700)
    parser.add_argument("--search_nearest_route", type=int, default=1)
    return parser.parse_args()


def main(
    cpp_binary_path: Path,
    rosbag_path: Path,
    vector_map_path: Path,
    save_dir: Path,
    step: int,
    limit: int,
    min_frames: int,
    search_nearest_route: bool,
):
    subprocess.run(
        [
            str(cpp_binary_path),
            str(rosbag_path),
            str(vector_map_path),
            str(save_dir),
            str(step),
            str(limit),
            str(min_frames),
            str(search_nearest_route),
        ]
    )


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
