#!/usr/bin/env python3
"""Parse a single rosbag into npz and visualize the result as an mp4.

Example:
    python3 test_scripts/test_parse_rosbag.py \
        /mnt/nvme/rosbags_from_label/x2_dev/2231_odaiba_shinagawa_copied_from_xx1/train/2026-01-07/11-06-10/

The vector map is resolved automatically from ``log_file_info.json``
(``area_map_version_id``) under the ``map`` directory of the dataset.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROS_SCRIPTS = PROJECT_ROOT / "ros_scripts"
UTIL_SCRIPTS = PROJECT_ROOT / "diffusion_planner" / "util_scripts"
DEFAULT_CPP_BINARY = (
    PROJECT_ROOT / "cpp_tools" / "build" / "autoware_diffusion_planner_tools" / "data_converter"
)
MAKE_MP4 = Path.home() / "misc" / "ffmpeg_lib" / "make_mp4_from_unsequential_png.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rosbag_dir",
        type=Path,
        help="rosbag directory (the one that contains metadata.yaml / log_file_info.json)",
    )
    parser.add_argument(
        "--vector_map_path", type=Path, default=None, help="override auto-resolved map"
    )
    parser.add_argument("--result_dir", type=Path, default=None, help="override output directory")
    parser.add_argument("--cpp_binary_path", type=Path, default=DEFAULT_CPP_BINARY)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--min_frames", type=int, default=0)
    parser.add_argument("--convert_yellow", type=int, default=0)
    parser.add_argument("--ego_wheel_base", type=float, default=2.75)
    parser.add_argument("--ego_length", type=float, default=4.34)
    parser.add_argument("--ego_width", type=float, default=1.70)
    return parser.parse_args()


def resolve_vector_map_path(bag_path: Path) -> Path:
    """Find lanelet2_map.osm for the given rosbag directory.

    area_map_version_id is read from log_file_info.json (metadata.yaml is not used).
    """
    info_path = bag_path / "log_file_info.json"
    date = bag_path.parent.name
    bag_time = bag_path.name

    map_version_id = None
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        map_version_id = info.get("area_map_version_id")

    # Search from near the bag path up to the dataset root to support multiple layouts.
    candidate_bases = []
    max_levels = min(len(bag_path.parents), 6)
    for i in range(1, max_levels):
        base = bag_path.parents[i]
        if base not in candidate_bases:
            candidate_bases.append(base)

    candidate_paths = []
    for base in candidate_bases:
        map_dir = base / "map"
        if not map_dir.is_dir():
            continue
        if map_version_id:
            candidate_paths.append(map_dir / map_version_id / "lanelet2_map.osm")
        # Legacy layouts.
        candidate_paths.append(map_dir / date / bag_time / "lanelet2_map.osm")
        candidate_paths.append(map_dir / date / "lanelet2_map.osm")
        candidate_paths.append(map_dir / bag_time / "lanelet2_map.osm")
        candidate_paths.append(map_dir / "lanelet2_map.osm")

    for path in candidate_paths:
        if path.is_file():
            return path

    searched = "\n".join(str(p) for p in candidate_paths) or "(no map dir found)"
    raise FileNotFoundError(
        f"lanelet2_map.osm was not found for bag: {bag_path}\n"
        f"log_file_info: {info_path}\n"
        f"area_map_version_id: {map_version_id}\n"
        f"searched:\n{searched}"
    )


def run(command: list[str], log_file: Path | None = None) -> None:
    print("+ " + " ".join(str(c) for c in command), flush=True)
    if log_file is None:
        subprocess.run([str(c) for c in command], check=True)
        return
    # Mirror stdout/stderr to both console and a log file (like `tee`).
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [str(c) for c in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)


def main() -> None:
    args = parse_args()

    bag_path = args.rosbag_dir.resolve()
    if not bag_path.is_dir():
        raise NotADirectoryError(f"rosbag_dir is not a directory: {bag_path}")

    vector_map_path = (
        args.vector_map_path.resolve()
        if args.vector_map_path is not None
        else resolve_vector_map_path(bag_path)
    )
    print(f"rosbag     : {bag_path}")
    print(f"vector map : {vector_map_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = (
        args.result_dir.resolve()
        if args.result_dir is not None
        else Path("/mnt/nvme/test") / f"{stamp}_test_{bag_path.name}"
    )
    npz_dir = result_dir / bag_path.name
    npz_dir.mkdir(parents=True, exist_ok=True)
    print(f"result dir : {result_dir}")

    start = time.perf_counter()

    # 1. Convert rosbag -> npz via the C++ data_converter.
    run(
        [
            sys.executable,
            ROS_SCRIPTS / "parse_rosbag_by_cpp.py",
            args.cpp_binary_path,
            bag_path,
            vector_map_path,
            npz_dir,
            f"--step={args.step}",
            f"--limit={args.limit}",
            f"--min_frames={args.min_frames}",
            f"--convert_yellow={args.convert_yellow}",
            f"--ego_wheel_base={args.ego_wheel_base}",
            f"--ego_length={args.ego_length}",
            f"--ego_width={args.ego_width}",
        ],
        log_file=result_dir / f"result_{stamp}.txt",
    )

    elapsed = int(time.perf_counter() - start)
    print(f"Conversion elapsed: {elapsed} s")

    # 2. Build the npz path list.
    run([sys.executable, UTIL_SCRIPTS / "create_train_set_path.py", npz_dir])

    # 3. Visualize the converted inputs.
    visualize_dir = result_dir / "visualize_result"
    run(
        [
            sys.executable,
            UTIL_SCRIPTS / "visualize_input.py",
            result_dir / "path_list.json",
            visualize_dir,
        ]
    )

    # 4. Make an mp4 out of the visualization pngs (best effort; needs ffmpeg).
    if not MAKE_MP4.is_file():
        print(f"Skip mp4: helper not found at {MAKE_MP4}")
    elif shutil.which("ffmpeg") is None:
        print("Skip mp4: ffmpeg is not installed (PNGs are in visualize_result/)")
    else:
        try:
            run([str(MAKE_MP4), visualize_dir])
        except subprocess.CalledProcessError as e:
            print(f"Skip mp4: ffmpeg step failed ({e}); PNGs are in visualize_result/")

    print(f"Done. Results in {result_dir}")


if __name__ == "__main__":
    main()
