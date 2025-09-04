import argparse
import subprocess
import sys
from pathlib import Path

from convert_cpp_bin_to_python_npz import process_single_file
from tqdm import tqdm


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
    # C++バイナリでrosbagを処理
    print("Running C++ binary to process rosbag...")
    print(f"{cpp_binary_path} {rosbag_path} {vector_map_path} {save_dir} {step} {limit} {min_frames} {search_nearest_route}")
    result = subprocess.run(
        [
            str(cpp_binary_path),
            str(rosbag_path),
            str(vector_map_path),
            str(save_dir),
            f"--step={step}",
            f"--limit={limit}",
            f"--min_frames={min_frames}",
            f"--search_nearest_route={search_nearest_route}",
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        print(f"C++ binary execution failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")
        return

    print("C++ binary execution completed successfully.")

    bin_files = list(save_dir.glob("*.bin"))
    print(f"Processing {len(bin_files)} files")

    for bin_file in tqdm(bin_files, desc="bin to npz"):
        process_single_file(bin_file, save_dir)

    # 処理後の.npzファイル数を表示
    npz_files = list(save_dir.glob("*.npz"))
    print(f"Generated {len(npz_files)} .npz files in {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
