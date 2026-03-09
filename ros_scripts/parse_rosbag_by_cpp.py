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
    parser.add_argument("--min_distance", type=float, default=50.0)
    parser.add_argument("--search_nearest_route", type=int, default=1)
    parser.add_argument("--convert_yellow", type=int, default=0)
    parser.add_argument("--convert_red", type=int, default=0)
    parser.add_argument("--interpolation", type=int, default=0)
    parser.add_argument("--ego_wheel_base", type=float, default=2.75)
    parser.add_argument("--ego_length", type=float, default=4.34)
    parser.add_argument("--ego_width", type=float, default=1.70)
    return parser.parse_args()


def main(
    cpp_binary_path: Path,
    rosbag_path: Path,
    vector_map_path: Path,
    save_dir: Path,
    step: int,
    limit: int,
    min_frames: int,
    min_distance: float,
    search_nearest_route: bool,
    convert_yellow: int,
    convert_red: int,
    interpolation: int,
    ego_wheel_base: float,
    ego_length: float,
    ego_width: float,
):
    # C++バイナリでrosbagを処理
    print("Running C++ binary to process rosbag...")
    command = [
        str(cpp_binary_path),
        str(rosbag_path),
        str(vector_map_path),
        str(save_dir),
        f"--step={step}",
        f"--limit={limit}",
        f"--min_frames={min_frames}",
        f"--min_distance={min_distance}",
        f"--search_nearest_route={search_nearest_route}",
        f"--convert_yellow={convert_yellow}",
        f"--convert_red={convert_red}",
        f"--interpolation={interpolation}",
        f"--ego_wheel_base={ego_wheel_base}",
        f"--ego_length={ego_length}",
        f"--ego_width={ego_width}",
    ]
    print(" ".join(command))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        print(f"C++ binary execution failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")
        print(f"{rosbag_path} processing failed.")
        raise RuntimeError("C++ binary execution failed")

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
