import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from convert_cpp_bin_to_python_npz import process_single_file_worker
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
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--skip_npz_conversion", action="store_true")
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
    num_workers: int,
    skip_npz_conversion: bool,
):
    # C++バイナリでrosbagを処理
    print("Running C++ binary to process rosbag...")
    result = subprocess.run(
        [
            str(cpp_binary_path),
            str(rosbag_path),
            str(vector_map_path),
            str(save_dir),
            str(step),
            str(limit),
            str(min_frames),
            str(search_nearest_route),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"C++ binary execution failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)

    print("C++ binary execution completed successfully.")

    # npz変換をスキップする場合は終了
    if skip_npz_conversion:
        print("Skipping npz conversion as requested.")
        return

    bin_files = list(save_dir.glob("*.bin"))
    print(
        f"Processing {len(bin_files)} files in parallel with {num_workers or 'CPU count'} workers..."
    )

    worker_args = [(bin_file, save_dir) for bin_file in bin_files]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_single_file_worker, worker_args),
                total=len(bin_files),
                desc="Processing files",
            )
        )

    print(f"Successfully processed {len(results)} files")

    # 処理後の.npzファイル数を表示
    npz_files = list(save_dir.glob("*.npz"))
    print(f"Generated {len(npz_files)} .npz files in {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
