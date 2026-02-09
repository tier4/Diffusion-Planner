"""This script performes recursive glob *.npz files and creates a train set path file."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir_list", type=Path, nargs="+")
    parser.add_argument("--save_path", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir_list = args.root_dir_list
    save_path = args.save_path
    if save_path is None:
        save_path = root_dir_list[0].parent / "path_list.json"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(save_path.with_suffix(".log"), "w")

    all_list = []

    for root_dir in root_dir_list:
        root_dir = root_dir.resolve()
        assert root_dir.is_absolute(), f"{root_dir} is not an absolute path."
        assert root_dir.exists(), f"{root_dir} does not exist."
        assert root_dir.is_dir(), f"{root_dir} is not a directory."

        npz_files = sorted(root_dir.rglob("*.npz"))
        print(f"Found {len(npz_files)} npz files in {root_dir}.")
        log.write(f"Found {len(npz_files)} npz files in {root_dir}.\n")

        all_list.extend(npz_files)

    print(f"Found {len(all_list)} npz files in total.")
    log.write(f"Found {len(all_list)} npz files in total.\n")

    root_dir = root_dir_list[0]

    with open(save_path, "w") as f:
        json.dump([str(npz_file) for npz_file in all_list], f, indent=4)

    print(f"Saved path list to {save_path}")
