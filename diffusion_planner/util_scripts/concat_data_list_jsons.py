"""This script performs recursive glob *.npz files and creates a train set path file."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_list", type=Path, nargs="+")
    parser.add_argument("--save_path", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    json_list = args.json_list
    save_path = args.save_path

    all_list = []

    assert len(json_list) > 1

    for json_path in json_list:
        json_path = json_path.resolve()
        assert json_path.exists(), f"{json_path} does not exist."

        npz_files = json.loads(json_path.read_text())
        print(f"len({json_path}) = {len(npz_files)}")

        all_list.extend(npz_files)

    print(f"{len(all_list)} npz files in total.")

    with open(save_path, "w") as f:
        json.dump([str(npz_file) for npz_file in all_list], f, indent=4)

    print(f"Saved set path to {save_path}")
