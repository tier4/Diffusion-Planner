"""This script performs recursive glob *.npz files and creates a train set path file."""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir_list", type=Path, nargs="+")
    parser.add_argument("--save_path", type=Path, default=None)
    parser.add_argument("--no_exclude_skipped", action="store_true")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    return parser.parse_args()


def _is_skipped(npz_file: Path):
    """Return the sibling json's is_skipped flag.

    Returns True/False when the sibling <npz>.json is readable, or None when it is
    missing/unreadable (such frames are kept, so filtering never silently drops data
    it could not verify).
    """
    json_file = npz_file.with_suffix(".json")
    try:
        with open(json_file, "r") as f:
            return bool(json.load(f).get("is_skipped", False))
    except (OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    args = parse_args()
    root_dir_list = args.root_dir_list
    save_path = args.save_path
    if save_path is None:
        save_path = root_dir_list[0].parent / "path_list.json"

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

    if not args.no_exclude_skipped:
        with Pool(args.num_workers) as pool:
            flags = pool.map(_is_skipped, all_list, chunksize=256)
        # keep is_skipped=False and unverifiable (None); drop is_skipped=True. order preserved.
        kept = [p for p, f in zip(all_list, flags) if f is not True]
        n_dropped = sum(1 for f in flags if f is True)
        n_unverified = sum(1 for f in flags if f is None)
        msg = (
            f"is_skipped filter: kept {len(kept)}, dropped {n_dropped}, "
            f"kept-but-unverified (missing json) {n_unverified}."
        )
        print(msg)
        log.write(msg + "\n")
        all_list = kept

    root_dir = root_dir_list[0]

    with open(save_path, "w") as f:
        json.dump([str(npz_file) for npz_file in all_list], f, indent=4)

    print(f"Saved path list to {save_path}")
