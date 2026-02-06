"""This script performs recursive glob *.npz files with time filtering and creates a train set path file."""

import argparse
import bisect
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--time_filter_json", type=Path, required=True)
    return parser.parse_args()


def extract_timestamp_from_path(file_path: Path) -> int | None:
    """Extract timestamp from corresponding JSON file.

    For an npz file path, reads the corresponding .json file and extracts the timestamp field.
    Expected format: .../YYYY-MM-DD/HH-MM-SS/HH-MM-SS_TIMESTAMP.npz
    Corresponding JSON: .../YYYY-MM-DD/HH-MM-SS/HH-MM-SS_TIMESTAMP.json
    Returns the timestamp as integer, or None if not found.
    """
    # Convert .npz path to .json path
    json_path = file_path.with_suffix(".json")

    # Check if JSON file exists
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
            if "timestamp" in data:
                return int(data["timestamp"])
    return None


def load_time_ranges(filter_json_path: Path) -> list[tuple[int, int]]:
    """Load all time ranges from a JSON file.

    Returns a sorted list of unique (start_time, end_time) tuples as integers.
    """
    time_ranges: set[tuple[int, int]] = set()

    with open(filter_json_path, "r") as f:
        data = json.load(f)

    # Extract time ranges from time_series
    if "time_series" in data:
        for time_key, time_data in data["time_series"].items():
            if "scenes" in time_data:
                for scene in time_data["scenes"]:
                    start_time = int(scene["start_time"])
                    end_time = int(scene["end_time"])
                    time_ranges.add((start_time, end_time))

    # Return sorted list for efficient binary search
    return sorted(time_ranges)


def is_timestamp_in_ranges(timestamp: int | None, time_ranges: list[tuple[int, int]]) -> bool:
    """Check if timestamp falls within any of the time ranges using binary search.

    Time ranges must be sorted by start_time.
    """
    if timestamp is None:
        return False

    # Use bisect to find the insertion point
    idx = bisect.bisect_right(time_ranges, (timestamp, float("inf")))

    # Check the range just before the insertion point
    if idx > 0:
        start_time, end_time = time_ranges[idx - 1]
        if start_time <= timestamp <= end_time:
            return True

    # Check the range at the insertion point (in case of overlapping ranges)
    if idx < len(time_ranges):
        start_time, end_time = time_ranges[idx]
        if start_time <= timestamp <= end_time:
            return True

    return False


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir
    save_path = args.save_path
    time_filter_json = args.time_filter_json

    log = open(save_path.with_suffix(".log"), "w")

    # Collect all npz files from root_dir
    root_dir = root_dir.resolve()
    assert root_dir.is_absolute(), f"{root_dir} is not an absolute path."
    assert root_dir.exists(), f"{root_dir} does not exist."
    assert root_dir.is_dir(), f"{root_dir} is not a directory."

    all_list = sorted(root_dir.rglob("*.npz"))
    print(f"Found {len(all_list)} npz files in {root_dir}.")
    log.write(f"Found {len(all_list)} npz files in {root_dir}.\n")

    # Apply time filter
    time_filter_json = time_filter_json.resolve()
    print(f"Loading time ranges from {time_filter_json}...")
    log.write(f"Loading time ranges from {time_filter_json}...\n")

    time_ranges = load_time_ranges(time_filter_json)
    print(f"Loaded {len(time_ranges)} time ranges")
    log.write(f"Loaded {len(time_ranges)} time ranges\n")

    filtered_list = []
    for file_path in all_list:
        timestamp = extract_timestamp_from_path(file_path)
        if is_timestamp_in_ranges(timestamp, time_ranges):
            filtered_list.append(file_path)

    print(f"Filtered: {len(filtered_list)} files in time range out of {len(all_list)} total")
    log.write(f"Filtered: {len(filtered_list)} files in time range out of {len(all_list)} total\n")

    all_list = filtered_list

    # Save the final list
    with open(save_path, "w") as f:
        json.dump([str(npz_file) for npz_file in all_list], f, indent=4)

    print(f"Saved path list to {save_path}")
    log.write(f"Saved path list to {save_path}\n")
    log.close()
