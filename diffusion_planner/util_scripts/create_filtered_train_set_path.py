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
    parser.add_argument("--no_exclude_skipped", action="store_true")
    return parser.parse_args()


def read_frame_meta(file_path: Path) -> tuple[int | None, bool | None]:
    """Read (timestamp, is_skipped) from the sibling JSON of an npz file.

    For an npz file path, reads the corresponding .json file.
    Expected format: .../YYYY-MM-DD/HH-MM-SS/HH-MM-SS_TIMESTAMP.npz
    Corresponding JSON: .../YYYY-MM-DD/HH-MM-SS/HH-MM-SS_TIMESTAMP.json
    Each element is None when it cannot be read, so the caller can tell a missing
    or unreadable json apart from an explicit value.
    """
    json_path = file_path.with_suffix(".json")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None

    timestamp = int(data["timestamp"]) if "timestamp" in data else None
    return timestamp, bool(data.get("is_skipped", False))


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
            # Support both old format ("scenes") and new format ("whitelist_scenes")
            if "whitelist_scenes" in time_data:
                scenes = time_data["whitelist_scenes"]
            elif "scenes" in time_data:
                scenes = time_data["scenes"]
            else:
                continue
            for scene in scenes:
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
    n_dropped_skipped = 0
    n_unverified = 0
    for file_path in all_list:
        timestamp, is_skipped = read_frame_meta(file_path)
        if not is_timestamp_in_ranges(timestamp, time_ranges):
            continue
        # Same policy as create_train_set_path.py: drop is_skipped=True, keep
        # unverifiable frames (missing/unreadable json) so nothing is silently lost.
        if not args.no_exclude_skipped:
            if is_skipped is True:
                n_dropped_skipped += 1
                continue
            if is_skipped is None:
                n_unverified += 1
        filtered_list.append(file_path)

    msg = f"Filtered: {len(filtered_list)} files in time range out of {len(all_list)} total"
    if not args.no_exclude_skipped:
        msg += (
            f" (is_skipped filter: dropped {n_dropped_skipped}, "
            f"kept-but-unverified (missing json) {n_unverified})"
        )
    print(msg)
    log.write(msg + "\n")

    all_list = filtered_list

    # Save the final list
    with open(save_path, "w") as f:
        json.dump([str(npz_file) for npz_file in all_list], f, indent=4)

    print(f"Saved path list to {save_path}")
    log.write(f"Saved path list to {save_path}\n")
    log.close()
