import argparse
import bisect
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument(
        "--string_filter",
        type=str,
        choices=[
            None,
            "us-nv-las-vegas",
            "sg-one-north",
            "us-pa-pittsburgh",
            "us-ma-boston",
            "shiojiri",
        ],
    )
    parser.add_argument("--num_filter", type=int, default=None)
    parser.add_argument(
        "--time_filter_jsons",
        type=Path,
        nargs="+",
        help="JSON files containing time_series with start_time and end_time for filtering",
    )
    return parser.parse_args()


def extract_timestamp_from_path(file_path: str) -> int | None:
    """Extract timestamp from corresponding JSON file.

    For an npz file path, reads the corresponding .json file and extracts the timestamp field.
    Expected format: .../YYYY-MM-DD/HH-MM-SS/HH-MM-SS_TIMESTAMP.npz
    Corresponding JSON: .../YYYY-MM-DD/HH-MM-SS/HH-MM-SS_TIMESTAMP.json
    Returns the timestamp as integer, or None if not found.
    """
    try:
        # Convert .npz path to .json path
        json_path = Path(file_path).with_suffix(".json")

        # Check if JSON file exists
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
                if "timestamp" in data:
                    return int(data["timestamp"])
    except (ValueError, IOError, json.JSONDecodeError):
        pass
    return None


def load_time_ranges(filter_json_paths: list[Path]) -> list[tuple[int, int]]:
    """Load all time ranges from multiple JSON files.

    Returns a sorted list of unique (start_time, end_time) tuples as integers.
    """
    time_ranges: set[tuple[int, int]] = set()

    for json_path in filter_json_paths:
        with open(json_path, "r") as f:
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
    input_json = args.input_json
    string_filter = args.string_filter
    num_filter = args.num_filter
    time_filter_jsons = args.time_filter_jsons

    input_json = input_json.resolve()

    with open(input_json, "r") as f:
        files = json.load(f)

    print(f"{len(files)=}")
    total = len(files)

    parent_dir = input_json.parent
    stem = input_json.stem

    # time range filter
    if time_filter_jsons is not None:
        print(f"Loading time ranges from {len(time_filter_jsons)} JSON files...")
        time_ranges = load_time_ranges(time_filter_jsons)
        print(f"Loaded {len(time_ranges)} time ranges")

        files_in_time_range = []
        files_out_of_time_range = []

        for file_path in files:
            timestamp = extract_timestamp_from_path(file_path)
            if is_timestamp_in_ranges(timestamp, time_ranges):
                files_in_time_range.append(file_path)
            else:
                files_out_of_time_range.append(file_path)

        print(f"{len(files_in_time_range)=}")
        print(f"{len(files_out_of_time_range)=}")

        with open(parent_dir / f"{stem}_in_time_range.json", "w") as f:
            print(f"Saving to {parent_dir / f'{stem}_in_time_range.json'}")
            json.dump(files_in_time_range, f, indent=4)

    # prefix filter
    if string_filter is not None:
        files_with_str = [f for f in files if string_filter in f]
        print(f"{len(files_with_str)=}")
        files_without_str = [f for f in files if string_filter not in f]
        print(f"{len(files_without_str)=}")
        with open(parent_dir / f"{stem}_with_{string_filter}.json", "w") as f:
            print(f"Saving to {parent_dir / f'{stem}_with_{string_filter}.json'}")
            json.dump(files_with_str, f, indent=4)
        with open(parent_dir / f"{stem}_without_{string_filter}.json", "w") as f:
            print(f"Saving to {parent_dir / f'{stem}_without_{string_filter}.json'}")
            json.dump(files_without_str, f, indent=4)

    # num filter
    if num_filter is not None:
        files = files[::num_filter]
        print(f"{len(files)=}")
        with open(parent_dir / f"{stem}_every_{num_filter}.json", "w") as f:
            print(f"Saving to {parent_dir / f'{stem}_every_{num_filter}.json'}")
            json.dump(files, f, indent=4)
