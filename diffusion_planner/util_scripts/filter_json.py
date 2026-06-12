import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--num_filter", type=int, default=None)
    parser.add_argument(
        "--num_filter_mode",
        type=str,
        choices=["head", "interval"],
        default="head",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_json = args.input_json
    num_filter = args.num_filter
    num_filter_mode = args.num_filter_mode

    input_json = input_json.resolve()

    with open(input_json, "r") as f:
        files = json.load(f)

    print(f"{len(files)=}")
    total = len(files)

    parent_dir = input_json.parent
    stem = input_json.stem

    # num filter
    if num_filter is not None:
        if num_filter_mode == "head":
            files = files[: len(files) // num_filter]
            output_path = parent_dir / f"{stem}_head_{num_filter}.json"
        elif num_filter_mode == "interval":
            files = files[::num_filter]
            output_path = parent_dir / f"{stem}_every_{num_filter}.json"
        else:
            raise ValueError(f"Unknown num_filter_mode: {num_filter_mode}")
        print(f"{len(files)=}")
        with open(output_path, "w") as f:
            print(f"Saving to {output_path}")
            json.dump(files, f, indent=4)
