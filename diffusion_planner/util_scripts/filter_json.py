import argparse
import json
from pathlib import Path


def parse_args():
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_json = args.input_json
    string_filter = args.string_filter
    num_filter = args.num_filter

    input_json = input_json.resolve()

    with open(input_json, "r") as f:
        files = json.load(f)

    print(f"{len(files)=}")
    total = len(files)

    parent_dir = input_json.parent
    stem = input_json.stem

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
