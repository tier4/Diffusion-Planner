import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_dir = args.target_dir

    map_path_list = sorted(target_dir.rglob("*.osm"))

    for map_path in map_path_list:
        # sha256 checksum for each file
        sha256_hash = hashlib.sha256()
        with open(map_path, "rb") as f:
            # check whole file
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        print(f"{map_path}: {sha256_hash.hexdigest()}")
