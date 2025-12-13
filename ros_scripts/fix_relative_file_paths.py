import argparse
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

import yaml
from natsort import natsorted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir_list", type=Path, nargs="+")
    parser.add_argument("--num_workers", type=int, default=32)
    return parser.parse_args()


def process_single_bag(bag_path: Path) -> None:
    logging.info(f"Processing bag: {bag_path}")

    metadata_path = bag_path / "metadata.yaml"

    # Read YAML file
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Sort relative_file_paths with natsort
    original_paths = data["rosbag2_bagfile_information"]["relative_file_paths"]
    sorted_paths = natsorted(original_paths)

    # Update with sorted paths
    data["rosbag2_bagfile_information"]["relative_file_paths"] = sorted_paths

    # Write back to file with yaml.dump
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    args = parse_args()
    target_dir_list = args.target_dir_list
    num_workers = args.num_workers or cpu_count()

    # search "metadata.yaml"
    metadata_list = []
    for target_dir in target_dir_list:
        metadata_list.extend(list(target_dir.glob("**/metadata.yaml")))
    bag_dir_list = [
        metadata_path.parent for metadata_path in metadata_list if metadata_path.is_file()
    ]
    bag_dir_list = list(set(bag_dir_list))  # Remove duplicates
    bag_dir_list.sort()

    logging.info(f"Found {len(bag_dir_list)} bag directories to process")
    logging.info(f"Using {num_workers} parallel workers")

    # Process bags in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_bag, bag_dir_list)

    logging.info("All processing completed")
