import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--num_filter", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_json = args.input_json
    num_filter = args.num_filter

    input_json = input_json.resolve()

    with open(input_json, "r") as f:
        files = json.load(f)

    print(f"{len(files)=}")
    total = len(files)

    parent_dir = input_json.parent
    stem = input_json.stem

    # num filter
    # 各ファイルは/mnt/nvme2/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_ver55_psimdata_cpp_INPUT_T_plus5/b_mobility_seed_200_poses_100_jpntaxi_vehicle_8_pilot-auto-v0.49.2/psim_training_bag_0_0/psim_training_bag_0_0_0000000000000036.npz
    # という感じなので、地点名で分割して、それぞれでnum_filterで割った前半だけを採用する

    file_list = defaultdict(list)
    for file_path in files:
        parts = file_path.split("/")
        target = parts[-3]  # b_mobility_seed_200_poses_100_jpntaxi_vehicle_8_pilot-auto-v0.49.2
        elements = target.split("_")
        seed_idx = elements.index("seed")
        location = "_".join(elements[:seed_idx])  # b_mobility
        file_list[location].append(file_path)

    files = []
    for location, loc_files in file_list.items():
        loc_files_sorted = sorted(loc_files)
        curr_num = len(loc_files_sorted) // num_filter
        files.extend(loc_files_sorted[:curr_num])

    print(f"{len(files)=}")
    with open(parent_dir / f"{stem}_filtered_{num_filter}.json", "w") as f:
        print(f"Saving to {parent_dir / f'{stem}_filtered_{num_filter}.json'}")
        json.dump(files, f, indent=4)
