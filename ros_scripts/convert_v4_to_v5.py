"""Convert a 32-neighbor args.json to 320-neighbor.

This script only updates args.json:
1. agent_num and predicted_neighbor_num -> 320
2. StateNormalizer mean/std from (33, 81, 4) to (321, 81, 4)
   by replicating the neighbor statistics.

Usage:
    python convert_v4_to_v5.py --model_dir path/to/model_dir --output_dir path/to/output_dir/
"""

import argparse
import json
from pathlib import Path

TARGET_NEIGHBORS = 320


def convert_args(model_dir: str, output_dir: str) -> None:
    args_path = Path(model_dir) / "args.json"
    with open(args_path, "r") as f:
        args_dict = json.load(f)

    old_neighbor_num = args_dict.get("predicted_neighbor_num", 32)
    print(f"Converting: agent_num {args_dict.get('agent_num', 32)} -> {TARGET_NEIGHBORS}")
    print(f"Converting: predicted_neighbor_num {old_neighbor_num} -> {TARGET_NEIGHBORS}")

    args_dict["agent_num"] = TARGET_NEIGHBORS
    args_dict["predicted_neighbor_num"] = TARGET_NEIGHBORS

    if "state_normalizer" in args_dict:
        sn = args_dict["state_normalizer"]
        old_mean = sn["mean"]
        old_std = sn["std"]

        # old_mean shape: [1 + old_N, horizon, 4]
        # Keep ego (index 0), replicate neighbor (index 1) to fill TARGET_NEIGHBORS.
        ego_mean = old_mean[0:1]
        ego_std = old_std[0:1]
        neighbor_mean = old_mean[1:2]
        neighbor_std = old_std[1:2]

        new_mean = ego_mean + neighbor_mean * TARGET_NEIGHBORS
        new_std = ego_std + neighbor_std * TARGET_NEIGHBORS

        sn["mean"] = new_mean
        sn["std"] = new_std
        print(f"StateNormalizer expanded: ({len(old_mean)}, ...) -> ({len(new_mean)}, ...)")

        args_dict["state_normalizer"] = sn

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_args_path = output_dir_path / "args.json"
    with open(output_args_path, "w") as f:
        json.dump(args_dict, f, indent=4)
    print(f"Saved updated args.json to {output_args_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 32-neighbor args.json to 320-neighbor")
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Path to 32-neighbor model directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for args.json"
    )
    args = parser.parse_args()

    convert_args(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
