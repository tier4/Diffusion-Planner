import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from diffusion_planner.dimensions import *
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_list_json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path_list_json = args.path_list_json
    with open(path_list_json) as f:
        npz_list = json.load(f)

    npz_list = npz_list[::50]

    mileage_list_in_yellow = []
    ng_path_list = []

    for npz_path in tqdm(npz_list):
        npz_data = np.load(npz_path)
        ego_current_state = npz_data["ego_current_state"]
        vx = ego_current_state[4]
        is_stopped = abs(vx) < 0.1

        route_lanes = npz_data["route_lanes"]

        is_yellow_light = bool(route_lanes[1, 0, TRAFFIC_LIGHT_YELLOW].item())  # next segment
        is_red_light = bool(route_lanes[1, 0, TRAFFIC_LIGHT_RED].item())  # next segment

        ego_agent_future = npz_data["ego_agent_future"]

        sum_mileage = 0.0
        for j in range(OUTPUT_T - 1):
            sum_mileage += np.linalg.norm(ego_agent_future[j, :2] - ego_agent_future[j + 1, :2])
        is_future_forward = sum_mileage > 0.5

        if is_yellow_light and is_future_forward and is_stopped:
            ng_path_list.append(npz_path)

        if is_yellow_light:
            mileage_list_in_yellow.append(sum_mileage)

    plt.hist(mileage_list_in_yellow, bins=30)
    plt.xlabel("Mileage")
    plt.ylabel("Frequency")
    plt.title("Mileage Distribution in Yellow Light")
    save_path = "./mileage_distribution_yellow_light.png"
    plt.savefig(save_path)
    print(f"Saved histogram to {save_path}")

    # output ng_path_list to a json file
    ng_path_json = path_list_json.parent / f"ng_paths_in_yellow_light.json"
    with open(ng_path_json, "w") as f:
        json.dump(ng_path_list, f, indent=4)
    print(f"Saved {len(ng_path_list)} paths to {ng_path_json}")
