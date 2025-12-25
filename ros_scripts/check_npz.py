import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from diffusion_planner.dimensions import *
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    npz_path = args.npz_path

    mileage_list_in_yellow = []
    ng_path_list = []

    npz_data = np.load(npz_path)
    for key in npz_data:
        print(f"{key}: {npz_data[key].shape}, dtype={npz_data[key].dtype}")

    print(npz_data["ego_shape"])
