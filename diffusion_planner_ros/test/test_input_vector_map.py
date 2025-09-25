from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import (
    convert_lanelet,
    create_lane_tensor,
)
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map_path",
        type=str,
        default="/home/shintarosakoda/data/misc/20250329_psim_rosbag/map/lanelet2_map.osm",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    map_path = args.map_path

    vector_map = convert_lanelet(map_path)
    print(f"{type(vector_map)=}")

    ego_x = 3734.4
    ego_y = 73680.015625
    ego_z = 19.519
    ego_qx = 0.0005874986553164094
    ego_qy = -0.0022261576737295824
    ego_qz = 0.2551699815886937
    ego_qw = 0.9668934685700217
    RANGE = 100

    map2bl_mat4x4 = np.eye(4)
    map2bl_mat4x4[0, 3] = ego_x
    map2bl_mat4x4[1, 3] = ego_y
    map2bl_mat4x4[2, 3] = ego_z
    map2bl_mat4x4[0:3, 0:3] = Rotation.from_quat([ego_qx, ego_qy, ego_qz, ego_qw]).as_matrix()
    map2bl_mat4x4 = np.linalg.inv(map2bl_mat4x4)

    lanes_tensor, lanes_speed_limit, lanes_has_speed_limit = create_lane_tensor(
        vector_map.lanelets.values(),
        map2bl_mat4x4,
        ego_x,
        ego_y,
        {},
        70,
        "cpu",
    )
    print(f"{lanes_tensor.shape=}")

    plt.figure(figsize=(10, 8))
    for i in range(lanes_tensor.shape[1]):
        result = lanes_tensor[0, i]
        plt.plot(result[:, 0], result[:, 1], "r-")
        plt.plot(result[:, 4] + result[:, 0], result[:, 5] + result[:, 1], "g-")
        plt.plot(result[:, 6] + result[:, 0], result[:, 7] + result[:, 1], "b-")

    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.xlim(-RANGE, RANGE)
    plt.ylim(-RANGE, RANGE)
    plt.grid(alpha=0.3)
    save_path = "./test_input_vector_map.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved plot to {save_path}")
