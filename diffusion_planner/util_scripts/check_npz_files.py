import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir
    limit = args.limit

    npz_path_list = sorted(root_dir.glob("**/*.npz"))

    print(f"Total {len(npz_path_list)} npz files")
    if limit > 0:
        div = len(npz_path_list) // limit
        npz_path_list = npz_path_list[::div]
    print(f"Check {len(npz_path_list)} npz files")

    npy_data_list = defaultdict(list)
    for npz_path in npz_path_list:
        # Load the npz file
        data = np.load(npz_path, allow_pickle=True)

        # Print the shape of each array
        for key in data.keys():
            val = data[key]
            npy_data_list[key].append(val)

    for key, val_list in npy_data_list.items():
        npy_data_list[key] = np.stack(val_list, axis=0)

    """
    ego_current_state	float32	(10, 10)
    ego_agent_future	float32	(10, 80, 3)
    neighbor_agents_past	float32	(10, 32, 21, 11)
    neighbor_agents_future	float32	(10, 32, 80, 3)
    static_objects	float32	(10, 5, 10)
    lanes	float32	(10, 70, 20, 13)
    lanes_speed_limit	float32	(10, 70, 1)
    lanes_has_speed_limit	bool	(10, 70, 1)
    route_lanes	float32	(10, 25, 20, 13)
    route_lanes_speed_limit	float32	(10, 25, 1)
    route_lanes_has_speed_limit	bool	(10, 25, 1)
    """

    BIN_NUM = 21

    # Check data distribution
    ###########
    # (1) ego #
    ###########
    ego_current_state = npy_data_list["ego_current_state"]
    plt.figure(figsize=(10, 8))
    ROW_NUM = 5
    plt.subplot(ROW_NUM, 2, 1)
    plt.hist(ego_current_state[:, 0], range=(-1, 1), bins=BIN_NUM)
    plt.xlabel("ego position x")
    plt.ylabel("count")

    plt.subplot(ROW_NUM, 2, 2)
    plt.hist(ego_current_state[:, 1], range=(-1, 1), bins=BIN_NUM)
    plt.xlabel("ego position y")
    plt.ylabel("count")

    plt.subplot(ROW_NUM, 2, 3)
    plt.hist(ego_current_state[:, 2], range=(-1, 1), bins=BIN_NUM)
    plt.xlabel("ego heading cos")
    plt.ylabel("count")

    plt.subplot(ROW_NUM, 2, 4)
    plt.hist(ego_current_state[:, 3], range=(-1, 1), bins=BIN_NUM)
    plt.xlabel("ego heading sin")
    plt.ylabel("count")

    plt.subplot(ROW_NUM, 2, 5)
    plt.hist(ego_current_state[:, 4], bins=BIN_NUM)
    plt.xlabel("ego velocity x")
    plt.ylabel("count")

    plt.subplot(ROW_NUM, 2, 6)
    plt.hist(ego_current_state[:, 5], bins=BIN_NUM)
    plt.xlabel("ego velocity y")
    plt.ylabel("count")

    plt.subplot(ROW_NUM, 2, 7)
    plt.hist(ego_current_state[:, 6], bins=BIN_NUM)
    plt.xlabel("ego acceleration x")
    plt.ylabel("count")

    plt.subplot(ROW_NUM, 2, 8)
    plt.hist(ego_current_state[:, 7], bins=BIN_NUM)
    plt.xlabel("ego acceleration y")
    plt.ylabel("count")

    plt.subplot(ROW_NUM, 2, 9)
    plt.hist(ego_current_state[:, 8], bins=BIN_NUM)
    plt.xlabel("ego steering angle")
    plt.ylabel("count")

    plt.subplot(ROW_NUM, 2, 10)
    plt.hist(ego_current_state[:, 9], bins=BIN_NUM)
    plt.xlabel("ego yaw rate")
    plt.ylabel("count")

    plt.tight_layout()
    save_path = root_dir.parent / "ego_current_state.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Save to {save_path}")

    ########################
    # (2) ego_agent_future #
    ########################
    ego_agent_future = npy_data_list["ego_agent_future"]

    # 2 seconds later
    plt.subplot(2, 2, 1)
    plt.scatter(ego_agent_future[:, 20 - 1, 0], ego_agent_future[:, 20 - 1, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("2 seconds later")
    plt.axis("equal")
    # 4 seconds later
    plt.subplot(2, 2, 2)
    plt.scatter(ego_agent_future[:, 40 - 1, 0], ego_agent_future[:, 40 - 1, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("4 seconds later")
    plt.axis("equal")
    # 6 seconds later
    plt.subplot(2, 2, 3)
    plt.scatter(ego_agent_future[:, 60 - 1, 0], ego_agent_future[:, 60 - 1, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("6 seconds later")
    plt.axis("equal")
    # 8 seconds later
    plt.subplot(2, 2, 4)
    plt.scatter(ego_agent_future[:, 80 - 1, 0], ego_agent_future[:, 80 - 1, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("8 seconds later")
    plt.axis("equal")
    plt.tight_layout()
    save_path = root_dir.parent / "ego_agent_future.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Save to {save_path}")

    ############################
    # (3) neighbor_agents_past #
    ############################
    neighbor_agents_past = npy_data_list["neighbor_agents_past"]
    # 2 seconds before
    # position
    plt.figure(figsize=(20, 10))
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        plt.scatter(neighbor_agents_past[:, i, 0, 0], neighbor_agents_past[:, i, 0, 1])
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(f"2 seconds before agent{i}")
    plt.tight_layout()
    save_path = root_dir.parent / "neighbor_agents_past.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Save to {save_path}")

    target = [2, 3, 4, 5, 6, 7]
    label_list = ["cos", "sin", "velocity_x", "velocity_y", "width", "length"]

    for index, label in zip(target, label_list):
        plt.figure(figsize=(20, 10))
        for i in range(32):
            plt.subplot(4, 8, i + 1)
            plt.hist(neighbor_agents_past[:, i, 0, index], bins=BIN_NUM)
            plt.xlabel(label)
            plt.ylabel("count")
            plt.title(f"agent{i}")
        plt.tight_layout()
        save_path = root_dir.parent / f"neighbor_agents_past_{label}_hist.png"
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"Save to {save_path}")

    ##############################
    # (4) neighbor_agents_future #
    ##############################
    neighbor_agents_future = npy_data_list["neighbor_agents_future"]
    # seconds later
    for timestep in [20, 40, 60, 80]:
        plt.figure(figsize=(20, 10))
        for i in range(32):
            plt.subplot(4, 8, i + 1)
            plt.scatter(
                neighbor_agents_future[:, i, timestep - 1, 0],
                neighbor_agents_future[:, i, timestep - 1, 1],
            )
            plt.xlim(-100, 100)
            plt.ylim(-100, 100)
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title(f"2 seconds later agent{i}")
        plt.tight_layout()
        save_path = root_dir.parent / f"neighbor_agents_future_{timestep // 10}sec_later.png"
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"Save to {save_path}")

        # heading cos
        plt.figure(figsize=(20, 10))
        for i in range(32):
            nf = neighbor_agents_future[:, i, timestep - 1]
            if nf.shape[-1] == 4:
                cos = nf[:, 2]
            elif nf.shape[-1] == 3:
                cos = np.cos(nf[:, 2])
            else:
                raise ValueError(f"neighbor future must be 3- or 4-col, got {nf.shape}")
            plt.subplot(4, 8, i + 1)
            plt.hist(cos, bins=BIN_NUM)
            plt.xlabel("heading cos")
            plt.ylabel("count")
            plt.title(f"agent{i}")
        plt.tight_layout()
        save_path = (
            root_dir.parent / f"neighbor_agents_future_cos_{timestep // 10}sec_later_hist.png"
        )
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"Save to {save_path}")

        # heading sin
        plt.figure(figsize=(20, 10))
        for i in range(32):
            nf = neighbor_agents_future[:, i, timestep - 1]
            if nf.shape[-1] == 4:
                sin = nf[:, 3]
            elif nf.shape[-1] == 3:
                sin = np.sin(nf[:, 2])
            else:
                raise ValueError(f"neighbor future must be 3- or 4-col, got {nf.shape}")
            plt.subplot(4, 8, i + 1)
            plt.hist(sin, bins=BIN_NUM)
            plt.xlabel("heading sin")
            plt.ylabel("count")
            plt.title(f"agent{i}")
        plt.tight_layout()
        save_path = (
            root_dir.parent / f"neighbor_agents_future_sin_{timestep // 10}sec_later_hist.png"
        )
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"Save to {save_path}")

    ######################
    # (5) static_objects #
    ######################
    static_objects = npy_data_list["static_objects"]
    target = [0, 1, 2, 3, 4, 5]
    label_list = ["pos_x", "pos_y", "head_cos", "head_sin", "obj_width", "obj_length"]
    for index, label in zip(target, label_list):
        plt.figure(figsize=(20, 10))
        for i in range(5):
            plt.subplot(3, 2, i + 1)
            plt.hist(static_objects[:, i, index], bins=BIN_NUM)
            plt.xlabel(label)
            plt.ylabel("count")
            plt.title(f"agent{i}")
        plt.tight_layout()
        save_path = root_dir.parent / f"static_object_{label}_hist.png"
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"Save to {save_path}")
