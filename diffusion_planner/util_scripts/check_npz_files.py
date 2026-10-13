import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from scenario_generation.canvas import Figure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


def panel_grid(size_in, rows, cols, n_used=None, **kwargs):
    """A figure's panels in row-major order, hiding the cells past ``n_used``."""
    fig = Figure(size_in, dpi=100)
    grid = fig.grid(rows, cols, margins=(0.06, 0.98, 0.07, 0.95), **kwargs)
    panels = [panel for row in grid for panel in row]
    for panel in panels[n_used if n_used is not None else len(panels) :]:
        panel.hide()
    return fig, panels


def scatter(panel, x, y):
    """Default matplotlib scatter: tab10 blue circles of 36 pt²."""
    panel.markers(np.stack([x, y], 1), "o", size=6, color="#1f77b4", z=1)


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
    ROW_NUM = 5
    fig, panels = panel_grid((10, 8), ROW_NUM, 2, hspace=0.8)
    ego_labels = [
        "ego position x",
        "ego position y",
        "ego heading cos",
        "ego heading sin",
        "ego velocity x",
        "ego velocity y",
        "ego acceleration x",
        "ego acceleration y",
        "ego steering angle",
        "ego yaw rate",
    ]
    for column, (panel, label) in enumerate(zip(panels, ego_labels)):
        hist_range = (-1, 1) if column < 4 else None
        panel.hist(ego_current_state[:, column], range=hist_range, bins=BIN_NUM)
        panel.xlabel(label)
        panel.ylabel("count")

    save_path = root_dir.parent / "ego_current_state.png"
    fig.save(save_path)
    print(f"Save to {save_path}")

    ########################
    # (2) ego_agent_future #
    ########################
    ego_agent_future = npy_data_list["ego_agent_future"]

    fig, panels = panel_grid((6.4, 4.8), 2, 2, hspace=0.6, equal=True)
    for panel, seconds in zip(panels, [2, 4, 6, 8]):
        scatter(
            panel,
            ego_agent_future[:, seconds * 10 - 1, 0],
            ego_agent_future[:, seconds * 10 - 1, 1],
        )
        panel.xlabel("x [m]")
        panel.ylabel("y [m]")
        panel.title(f"{seconds} seconds later")
    save_path = root_dir.parent / "ego_agent_future.png"
    fig.save(save_path)
    print(f"Save to {save_path}")

    ############################
    # (3) neighbor_agents_past #
    ############################
    neighbor_agents_past = npy_data_list["neighbor_agents_past"]
    # 2 seconds before
    # position
    fig, panels = panel_grid((20, 10), 4, 8, hspace=0.6, xlim=(-100, 100), ylim=(-100, 100))
    for i in range(32):
        scatter(panels[i], neighbor_agents_past[:, i, 0, 0], neighbor_agents_past[:, i, 0, 1])
        panels[i].xlabel("x [m]")
        panels[i].ylabel("y [m]")
        panels[i].title(f"2 seconds before agent{i}")
    save_path = root_dir.parent / "neighbor_agents_past.png"
    fig.save(save_path)
    print(f"Save to {save_path}")

    target = [2, 3, 4, 5, 6, 7]
    label_list = ["cos", "sin", "velocity_x", "velocity_y", "width", "length"]

    for index, label in zip(target, label_list):
        fig, panels = panel_grid((20, 10), 4, 8, hspace=0.6)
        for i in range(32):
            panels[i].hist(neighbor_agents_past[:, i, 0, index], bins=BIN_NUM)
            panels[i].xlabel(label)
            panels[i].ylabel("count")
            panels[i].title(f"agent{i}")
        save_path = root_dir.parent / f"neighbor_agents_past_{label}_hist.png"
        fig.save(save_path)
        print(f"Save to {save_path}")

    ##############################
    # (4) neighbor_agents_future #
    ##############################
    neighbor_agents_future = npy_data_list["neighbor_agents_future"]
    # seconds later
    for timestep in [20, 40, 60, 80]:
        fig, panels = panel_grid((20, 10), 4, 8, hspace=0.6, xlim=(-100, 100), ylim=(-100, 100))
        for i in range(32):
            scatter(
                panels[i],
                neighbor_agents_future[:, i, timestep - 1, 0],
                neighbor_agents_future[:, i, timestep - 1, 1],
            )
            panels[i].xlabel("x [m]")
            panels[i].ylabel("y [m]")
            panels[i].title(f"2 seconds later agent{i}")
        save_path = root_dir.parent / f"neighbor_agents_future_{timestep // 10}sec_later.png"
        fig.save(save_path)
        print(f"Save to {save_path}")

        # heading cos
        fig, panels = panel_grid((20, 10), 4, 8, hspace=0.6)
        for i in range(32):
            nf = neighbor_agents_future[:, i, timestep - 1]
            if nf.shape[-1] == 4:
                cos = nf[:, 2]
            elif nf.shape[-1] == 3:
                cos = np.cos(nf[:, 2])
            else:
                raise ValueError(f"neighbor future must be 3- or 4-col, got {nf.shape}")
            panels[i].hist(cos, bins=BIN_NUM)
            panels[i].xlabel("heading cos")
            panels[i].ylabel("count")
            panels[i].title(f"agent{i}")
        save_path = (
            root_dir.parent / f"neighbor_agents_future_cos_{timestep // 10}sec_later_hist.png"
        )
        fig.save(save_path)
        print(f"Save to {save_path}")

        # heading sin
        fig, panels = panel_grid((20, 10), 4, 8, hspace=0.6)
        for i in range(32):
            nf = neighbor_agents_future[:, i, timestep - 1]
            if nf.shape[-1] == 4:
                sin = nf[:, 3]
            elif nf.shape[-1] == 3:
                sin = np.sin(nf[:, 2])
            else:
                raise ValueError(f"neighbor future must be 3- or 4-col, got {nf.shape}")
            panels[i].hist(sin, bins=BIN_NUM)
            panels[i].xlabel("heading sin")
            panels[i].ylabel("count")
            panels[i].title(f"agent{i}")
        save_path = (
            root_dir.parent / f"neighbor_agents_future_sin_{timestep // 10}sec_later_hist.png"
        )
        fig.save(save_path)
        print(f"Save to {save_path}")

    ######################
    # (5) static_objects #
    ######################
    static_objects = npy_data_list["static_objects"]
    target = [0, 1, 2, 3, 4, 5]
    label_list = ["pos_x", "pos_y", "head_cos", "head_sin", "obj_width", "obj_length"]
    for index, label in zip(target, label_list):
        fig, panels = panel_grid((20, 10), 3, 2, n_used=5, hspace=0.6)
        for i in range(5):
            panels[i].hist(static_objects[:, i, index], bins=BIN_NUM)
            panels[i].xlabel(label)
            panels[i].ylabel("count")
            panels[i].title(f"agent{i}")
        save_path = root_dir.parent / f"static_object_{label}_hist.png"
        fig.save(save_path)
        print(f"Save to {save_path}")
