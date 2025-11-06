import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.visualize_input import visualize_inputs
from parse_prediction_results import calc_loss
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=Path, required=True)
    parser.add_argument("--valid_data_list", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--only_top_p", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictions_dir = args.predictions_dir
    valid_data_list = args.valid_data_list
    save_dir = args.save_dir
    only_top_p = args.only_top_p

    if save_dir is None:
        save_dir = predictions_dir.parent / f"visualization"

    with open(valid_data_list, "r") as f:
        valid_data_path_list = json.load(f)

    prediction_path_list = sorted(predictions_dir.glob("**/*.npz"))
    loss_path_list = sorted(predictions_dir.glob("**/*.json"))

    # prediction_path_list, loss_path_listとの対応付けを保ったままvalid_data_path_listをソート
    list_of_tuple = [
        (valid_data_path, prediction_path, loss_path)
        for valid_data_path, prediction_path, loss_path in zip(
            valid_data_path_list, prediction_path_list, loss_path_list
        )
    ]
    list_of_tuple.sort(key=lambda x: x[0])
    valid_data_path_list, prediction_path_list, loss_path_list = zip(*list_of_tuple)

    info_path_list = [
        Path(valid_data_path).parent / f"{Path(valid_data_path).stem}.json"
        for valid_data_path in valid_data_path_list
    ]
    trajectory_dict_x = defaultdict(list)
    trajectory_dict_y = defaultdict(list)
    loss_ego_3sec = defaultdict(list)
    loss_ego_position_lat = defaultdict(list)
    loss_ego_lane_boundary_margin_loss = defaultdict(list)
    loss_ego_neighbor_margin_loss = defaultdict(list)
    loss_list = []
    for info_path, loss_path in zip(info_path_list, loss_path_list):
        assert info_path.is_file()
        time_str = info_path.stem.split("_")[0]

        pose_data = json.load(open(info_path, "r"))
        trajectory_dict_x[time_str].append(pose_data["x"])
        trajectory_dict_y[time_str].append(pose_data["y"])

        loss_data = json.load(open(loss_path, "r"))
        loss_ego_3sec[time_str].append(loss_data["loss_ego_3sec"])
        loss_list.append(loss_data["loss_ego_3sec"])
        loss_ego_position_lat[time_str].append(loss_data["ego_position_lat_loss"])
        loss_ego_lane_boundary_margin_loss[time_str].append(
            loss_data["ego_lane_boundary_margin_loss"]
        )
        loss_ego_neighbor_margin_loss[time_str].append(loss_data["ego_neighbor_margin_loss"])

    assert len(prediction_path_list) == len(valid_data_path_list)

    top_k_num = int(len(loss_list) * only_top_p)
    print(f"{top_k_num=}, {len(loss_list)=}, {only_top_p=}")
    max_indices = np.argpartition(-np.array(loss_list), min(top_k_num, len(loss_list) - 1))[
        :top_k_num
    ]

    # top_p_loss以上のもの、またその前後を保存する
    width = 20
    use_set = set()
    for i in max_indices:
        for j in range(max(0, i - width), min(len(loss_list), i + width + 1)):
            use_set.add(valid_data_path_list[j])
    print(f"use {len(use_set):,}/{len(valid_data_path_list):,}")

    save_dir.mkdir(parents=True, exist_ok=True)
    assert save_dir.is_dir()
    rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def process_one_pair(pair):
        valid_data_path, prediction_path = pair
        if valid_data_path not in use_set:
            return
        valid_data_path = Path(valid_data_path)
        prediction_path = Path(prediction_path)
        valid_loss_path = (
            prediction_path.parent / f"{prediction_path.stem.replace('prediction', 'loss')}.json"
        )
        info_data_path = valid_data_path.parent / f"{valid_data_path.stem}.json"
        valid_data = np.load(valid_data_path)
        output_dict = np.load(prediction_path)
        info_data = json.load(open(info_data_path, "r"))
        valid_loss = json.load(open(valid_loss_path, "r"))
        ego_x = info_data["x"]
        ego_y = info_data["y"]

        # valid_data_path = (...)/2025-06-12/10-19-35/10-19-35_0000000000000021.npz
        date_str = valid_data_path.parent.parent.name
        time_str = valid_data_path.parent.name

        valid_data_dict = {}
        for key, value in valid_data.items():
            if key == "map_name" or key == "token":
                continue
            # add batch size axis
            valid_data_dict[key] = torch.tensor(np.expand_dims(value, axis=0))
        valid_data_dict["ego_agent_past"] = heading_to_cos_sin(valid_data_dict["ego_agent_past"])
        valid_data_dict["goal_pose"] = heading_to_cos_sin(valid_data_dict["goal_pose"])

        prediction = output_dict["prediction"]  # (1 + P, T, D)
        turn_indicator = int(output_dict["turn_indicator"])  # ()
        valid_data_dict["turn_indicator_pred"] = turn_indicator
        (
            loss_ego,
            loss_nei,
            neighbors_future_valid,
            lat_error_ego,
            lon_error_ego,
            angle_error_ego,
            lat_error_nei,
            lon_error_nei,
            angle_error_nei,
        ) = calc_loss(valid_data, prediction)
        # loss_ego (T, 4)
        # loss_nei (P, T, 4)
        loss_ego = np.sqrt(loss_ego)
        loss_nei = np.sqrt(loss_nei)
        loss_ego_mean = np.mean(loss_ego)

        fig, ax = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={"width_ratios": [2, 1]})
        visualize_inputs(valid_data_dict, save_path=None, ax=ax[0])

        # plot prediction
        # Ego
        ax[0].plot(
            prediction[0, :, 0],
            prediction[0, :, 1],
            color="orange",
            label="prediction",
            linewidth=2,
        )
        # 3sec, 5sec, 8sec
        title = f"{valid_data_path.stem.replace('_', ' ')}"
        for timestep in [30, 50, 80]:
            index = timestep - 1
            diff_m = np.sqrt(loss_ego[index, 0] ** 2 + loss_ego[index, 1] ** 2)
            ax[0].plot(prediction[0, index, 0], prediction[0, index, 1], color="black", marker="x")
            if timestep == 30:
                title += (
                    f"\nloss{timestep // 10}sec={diff_m:.2f}[m]\n"
                    f"lat={lat_error_ego[index]:.2f}[m], lon={lon_error_ego[index]:.2f}[m], angle={angle_error_ego[index]:.2f}[rad]"
                )
        # title += f"\nego_lane_boundary_margin_loss={valid_loss['ego_lane_boundary_margin_loss']:.2f}, ego_neighbor_margin_loss={valid_loss['ego_neighbor_margin_loss']:.2f}"

        # Neighbors
        neighbors = valid_data_dict["neighbor_agents_past"][0]
        for i in range(prediction.shape[0] - 1):
            neighbor = neighbors[i, -1]
            if np.sum(np.abs(neighbor[:4])).item() < 1e-6:
                continue
            ax[0].plot(
                prediction[i + 1, :, 0],
                prediction[i + 1, :, 1],
                color="teal",
                alpha=0.5,
            )
            # # Prediction text
            # ax[0].text(
            #     prediction[i + 1, 0, 0] + i,
            #     prediction[i + 1, 0, 1] + i,
            #     f"Prediction{i}",
            #     fontsize=8,
            #     color="teal",
            # )
            # ax[0].text(
            #     prediction[i + 1, -1, 0] + i,
            #     prediction[i + 1, -1, 1] + i,
            #     f"Prediction{i}",
            #     fontsize=8,
            #     color="teal",
            # )

        ax[0].set_title(title)

        # scaling sizes based on loss_values
        loss_values = np.array(loss_ego_position_lat[time_str])
        loss_min = 0.0
        loss_max = 3.0
        clipped_loss_values = np.clip(loss_values, loss_min, loss_max)
        sizes = 10 + (clipped_loss_values - loss_min) / (loss_max - loss_min) * 10

        ax[1].scatter(
            trajectory_dict_x[time_str],
            trajectory_dict_y[time_str],
            c=loss_ego_position_lat[time_str],
            marker="o",
            s=sizes,
            vmin=loss_min,
            vmax=loss_max,
        )
        ax[1].scatter(
            ego_x,
            ego_y,
            color="red",
            marker="+",
            s=50,
        )
        ax[1].set_xlabel("x[m]")
        ax[1].set_ylabel("y[m]")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].grid(True)
        ax[1].set_title("lateral error")
        ax[1].set_aspect("equal")

        plt.colorbar(ax[1].collections[0], ax=ax[1])

        curr_save_dir = save_dir / date_str / time_str
        curr_save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(curr_save_dir / f"{valid_data_path.stem}.png")
        plt.close()

    pool = Pool(16)
    with tqdm(total=len(valid_data_path_list)) as pbar:
        for _ in pool.imap_unordered(
            process_one_pair, zip(valid_data_path_list, prediction_path_list)
        ):
            pbar.update(1)
