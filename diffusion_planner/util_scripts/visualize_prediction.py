import argparse
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.path_key import data_path_to_rel
from diffusion_planner.utils.visualize_input import visualize_inputs
from tqdm import tqdm
from util_scripts.parse_prediction_results import calc_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=Path, required=True)
    parser.add_argument("--valid_data_list", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--only_top_p", type=float, default=1.0)
    return parser.parse_args()


def leaf_dir_and_png(valid_data_path: Path, save_dir: Path) -> tuple[Path, Path]:
    """Reproduce the per-frame PNG path chosen in ``process_one_pair`` (kept in sync with it).

    Returns ``(leaf_dir, png_path)`` where ``leaf_dir`` is the per-clip directory the frames land in
    (``save_dir/<location>/<date>/<time>``); the clip's MP4 is written next to it as
    ``<leaf_dir>.mp4``.
    """
    parts = valid_data_path.parts
    split_idx = next((i for i, p in enumerate(parts) if p in ("valid", "train")), len(parts) - 4)
    leaf_dir = save_dir / parts[split_idx - 1] / parts[split_idx + 1] / parts[split_idx + 2]
    return leaf_dir, leaf_dir / f"{valid_data_path.stem}.png"


def encode_mp4(png_paths: list[Path], mp4_path: Path, fps: int) -> None:
    """Encode an ORDERED list of PNGs into an MP4, replacing the external ffmpeg_lib shell scripts.

    Uses ffmpeg's concat demuxer so the exact given order is honored (no re-sorting). Encoding
    params match the former make_mp4_from_unsequential_png.sh (libx264, yuv420p, even dimensions).
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in png_paths:
            f.write(f"file '{p.resolve()}'\n")
        list_path = f.name
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-r",
                str(fps),
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-r",
                str(fps),
                str(mp4_path),
            ],
            check=True,
        )
    finally:
        os.unlink(list_path)


def process_one_pair(
    pair, use_set, save_dir, trajectory_dict_x, trajectory_dict_y, loss_ego_position_lat
):
    valid_data_path, prediction_path = pair
    if valid_data_path not in use_set:
        return
    valid_data_path = Path(valid_data_path)
    prediction_path = Path(prediction_path)
    valid_loss_path = prediction_path.with_suffix(".json")
    info_data_path = valid_data_path.parent / f"{valid_data_path.stem}.json"
    valid_data = np.load(valid_data_path)
    output_dict = np.load(prediction_path)
    info_data = json.load(open(info_data_path, "r"))
    valid_loss = json.load(open(valid_loss_path, "r"))
    ego_x = info_data["x"]
    ego_y = info_data["y"]

    # valid_data_path = (...)/<project>/<location>/<train_or_val>/<date>/<time>/<frame>.npz
    # 'valid'/'train' を起点に組み立てる（同じ date/time が複数 location に出るので
    #  location も含めて保存先を分けないと衝突する）。
    parts = valid_data_path.parts
    split_idx = next((i for i, p in enumerate(parts) if p in ("valid", "train")), len(parts) - 4)
    location_str = parts[split_idx - 1]
    date_str = parts[split_idx + 1]
    time_str = parts[split_idx + 2]

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

    fig, ax = plt.subplots(1, 2, figsize=(8, 5.5), gridspec_kw={"width_ratios": [2, 1]})
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
    nbr_val = valid_loss["ego_neighbor_margin_loss"]
    rb_val = valid_loss["ego_road_border_loss"]
    title += f"\nnbr={nbr_val:.2f}, rb={rb_val:.2f}"

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

    curr_save_dir = save_dir / location_str / date_str / time_str
    curr_save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(curr_save_dir / f"{valid_data_path.stem}.png")
    plt.close()


def visualize_predictions(
    predictions_dir: Path,
    valid_data_list: Path,
    save_dir: Path | None = None,
    only_top_p: float = 1.0,
) -> None:
    """Render per-frame prediction PNGs (+ one MP4 per clip) for a saved predictions dir.

    Extracted from the CLI so callers (e.g. ``valid_predictor.py``) can run it in-process with these
    defaults instead of shelling out. ``save_dir`` defaults to ``<predictions_dir>/../visualization``.
    """
    predictions_dir = Path(predictions_dir)
    valid_data_list = Path(valid_data_list)
    if save_dir is None:
        save_dir = predictions_dir.parent / "visualization"
    save_dir = Path(save_dir)

    with open(valid_data_list, "r") as f:
        valid_data_path_list = json.load(f)

    # Output files mirror the input hierarchy via data_path_to_rel, so map each input
    # directly to its prediction/loss file instead of relying on a fragile sort-order
    # correspondence. Skip inputs without a prediction file.
    list_of_tuple = []
    for valid_data_path in sorted(valid_data_path_list):
        rel = data_path_to_rel(valid_data_path)
        prediction_path = (predictions_dir / rel).with_suffix(".npz")
        loss_path = (predictions_dir / rel).with_suffix(".json")
        if not prediction_path.is_file():
            continue
        list_of_tuple.append((valid_data_path, prediction_path, loss_path))
    if not list_of_tuple:
        raise SystemExit(
            f"No prediction .npz files found under {predictions_dir} for any entry in "
            f"{valid_data_list}. Did the prediction step complete and write its outputs?"
        )
    valid_data_path_list, prediction_path_list, loss_path_list = (
        list(x) for x in zip(*list_of_tuple)
    )

    info_path_list = [
        Path(valid_data_path).parent / f"{Path(valid_data_path).stem}.json"
        for valid_data_path in valid_data_path_list
    ]
    trajectory_dict_x = defaultdict(list)
    trajectory_dict_y = defaultdict(list)
    loss_ego_3sec = defaultdict(list)
    loss_ego_position_lat = defaultdict(list)
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

    # Bind the shared read-only state onto the worker explicitly (no module-level globals). A coarse
    # chunksize keeps that bound state pickled only ~a couple times per worker rather than per frame.
    render = partial(
        process_one_pair,
        use_set=use_set,
        save_dir=save_dir,
        trajectory_dict_x=trajectory_dict_x,
        trajectory_dict_y=trajectory_dict_y,
        loss_ego_position_lat=loss_ego_position_lat,
    )
    n = len(valid_data_path_list)
    chunksize = max(1, n // (os.cpu_count() * 2))
    # "spawn", not the default fork: valid_predictor.py calls this in-process on rank 0, where
    # CUDA and the NCCL process group are already initialized. Forking there copies the memory of
    # a process whose helper threads hold locks those threads no longer exist to release, and the
    # workers deadlock before writing a single PNG. spawn gives each worker a clean interpreter.
    with get_context("spawn").Pool(os.cpu_count()) as pool:
        with tqdm(total=n) as pbar:
            for _ in pool.imap_unordered(
                render, zip(valid_data_path_list, prediction_path_list), chunksize=chunksize
            ):
                pbar.update(1)

    # Encode one MP4 per clip from the frames just rendered, in the (sorted) order they were
    # accumulated -- no dependency on the external ffmpeg_lib shell scripts. valid_data_path_list is
    # already sorted, so both the clip order and each clip's frame order follow it.
    mp4_frame_lists: dict[Path, list[Path]] = defaultdict(list)
    for valid_data_path in valid_data_path_list:
        if valid_data_path not in use_set:
            continue
        leaf_dir, png_path = leaf_dir_and_png(Path(valid_data_path), save_dir)
        mp4_frame_lists[leaf_dir].append(png_path)

    for leaf_dir, png_paths in mp4_frame_lists.items():
        frames = [p for p in png_paths if p.is_file()]
        if not frames:
            continue
        encode_mp4(frames, leaf_dir.parent / f"{leaf_dir.name}.mp4", fps=10)


if __name__ == "__main__":
    args = parse_args()
    visualize_predictions(
        predictions_dir=args.predictions_dir,
        valid_data_list=args.valid_data_list,
        save_dir=args.save_dir,
        only_top_p=args.only_top_p,
    )
