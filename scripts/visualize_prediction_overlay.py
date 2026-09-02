"""Overlay model-predicted and recorded ego/neighbor trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.visualize_input import visualize_inputs
from token_analysis_common import load_model, prepare_inputs
from visualize_neighbor_attention import sample_to_batch


def selected(mode: str, name: str) -> bool:
    return mode in (name, "both")


def draw_prediction_paths(ax, sample, prediction, ego=True, neighbors=True):
    """Overlay model-predicted future paths on an existing scene axis."""
    prediction = np.asarray(prediction)
    if ego and prediction.ndim == 3 and prediction.shape[0] > 0:
        ax.plot(
            prediction[0, :, 0], prediction[0, :, 1], color="orange", linewidth=2.5,
            label="ego model prediction", zorder=35,
        )
    if not neighbors or prediction.ndim != 3:
        return
    neighbor_agents = np.asarray(sample["neighbor_agents_past"])
    for i in range(min(neighbor_agents.shape[0], prediction.shape[0] - 1)):
        if not np.any(np.abs(neighbor_agents[i, :, :8]) > 1e-8):
            continue
        ax.plot(
            prediction[i + 1, :, 0], prediction[i + 1, :, 1], color="teal", alpha=0.55,
            linewidth=1.0, zorder=34,
        )


def draw_overlay(
    sample,
    batch,
    prediction,
    sample_index,
    sample_path,
    view_range,
    ego_mode,
    neighbor_mode,
    out_png,
):
    fig, ax = plt.subplots(figsize=(12, 10))
    visual_inputs = {key: value.clone() for key, value in batch.items()}
    visual_inputs["ego_agent_past"] = heading_to_cos_sin(visual_inputs["ego_agent_past"])
    visual_inputs["goal_pose"] = heading_to_cos_sin(visual_inputs["goal_pose"])
    visualize_inputs(visual_inputs, ax=ax, view_ranges=[view_range])
    for artist in list(ax.texts):
        artist.remove()

    if selected(ego_mode, "ground_truth"):
        gt = np.asarray(sample["ego_agent_future"])
        ax.plot(gt[:, 0], gt[:, 1], color="deepskyblue", linewidth=2.5, label="ego GT future")
    if selected(ego_mode, "prediction"):
        draw_prediction_paths(ax, sample, prediction, ego=True, neighbors=False)

    neighbors = np.asarray(sample["neighbor_agents_past"])
    for i in range(min(neighbors.shape[0], prediction.shape[0] - 1)):
        if not np.any(np.abs(neighbors[i, :, :8]) > 1e-8):
            continue
        if selected(neighbor_mode, "ground_truth"):
            gt = np.asarray(sample["neighbor_agents_future"])[i]
            ax.plot(gt[:, 0], gt[:, 1], color="steelblue", alpha=0.35, linewidth=1.0)
    if selected(neighbor_mode, "prediction"):
        draw_prediction_paths(ax, sample, prediction, ego=False, neighbors=True)

    ax.legend(loc="upper right")
    ax.set_title(
        f"Trajectory overlay — dataset index {sample_index}\n"
        f"ego={ego_mode}, neighbors={neighbor_mode} — {Path(sample_path).name}"
    )
    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal", adjustable="box")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--valid_set_list", required=True)
    parser.add_argument("--sample_index", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--view_range", type=float, default=80.0)
    parser.add_argument(
        "--ego", choices=("none", "ground_truth", "prediction", "both"), default="prediction"
    )
    parser.add_argument(
        "--neighbors", choices=("none", "ground_truth", "prediction", "both"), default="prediction"
    )
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--out_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.view_range <= 0:
        raise ValueError("--view_range must be positive")
    dataset = DiffusionPlannerData(args.valid_set_list)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"--sample_index must be in 0..{len(dataset) - 1}")
    sample = dataset[args.sample_index]
    batch = sample_to_batch(sample)
    model, cfg, _ = load_model(Path(args.run_dir).resolve(), args.device)
    with torch.no_grad():
        inputs = prepare_inputs(dict(batch), cfg, args.device)
        encoding = model.encoder(inputs)
        output = model.decoder(encoding, inputs)
    prediction = output["prediction"][0].detach().cpu().numpy()
    draw_overlay(
        sample,
        batch,
        prediction,
        args.sample_index,
        dataset.data_list[args.sample_index],
        args.view_range,
        args.ego,
        args.neighbors,
        Path(args.out_png),
    )
    with Path(args.out_json).open("w") as file:
        json.dump(
            {
                "sample_index": args.sample_index,
                "sample_path": dataset.data_list[args.sample_index],
                "ego_mode": args.ego,
                "neighbor_mode": args.neighbors,
                "prediction": prediction.tolist(),
            },
            file,
            indent=2,
        )
    print(f"wrote {args.out_png}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
