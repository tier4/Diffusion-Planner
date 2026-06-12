#!/usr/bin/env python3
"""Visualize exploration policy guidance on scenes with road borders.

Shows ref trajectory (no guidance) vs shifted trajectory (with policy η)
overlaid on road borders from lane data.

Usage:
    source .venv/bin/activate
    python rlvr/viz_policy_guidance.py \
        --model_path /path/to/best_model.pth \
        --policy_path /path/to/lora_epoch_NNN/exploration_policy.pth \
        --scenes /path/to/scenes.json \
        --output /path/to/output.png \
        [--n_scenes 6] [--raw_scale 10.0] [--lambda_lat 2.5]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "diffusion_planner"))
sys.path.insert(0, str(PROJECT_ROOT / "preference_optimization"))

from model_utils import load_model
from utils import load_npz_data

from exploration_policy import ExplorationPolicy, ExplorationPolicyConfig
from exploration_policy.utils import generate_reference_trajectory, get_frozen_encoder


def draw_road_borders(ax, data_raw, max_dist=25):
    """Draw lane boundaries from NPZ lane data."""
    # All lanes (orange, thin)
    if "lanes" in data_raw:
        lanes = data_raw["lanes"]
        for s in range(lanes.shape[0]):
            for side, (bx_idx, by_idx) in [("left", (4, 5)), ("right", (6, 7))]:
                pts = []
                for p in range(lanes.shape[1]):
                    cx, cy = lanes[s, p, 0], lanes[s, p, 1]
                    bx, by = lanes[s, p, bx_idx], lanes[s, p, by_idx]
                    if abs(cx) > max_dist or abs(cy) > max_dist:
                        continue
                    if abs(cx) < 0.01 and abs(cy) < 0.01:
                        continue
                    if abs(bx) > 0.01 or abs(by) > 0.01:
                        pts.append([cx + bx, cy + by])
                if pts:
                    pts = np.array(pts)
                    ax.plot(pts[:, 0], pts[:, 1], "-", color="orange", linewidth=0.8, alpha=0.4)

    # Route lanes (red, thicker)
    if "route_lanes" in data_raw:
        rl = data_raw["route_lanes"]
        for s in range(rl.shape[0]):
            for side, (bx_idx, by_idx) in [("left", (4, 5)), ("right", (6, 7))]:
                pts = []
                for p in range(rl.shape[1]):
                    cx, cy = rl[s, p, 0], rl[s, p, 1]
                    bx, by = rl[s, p, bx_idx], rl[s, p, by_idx]
                    if abs(cx) > max_dist or abs(cy) > max_dist:
                        continue
                    if abs(cx) < 0.01 and abs(cy) < 0.01:
                        continue
                    if abs(bx) > 0.01 or abs(by) > 0.01:
                        pts.append([cx + bx, cy + by])
                if pts:
                    pts = np.array(pts)
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        "-",
                        color="red",
                        linewidth=1.5,
                        alpha=0.6,
                        label="road border" if s == 0 and side == "left" else "",
                    )


def shift_trajectory(ref, eta_lat, lambda_lat):
    """Shift trajectory laterally by eta_lat * lambda_lat metres."""
    offset_m = eta_lat * lambda_lat
    shifted = ref.copy()
    for t in range(ref.shape[0]):
        cos_h, sin_h = ref[t, 2], ref[t, 3]
        shifted[t, 0] = ref[t, 0] + (-sin_h) * offset_m
        shifted[t, 1] = ref[t, 1] + cos_h * offset_m
    return shifted, offset_m


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Visualize policy guidance on scenes")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_scenes", type=int, default=6)
    parser.add_argument("--raw_scale", type=float, default=10.0)
    parser.add_argument("--lambda_lat", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific scene indices instead of random",
    )
    parser.add_argument("--T", type=int, default=50, help="Timesteps to plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(Path(args.model_path), device)
    model.eval()

    policy_config = ExplorationPolicyConfig(
        hidden_dim=128, head_init="zeros", head_raw_scale=args.raw_scale
    )
    policy = ExplorationPolicy(policy_config).to(device)
    ckpt = torch.load(args.policy_path, map_location=device)
    missing, unexpected = policy.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f"Warning: missing={missing}, unexpected={unexpected}")
    policy.eval()
    encoder = get_frozen_encoder(model)

    with open(args.scenes) as f:
        all_scenes = json.load(f)

    if args.indices:
        picks = [all_scenes[i] for i in args.indices]
    else:
        rng = np.random.default_rng(args.seed)
        picks = [all_scenes[i] for i in rng.choice(len(all_scenes), args.n_scenes, replace=False)]

    n = len(picks)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, npz_path in enumerate(picks):
        data_raw = np.load(npz_path)
        data = load_npz_data(npz_path, device)
        norm_data = {k: v.clone() for k, v in data.items()}
        norm_data = model_args.observation_normalizer(norm_data)

        scene_enc = encoder(norm_data)
        x_ref = generate_reference_trajectory(model, model_args, norm_data, device)
        if isinstance(x_ref, np.ndarray):
            x_ref_t = torch.from_numpy(x_ref).float().to(device)
        else:
            x_ref_t = x_ref
        if x_ref_t.dim() == 2:
            x_ref_t = x_ref_t.unsqueeze(0)

        output = policy(scene_enc, x_ref_t, deterministic=True)
        eta_lat = output.lat_dist.mean.item() * 2 - 1
        eta_lon = output.lon_dist.mean.item() * 2 - 1

        ref = x_ref if isinstance(x_ref, np.ndarray) else x_ref.cpu().numpy()
        if ref.ndim == 3:
            ref = ref[0]

        shifted, offset_m = shift_trajectory(ref, eta_lat, args.lambda_lat)
        gt = data_raw["ego_agent_future"]
        ego_past = data_raw["ego_agent_past"]
        T = min(args.T, ref.shape[0])

        ax = axes[idx]
        draw_road_borders(ax, data_raw)

        ax.plot(ego_past[:, 0], ego_past[:, 1], "k-", linewidth=3, label="ego past", zorder=5)
        ax.plot(gt[:T, 0], gt[:T, 1], "g-", linewidth=2.5, label="GT", zorder=4)
        ax.plot(ref[:T, 0], ref[:T, 1], "b-", linewidth=2.5, label="ref (no guidance)", zorder=3)
        ax.plot(
            shifted[:T, 0],
            shifted[:T, 1],
            "m--",
            linewidth=2.5,
            label=f"guided ({offset_m * 100:.0f}cm)",
            zorder=6,
        )

        for t in [10, 20, 30]:
            if t < T:
                ax.annotate(
                    "",
                    xy=(shifted[t, 0], shifted[t, 1]),
                    xytext=(ref[t, 0], ref[t, 1]),
                    arrowprops=dict(arrowstyle="->", color="magenta", lw=2),
                )

        name = Path(npz_path).stem[-25:]
        ax.set_title(
            f"{name}\nη_lat={eta_lat:.4f} ({offset_m * 100:.1f}cm), η_lon={eta_lon:.4f}",
            fontsize=10,
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

        all_x = np.concatenate([ref[:T, 0], gt[:T, 0], ego_past[:, 0]])
        all_y = np.concatenate([ref[:T, 1], gt[:T, 1], ego_past[:, 1]])
        margin = 3
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(
        f"Policy Guidance Visualization\nBlue=ref, Magenta=guided, Green=GT, Red=road borders",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
