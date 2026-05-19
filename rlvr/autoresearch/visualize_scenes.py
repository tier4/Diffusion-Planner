#!/usr/bin/env python3
"""Visualize deterministic trajectories on scenes with road borders and ego footprints.

Compares baseline vs LoRA model side-by-side on specified scenes.
All paths via CLI args — no hardcoded paths.

Usage:
    # Compare baseline vs LoRA on prob scenes
    python -m rlvr.autoresearch.visualize_scenes \
      --model_path /path/to/best_model.pth \
      --scenes /path/to/scenes.json \
      --lora_path /path/to/lora_epoch_004 \
      --output_dir ~/Pictures/viz \
      --indices 0 5 10 15 20 25

    # Baseline only
    python -m rlvr.autoresearch.visualize_scenes \
      --model_path /path/to/best_model.pth \
      --scenes /path/to/scenes.json \
      --output_dir ~/Pictures/viz \
      --n_scenes 12
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch
from matplotlib.patches import Rectangle

_repo = str(Path(__file__).resolve().parent.parent.parent)
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

from guidance_gui.generate_samples import generate_samples
from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import RewardConfig, compute_reward_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path, lora_path=None):
    args_file = str(Path(model_path).parent / "args.json")
    args = Config(args_file)
    model = Diffusion_Planner(args)
    ckpt = torch.load(model_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE)
    if lora_path:
        model = load_lora_checkpoint(model, lora_path)
    model.eval()
    return model, args


def infer(model, args, npz_path, reward_config=None):
    if reward_config is None:
        reward_config = RewardConfig(enable_overprogress=True)
    data = load_npz_data(npz_path, DEVICE)
    norm_dict = args.observation_normalizer._normalization_dict
    for k, v in norm_dict.items():
        if k in data and isinstance(data[k], torch.Tensor):
            expected_dim = v["mean"].shape[-1]
            actual_dim = data[k].shape[-1]
            if actual_dim != expected_dim:
                raise ValueError(
                    f"NPZ '{npz_path}': field '{k}' has {actual_dim} columns "
                    f"but the model normalizer expects {expected_dim}. "
                    f"Fix the NPZ upstream -- do not pad."
                )
    for k in data:
        if isinstance(data[k], torch.Tensor) and data[k].dtype == torch.float64:
            data[k] = data[k].float()
    norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    norm = args.observation_normalizer(norm)
    traj = generate_samples(model, args, norm, 0.0, 1, None, DEVICE)[0]
    traj_t = torch.tensor(traj[None], device=DEVICE, dtype=torch.float32)
    r = compute_reward_batch(traj_t, data, reward_config)[0]
    return traj, data, r


def draw_scene(ax, npz_path, traj, label, color, r, show_gt=True):
    npz = np.load(npz_path)
    es = npz.get("ego_shape", None)
    if es is None or len(es) < 3:
        raise ValueError(
            f"NPZ at '{npz_path}' is missing 'ego_shape'. "
            "Inject it upstream before visualizing."
        )
    wb, length, width = float(es[0]), float(es[1]), float(es[2])
    ro = (length - wb) / 2

    # Lanes
    lanes = npz["lanes"]
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        if lane.shape[1] > 7:
            pts = lane[:, :2]
            lb, rb = lane[:, 4:6], lane[:, 6:8]
            v = np.abs(pts).sum(axis=1) > 0.1
            if v.sum() > 1:
                ax.plot((pts + lb)[v, 0], (pts + lb)[v, 1], "-", color="#bbb", alpha=0.5, lw=0.7)
                ax.plot((pts + rb)[v, 0], (pts + rb)[v, 1], "-", color="#bbb", alpha=0.5, lw=0.7)

    # Road borders (from line_strings channel 3 — same as reward uses)
    ls = npz["line_strings"]
    for i in range(ls.shape[0]):
        line = ls[i]
        if np.abs(line[:, :2]).sum() < 1e-6:
            continue
        if ls.shape[-1] >= 4 and line[:, 3].max() > 0.5:
            # Only plot points with border flag and nonzero coords
            valid = (line[:, 3] > 0.5) & (np.abs(line[:, :2]).sum(axis=1) > 0.01)
            if valid.sum() > 1:
                ax.plot(line[valid, 0], line[valid, 1], color="red", lw=3, alpha=0.7, zorder=4)

    # GT
    if show_gt:
        gt = npz["ego_agent_future"]
        ax.plot(gt[:, 0], gt[:, 1], "g-", lw=2, alpha=0.5, zorder=5)
        ax.plot(gt[::3, 0], gt[::3, 1], "go", ms=3, alpha=0.7, mew=0, zorder=6, label="GT")

    # Ego box at t=0
    ax.add_patch(Rectangle((-ro, -width / 2), length, width, lw=2, ec="black", fc="#3366cc", alpha=0.9, zorder=20))

    # Neighbors at t=0
    nb_past = npz["neighbor_agents_past"]  # (N, 31, 11)
    for i in range(nb_past.shape[0]):
        nb = nb_past[i]
        if np.abs(nb).sum() < 1e-6:
            continue
        nx, ny = nb[-1, 0], nb[-1, 1]
        ncos, nsin = nb[-1, 2], nb[-1, 3]
        nw, nl = nb[-1, 6], nb[-1, 7]
        if nl < 0.1 or nw < 0.1:
            continue
        nro = nl * 0.175  # ~(1 - 0.65) / 2; neighbors lack wheelbase, assume wb/l ~ 0.65
        nh = np.arctan2(nsin, ncos)
        t_rot = mtransforms.Affine2D().rotate(nh).translate(nx, ny) + ax.transData
        ax.add_patch(Rectangle((-nro, -nw / 2), nl, nw, lw=1.5, ec="#cc4400",
                               fc="#ff8844", alpha=0.7, zorder=15, transform=t_rot))

    # Trajectory + footprints
    pl = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()
    ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=2, alpha=0.5, zorder=10)
    ax.plot(traj[::3, 0], traj[::3, 1], "o", color=color, ms=3.5, alpha=0.9, mew=0, zorder=11,
            label=f"{label} ({pl:.1f}m)")

    # Footprints every 10 steps
    for ts in range(5, len(traj), 10):
        cx, cy = traj[ts, 0], traj[ts, 1]
        cos_h, sin_h = traj[ts, 2], traj[ts, 3]
        hn = np.sqrt(cos_h ** 2 + sin_h ** 2)
        if hn > 0.01:
            heading = np.arctan2(sin_h / hn, cos_h / hn)
            t_rot = mtransforms.Affine2D().rotate(heading).translate(cx, cy) + ax.transData
            ax.add_patch(Rectangle((-ro, -width / 2), length, width, lw=0.5, ec=color, fc=color,
                                   alpha=0.15, zorder=8, transform=t_rot))

    # Endpoint footprint
    t_end = len(traj) - 1
    cx, cy = traj[t_end, 0], traj[t_end, 1]
    cos_h, sin_h = traj[t_end, 2], traj[t_end, 3]
    hn = np.sqrt(cos_h ** 2 + sin_h ** 2)
    if hn > 0.01:
        heading = np.arctan2(sin_h / hn, cos_h / hn)
        t_rot = mtransforms.Affine2D().rotate(heading).translate(cx, cy) + ax.transData
        ax.add_patch(Rectangle((-ro, -width / 2), length, width, lw=1.5, ec=color, fc=color,
                                alpha=0.4, zorder=9, transform=t_rot))

    # Auto-zoom
    gt_pts = npz["ego_agent_future"][:, :2] if show_gt else np.zeros((1, 2))
    all_pts = np.vstack([traj[:, :2], gt_pts, [[0, 0]]])
    cx, cy = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])
    half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 8
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=7, loc="upper left")

    rb_tag = f"rb_x={'Y' if r.rb_crossing else 'N'} rb_n={r.rb_near_penalty:.2f}"
    ax.set_title(f"{rb_tag} rw={r.total:.1f}", fontsize=8)


def main():
    parser = argparse.ArgumentParser(description="Visualize trajectories with road borders")
    parser.add_argument("--model_path", type=Path, required=True, help="Base model .pth")
    parser.add_argument("--scenes", type=Path, required=True, help="JSON list of NPZ paths")
    parser.add_argument("--lora_path", type=Path, default=None, help="LoRA checkpoint dir (optional)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Save images here")
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="Scene indices to visualize")
    parser.add_argument("--n_scenes", type=int, default=12, help="Number of scenes if --indices not given")
    parser.add_argument("--cols", type=int, default=3, help="Columns in grid")
    parser.add_argument("--config", type=Path, required=True,
                        help="GRPO training config JSON. Reward flags in titles "
                             "(rb_crossing, lane_crossing) use the training run's "
                             "thresholds/gate settings.")
    args = parser.parse_args()

    reward_config = load_reward_config(args.config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.scenes) as f:
        scenes = json.load(f)

    if args.indices:
        indices = args.indices
    else:
        step = max(1, len(scenes) // args.n_scenes)
        indices = list(range(0, len(scenes), step))[: args.n_scenes]

    print(f"Loading base model from {args.model_path}")
    model_base, model_args = load_model(str(args.model_path))

    model_lora = None
    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}")
        model_lora, _ = load_model(str(args.model_path), str(args.lora_path))

    # If comparing, do side-by-side overlapped
    if model_lora:
        n = len(indices)
        rows = (n + args.cols - 1) // args.cols
        fig, axes = plt.subplots(rows, args.cols, figsize=(8 * args.cols, 8 * rows))
        if rows == 1 and args.cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[None, :]
        elif args.cols == 1:
            axes = axes[:, None]
        axes_flat = axes.flatten()

        for plot_idx, si in enumerate(indices):
            if plot_idx >= len(axes_flat):
                break
            ax = axes_flat[plot_idx]
            traj_b, data_b, r_b = infer(model_base, model_args, scenes[si], reward_config)
            traj_l, data_l, r_l = infer(model_lora, model_args, scenes[si], reward_config)

            # Draw baseline with borders, GT, and footprints
            draw_scene(ax, scenes[si], traj_b, "Baseline", "blue", r_b, show_gt=True)
            # Overlay LoRA trajectory with footprints
            npz = np.load(scenes[si], allow_pickle=True)
            es = npz.get("ego_shape", None)
            if es is None or len(es) < 3:
                raise ValueError(
                    f"NPZ at '{scenes[si]}' is missing 'ego_shape'."
                )
            wb, length, width = float(es[0]), float(es[1]), float(es[2])
            ro = (length - wb) / 2
            pl_l = np.linalg.norm(np.diff(traj_l[:, :2], axis=0), axis=1).sum()
            ax.plot(traj_l[:, 0], traj_l[:, 1], "-", color="orange", lw=2, alpha=0.6, zorder=12)
            ax.plot(traj_l[::3, 0], traj_l[::3, 1], "o", color="orange", ms=3.5, alpha=0.9, mew=0,
                    zorder=13, label=f"LoRA ({pl_l:.1f}m)")
            # LoRA footprints every 10 steps
            for ts in range(5, len(traj_l), 10):
                cx, cy = traj_l[ts, 0], traj_l[ts, 1]
                cos_h, sin_h = traj_l[ts, 2], traj_l[ts, 3]
                hn = np.sqrt(cos_h ** 2 + sin_h ** 2)
                if hn > 0.01:
                    heading = np.arctan2(sin_h / hn, cos_h / hn)
                    t_rot = mtransforms.Affine2D().rotate(heading).translate(cx, cy) + ax.transData
                    ax.add_patch(Rectangle((-ro, -width / 2), length, width, lw=0.5,
                                           ec="orange", fc="orange", alpha=0.15, zorder=12, transform=t_rot))
            # LoRA endpoint footprint
            t_end = len(traj_l) - 1
            cx, cy = traj_l[t_end, 0], traj_l[t_end, 1]
            cos_h, sin_h = traj_l[t_end, 2], traj_l[t_end, 3]
            hn = np.sqrt(cos_h ** 2 + sin_h ** 2)
            if hn > 0.01:
                heading = np.arctan2(sin_h / hn, cos_h / hn)
                t_rot = mtransforms.Affine2D().rotate(heading).translate(cx, cy) + ax.transData
                ax.add_patch(Rectangle((-ro, -width / 2), length, width, lw=1.5,
                                       ec="orange", fc="orange", alpha=0.4, zorder=12, transform=t_rot))
            # Expand view to include LoRA trajectory
            all_pts = np.vstack([traj_b[:, :2], traj_l[:, :2], npz["ego_agent_future"][:, :2], [[0, 0]]])
            cx_v, cy_v = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])
            half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 8
            ax.set_xlim(cx_v - half, cx_v + half)
            ax.set_ylim(cy_v - half, cy_v + half)
            ax.legend(fontsize=7, loc="upper left")
            ld_b = "LD" if r_b.lane_crossing else "OK"
            ld_l = "LD" if r_l.lane_crossing else "OK"
            ax.set_title(f"[{si}] base:{ld_b} lora:{ld_l}", fontsize=9)

        for j in range(len(indices), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Baseline (blue) vs LoRA (orange)", fontsize=14)
        fig.tight_layout()
        out = args.output_dir / "comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    else:
        # Single model visualization
        n = len(indices)
        rows = (n + args.cols - 1) // args.cols
        fig, axes = plt.subplots(rows, args.cols, figsize=(8 * args.cols, 8 * rows))
        if rows == 1 and args.cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[None, :]
        elif args.cols == 1:
            axes = axes[:, None]
        axes_flat = axes.flatten()

        for plot_idx, si in enumerate(indices):
            if plot_idx >= len(axes_flat):
                break
            ax = axes_flat[plot_idx]
            traj, data, r = infer(model_base, model_args, scenes[si], reward_config)
            draw_scene(ax, scenes[si], traj, "Det", "blue", r)
            ax.set_title(f"[{si}] rb_x={'Y' if r.rb_crossing else 'N'} "
                         f"rb_n={r.rb_near_penalty:.2f} rw={r.total:.1f}", fontsize=8)

        for j in range(len(indices), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Deterministic trajectories with road borders", fontsize=14)
        fig.tight_layout()
        out = args.output_dir / "scenes.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
