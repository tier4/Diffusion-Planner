#!/usr/bin/env python3
"""Visualize the val scenes where a model's DET trajectory crosses the road border.

For a given base model + LoRA checkpoint, runs deterministic inference on every
scene in the val list, scores it with the training reward config, filters the
scenes where `r.rb_crossing=True`, and draws the worst N (by `rb_min_dist`) on
a grid with: lane boundaries / road borders / route centerline / GT / DET
trajectory + ego OBB rendered at the FIRST crossing step.

Usage:
    python -m rlvr.autoresearch.tools.viz_rb_cross_scenes \
        --model_path <base.pth> \
        --lora_dir <run_dir>/lora_epoch_NNN \
        --scenes <val.json> \
        --config <grpo_config.json> \
        --output_dir <out_dir> \
        [--max_scenes 16] \
        [--cols 4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

from guidance_gui.generate_samples import generate_samples
from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.autoresearch.tools.viz_cl_recovery import draw_scene_base
from rlvr.reward import _build_ego_bbox_corners, compute_reward_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _draw_ego_obb(ax, xy_yaw_4, ego_shape, color="red", lw=2.0, label=None):
    """Draw a single ego bounding box at given [x, y, cos, sin] pose."""
    traj = torch.tensor(xy_yaw_4, dtype=torch.float32).reshape(1, 1, 4)
    corners = _build_ego_bbox_corners(traj, ego_shape)[0, 0].cpu().numpy()
    corners_closed = np.vstack([corners, corners[:1]])
    ax.plot(corners_closed[:, 0], corners_closed[:, 1],
            color=color, lw=lw, label=label, zorder=15)


def _draw_gt(ax, path):
    d = np.load(path, allow_pickle=True)
    gt = d["ego_agent_future"][:, :2]
    valid = ~((gt[:, 0] == 0) & (gt[:, 1] == 0))
    if valid.sum() >= 2:
        ax.plot(gt[valid, 0], gt[valid, 1], "--",
                color="black", lw=1.2, alpha=0.6, label="GT")


def _draw_traj(ax, traj, label, color):
    ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=2.0, label=label, zorder=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--lora_dir", type=Path, default=None)
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--max_scenes", type=int, default=16,
                   help="Plot at most N rb-cross scenes (worst by rb_min_dist first)")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(DEVICE)

    # Load model + LoRA
    model_dir = args.model_path.parent
    args_json = model_dir / "args.json"
    if not args_json.exists():
        args_json = model_dir.parent / "args.json"
    model_args = Config(str(args_json))
    model = Diffusion_Planner(model_args)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    if args.lora_dir:
        model = load_lora_checkpoint(model, args.lora_dir)
        model.eval()

    rcfg = load_reward_config(args.config)
    with open(args.scenes) as f:
        scenes = json.load(f)
    print(f"Loaded {len(scenes)} val scenes. Scanning for rb_crossing=True...")

    offenders = []  # (scene_idx, rb_min_dist, first_cross_step, det_traj, path)
    for si, path in enumerate(scenes):
        data = load_npz_data(path, device)
        det_norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        det_norm = model_args.observation_normalizer(det_norm)
        det_traj = generate_samples(model, model_args, det_norm, 0.0, 1, None, device)[0]
        det_t = torch.tensor(det_traj[None], device=device, dtype=torch.float32)
        r = compute_reward_batch(det_t, data, rcfg)[0]
        if r.rb_crossing:
            # Find first crossing step — traj pose with min rb distance across t
            # r.rb_min_dist is the scalar min across t>=1
            offenders.append((si, float(r.rb_min_dist), det_traj, path))
        if (si + 1) % 50 == 0:
            print(f"  ... scanned {si+1}/{len(scenes)}, offenders so far: {len(offenders)}")

    print(f"Total rb_crossing scenes: {len(offenders)}/{len(scenes)} "
          f"({100*len(offenders)/len(scenes):.1f}%)")

    # Sort by rb_min_dist ascending (worst first — min distance is most negative/zero)
    offenders.sort(key=lambda x: x[1])
    offenders = offenders[:args.max_scenes]
    if not offenders:
        print("No rb_crossing scenes found. Exiting.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n = len(offenders)
    cols = min(args.cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    if rows == 1 and cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes[None, :]
    elif cols == 1: axes = axes[:, None]
    axes_flat = axes.flatten()

    # Ego shape from first scene (consistent across all scenes)
    data0 = load_npz_data(offenders[0][3], device)
    es = data0.get("ego_shape")
    ego_shape = (es[0] if es is not None and es.dim() > 1 else es).cpu()

    for plot_idx, (si, rb_min_d, det_traj, path) in enumerate(offenders):
        ax = axes_flat[plot_idx]
        # Map background
        draw_scene_base(ax, path)
        _draw_gt(ax, path)
        _draw_traj(ax, det_traj, f"DET (scene {si})", "#d62728")
        # Ego OBB at t=0 (start)
        _draw_ego_obb(ax, [0.0, 0.0, 1.0, 0.0], ego_shape, color="blue", lw=1.5,
                      label="ego t=0")
        # Ego OBB at the worst step (min rb_dist) — approximate by finding step in
        # det_traj where the ego-to-border distance is smallest. Compute per-step
        # using compute_reward_batch once more with the single traj.
        data = load_npz_data(path, device)
        det_t = torch.tensor(det_traj[None], device=device, dtype=torch.float32)
        r = compute_reward_batch(det_t, data, rcfg)[0]
        # Use first crossing step via rb_per_ts recomputation from reward internals:
        # We don't expose per-step easily; just use argmin over stepwise rb approximation.
        # Use t=40 (mid) as a reasonable proxy ego OBB marker for the worst area,
        # OR: evaluate per-step using direct call.
        try:
            from rlvr.reward import compute_road_border_penalty
            _, _, _, _, _, per_ts_min = compute_road_border_penalty(
                det_t, ego_shape.to(device), data,
            )
            worst_t = int(per_ts_min[0].argmin().item())
        except Exception:
            worst_t = 40
        if 0 < worst_t < det_traj.shape[0]:
            pose = det_traj[worst_t]  # [x, y, cos, sin]
            _draw_ego_obb(ax, pose.tolist(), ego_shape, color="red", lw=2.2,
                          label=f"ego t={worst_t} (worst)")

        # Frame
        pts = np.vstack([det_traj[:, :2], [[0, 0]]])
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        half = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 0.6 + 8
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.legend(fontsize=6, loc="upper left")
        ax.set_title(
            f"scene {si}  rb_min={rb_min_d:.2f}m  worst_t={worst_t}",
            fontsize=9,
        )

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    ckpt_tag = args.lora_dir.name if args.lora_dir else "base"
    out = args.output_dir / f"rb_cross_{ckpt_tag}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
