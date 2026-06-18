#!/usr/bin/env python3
"""Evaluate driving quality metrics (speed, lat accel, path length, stopped).

Reports:
  - max/mean/p95 speed
  - max/mean/p95 lateral accel (curvature-based, accurate)
  - max/mean/p95 lateral accel (current reward.py method, for comparison)
  - reward, collision, rb_cross, path length
  - per-scene stopped count

Usage:
    source .venv/bin/activate
    python -m rlvr.autoresearch.tools.eval_driving_metrics \
        --model_path /path/to/best_model.pth \
        --scenes /path/to/scenes.json \
        [--lora_path /path/to/lora_epoch_NNN] \
        [--tag "ep4"]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data
from rlvr.reward import _build_sg_diff_kernel

DT = 0.1


def lat_accel_curvature(trajectory: np.ndarray) -> np.ndarray:
    """trajectory: [T, 4] (x, y, cos, sin) -> lat_accel: [T-1]"""
    x, y = trajectory[:, 0], trajectory[:, 1]
    cos_h, sin_h = trajectory[:, 2], trajectory[:, 3]
    theta = np.arctan2(sin_h, cos_h)
    dx = np.diff(x) / DT
    dy = np.diff(y) / DT
    speed = np.sqrt(dx**2 + dy**2)
    dtheta = np.diff(theta)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    omega = dtheta / DT
    return speed * omega


def lat_accel_current(positions: np.ndarray) -> np.ndarray:
    """positions: [T, 2] -> lat_accel: [T-2] (reward.py method)"""
    vel = np.diff(positions, axis=0) / DT
    acc_vec = np.diff(vel, axis=0) / DT
    heading = vel[:-1]
    heading_norm = heading / (np.linalg.norm(heading, axis=-1, keepdims=True) + 1e-6)
    lat_dir = np.stack([-heading_norm[:, 1], heading_norm[:, 0]], axis=-1)
    return np.sum(acc_vec * lat_dir, axis=-1)


def lat_accel_smoothed(positions: np.ndarray, window: int = 11, order: int = 3) -> np.ndarray:
    """positions: [T, 2] -> lat_accel: [T] using SG derivative (no finite diff noise).

    Uses the same torch conv1d approach as reward.py for consistency.
    Runs on CPU (no GPU needed for eval).
    """
    if positions.shape[0] < window:
        return lat_accel_current(positions)
    # Use torch conv1d on CPU (same as reward.py for exact consistency)
    vel_kernel = _build_sg_diff_kernel(window, order, deriv=1, delta=DT)
    accel_kernel = _build_sg_diff_kernel(window, order, deriv=2, delta=DT)
    pad = window // 2
    pos_t = torch.from_numpy(positions).float().unsqueeze(0).permute(0, 2, 1)  # [1, 2, T]
    pos_padded = torch.nn.functional.pad(pos_t, (pad, pad), mode="replicate")
    vel_t = torch.nn.functional.conv1d(
        pos_padded, vel_kernel.view(1, 1, -1).expand(2, 1, -1), groups=2
    )
    accel_t = torch.nn.functional.conv1d(
        pos_padded, accel_kernel.view(1, 1, -1).expand(2, 1, -1), groups=2
    )
    vx = vel_t[0, 0].numpy()
    vy = vel_t[0, 1].numpy()
    ax = accel_t[0, 0].numpy()
    ay = accel_t[0, 1].numpy()

    speed = np.sqrt(vx**2 + vy**2).clip(min=1e-6)
    # Lateral acceleration = |v × a| / |v|
    cross = np.abs(vx * ay - vy * ax)
    lat_acc = cross / speed
    return lat_acc


@torch.no_grad()
def generate_trajectory(model, model_args, data, device):
    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    norm_data = {k: v.clone() for k, v in data.items()}
    norm_data = model_args.observation_normalizer(norm_data)
    norm_data["sampled_trajectories"] = torch.zeros(B, P, future_len + 1, 4, device=device)

    _orig_gfn = model.decoder._guidance_fn
    model.decoder._guidance_fn = None
    _, outputs = model(norm_data)
    model.decoder._guidance_fn = _orig_gfn

    return outputs["prediction"][0, 0].cpu().numpy()  # [T, 4]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(Path(args.model_path), device)
    model.eval()

    # Load LoRA if specified
    if args.lora_path:
        from preference_optimization.lora_utils import load_lora_checkpoint

        model = load_lora_checkpoint(model, args.lora_path)
        model.eval()
        print(f"Loaded LoRA from {args.lora_path}")

    with open(args.scenes) as f:
        scene_paths = json.load(f)
        from diffusion_planner.utils.scene_skip import filter_scene_list

        scene_paths = filter_scene_list(scene_paths, label="eval_driving_metrics")

    print(f"Evaluating {len(scene_paths)} scenes [{args.tag}]\n")

    # Per-scene accumulators
    all_speeds = []
    all_lat_accel_curv = []
    all_lat_accel_curr = []
    all_lat_accel_smooth = []
    all_path_lengths = []
    stopped_count = 0

    scene_results = []

    for i, npz_path in enumerate(scene_paths):
        data = load_npz_data(npz_path, device)
        traj = generate_trajectory(model, model_args, data, device)  # [T, 4]

        pos = traj[:, :2]
        # Speed
        vel = np.diff(pos, axis=0) / DT
        speed = np.linalg.norm(vel, axis=-1)
        all_speeds.append(speed)

        # Path length
        path_len = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=-1))
        all_path_lengths.append(path_len)

        # Stopped check (max speed < 0.5 m/s)
        if speed.max() < 0.5:
            stopped_count += 1

        # Lat accel - curvature based (accurate)
        la_curv = lat_accel_curvature(traj)
        all_lat_accel_curv.append(la_curv)

        # Lat accel - current method (reward.py)
        la_curr = lat_accel_current(pos)
        all_lat_accel_curr.append(la_curr)

        # Lat accel - smoothed
        la_smooth = lat_accel_smoothed(pos)
        all_lat_accel_smooth.append(la_smooth)

        scene_results.append(
            {
                "scene": Path(npz_path).stem,
                "max_speed": float(speed.max()),
                "mean_speed": float(speed.mean()),
                "path_length": float(path_len),
                "max_lat_accel_curv": float(np.abs(la_curv).max()),
                "max_lat_accel_curr": float(np.abs(la_curr).max()),
                "max_lat_accel_smooth": float(np.abs(la_smooth).max()),
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(scene_paths)} scenes...")

    # Aggregate
    speeds_all = np.concatenate(all_speeds)
    la_curv_all = np.concatenate(all_lat_accel_curv)
    la_curr_all = np.concatenate(all_lat_accel_curr)
    la_smooth_all = np.concatenate(all_lat_accel_smooth)

    print(f"\n{'=' * 70}")
    print(f"TELEPORT METRICS — {args.tag} ({len(scene_paths)} scenes)")
    print(f"{'=' * 70}")

    print(f"\n--- Speed ---")
    print(f"  max:  {speeds_all.max():.2f} m/s")
    print(f"  mean: {speeds_all.mean():.2f} m/s")
    print(f"  p95:  {np.percentile(speeds_all, 95):.2f} m/s")

    print(f"\n--- Lateral Acceleration (curvature-based, ACCURATE) ---")
    print(f"  max:  {np.abs(la_curv_all).max():.2f} m/s²")
    print(f"  mean: {np.abs(la_curv_all).mean():.2f} m/s²")
    print(f"  p95:  {np.percentile(np.abs(la_curv_all), 95):.2f} m/s²")

    print(f"\n--- Lateral Acceleration (smoothed finite diff) ---")
    print(f"  max:  {np.abs(la_smooth_all).max():.2f} m/s²")
    print(f"  mean: {np.abs(la_smooth_all).mean():.2f} m/s²")
    print(f"  p95:  {np.percentile(np.abs(la_smooth_all), 95):.2f} m/s²")

    print(f"\n--- Lateral Acceleration (current reward.py, NOISY) ---")
    print(f"  max:  {np.abs(la_curr_all).max():.2f} m/s²")
    print(f"  mean: {np.abs(la_curr_all).mean():.2f} m/s²")
    print(f"  p95:  {np.percentile(np.abs(la_curr_all), 95):.2f} m/s²")

    print(f"\n--- Other ---")
    print(f"  mean path length: {np.mean(all_path_lengths):.1f} m")
    print(f"  stopped scenes:   {stopped_count}/{len(scene_paths)}")

    # Per-scene worst offenders
    scene_results.sort(key=lambda x: x["max_lat_accel_curv"], reverse=True)
    print(f"\n--- Top 5 worst lat accel scenes (curvature) ---")
    for sr in scene_results[:5]:
        print(
            f"  {sr['scene']}: curv={sr['max_lat_accel_curv']:.2f}  speed={sr['max_speed']:.2f}  path={sr['path_length']:.1f}m"
        )


if __name__ == "__main__":
    main()
