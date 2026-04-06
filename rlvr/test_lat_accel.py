#!/usr/bin/env python3
"""Standalone test comparing three lateral acceleration calculation methods.

Methods compared:
  (a) Current: double finite difference on raw positions (what reward.py does)
  (b) Smoothed: Savitzky-Golay filter on positions, then finite difference
  (c) Curvature-based: heading from cos/sin output, kappa = dtheta/ds, lat_accel = v^2 * kappa

Usage:
    source .venv/bin/activate
    python rlvr/test_lat_accel.py \
        --model_path /path/to/best_model.pth \
        --scenes /path/to/scenes.json \
        --n_scenes 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rlvr.reward import _build_sg_diff_kernel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "diffusion_planner"))
sys.path.insert(0, str(PROJECT_ROOT / "preference_optimization"))

from model_utils import load_model
from utils import load_npz_data

DT = 0.1  # 10 Hz


# ---------------------------------------------------------------------------
# Method A: Current double finite diff (same as reward.py lines 679-693)
# ---------------------------------------------------------------------------
def lat_accel_current(positions: np.ndarray) -> np.ndarray:
    """positions: [T, 2] -> lat_accel: [T-2]"""
    vel = np.diff(positions, axis=0) / DT          # (T-1, 2)
    acc_vec = np.diff(vel, axis=0) / DT            # (T-2, 2)
    heading = vel[:-1]                              # (T-2, 2)
    heading_norm = heading / (np.linalg.norm(heading, axis=-1, keepdims=True) + 1e-6)
    lat_dir = np.stack([-heading_norm[:, 1], heading_norm[:, 0]], axis=-1)
    lat_accel = np.sum(acc_vec * lat_dir, axis=-1)  # (T-2,)
    return lat_accel


# ---------------------------------------------------------------------------
# Method B: Savitzky-Golay smoothed positions, then same finite diff
# ---------------------------------------------------------------------------
def lat_accel_smoothed(positions: np.ndarray, window: int = 7, order: int = 3) -> np.ndarray:
    """positions: [T, 2] -> lat_accel: [T-2]"""
    if positions.shape[0] < window:
        return lat_accel_current(positions)
    # SG smoothing (deriv=0) via numpy convolution
    smooth_kernel = _build_sg_diff_kernel(window, order, deriv=0, delta=1.0).numpy()
    pad = window // 2
    def _sg_smooth(signal):
        padded = np.pad(signal, pad, mode='edge')
        return np.convolve(padded, smooth_kernel, mode='valid')
    smoothed = np.stack([
        _sg_smooth(positions[:, 0]),
        _sg_smooth(positions[:, 1]),
    ], axis=-1)
    return lat_accel_current(smoothed)


# ---------------------------------------------------------------------------
# Method C: Curvature-based using heading from cos/sin
# ---------------------------------------------------------------------------
def lat_accel_curvature(trajectory: np.ndarray) -> np.ndarray:
    """trajectory: [T, 4] (x, y, cos, sin) -> lat_accel: [T-1]

    Uses heading directly from the model output (cos/sin columns).
    kappa = dtheta / ds, lat_accel = v^2 * kappa
    """
    x, y = trajectory[:, 0], trajectory[:, 1]
    cos_h, sin_h = trajectory[:, 2], trajectory[:, 3]
    theta = np.arctan2(sin_h, cos_h)  # (T,)

    # Velocity from positions
    dx = np.diff(x) / DT
    dy = np.diff(y) / DT
    speed = np.sqrt(dx**2 + dy**2)  # (T-1,)

    # Angular rate from heading
    dtheta = np.diff(theta)  # (T-1,)
    # Unwrap large jumps
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    omega = dtheta / DT  # angular velocity (T-1,)

    # lat_accel = v * omega  (equivalent to v^2 * kappa since kappa = omega/v)
    lat_accel = speed * omega  # (T-1,)
    return lat_accel


def lat_accel_curvature_from_xy(positions: np.ndarray) -> np.ndarray:
    """For GT trajectories that only have [x, y, yaw_rad].
    Derives heading from yaw column. positions: [T, 3] -> lat_accel: [T-1]
    """
    x, y, yaw = positions[:, 0], positions[:, 1], positions[:, 2]
    dx = np.diff(x) / DT
    dy = np.diff(y) / DT
    speed = np.sqrt(dx**2 + dy**2)

    dtheta = np.diff(yaw)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    omega = dtheta / DT

    return speed * omega


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------
def compute_stats(values: np.ndarray) -> dict:
    abs_vals = np.abs(values)
    return {
        "max": float(np.max(abs_vals)) if len(abs_vals) > 0 else 0.0,
        "mean": float(np.mean(abs_vals)) if len(abs_vals) > 0 else 0.0,
        "p95": float(np.percentile(abs_vals, 95)) if len(abs_vals) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Compare lateral acceleration methods")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--n_scenes", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, model_args = load_model(Path(args.model_path), device)
    model.eval()

    # Load scene list
    with open(args.scenes) as f:
        scene_paths = json.load(f)
    scene_paths = scene_paths[: args.n_scenes]

    print(f"\nLoaded model from {args.model_path}")
    print(f"Processing {len(scene_paths)} scenes\n")

    # Accumulators per method
    all_results = {
        "model": {"current": [], "smoothed": [], "curvature": []},
        "gt": {"current": [], "smoothed": [], "curvature": []},
    }

    for i, npz_path in enumerate(scene_paths):
        print(f"--- Scene {i+1}: {Path(npz_path).stem} ---")
        data = load_npz_data(npz_path, device)

        # Generate deterministic trajectory
        B = data["ego_current_state"].shape[0]
        P = 1 + model_args.predicted_neighbor_num
        future_len = model_args.future_len

        norm_data = {k: v.clone() for k, v in data.items()}
        norm_data = model_args.observation_normalizer(norm_data)
        norm_data["sampled_trajectories"] = torch.zeros(
            B, P, future_len + 1, 4, device=device
        )

        _orig_gfn = model.decoder._guidance_fn
        model.decoder._guidance_fn = None
        _, outputs = model(norm_data)
        model.decoder._guidance_fn = _orig_gfn

        pred_traj = outputs["prediction"][0, 0].cpu().numpy()  # [T, 4] x,y,cos,sin

        # GT trajectory
        gt_raw = np.load(str(npz_path))
        gt_future = gt_raw["ego_agent_future"]  # [80, 3] x,y,yaw_rad

        # --- Model predictions ---
        model_pos = pred_traj[:, :2]
        mc = lat_accel_current(model_pos)
        ms = lat_accel_smoothed(model_pos)
        mk = lat_accel_curvature(pred_traj)

        all_results["model"]["current"].append(mc)
        all_results["model"]["smoothed"].append(ms)
        all_results["model"]["curvature"].append(mk)

        mc_s, ms_s, mk_s = compute_stats(mc), compute_stats(ms), compute_stats(mk)
        print(f"  MODEL  current:   max={mc_s['max']:6.2f}  mean={mc_s['mean']:5.2f}  p95={mc_s['p95']:5.2f}")
        print(f"  MODEL  smoothed:  max={ms_s['max']:6.2f}  mean={ms_s['mean']:5.2f}  p95={ms_s['p95']:5.2f}")
        print(f"  MODEL  curvature: max={mk_s['max']:6.2f}  mean={mk_s['mean']:5.2f}  p95={mk_s['p95']:5.2f}")

        # --- GT ---
        gt_pos = gt_future[:, :2]
        gc = lat_accel_current(gt_pos)
        gs = lat_accel_smoothed(gt_pos)
        gk = lat_accel_curvature_from_xy(gt_future)

        all_results["gt"]["current"].append(gc)
        all_results["gt"]["smoothed"].append(gs)
        all_results["gt"]["curvature"].append(gk)

        gc_s, gs_s, gk_s = compute_stats(gc), compute_stats(gs), compute_stats(gk)
        print(f"  GT     current:   max={gc_s['max']:6.2f}  mean={gc_s['mean']:5.2f}  p95={gc_s['p95']:5.2f}")
        print(f"  GT     smoothed:  max={gs_s['max']:6.2f}  mean={gs_s['mean']:5.2f}  p95={gs_s['p95']:5.2f}")
        print(f"  GT     curvature: max={gk_s['max']:6.2f}  mean={gk_s['mean']:5.2f}  p95={gk_s['p95']:5.2f}")
        print()

    # ---------------------------------------------------------------------------
    # Aggregate summary
    # ---------------------------------------------------------------------------
    print("=" * 78)
    print("AGGREGATE SUMMARY (across all scenes)")
    print("=" * 78)
    header = f"{'Source':<8} {'Method':<12} {'Max':>8} {'Mean':>8} {'P95':>8}"
    print(header)
    print("-" * len(header))

    for source in ["model", "gt"]:
        for method in ["current", "smoothed", "curvature"]:
            combined = np.concatenate(all_results[source][method])
            s = compute_stats(combined)
            print(f"{source:<8} {method:<12} {s['max']:8.2f} {s['mean']:8.2f} {s['p95']:8.2f}")
        print()

    # Noise amplification ratio
    print("NOISE AMPLIFICATION (current_max / curvature_max):")
    for source in ["model", "gt"]:
        c_max = compute_stats(np.concatenate(all_results[source]["current"]))["max"]
        k_max = compute_stats(np.concatenate(all_results[source]["curvature"]))["max"]
        ratio = c_max / k_max if k_max > 1e-6 else float("inf")
        print(f"  {source}: {c_max:.2f} / {k_max:.2f} = {ratio:.1f}x")


if __name__ == "__main__":
    main()
