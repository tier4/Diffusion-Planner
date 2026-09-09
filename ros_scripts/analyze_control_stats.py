#!/usr/bin/env python3
"""Analyze acceleration/curvature statistics from a dataset (ego + neighbor).

Usage:
    python3 ros_scripts/analyze_control_stats.py <path_list.json>
"""

import json
import sys

import numpy as np
import torch
from diffusion_planner.utils.unicycle_accel_curvature import (
    UnicycleAccelCurvatureActionSpace,
    traj4d_to_action,
)
from tqdm import tqdm


def heading_to_cos_sin_np(traj_3d: np.ndarray) -> np.ndarray:
    """Convert (x, y, heading) to (x, y, cos, sin)."""
    out = np.zeros((*traj_3d.shape[:-1], 4), dtype=np.float32)
    out[..., :2] = traj_3d[..., :2]
    out[..., 2] = np.cos(traj_3d[..., 2])
    out[..., 3] = np.sin(traj_3d[..., 2])
    return out


def transform_to_local_frame(history_4d, future_4d):
    """Transform history and future from ego-centric to agent-local frame.

    Args:
        history_4d: [Pn, T_hist, 4] (x, y, cos, sin) in ego-centric frame
        future_4d: [Pn, T_fut, 4] (x, y, cos, sin) in ego-centric frame

    Returns:
        history_local: [Pn, T_hist, 4]
        future_local: [Pn, T_fut, 4]
    """
    # Reference position and heading from last history timestep
    pos = history_4d[:, -1:, :2]  # [Pn, 1, 2]
    cos_h = history_4d[:, -1:, 2:3]  # [Pn, 1, 1]
    sin_h = history_4d[:, -1:, 3:4]  # [Pn, 1, 1]

    # History to local
    h_xy = history_4d[..., :2] - pos
    h_x = h_xy[..., 0:1] * cos_h + h_xy[..., 1:2] * sin_h
    h_y = -h_xy[..., 0:1] * sin_h + h_xy[..., 1:2] * cos_h
    h_cos = history_4d[..., 2:3] * cos_h + history_4d[..., 3:4] * sin_h
    h_sin = -history_4d[..., 2:3] * sin_h + history_4d[..., 3:4] * cos_h
    history_local = np.concatenate([h_x, h_y, h_cos, h_sin], axis=-1)

    # Future to local
    f_xy = future_4d[..., :2] - pos
    f_x = f_xy[..., 0:1] * cos_h + f_xy[..., 1:2] * sin_h
    f_y = -f_xy[..., 0:1] * sin_h + f_xy[..., 1:2] * cos_h
    f_cos = future_4d[..., 2:3] * cos_h + future_4d[..., 3:4] * sin_h
    f_sin = -future_4d[..., 2:3] * sin_h + future_4d[..., 3:4] * cos_h
    future_local = np.concatenate([f_x, f_y, f_cos, f_sin], axis=-1)

    return history_local, future_local


def print_stats(name, data):
    print(f"=== {name} ===")
    print(f"  mean:   {np.mean(data):.6f}")
    print(f"  std:    {np.std(data):.6f}")
    print(f"  min:    {np.min(data):.6f}")
    print(f"  max:    {np.max(data):.6f}")
    print(f"  p1:     {np.percentile(data, 1):.6f}")
    print(f"  p5:     {np.percentile(data, 5):.6f}")
    print(f"  p25:    {np.percentile(data, 25):.6f}")
    print(f"  p50:    {np.percentile(data, 50):.6f}")
    print(f"  p75:    {np.percentile(data, 75):.6f}")
    print(f"  p95:    {np.percentile(data, 95):.6f}")
    print(f"  p99:    {np.percentile(data, 99):.6f}")
    print()


def main():
    path_list_json = sys.argv[1]
    with open(path_list_json) as f:
        paths = json.load(f)
    print(f"Total files: {len(paths)}")

    action_space = UnicycleAccelCurvatureActionSpace(n_waypoints=80)

    # Ego stats
    ego_accels = []
    ego_curvatures = []
    ego_v0s = []
    ego_errors = 0

    # Neighbor stats
    nei_accels = []
    nei_curvatures = []
    nei_v0s = []
    nei_errors = 0
    nei_valid_count = 0

    # Per-neighbor metadata for outlier analysis
    nei_metadata = []  # list of dicts with v0, valid_count, max_abs_accel, max_abs_kappa, file_path, neighbor_idx

    for p in tqdm(paths, desc="Converting trajectories"):
        try:
            d = np.load(p)
        except Exception as e:
            ego_errors += 1
            if ego_errors <= 5:
                print(f"Error loading {p}: {e}")
            continue

        # --- Ego ---
        try:
            past_4d = heading_to_cos_sin_np(d["ego_agent_past"])  # (31, 4)
            future_4d = heading_to_cos_sin_np(d["ego_agent_future"])  # (80, 4)
            v0 = float(d["ego_current_state"][4])

            past_t = torch.from_numpy(past_4d).unsqueeze(0)
            future_t = torch.from_numpy(future_4d).unsqueeze(0)
            t0_states = {"v": torch.tensor([v0])}

            with torch.no_grad():
                actions = traj4d_to_action(action_space, past_t, future_t, t0_states=t0_states)

            actions_np = actions.squeeze(0).numpy()  # (80, 2)
            ego_accels.append(actions_np[:, 0])
            ego_curvatures.append(actions_np[:, 1])
            ego_v0s.append(v0)
        except Exception as e:
            ego_errors += 1
            if ego_errors <= 5:
                print(f"Ego error on {p}: {e}")

        # --- Neighbors ---
        try:
            neighbor_past = d["neighbor_agents_past"]  # (Pn, T_hist, 11)
            neighbor_future = d["neighbor_agents_future"]  # (Pn, T_fut, 3)

            for ni in range(neighbor_past.shape[0]):
                # Check if neighbor is valid (last history timestep non-zero)
                if np.abs(neighbor_past[ni, -1, :4]).sum() < 1e-6:
                    continue

                # Check if neighbor has valid future
                future_3d = neighbor_future[ni]  # (T_fut, 3)
                future_valid = np.abs(future_3d).sum(axis=-1) > 0
                if future_valid.sum() == 0:
                    continue

                # neighbor_past has 11 dims: (x, y, cos, sin, ...) — first 4 are already cos/sin
                n_hist_4d = neighbor_past[ni, :, :4].astype(np.float32)  # (T_hist, 4)
                n_fut_4d = heading_to_cos_sin_np(future_3d)  # (T_fut, 4)

                # Zero out invalid future timesteps
                n_fut_4d[~future_valid] = 0.0

                # Transform to neighbor-local frame
                n_hist_local, n_fut_local = transform_to_local_frame(
                    n_hist_4d[None], n_fut_4d[None]
                )  # each [1, T, 4]

                # Restore zeros for invalid timesteps
                n_fut_local[0, ~future_valid] = 0.0

                past_t = torch.from_numpy(n_hist_local)  # [1, T_hist, 4]
                future_t = torch.from_numpy(n_fut_local)  # [1, T_fut, 4]

                with torch.no_grad():
                    actions = traj4d_to_action(action_space, past_t, future_t)

                actions_np = actions.squeeze(0).numpy()  # (T_fut, 2)

                # Only collect stats for valid timesteps
                valid_actions = actions_np[future_valid]
                if len(valid_actions) == 0:
                    continue

                # Skip if NaN/inf
                if np.any(~np.isfinite(valid_actions)):
                    nei_errors += 1
                    continue

                nei_accels.append(valid_actions[:, 0])
                nei_curvatures.append(valid_actions[:, 1])

                # Estimate v0 from last two history positions
                n_v0 = np.linalg.norm(neighbor_past[ni, -1, :2] - neighbor_past[ni, -2, :2]) / 0.1
                nei_v0s.append(n_v0)
                nei_valid_count += 1

                # Track per-neighbor metadata for outlier analysis
                nei_metadata.append(
                    {
                        "file_path": p,
                        "neighbor_idx": ni,
                        "v0": n_v0,
                        "valid_count": int(future_valid.sum()),
                        "max_abs_accel": float(np.max(np.abs(valid_actions[:, 0]))),
                        "max_abs_kappa": float(np.max(np.abs(valid_actions[:, 1]))),
                    }
                )

        except Exception as e:
            nei_errors += 1
            if nei_errors <= 5:
                print(f"Neighbor error on {p}: {e}")

    # --- Print results ---
    print(f"\nProcessed: {len(paths) - ego_errors} files, Ego errors: {ego_errors}")
    print(f"Valid neighbors: {nei_valid_count}, Neighbor errors: {nei_errors}")
    print()

    # Ego
    ego_accels = np.concatenate(ego_accels)
    ego_curvatures = np.concatenate(ego_curvatures)
    ego_v0s = np.array(ego_v0s)

    print("=" * 60)
    print("EGO")
    print("=" * 60)
    print_stats("Ego Acceleration (m/s^2)", ego_accels)
    print_stats("Ego Curvature (1/m)", ego_curvatures)
    print_stats("Ego Initial velocity v0 (m/s)", ego_v0s)

    # Neighbor
    if nei_valid_count > 0:
        nei_accels = np.concatenate(nei_accels)
        nei_curvatures = np.concatenate(nei_curvatures)
        nei_v0s = np.array(nei_v0s)

        print("=" * 60)
        print("NEIGHBOR")
        print("=" * 60)
        print_stats("Neighbor Acceleration (m/s^2)", nei_accels)
        print_stats("Neighbor Curvature (1/m)", nei_curvatures)
        print_stats("Neighbor Initial velocity v0 (m/s)", nei_v0s)
    else:
        print("No valid neighbors found.")

    # Comparison
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    ego_a_std = np.std(ego_accels)
    ego_k_std = np.std(ego_curvatures)
    print(
        f"  Ego:      accel_mean={np.mean(ego_accels):.6f}, accel_std={ego_a_std:.6f}, "
        f"kappa_mean={np.mean(ego_curvatures):.6f}, kappa_std={ego_k_std:.6f}"
    )
    if nei_valid_count > 0:
        nei_a_std = np.std(nei_accels)
        nei_k_std = np.std(nei_curvatures)
        print(
            f"  Neighbor: accel_mean={np.mean(nei_accels):.6f}, accel_std={nei_a_std:.6f}, "
            f"kappa_mean={np.mean(nei_curvatures):.6f}, kappa_std={nei_k_std:.6f}"
        )
        print(
            f"  Ratio (nei/ego): accel_std={nei_a_std / ego_a_std:.2f}x, "
            f"kappa_std={nei_k_std / ego_k_std:.2f}x"
        )

    # Outlier pattern analysis
    if nei_valid_count > 0 and len(nei_metadata) > 0:
        print()
        print("=" * 60)
        print("NEIGHBOR OUTLIER PATTERN ANALYSIS")
        print("=" * 60)

        meta_v0s = np.array([m["v0"] for m in nei_metadata])
        meta_valid_counts = np.array([m["valid_count"] for m in nei_metadata])
        meta_max_abs_accel = np.array([m["max_abs_accel"] for m in nei_metadata])
        meta_max_abs_kappa = np.array([m["max_abs_kappa"] for m in nei_metadata])

        # 1. Outlier breakdown by speed
        print()
        print("--- Outlier breakdown by v0 speed ---")
        speed_buckets = [
            ("0-0.5 m/s", 0.0, 0.5),
            ("0.5-2 m/s", 0.5, 2.0),
            ("2-5 m/s", 2.0, 5.0),
            ("5-10 m/s", 5.0, 10.0),
            ("10+ m/s", 10.0, float("inf")),
        ]
        for label, lo, hi in speed_buckets:
            mask = (meta_v0s >= lo) & (meta_v0s < hi)
            n = mask.sum()
            if n == 0:
                print(f"  {label:12s}: n=0")
                continue
            a = meta_max_abs_accel[mask]
            k = meta_max_abs_kappa[mask]
            print(
                f"  {label:12s}: n={n:6d}  "
                f"accel(mean={np.mean(a):.2f}, p95={np.percentile(a, 95):.2f}, "
                f"p99={np.percentile(a, 99):.2f}, max={np.max(a):.2f})  "
                f"kappa(mean={np.mean(k):.4f}, p95={np.percentile(k, 95):.4f}, "
                f"p99={np.percentile(k, 99):.4f}, max={np.max(k):.4f})"
            )

        # 2. Outlier breakdown by valid future length
        print()
        print("--- Outlier breakdown by valid future timestep count ---")
        valid_buckets = [
            ("1-10", 1, 10),
            ("11-30", 11, 30),
            ("31-60", 31, 60),
            ("61-80", 61, 80),
        ]
        for label, lo, hi in valid_buckets:
            mask = (meta_valid_counts >= lo) & (meta_valid_counts <= hi)
            n = mask.sum()
            if n == 0:
                print(f"  {label:6s}: n=0")
                continue
            a = meta_max_abs_accel[mask]
            k = meta_max_abs_kappa[mask]
            print(
                f"  {label:6s}: n={n:6d}  "
                f"accel(mean={np.mean(a):.2f}, p95={np.percentile(a, 95):.2f}, "
                f"p99={np.percentile(a, 99):.2f}, max={np.max(a):.2f})  "
                f"kappa(mean={np.mean(k):.4f}, p95={np.percentile(k, 95):.4f}, "
                f"p99={np.percentile(k, 99):.4f}, max={np.max(k):.4f})"
            )

        # 3. Top 10 most extreme outlier samples
        print()
        print("--- Top 10 most extreme ACCEL outliers ---")
        accel_order = np.argsort(meta_max_abs_accel)[::-1]
        for rank, idx in enumerate(accel_order[:10]):
            m = nei_metadata[idx]
            print(
                f"  #{rank + 1:2d}: max_abs_accel={m['max_abs_accel']:.2f}, "
                f"max_abs_kappa={m['max_abs_kappa']:.4f}, "
                f"v0={m['v0']:.3f}, valid_count={m['valid_count']}, "
                f"nei_idx={m['neighbor_idx']}, file={m['file_path']}"
            )

        print()
        print("--- Top 10 most extreme KAPPA outliers ---")
        kappa_order = np.argsort(meta_max_abs_kappa)[::-1]
        for rank, idx in enumerate(kappa_order[:10]):
            m = nei_metadata[idx]
            print(
                f"  #{rank + 1:2d}: max_abs_kappa={m['max_abs_kappa']:.4f}, "
                f"max_abs_accel={m['max_abs_accel']:.2f}, "
                f"v0={m['v0']:.3f}, valid_count={m['valid_count']}, "
                f"nei_idx={m['neighbor_idx']}, file={m['file_path']}"
            )

    # Suggested normalizer values
    print()
    print("=" * 60)
    print("SUGGESTED NORMALIZER VALUES")
    print("=" * 60)
    print(f"  ego_control_normalizer:")
    print(f"    mean: [{np.mean(ego_accels):.6f}, {np.mean(ego_curvatures):.6f}]")
    print(f"    std:  [{ego_a_std:.6f}, {ego_k_std:.6f}]")
    if nei_valid_count > 0:
        print(f"  neighbor_control_normalizer:")
        print(f"    mean: [{np.mean(nei_accels):.6f}, {np.mean(nei_curvatures):.6f}]")
        print(f"    std:  [{nei_a_std:.6f}, {nei_k_std:.6f}]")


if __name__ == "__main__":
    main()
