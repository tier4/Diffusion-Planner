#!/usr/bin/env python3
"""Cleanse scene lists by removing scenes where ego starts outside lane at t=0.

Optionally checks that GT trajectory stays in lane for the first N timesteps
(--check_gt_lane --gt_max_t 20) and that GT path is at least --min_gt_path meters.

Uses compute_lane_departure_penalty on a single-timestep trajectory at the
ego starting position to check if the ego perimeter is inside a lane.

Usage:
    python -m rlvr.autoresearch.tools.cleanse_lane_scenes \
        --scenes /path/to/scenes.json \
        --output /path/to/clean_scenes.json \
        [--threshold 0.15] \
        [--also_check_road_border] \
        [--check_gt_lane] [--gt_max_t 20] [--min_gt_path 5.0]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.reward import compute_lane_departure_penalty, compute_road_border_penalty

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def check_scene_t0(npz_path: str, device: torch.device, threshold: float = 0.15,
                    check_road_border: bool = False) -> tuple[bool, float, float]:
    """Check if ego is in-lane at t=0.

    Returns: (is_clean, lane_clearance, rb_clearance)
    """
    data = load_npz_data(npz_path, device)

    es = data.get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es
    if ego_shape is None:
        ego_shape = torch.tensor([2.75, 4.34, 1.70], device=device)

    # Create a 2-timestep trajectory at origin (ego at t=0)
    # t=0 is skipped by the penalty, so we need t=0 and t=1 both at origin
    traj_t0 = torch.zeros(1, 2, 4, device=device)
    traj_t0[0, 0, 2] = 1.0  # cos(0) = 1
    traj_t0[0, 1, 2] = 1.0  # cos(0) = 1

    # Lane check — we need t=1 to not be skipped
    # Temporarily patch: compute on full trajectory but only t=1 matters
    lane_gate, lane_near, lane_wide, _, lane_cont = compute_lane_departure_penalty(
        traj_t0, ego_shape, data
    )

    # Lane clearance: if near_frac > 0 at t=1, ego is within 25cm of lane edge
    # If gate=0, ego is outside lane
    if lane_gate.item() < 0.5:
        lane_clearance = 0.0  # crossing
    elif lane_near.item() > 0:
        lane_clearance = 0.10  # near edge
    elif lane_wide.item() > 0:
        lane_clearance = 0.30  # between near and wide
    else:
        lane_clearance = 0.50  # safe

    # Road border check
    rb_clearance = 1.0
    if check_road_border:
        rb_gate, rb_near, rb_wide, _, _, _ = compute_road_border_penalty(
            traj_t0, ego_shape, data
        )
        if rb_gate.item() < 0.5:
            rb_clearance = 0.0
        elif rb_near.item() > 0:
            rb_clearance = 0.10
        elif rb_wide.item() > 0:
            rb_clearance = 0.30
        else:
            rb_clearance = 0.50

    is_clean = lane_clearance >= threshold
    if check_road_border:
        is_clean = is_clean and rb_clearance >= threshold

    return is_clean, lane_clearance, rb_clearance


@torch.no_grad()
def check_gt_lane_departure(npz_path: str, device: torch.device,
                             gt_max_t: int = 20, min_gt_path: float = 5.0
                             ) -> tuple[bool, float, bool]:
    """Check if GT trajectory stays in lane for first gt_max_t timesteps
    and has sufficient path length.

    Returns: (gt_in_lane, gt_path_len, gt_path_ok)
    """
    with np.load(npz_path, allow_pickle=True) as raw:
        gt = raw["ego_agent_future"].copy()  # [T, 3] = x, y, heading

    # GT path length — only over valid (non-padded) timesteps
    gt_xy = gt[:, :2]
    valid_mask = np.abs(gt_xy).sum(axis=-1) > 0.1
    valid_xy = gt_xy[valid_mask]
    if len(valid_xy) >= 2:
        gt_path = float(np.sqrt(np.diff(valid_xy[:, 0])**2 + np.diff(valid_xy[:, 1])**2).sum())
    else:
        gt_path = 0.0
    gt_path_ok = gt_path >= min_gt_path

    data = load_npz_data(npz_path, device)
    es = data.get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es
    if ego_shape is None:
        ego_shape = torch.tensor([2.75, 4.34, 1.70], device=device)

    # Pad GT to [T, 4] = [x, y, cos_yaw, sin_yaw] for compute_lane_departure_penalty
    gt_padded = np.zeros((gt.shape[0], 4), dtype=np.float32)
    gt_padded[:, :2] = gt[:, :2]
    gt_padded[:, 2] = np.cos(gt[:, 2])  # heading radians → cos_yaw
    gt_padded[:, 3] = np.sin(gt[:, 2])  # heading radians → sin_yaw

    # Check first gt_max_t timesteps
    t_check = min(gt_max_t, gt.shape[0])
    gt_short = torch.tensor(gt_padded[:t_check][None], device=device, dtype=torch.float32)
    gate, _, _, _, _ = compute_lane_departure_penalty(gt_short, ego_shape, data)
    gt_in_lane = gate.item() >= 0.5

    return gt_in_lane, gt_path, gt_path_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Min clearance to lane edge at t=0 (meters)")
    parser.add_argument("--also_check_road_border", action="store_true")
    parser.add_argument("--check_gt_lane", action="store_true",
                        help="Also check that GT stays in lane for first --gt_max_t steps")
    parser.add_argument("--gt_max_t", type=int, default=20,
                        help="Number of GT timesteps to check for lane departure (default: 20)")
    parser.add_argument("--min_gt_path", type=float, default=5.0,
                        help="Minimum GT path length in meters (default: 5.0)")
    args = parser.parse_args()

    device = torch.device(DEVICE)

    with open(args.scenes) as f:
        scenes = json.load(f)

    good = []
    bad = []
    gt_bad = []
    gt_short_path = []

    for i, path in enumerate(scenes):
        is_clean, lane_cl, rb_cl = check_scene_t0(
            path, device, args.threshold, args.also_check_road_border
        )
        if not is_clean:
            bad.append((i, lane_cl, rb_cl, path, "t0_lane"))
            continue

        # Optional GT lane departure check
        if args.check_gt_lane:
            gt_in_lane, gt_path, gt_path_ok = check_gt_lane_departure(
                path, device, args.gt_max_t, args.min_gt_path
            )
            if not gt_path_ok:
                gt_short_path.append((i, gt_path, path))
                continue
            if not gt_in_lane:
                gt_bad.append((i, gt_path, path))
                continue

        good.append(path)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(scenes)} ({len(bad)+len(gt_bad)+len(gt_short_path)} removed)")

    print(f"\nTotal: {len(scenes)} scenes")
    print(f"Clean: {len(good)}")
    print(f"Bad t=0 lane: {len(bad)}")
    if args.check_gt_lane:
        print(f"Bad GT lane (first {args.gt_max_t} steps): {len(gt_bad)}")
        print(f"Bad GT path (<{args.min_gt_path}m): {len(gt_short_path)}")

    if bad:
        print(f"\nBad t=0 scenes (first 10):")
        for i, lane_cl, rb_cl, path, reason in bad[:10]:
            print(f"  scene {i}: lane={lane_cl:.2f}m rb={rb_cl:.2f}m — {Path(path).stem[-30:]}")
    if gt_bad:
        print(f"\nBad GT lane scenes (first 10):")
        for i, gt_path, path in gt_bad[:10]:
            print(f"  scene {i}: gt_path={gt_path:.1f}m — {Path(path).stem[-30:]}")

    with open(args.output, 'w') as f:
        json.dump(good, f, indent=2)
    print(f"\nSaved {len(good)} clean scenes to {args.output}")


if __name__ == "__main__":
    main()
