#!/usr/bin/env python3
"""Cleanse scene lists by removing scenes where ego starts outside lane at t=0.

Optional checks:
  --check_gt_lane  --gt_max_t 20   reject if GT trajectory leaves lane within
                                    the first N timesteps (default 20).
  --min_gt_path 5.0                reject if GT path length < N metres.
  --check_rb_future --rb_thresh 0.4 --rb_max_t 10
                                    reject if the GT ego footprint comes within
                                    `--rb_thresh` metres of any road border over
                                    the first `--rb_max_t` timesteps (t=0..N).
                                    Catches midcurve apex frames where the
                                    recorded driver threads a curb.

Uses compute_lane_departure_penalty and compute_road_border_penalty directly,
so the thresholds and ego-footprint sampling match training exactly.

Usage:
    python -m rlvr.autoresearch.tools.cleanse_lane_scenes \
        --scenes /path/to/scenes.json \
        --output /path/to/clean_scenes.json \
        [--threshold 0.15] \
        [--also_check_road_border] \
        [--check_gt_lane] [--gt_max_t 20] [--min_gt_path 5.0] \
        [--check_rb_future] [--rb_thresh 0.4] [--rb_max_t 10]
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
def check_scene_t0(
    npz_path: str, device: torch.device, threshold: float = 0.15, check_road_border: bool = False
) -> tuple[bool, float, float]:
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
        rb_gate, rb_near, rb_wide, _, _, _ = compute_road_border_penalty(traj_t0, ego_shape, data)
        if rb_gate.item() < 0.5:
            rb_clearance = 0.0
        elif rb_near.item() > 0:
            rb_clearance = 0.10
        elif rb_wide.item() > 0:
            rb_clearance = 0.30
        else:
            rb_clearance = 0.50

    # Zero clearance = gate fired = crossing. Always reject — even with
    # threshold=0, where `>= threshold` would otherwise let crossings through.
    is_clean = lane_clearance > 0.0 and lane_clearance >= threshold
    if check_road_border:
        is_clean = is_clean and rb_clearance > 0.0 and rb_clearance >= threshold

    return is_clean, lane_clearance, rb_clearance


@torch.no_grad()
def check_gt_lane_departure(
    npz_path: str, device: torch.device, gt_max_t: int = 20, min_gt_path: float = 5.0
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
        gt_path = float(np.sqrt(np.diff(valid_xy[:, 0]) ** 2 + np.diff(valid_xy[:, 1]) ** 2).sum())
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


@torch.no_grad()
def check_rb_future_margin(
    npz_path: str, device: torch.device, rb_thresh: float = 0.4, rb_max_t: int = 10
) -> tuple[bool, float]:
    """Check min ego-to-road-border distance across [t=0 .. t=rb_max_t] of GT.

    Builds a (rb_max_t + 1)-step trajectory from GT positions and calls
    compute_road_border_penalty, then takes the min over all checked timesteps
    (including t=0 — reward.py no longer sentinels the first slot).

    Returns (is_safe, min_dist_m). is_safe = min_dist >= rb_thresh.
    """
    with np.load(npz_path, allow_pickle=True) as raw:
        gt = raw["ego_agent_future"].copy()  # [T, 3] = x, y, heading

    data = load_npz_data(npz_path, device)
    es = data.get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es
    if ego_shape is None:
        ego_shape = torch.tensor([2.75, 4.34, 1.70], device=device)

    # Build (rb_max_t + 1)-step trajectory: t=0 at origin + first rb_max_t GT steps
    T_full = rb_max_t + 1
    traj = np.zeros((T_full, 4), dtype=np.float32)
    traj[0, 2] = 1.0  # cos_yaw = 1 (heading = 0) at origin
    n_gt = min(rb_max_t, gt.shape[0])
    traj[1 : 1 + n_gt, :2] = gt[:n_gt, :2]
    traj[1 : 1 + n_gt, 2] = np.cos(gt[:n_gt, 2])
    traj[1 : 1 + n_gt, 3] = np.sin(gt[:n_gt, 2])
    if n_gt < rb_max_t:
        traj[1 + n_gt :, 2] = 1.0

    traj_t = torch.tensor(traj[None], device=device)
    _, _, _, _, _, per_ts_min = compute_road_border_penalty(traj_t, ego_shape, data)
    min_dist = float(per_ts_min[0, :].min().item())
    return min_dist >= rb_thresh, min_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--threshold", type=float, default=0.15, help="Min clearance to lane edge at t=0 (meters)"
    )
    parser.add_argument("--also_check_road_border", action="store_true")
    parser.add_argument(
        "--check_gt_lane",
        action="store_true",
        help="Also check that GT stays in lane for first --gt_max_t steps",
    )
    parser.add_argument(
        "--gt_max_t",
        type=int,
        default=20,
        help="Number of GT timesteps to check for lane departure (default: 20)",
    )
    parser.add_argument(
        "--min_gt_path",
        type=float,
        default=5.0,
        help="Minimum GT path length in meters (default: 5.0)",
    )
    parser.add_argument(
        "--check_rb_future",
        action="store_true",
        help="Also drop scenes where ego is within --rb_thresh of a road "
        "border at any of the first --rb_max_t GT timesteps.",
    )
    parser.add_argument(
        "--rb_thresh",
        type=float,
        default=0.4,
        help="Min ego-to-border distance required at every checked timestep (default: 0.4 m)",
    )
    parser.add_argument(
        "--rb_max_t",
        type=int,
        default=10,
        help="Number of timesteps to check (default: 10, i.e. t=0..10)",
    )
    args = parser.parse_args()

    device = torch.device(DEVICE)

    with open(args.scenes) as f:
        scenes = json.load(f)

    good = []
    bad = []
    gt_bad = []
    gt_short_path = []
    rb_future_bad = []

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

        # Optional rb-future margin check
        if args.check_rb_future:
            is_safe, min_dist = check_rb_future_margin(path, device, args.rb_thresh, args.rb_max_t)
            if not is_safe:
                rb_future_bad.append((i, min_dist, path))
                continue

        good.append(path)

        if (i + 1) % 20 == 0:
            n_removed = len(bad) + len(gt_bad) + len(gt_short_path) + len(rb_future_bad)
            print(f"  Processed {i + 1}/{len(scenes)} ({n_removed} removed)")

    print(f"\nTotal: {len(scenes)} scenes")
    print(f"Clean: {len(good)}")
    print(f"Bad t=0 lane: {len(bad)}")
    if args.check_gt_lane:
        print(f"Bad GT lane (first {args.gt_max_t} steps): {len(gt_bad)}")
        print(f"Bad GT path (<{args.min_gt_path}m): {len(gt_short_path)}")
    if args.check_rb_future:
        print(
            f"Bad rb-future margin (<{args.rb_thresh}m within t=0..{args.rb_max_t}): {len(rb_future_bad)}"
        )

    if bad:
        print(f"\nBad t=0 scenes (first 10):")
        for i, lane_cl, rb_cl, path, reason in bad[:10]:
            print(f"  scene {i}: lane={lane_cl:.2f}m rb={rb_cl:.2f}m — {Path(path).stem[-30:]}")
    if gt_bad:
        print(f"\nBad GT lane scenes (first 10):")
        for i, gt_path, path in gt_bad[:10]:
            print(f"  scene {i}: gt_path={gt_path:.1f}m — {Path(path).stem[-30:]}")
    if rb_future_bad:
        print(f"\nBad rb-future scenes (first 10):")
        for i, min_dist, path in rb_future_bad[:10]:
            print(f"  scene {i}: rb_min={min_dist:.3f}m — {Path(path).stem[-30:]}")

    with open(args.output, "w") as f:
        json.dump(good, f, indent=2)
    print(f"\nSaved {len(good)} clean scenes to {args.output}")


if __name__ == "__main__":
    main()
