#!/usr/bin/env python3
"""Cleanse scene lists by removing scenes where ego starts outside lane at t=0.

Uses compute_lane_departure_penalty on a single-timestep trajectory at the
ego starting position to check if the ego perimeter is inside a lane.

Usage:
    python -m rlvr.autoresearch.tools.cleanse_lane_scenes \
        --scenes /path/to/scenes.json \
        --output /path/to/clean_scenes.json \
        [--threshold 0.15] \
        [--also_check_road_border]
"""

import argparse
import json
from pathlib import Path

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
        rb_gate, rb_near, rb_wide, _, _ = compute_road_border_penalty(
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Min clearance to lane edge at t=0 (meters)")
    parser.add_argument("--also_check_road_border", action="store_true")
    args = parser.parse_args()

    device = torch.device(DEVICE)

    with open(args.scenes) as f:
        scenes = json.load(f)

    good = []
    bad = []

    for i, path in enumerate(scenes):
        is_clean, lane_cl, rb_cl = check_scene_t0(
            path, device, args.threshold, args.also_check_road_border
        )
        if is_clean:
            good.append(path)
        else:
            bad.append((i, lane_cl, rb_cl, path))

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(scenes)} ({len(bad)} bad so far)")

    print(f"\nTotal: {len(scenes)} scenes")
    print(f"Clean: {len(good)} (lane clearance >= {args.threshold}m at t=0)")
    print(f"Bad: {len(bad)}")

    if bad:
        print(f"\nBad scenes:")
        for i, lane_cl, rb_cl, path in bad:
            print(f"  scene {i}: lane={lane_cl:.2f}m rb={rb_cl:.2f}m — {Path(path).stem[-30:]}")

    with open(args.output, 'w') as f:
        json.dump(good, f, indent=2)
    print(f"\nSaved {len(good)} clean scenes to {args.output}")


if __name__ == "__main__":
    main()
