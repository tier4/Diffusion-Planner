#!/usr/bin/env python3
"""Filter avoidance scenes to those that are sc-clean at t=0.

The static-collision gate ignores t=0, so a scene whose ego ALREADY overlaps
(or nearly touches) a stopped neighbor at t=0 silently passes every gate and
poisons recovery/avoidance training (can't train avoidance from an
already-failed start). This tool drops such scenes, plus scenes with no
stopped neighbor at all (nothing to avoid).

Mechanism (canonical reward path, no hand-rolled geometry): score a static
trajectory pinned at the t=0 ego pose (origin of the ego frame) through
``compute_reward_batch``; its ``sc_min_dist`` IS the t=0 OBB clearance to the
nearest stopped neighbor. Keep scenes with sc_min_dist >= sc_cross_thresh.

Usage:
    python -m rlvr.autoresearch.tools.filter_sc_t0_clean \
        --scenes <scenes.json> --config <reward_config.json> \
        --out <filtered.json>

Outputs <out> (kept scene list) and <out>.report.json (kept / dropped lists).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import compute_reward_batch


def t0_sc_clearance(data: dict, rcfg, device: torch.device) -> tuple[float, int]:
    """t=0 OBB clearance to the nearest stopped neighbor + stopped count.

    Scores a static trajectory held at the ego-frame origin (the t=0 pose):
    every future step sits at the t=0 footprint, so the breakdown's
    sc_min_dist (min over t>=1) equals the t=0 clearance.
    """
    T = data["ego_agent_future"].shape[-2] if "ego_agent_future" in data else 80
    static_traj = torch.zeros(1, T, 4, device=device)
    static_traj[..., 2] = 1.0  # heading 0 -> (cos, sin) = (1, 0)
    r = compute_reward_batch(static_traj, data, rcfg)[0]
    return float(r.sc_min_dist), int(r.sc_n_stopped)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require_stopped", action="store_true", default=True,
                        help="Drop scenes with no stopped neighbor (default on)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rcfg = load_reward_config(args.config)
    if not rcfg.static_collision_enabled:
        raise SystemExit(
            "--config must have static_collision_enabled=true: without it the "
            "sc fields this filter reads are never computed."
        )

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    kept, dropped_t0, dropped_no_stopped, errors = [], [], [], []
    for p in scene_paths:
        try:
            data = load_npz_data(p, device)
            clearance, n_stopped = t0_sc_clearance(data, rcfg, device)
        except Exception as e:  # noqa: BLE001
            errors.append({"scene": p, "error": str(e)})
            print(f"  [err ] {Path(p).name}: {e}")
            continue
        if n_stopped == 0 and args.require_stopped:
            dropped_no_stopped.append({"scene": p})
            print(f"  [drop] {Path(p).name}: no stopped neighbor")
        elif clearance < rcfg.sc_cross_thresh:
            dropped_t0.append({"scene": p, "t0_clearance": clearance})
            print(f"  [drop] {Path(p).name}: t0 clearance {clearance:+.3f}m "
                  f"< {rcfg.sc_cross_thresh}m")
        else:
            kept.append(p)
            print(f"  [keep] {Path(p).name}: t0 clearance {clearance:+.3f}m, "
                  f"{n_stopped} stopped")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(kept, f, indent=1)
    with open(str(out) + ".report.json", "w") as f:
        json.dump({
            "n_input": len(scene_paths), "n_kept": len(kept),
            "n_dropped_t0": len(dropped_t0),
            "n_dropped_no_stopped": len(dropped_no_stopped),
            "n_errors": len(errors),
            "dropped_t0": dropped_t0,
            "dropped_no_stopped": dropped_no_stopped,
            "errors": errors,
        }, f, indent=1)
    print(f"\nKept {len(kept)}/{len(scene_paths)} "
          f"(dropped {len(dropped_t0)} t0-violating, "
          f"{len(dropped_no_stopped)} no-stopped, {len(errors)} errors)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
