#!/usr/bin/env python3
"""
RLVR Phase 1 acceptance test: ghost replay validator.

Loads a single .npz/.json pair, runs the ego along the ground-truth future
trajectory while NDE background traffic moves naturally in TeraSim, and
asserts that:
  1. No collision at any of the 80 steps (AV stays in the simulation).
  2. The AV's final simulated position is within 2 m of the GT final position
     (verifies MGRS coordinate alignment between the data pipeline and SUMO).

Usage:
    source .venv/bin/activate
    python3 rlvr/scripts/validate_ghost_replay.py \\
        --npz_path <path>.npz \\
        --json_path <path>.json
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np

from rlvr.npz_utils import extract_spawn_states
from rlvr.terasim_bridge import TeraSimBridge

_REPO_ROOT = Path(__file__).parents[2]
_SIM_CONFIG_DIR = _REPO_ROOT / "rlvr" / "sim_config"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ghost replay validator for RLVR Phase 1."
    )
    parser.add_argument("--npz_path", required=True, help="Path to .npz data file")
    parser.add_argument(
        "--json_path",
        default=None,
        help="Path to companion .json sidecar (defaults to <npz_path>.json)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable TeraSim Dash visualizer on http://localhost:8050",
    )
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.0,
        help="Sleep this many seconds between each simulation step (default: 0 = as fast as possible). "
             "Use 0.1 to replay at real-time speed, or higher to slow down for visualization.",
    )
    args = parser.parse_args()

    npz_path = args.npz_path
    json_path = args.json_path or npz_path.replace(".npz", ".json")

    print(f"NPZ:  {npz_path}")
    print(f"JSON: {json_path}")

    # -----------------------------------------------------------------------
    # Extract spawn states
    # -----------------------------------------------------------------------
    print("Extracting spawn states…")
    spawn = extract_spawn_states(npz_path, json_path)
    print(
        f"  Ego t=0: x={spawn['ego']['x']:.2f}  y={spawn['ego']['y']:.2f}"
        f"  yaw={math.degrees(spawn['ego']['yaw_rad']):.1f}°"
        f"  speed={spawn['ego']['vx']:.2f} m/s"
    )
    print(f"  Active NPCs: {len(spawn['npcs'])}")
    print(f"  GT future steps: {spawn['ego_future_map'].shape[0]}")

    # -----------------------------------------------------------------------
    # Run ghost replay
    # -----------------------------------------------------------------------
    if args.viz:
        print("Visualization enabled → http://localhost:8050")

    with TeraSimBridge(sim_config_host_dir=str(_SIM_CONFIG_DIR)) as sim:
        print("Starting simulation episode…")
        sim.start_episode(spawn, enable_viz=args.viz)
        print("  Episode started.")
        if args.viz:
            print("  >>> Open http://localhost:8050 in your browser now <<<")
            print("  Waiting 3 s for the Dash app to initialise…")
            time.sleep(3)

        for step_idx in range(len(spawn["ego_future_map"])):
            x, y, yaw_rad = spawn["ego_future_map"][step_idx]
            result = sim.step((float(x), float(y)), float(yaw_rad))

            if args.step_delay > 0:
                time.sleep(args.step_delay)

            if not result["av_in_sim"]:
                raise AssertionError(
                    f"AV removed from simulation at step {step_idx} "
                    f"(t={result['sim_time']:.1f}s) — collision or out-of-bounds.\n"
                    f"  AV position target: ({x:.2f}, {y:.2f})"
                )

            if (step_idx + 1) % 10 == 0:
                print(
                    f"  step {step_idx + 1:3d}/80  "
                    f"t={result['sim_time']:.1f}s  "
                    f"NPCs={len(result['npc_states'])}"
                )

        # -------------------------------------------------------------------
        # Final position check
        # -------------------------------------------------------------------
        final_state = sim._last_state
        av_state = final_state["agent_details"]["vehicle"]["AV"]
        av_x, av_y = av_state["x"], av_state["y"]
        gt_x, gt_y = float(spawn["ego_future_map"][-1, 0]), float(
            spawn["ego_future_map"][-1, 1]
        )
        dist = math.sqrt((av_x - gt_x) ** 2 + (av_y - gt_y) ** 2)
        print(
            f"\nFinal position:"
            f"  simulated=({av_x:.2f}, {av_y:.2f})"
            f"  GT=({gt_x:.2f}, {gt_y:.2f})"
            f"  error={dist:.3f}m"
        )
        assert dist < 2.0, (
            f"Final position error too large: {dist:.3f}m > 2.0m threshold.\n"
            f"  AV at ({av_x:.2f}, {av_y:.2f}), GT at ({gt_x:.2f}, {gt_y:.2f})\n"
            f"  This indicates a coordinate alignment problem."
        )

    print("\nGhost replay validation PASSED")


if __name__ == "__main__":
    main()
