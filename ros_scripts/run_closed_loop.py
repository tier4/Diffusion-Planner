#!/usr/bin/env python3
"""Closed-Loop (Perception Reproducer) one-shot runner.

Stage1 (extract_scene.build_scene_pkl) -> Stage2 (perception_reproducer.run_reproducer) run in
the same process via import. Stage1 is skipped if scene.pkl already exists.

Prepare the env beforehand, same as the other ros_scripts:
    cd <meta-repo> && uv sync                       # first time only (.venv is created at python3.10)
    source <meta-repo>/.venv/bin/activate
    source Diffusion-Planner/cpp_tools/install/setup.bash

Usage:
    python3 ros_scripts/run_closed_loop.py --bag <BAG_DIR> --model_dir <MODEL_DIR>
    python3 ros_scripts/run_closed_loop.py --bag <BAG_DIR> --model_dir <MODEL_DIR> --num_steps 50

Do not embed site-specific paths here (this file is public). To reuse default paths, pass
--bag / --model_dir from a private launcher (e.g. meta-repo local/run_closed_loop.sh).
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Make the sibling ros_scripts modules importable (same approach as parse_rosbag, etc.).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from extract_scene import build_scene_pkl  # noqa: E402
from perception_reproducer import (  # noqa: E402
    DEFAULT_EGO_LENGTH,
    DEFAULT_EGO_WIDTH,
    DEFAULT_MAX_STUCK_STEPS,
    DEFAULT_OFFROUTE_THRESHOLD,
    DEFAULT_TRAJ_STEP,
    DEFAULT_WHEEL_BASE,
    run_reproducer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bag", type=Path, required=True, help="rosbag directory")
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="directory containing args.json + best_model.pth",
    )
    parser.add_argument("--num_steps", type=int, default=None, help="default: all frames")
    parser.add_argument(
        "--traj_step",
        type=int,
        default=DEFAULT_TRAJ_STEP,
        help="advance to the n-th predicted waypoint per iteration (1 = +0.1 s)",
    )
    parser.add_argument(
        "--max_stuck_steps",
        type=int,
        default=DEFAULT_MAX_STUCK_STEPS,
        help="end after this many consecutive no-progress steps (0 disables)",
    )
    parser.add_argument(
        "--scene", type=Path, default=None, help="default: ~/data/closed_loop/scene_<tag>.pkl"
    )
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=None,
        help="default: <model_dir>/closed_loop/<tag>_<timestamp> (new dir per run)",
    )
    parser.add_argument("--no_video", action="store_true")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    bag = args.bag.resolve()

    tag = f"{bag.parent.name}_{bag.name}"  # e.g. 2026-01-15_13-42-45
    if args.scene is not None:
        scene = args.scene
    else:
        scene = Path.home() / "data" / "closed_loop" / f"scene_{tag}.pkl"
    if args.result_dir is not None:
        result_dir = args.result_dir
    else:
        # Results live with the model (closed-loop evaluation of that checkpoint).
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = args.model_dir.resolve() / "closed_loop" / f"{tag}_{stamp}"

    print(f"BAG       : {bag}")
    print(f"MODEL_DIR : {args.model_dir}")
    print(f"SCENE     : {scene}")
    print(f"RESULT    : {result_dir}")

    # --- Stage 1: scene.pkl (skip if it already exists) ---
    if scene.is_file():
        print(f"=== Stage1 skip (scene exists): {scene} ===")
    else:
        print("=== Stage1: extract_scene ===")
        build_scene_pkl(bag, scene, None, -1)

    # --- Stage 2: closed-loop ---
    print("=== Stage2: perception_reproducer ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_reproducer(
        args.model_dir,
        None,
        scene,
        result_dir,
        args.num_steps,
        args.traj_step,
        device,
        not args.no_video,
        DEFAULT_WHEEL_BASE,
        DEFAULT_EGO_LENGTH,
        DEFAULT_EGO_WIDTH,
        DEFAULT_OFFROUTE_THRESHOLD,
        args.max_stuck_steps,
    )

    print(f"=== done. result: {result_dir} ===")


if __name__ == "__main__":
    main()
