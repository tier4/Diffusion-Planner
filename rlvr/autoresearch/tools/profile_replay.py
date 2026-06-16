#!/usr/bin/env python3
"""cProfile wrapper around ``scenario_generation.replay.run_route_replay``.

Answers questions like "where is the MPC replay spending its wall clock?".
Covers the full step loop (model inference, MPC solve per agent, map
rebuild, PNG render submission, NPZ dump, live metric scoring). Run on a
short horizon so the dump + profile stays manageable.

Usage:
    python -m rlvr.autoresearch.tools.profile_replay \\
        --route /path/to/route.pkl \\
        --model_path /path/to/model.pth \\
        --config /path/to/spawn_config.json \\
        --output_dir /path/to/profile_out/ \\
        --steps 200 \\
        [--top 40]
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Override spawn_config.max_steps for the profile run.",
    )
    parser.add_argument(
        "--top", type=int, default=40, help="How many top consumers to print per sort order."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Lazy imports — keep argparse errors fast.
    import torch

    from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
    from scenario_generation.replay import SpawnConfig, run_route_replay
    from scenario_generation.route import Route
    from scenario_generation.simulate import load_model

    print(f"Loading route from {args.route}")
    route = Route.load(args.route)

    print(f"Loading lanelet2 builder from {route.map_path}")
    builder = LaneletSceneBuilder(route.map_path)

    print(f"Loading model {args.model_path}")
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model, model_args = load_model(str(args.model_path), device)

    cfg = SpawnConfig.from_json(args.config)
    cfg.max_steps = args.steps
    cfg.seed = args.seed
    cfg.validate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / "replay.prof"

    print(f"Profiling replay for {args.steps} steps → {profile_path}")
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run_route_replay(
            model=model,
            model_args=model_args,
            builder=builder,
            route=route,
            output_dir=args.output_dir,
            spawn_config=cfg,
            device=device,
        )
    finally:
        profiler.disable()

    profiler.dump_stats(str(profile_path))

    stats = pstats.Stats(str(profile_path))
    for sort_key, label in (
        ("cumulative", "by cumulative time"),
        ("tottime", "by self time"),
    ):
        print(f"\n─── Top {args.top} {label} ───")
        stats.sort_stats(sort_key)
        stats.print_stats(args.top)

    print(f"\nFull profile: {profile_path}")
    print(f"Inspect interactively with: python -m pstats {profile_path}")


if __name__ == "__main__":
    main()
