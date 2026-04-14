"""Batch scene generation and closed-loop simulation from saved lanelet selections.

The GUI saves lanelet selections (lanelet IDs + optional ego pose) as pickles.
This script loads them, generates N random scenes per selection, and optionally
runs closed-loop simulation with the Diffusion-Planner model.

Routes and history that extend beyond the saved lanelet set are retroactively
added to the map data (same behavior as the GUI).

Usage:
    python -m scenario_generation.batch_generate \
        --config scenario_generation/configs/example.json \
        --map_path /path/to/lanelet2_map.osm \
        --output_dir /path/to/output \
        [--model_path /path/to/best_model.pth]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.gui.scene_renderer import render_scene_figure
from scenario_generation.scene_context import SceneContext


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def load_selection(sel_path: str | Path) -> dict:
    with open(sel_path, "rb") as f:
        return pickle.load(f)


def generate_scenes(
    builder: LaneletSceneBuilder,
    lanelet_ids: list[int],
    gen_params: dict,
    n_scenes: int = 1,
    ego_pose: tuple[float, float, float] | None = None,
) -> list[SceneContext]:
    scenes = []
    for i in range(n_scenes):
        try:
            scene = builder.build_scene_context(
                lanelet_ids=lanelet_ids,
                n_neighbors=gen_params.get("n_neighbors", 5),
                min_separation_m=gen_params.get("min_separation", 8.0),
                min_speed=gen_params.get("min_speed", 3.0),
                max_speed=gen_params.get("max_speed", 12.0),
                route_length_m=gen_params.get("route_length", 120.0),
                ego_pose=ego_pose,
            )
            scenes.append(scene)
        except ValueError as e:
            print(f"  Scene {i}: generation failed: {e}")
    return scenes


def run_batch(config: dict, builder: LaneletSceneBuilder, output_dir: Path,
              model_path: str | None = None, device: str = "cuda"):
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_params = config.get("generation", {})
    sim_config = config.get("simulation", {})
    sim_enabled = sim_config.get("enabled", False) and model_path is not None
    sim_steps = sim_config.get("steps", 80)
    per_agent = sim_config.get("per_agent_views", False)

    model, model_args = None, None
    if sim_enabled:
        from scenario_generation.simulate import load_model
        print(f"Loading model from {model_path}...")
        model, model_args = load_model(model_path, device=device)

    selections = config.get("selections", [])
    total_scenes = sum(s.get("n_scenes", 1) for s in selections)
    print(f"Generating {total_scenes} scenes across {len(selections)} selections")

    scene_idx = 0
    for sel in selections:
        name = sel.get("name", f"selection_{scene_idx}")
        sel_path = sel["path"]
        n_scenes = sel.get("n_scenes", 1)
        sel_dir = output_dir / name
        sel_dir.mkdir(parents=True, exist_ok=True)

        sel_data = load_selection(sel_path)
        lanelet_ids = sel_data["lanelet_ids"]
        ego_pose = sel_data.get("ego_pose")
        if ego_pose is not None:
            ego_pose = tuple(ego_pose)

        print(f"\n--- {name} ({len(lanelet_ids)} lanelets, {n_scenes} scenes) ---")
        t0 = time.time()
        scenes = generate_scenes(builder, lanelet_ids, gen_params, n_scenes, ego_pose)
        print(f"  Generated {len(scenes)} scenes in {time.time() - t0:.1f}s")

        for i, scene in enumerate(scenes):
            scene_dir = sel_dir / f"scene_{i:03d}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            with open(scene_dir / "scene.pkl", "wb") as f:
                pickle.dump(scene, f)

            info = {
                "selection": name,
                "n_agents": len(scene.agents),
                "n_lanes": int(scene.map_data.lanes.shape[0]),
                "agents": [
                    {"id": a.id, "pos": a.current_position.tolist(),
                     "heading_deg": float(np.degrees(a.current_heading)),
                     "speed": float(np.linalg.norm(a.current_velocity))}
                    for a in scene.agents
                ],
            }
            with open(scene_dir / "info.json", "w") as f:
                json.dump(info, f, indent=2)

            fig = render_scene_figure(scene)
            fig.savefig(scene_dir / "initial.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

            if sim_enabled:
                from scenario_generation.simulate import run_simulation
                print(f"  Scene {i}: running {sim_steps}-step simulation...")
                t1 = time.time()
                run_simulation(
                    model, model_args, scene, sim_steps,
                    scene_dir / "simulation", device=device,
                    per_agent=per_agent,
                )
                print(f"  Scene {i}: simulation done in {time.time() - t1:.1f}s")

            scene_idx += 1

    print(f"\nBatch complete. {scene_idx} scenes saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch scene generation + simulation")
    parser.add_argument("--config", type=Path, required=True, help="Config JSON path")
    parser.add_argument("--map_path", type=str, required=True, help="Lanelet2 map .osm")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path for simulation (skip if not provided)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    print("Loading map...")
    builder = LaneletSceneBuilder(args.map_path)

    config = load_config(args.config)
    run_batch(config, builder, args.output_dir,
              model_path=args.model_path, device=device)


if __name__ == "__main__":
    main()
