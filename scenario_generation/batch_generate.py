"""Batch scene generation and closed-loop simulation from saved map snippets.

The GUI saves lanelet selections (map snippets) as pickles in a configurable
directory. This script discovers all snippets, generates N random scenes per
snippet, and runs both closed-loop and semi-closed-loop simulation.

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


def discover_snippets(snippets_dir: str | Path) -> list[Path]:
    d = Path(snippets_dir)
    if not d.exists():
        return []
    return sorted(d.glob("*.pkl"))


def generate_scenes(
    builder: LaneletSceneBuilder,
    lanelet_ids: list[int],
    gen_params: dict,
    n_scenes: int,
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


def _load_snippet(snip_path: Path) -> tuple[str, list[int], tuple[float, float, float] | None]:
    """Load a snippet pickle and return (name, lanelet_ids, ego_pose)."""
    with open(snip_path, "rb") as f:
        snip_data = pickle.load(f)
    ego_pose = snip_data.get("ego_pose")
    if ego_pose is not None:
        ego_pose = tuple(ego_pose)
    return snip_path.stem, snip_data["lanelet_ids"], ego_pose


def _save_scene(scene: SceneContext, scene_dir: Path, snippet_name: str) -> None:
    """Save scene pickle, info JSON, and initial visualization."""
    scene_dir.mkdir(parents=True, exist_ok=True)

    with open(scene_dir / "scene.pkl", "wb") as f:
        pickle.dump(scene, f)

    info = {
        "snippet": snippet_name,
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


def run_batch(config: dict, builder: LaneletSceneBuilder, output_dir: Path,
              model_path: str | None = None, device: str = "cuda"):
    from concurrent.futures import ThreadPoolExecutor

    output_dir.mkdir(parents=True, exist_ok=True)

    gen_params = config.get("generation", {})
    n_scenes_per = gen_params.get("n_scenes_per_snippet", 3)
    sim_config = config.get("simulation", {})
    sim_enabled = sim_config.get("enabled", False) and model_path is not None
    sim_steps = sim_config.get("steps", 80)
    per_agent = sim_config.get("per_agent_views", False)
    sim_modes = sim_config.get("modes", ["closed_loop", "semi_closed_loop"])

    snippets_dir = config.get("snippets_dir", ".map_snippets")
    snippet_files = discover_snippets(snippets_dir)
    if not snippet_files:
        raise ValueError(f"No .pkl snippets found in {snippets_dir}")

    model, model_args = None, None
    if sim_enabled:
        from scenario_generation.simulate import load_model
        print(f"Loading model from {model_path}...")
        model, model_args = load_model(model_path, device=device)

    print(f"Found {len(snippet_files)} snippets, generating {n_scenes_per} scenes each")
    if sim_enabled:
        print(f"Simulation modes: {sim_modes}, {sim_steps} steps each")

    scene_idx = 0

    # Pipeline: generate next snippet's scenes on CPU while simulating current on GPU
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene_gen") as gen_pool:
        pending_future = None

        def _gen_for_snippet(snip_path):
            name, lanelet_ids, ego_pose = _load_snippet(snip_path)
            t0 = time.time()
            scenes = generate_scenes(builder, lanelet_ids, gen_params, n_scenes_per, ego_pose)
            elapsed = time.time() - t0
            return name, lanelet_ids, scenes, elapsed

        def _process_snippet(name, scenes, snip_dir):
            nonlocal scene_idx
            for i, scene in enumerate(scenes):
                scene_dir = snip_dir / f"scene_{i:03d}"
                _save_scene(scene, scene_dir, name)

                if sim_enabled:
                    from scenario_generation.simulate import run_simulation
                    for sim_mode in sim_modes:
                        sim_out = scene_dir / sim_mode
                        print(f"  Scene {i}: {sim_mode} ({sim_steps} steps)...")
                        t1 = time.time()
                        run_simulation(
                            model, model_args, scene, sim_steps,
                            sim_out, device=device,
                            per_agent=per_agent, mode=sim_mode,
                        )
                        print(f"  Scene {i}: {sim_mode} done in {time.time() - t1:.1f}s")

                scene_idx += 1

        # Submit first snippet generation
        if snippet_files:
            pending_future = gen_pool.submit(_gen_for_snippet, snippet_files[0])

        for next_idx in range(1, len(snippet_files) + 1):
            # Wait for current generation to finish
            name, lanelet_ids, scenes, elapsed = pending_future.result()
            snip_dir = output_dir / name
            snip_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--- {name} ({len(lanelet_ids)} lanelets, {n_scenes_per} scenes) ---")
            print(f"  Generated {len(scenes)} scenes in {elapsed:.1f}s")

            # Submit next snippet generation (overlaps with current simulation)
            if next_idx < len(snippet_files):
                pending_future = gen_pool.submit(_gen_for_snippet, snippet_files[next_idx])

            # Process current snippet (save + simulate) on main thread
            _process_snippet(name, scenes, snip_dir)

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
