# scenario_generation

Structured scene representation for Diffusion-Planner with multi-agent closed-loop simulation.

## Overview

This module provides `SceneContext`, an intermediate format that stores all scene data (agents, map, routes) in a world coordinate frame. Any agent can be promoted to ego during tensor conversion, enabling the Diffusion-Planner model to generate trajectories for all vehicles in a scene.

### Key components

- **SceneContext / Agent / MapData** (`scene_context.py`) -- Core dataclasses. Agents store trajectory history, shape, velocities, and optional extensible fields (goal, route, turn indicators). Heading stored as radians for readability.
- **NPZ loader** (`npz_loader.py`) -- Converts NPZ training files to SceneContext. Derives neighbor wheelbase from length, converts cos/sin headings to radians, corrects 180-degree heading errors using velocity direction.
- **Tensor converter** (`tensor_converter.py`) -- Transforms SceneContext to normalized model input tensors for any chosen ego agent. Applies ego-centric coordinate transform and full observation normalization.
- **GT route extractor** (`gt_route_extractor.py`) -- Assigns goal poses and route lanelets to agents from their ground truth future trajectories. Trims zero-padded GT data and removes short-lived agents (misdetections).
- **Visualization** (`visualize.py`) -- Renders scenes with lanes, road borders, stop lines, agent bounding boxes, trajectories, goals, and routes.
- **Simulation** (`simulate.py`) -- Closed-loop simulation: at each timestep, every vehicle agent gets its own model forward pass as ego, then all agents advance by one step.

## Usage

### Visualize a scene from NPZ

```bash
python -m scenario_generation.visualize /path/to/scene.npz
python -m scenario_generation.visualize /path/to/*.npz --cols 3 -o grid.png
python -m scenario_generation.visualize scene.npz --ego neighbor_0
```

### Run closed-loop simulation

```bash
python -m scenario_generation.simulate \
    --model_path /path/to/best_model.pth \
    --npz /path/to/scene.npz \
    --output_dir /path/to/output \
    --steps 80 \
    --use_gt_goals \
    --per_agent
```

Flags:
- `--use_gt_goals` -- Set neighbor goals and routes from GT future trajectories. Removes agents with <10 valid GT timesteps (misdetections).
- `--per_agent` -- Save per-agent zoomed images in `<output_dir>/<agent_id>/`.

### Python API

```python
from scenario_generation import from_npz, to_model_tensors, assign_gt_goals_and_routes

scene = from_npz("scene.npz")
assign_gt_goals_and_routes(scene)

# Extend a neighbor
scene.get_agent("neighbor_0").goal_pose = np.array([100.0, 50.0, 0.5])

# Get model input for any agent as ego
tensors = to_model_tensors(scene, ego_agent_id="neighbor_0",
                           model_args=model_args, device="cuda")
_, outputs = model(tensors)
```

### Run tests

```bash
python -m pytest scenario_generation/tests/ -v
```

## Design notes

- **World frame**: All coordinates in a scene-level frame (ego's frame when loaded from NPZ). Re-centering to any agent happens during tensor conversion.
- **Lane boundary convention**: Indices [4:6] and [6:8] in the 33-dim lane format are offsets from centerline, not absolute positions. They get rotation-only transforms (no translation), matching `state_update.py`.
- **Heading correction**: On NPZ load, neighbor headings are checked against velocity direction. If off by >90 degrees, all headings are flipped by pi.
- **Static agents**: Vehicles with speed <0.5 m/s and goal distance <1.0m are kept in the scene but not simulated (no model forward pass, no position update).
- **Misdetection filtering**: When using `--use_gt_goals`, agents with fewer than 10 valid GT future timesteps are removed from the scene.
