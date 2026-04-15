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
    --per_agent \
    --mode closed_loop
```

Flags:
- `--use_gt_goals` -- Set neighbor goals and routes from GT future trajectories. Removes agents with <10 valid GT timesteps (misdetections).
- `--per_agent` -- Save per-agent zoomed images in `<output_dir>/<agent_id>/`.
- `--mode` -- `closed_loop` (all agents re-planned each step) or `semi_closed_loop` (ego follows its initial trajectory, only neighbors re-planned).

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

### Generate synthetic scenes (GUI)

Interactive GUI for generating driving scenes from a Lanelet2 map. Select a map region, choose the number of neighbors, and generate agents with feasible routes, realistic history, and collision-free placement.

```bash
source /opt/ros/humble/setup.bash
source ~/autoware/install/setup.bash
source .venv/bin/activate
python -m scenario_generation.gui --map_path /path/to/lanelet2_map.osm [--port 7862]
```

The GUI provides:
- Pan/zoom map canvas with Ctrl+drag rectangle selection for the scene area
- Configurable number of neighbors, speed range, separation distance, and route length
- Focus mode dropdown to inspect individual agents (full detail for the selected agent, minimal for others)
- Heading arrows, footprint history, route highlighting, and goal markers for all agents
- Alt+drag to rotate the map view, Shift+drag to set ego pose and heading
- Zoom slider for the scene preview, rotation carried over to the rendered view
- Save lanelet selections as map snippets (.map_snippets/) for batch processing
- Export generated SceneContext as pickle for downstream use

Key modules in `gui/`:
- `lanelet_scene_builder.py` -- Loads Lanelet2 map, builds routing graph, generates agents with OBB collision-free placement, backward centerline history tracing, and route finding via the lanelet2 routing API. Accepts either a rectangle or pre-saved lanelet IDs.
- `scene_renderer.py` -- Matplotlib rendering with all-agents and focus modes, with rotation support.
- `app.py` -- Gradio web interface with shared interactive map canvas (also used by scene_search).

### Batch scene generation

Generate N scenes per saved map snippet and run closed-loop + semi-closed-loop simulation:

```bash
python -m scenario_generation.batch_generate \
    --config scenario_generation/configs/example.json \
    --map_path /path/to/lanelet2_map.osm \
    --output_dir /path/to/output \
    --model_path /path/to/best_model.pth
```

The config JSON (`configs/example.json`) controls:
- `snippets_dir` -- directory containing .pkl map snippets saved from the GUI (default `.map_snippets/`)
- `generation` -- n_neighbors, speed range, separation, route length, n_scenes_per_snippet
- `simulation` -- steps, modes (`closed_loop`, `semi_closed_loop`), per_agent_views

Output structure: `<output_dir>/<snippet_name>/scene_NNN/{initial.png, scene.pkl, closed_loop/, semi_closed_loop/}`

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
