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

### Route-based replay

Pick a start, goal, and zero or more waypoints in the GUI; save the result as
a reusable `Route` pickle; then replay it in closed-loop with the diffusion
planner. A background NPC manager spawns and despawns neighbors around the
ego as the simulation progresses.

**Save a Route in the GUI.** Launch `python -m scenario_generation.gui
--map_path <lanelet2_map.osm>`. The left sidebar has a **Mode** radio
(`Pan` / `Set Start` / `Set Goal` / `Add Waypoint`) that controls what a
plain drag on the map places. Pan is the default. `Add Waypoint` stays
active until you switch modes, so you can chain multiple waypoints.

| Interaction | How to trigger | Purpose | Visual |
|---|---|---|---|
| Pan | drag (Mode = Pan) | navigate the map | grab cursor |
| Zoom / rotate | scroll / Alt+drag | view controls | — |
| Rectangle select | Ctrl+drag | existing snippet workflow (works in any mode) | blue dashed box |
| **Ego start pose** | drag (Mode = Set Start) *or* Shift+drag | singular, overwrites | blue arrow |
| **Ego goal pose** | drag (Mode = Set Goal) | singular, overwrites | red arrow |
| **Waypoint** | drag (Mode = Add Waypoint) | appends to ordered list | yellow arrow, numbered |

Shift+drag stays as a power-user shortcut for the start pose so the existing
snippet workflow continues to work unchanged. Sidebar **Clear Start**,
**Clear Goal** and **Clear Waypoints** buttons reset each part independently.

The resolved route (`lanelet2.routing.RoutingGraph.shortestPathWithVia` under
the hood) draws as a green polyline on the canvas every time start, goal, or
waypoints change. Click **Save Route** to pickle the full spec (including
`route_lanelet_ids`) to disk.

Inspect a saved route:

```bash
python -m scenario_generation.tools.inspect_route my_route.pkl
```

**Run closed-loop replay.**

```bash
python -m scenario_generation.replay \
    --route my_route.pkl \
    --model_path /path/to/best_model.pth \
    --output_dir ./replay_out \
    [--map_path <override>] \
    [--steps 6000] \
    [--max_npcs 8] \
    [--spawn_probability 0.3] \
    [--config scenario_generation/configs/replay_default.json] \
    [--seed 42]
```

**Advance modes.** The `advance_mode` field in `SpawnConfig` (or the JSON
config) controls how the vehicle moves each step:

| Mode | Description | Per-agent cost |
|---|---|---|
| `teleport` (default) | Snap to `pred[0]` each step. Original behaviour — fast but can produce aggressive driving (lane invasion, red-light running) because there are no kinematic constraints. | ~0 ms |
| `perfect` | Euler integration with velocity from the reference trajectory and heading snap. Inspired by Autoware's `autoware_perfect_tracker`. Velocity limits how far the vehicle can move per step, preventing unphysical jumps. | ~0.01 ms |
| `mpc` | Bicycle-model MPC via scipy L-BFGS-B. Optimises acceleration and steering over a 2 s lookahead horizon (20 steps, 5 control knots). Enforces kinematic constraints (max accel, steering limits, speed bounds). | ~13 ms |

Both `perfect` and `mpc` modes apply C++-style post-processing to the
reference trajectory before tracking: velocity moving average (window=8) and
force-stop logic (ported from the C++ `postprocessing_utils.cpp`).

Example config enabling MPC:

```json
{"advance_mode": "mpc", "seed": 42, "max_steps": 6000, "mpc_horizon_steps": 20, "mpc_n_knots": 5}
```

Key files for trajectory tracking:

| File | Role |
|---|---|
| `mpc_tracker.py` | `MPCTracker` (bicycle MPC), `PerfectTracker` (Euler follower), `postprocess_reference` |
| `simulate.py` | `advance_scene` (teleport), `advance_scene_mpc` (MPC/perfect tracked advance) |

Per-step PNG `step_NNNN.png` is written to `output_dir`. The simulation ends
when one of:

- The ego reaches within `goal_tolerance_m` (default 2 m) of the goal —
  reason `goal_reached`.
- The ego *passes* the goal (vector ego→goal points behind ego) AND its
  closest-approach to the goal seen so far is within `goal_pass_window_m`
  (default 25 m) — reason `goal_passed`. Catches the common case where the
  diffusion planner doesn't perfectly stop at the goal but visibly drives
  past it.
- `max_steps` ticks have elapsed (default 6000 = 10 min of simulated data
  at `dt = 0.1 s`) — reason `max_steps`.

**NPC manager.** By default the manager holds the neighbor count in
`[0, max_active_npcs]` (hard cap = 8), spawning with 30% per-tick probability
only when the count is below the cap. Neighbors further than
`despawn_distance` (default 120 m) are dropped every tick. 70% of newly
spawned neighbors get a random forward route; 30% are biased to share at
least one lanelet with the ego's route. See
`scenario_generation/configs/replay_default.json` for the full knob list.

**Live map tensor refresh.** Every `map_refresh_steps` (default 5) ticks,
`scene.map_data.lanes` is rebuilt from the closest lanelets to the ego plus
the ego route + each alive NPC's current lanelet. This mirrors the
Diffusion-Planner ROS node's per-frame lane filter — the model never sees a
stale lane tensor as the ego moves across the map. Knobs:

- `map_mask_range_m` (default 200): half-side of the AABB around the ego; a
  lanelet passes when any of (center, first, last) centerline point is in
  the square. The ROS node uses 100 m, but that yields only ~22 lanelets on
  the Shinagawa map — below the training distribution (median 61). 200 m
  matches the training median.
- `max_map_lanelets` (default 140): hard cap, matches
  `tensor_converter._NUM_LANES`. Ego-closest fill first; pinned ids (ego
  route, history, NPC lanelets) fill the remaining slots.
- `map_refresh_steps` (default 5 = 0.5 s at `dt=0.1`): refresh period. A
  vectorised spatial query keeps the cost around 0.2 ms/call even on a
  6 000-lanelet map, so lowering this is cheap if you want a stricter match
  to the ROS node's per-frame refresh.

**Traffic lights.** A `TrafficLightController` manages signal group state
machines per step. Route-facing groups cycle through green/yellow/red phases;
perpendicular signals run in phase opposition. TL state is written into the
5-dim one-hot block at lane-dim indices `[8:13]` of `scene.map_data.lanes`
and into each agent's `route_lanes` tensor every step.

Relevant modules:

| Path | Role |
|---|---|
| `scenario_generation/route.py` | `Route` dataclass + pickle save/load |
| `scenario_generation/replay.py` | `run_route_replay`, `SceneNPCManager`, `SpawnConfig`, CLI |
| `scenario_generation/simulate.py` | `advance_scene`, `advance_scene_mpc`, model inference helpers |
| `scenario_generation/mpc_tracker.py` | `MPCTracker`, `PerfectTracker`, `postprocess_reference` |
| `scenario_generation/traffic_light.py` | `TrafficLightController` + signal group state machines |
| `scenario_generation/configs/replay_default.json` | Default `SpawnConfig` values |
| `scenario_generation/tools/inspect_route.py` | CLI to dump a saved route pickle |
| `scenario_generation/gui/app.py` | GUI panels + live route overlay wiring |
| `scene_search/map_canvas_js.py` | JS canvas: start / goal / waypoint arrows + route polyline |

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
