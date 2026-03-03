# RLVR Integration — Implementation Status

This document describes the current state of the RLVR (Reinforcement Learning with Verifiable Rewards)
integration with TeraSim for the Diffusion Planner project.
It is intended as a handoff document for a new model/session picking up this work.

---

## Repository Overview

| Path | Purpose |
|---|---|
| `diffusion_planner/` | Core PyTorch ML package (Encoder + DiT Decoder) |
| `preference_optimization/` | DPO pipeline — Direct Preference Optimization |
| `rlvr/` | **RLVR integration — all new code lives here** |
| `/home/danielsanchez/TeraSim/` | TeraSim simulator (tier4 fork, NOT inside Diffusion-Planner) |

---

## What Has Been Implemented (Phase 1: Ghost Replay Validator)

### Goal of Phase 1
Establish that the training data coordinate system (`.npz` files from rosbags) and the TeraSim
simulation coordinate system are perfectly aligned. This is done via a "ghost replay": the ego vehicle
is driven along its recorded ground-truth trajectory while NDE background traffic moves naturally,
and we verify no coordinate errors accumulate.

---

### 1. Coordinate Utilities — `rlvr/npz_utils.py`

Every training sample is a pair of files:
- `<stem>.npz` — all data in ego base_link frame at t=0
- `<stem>.json` — ego world-frame pose at t=0 in MGRS local Cartesian coordinates

Key functions:
- `load_bl2map(json_path)` → 4×4 transform matrix (base_link → MGRS map frame)
- `ego_centric_to_map(xy_bl, bl2map)` → converts (N,2) ego-centric positions to map frame
- `heading_bl_to_map(cos_h, sin_h, bl2map)` → converts heading to map-frame yaw
- `ros_yaw_to_sumo_angle(yaw_rad)` → converts ROS yaw (CCW from +X, radians) to SUMO angle (CW from +Y/North, degrees)
- `extract_spawn_states(npz_path, json_path)` → returns dict with:
  - `ego`: {x, y, yaw_rad, sumo_angle, vx, length, width} at t=0 in map frame
  - `npcs`: list of NPC states at t=0 in map frame
  - `ego_future_map`: (80, 3) array of [x, y, yaw_rad] in map frame — the GT trajectory

**Verified:** ego x/y from `extract_spawn_states` exactly matches the `.json` sidecar values.

---

### 2. SUMO Road Network — `rlvr/sim_config/maps/shinagawa_odaiba.net.xml`

The Shinagawa-Odaiba Lanelet2 map is converted to a SUMO `.net.xml` network file.

**Conversion script:** `rlvr/scripts/convert_lanelet2_to_sumo.py`

Key design decisions:
- Uses `lanelet2` with `MGRSProjector` from the Autoware installation to load the map in MGRS local
  Cartesian coordinates — the same coordinate system as the `.json` sidecars.
- All junctions use `type="unregulated"` — this avoids SUMO's internal crossing-lane computation
  that would otherwise require running `netconvert`, which destroys curved lane geometry.
- `projParameter="!"` in the net.xml location element tells SUMO to use coordinates as-is (no reprojection).
- **Do NOT run `netconvert` post-processing** — it collapses multi-point lane shapes to just start/end,
  making all roads appear as straight sticks.

**Dependencies (not in `.venv`, accessed via sys.path):**
```python
sys.path.insert(0, '/opt/ros/humble/lib/python3.10/site-packages')          # lanelet2
sys.path.insert(0, '/home/danielsanchez/autoware/install/autoware_lanelet2_extension_python/local/lib/python3.10/dist-packages')  # MGRSProjector
```

Run (once, or when the Autoware map changes):
```bash
source .venv/bin/activate
python3 rlvr/scripts/convert_lanelet2_to_sumo.py \
  --osm /home/danielsanchez/autoware_map/shinagawa_odaiba_stable/lanelet2_map.osm \
  --output rlvr/sim_config/maps/shinagawa_odaiba.net.xml
```

---

### 3. TeraSim Docker — `terasim:latest` image

TeraSim runs entirely inside Docker. The image is built from the **tier4 fork**:
```
/home/danielsanchez/TeraSim/    (git@github.com:tier4/TeraSim.git)
```

Build command:
```bash
cd /home/danielsanchez/TeraSim && docker build -t terasim:latest .
```

**Bugs fixed in the tier4 fork** (all committed to `/home/danielsanchez/TeraSim` main branch):

| File | Bug | Fix |
|---|---|---|
| `Dockerfile` | Missing `libxrender1`, `libgl1-mesa-glx` in runtime stage → `sumo` binary fails to start | Added both packages |
| `cosim.py` | `moveToXY(..., keepRoute=2)` places AV on internal junction lanes (`''`) after a few steps, causing TraCIException cascade | Changed to `keepRoute=0` |
| `nade_with_av.py` | `NDE_decision`: AV appears in its own context subscription results, causing KeyError cascade | Filter AV out of `terasim_controlled_vehicle_ids` |
| `nade_with_av.py` | `NDE_decision`: AV not in `vehicle_list` so no observation is populated for it, but NADE needs its position/velocity | Inject AV observation from TraCI directly |
| `nade_with_av.py` | `NADE_decision`/`NADE_decision_and_control`: uses `dict[AV_ID]` (KeyError) instead of `.get()` after `executeMove()` converts addict.Dict to plain dict | Use `.get()` |
| `nade_with_av.py` | `predict_av_control_command`: crashes when AV is on internal lane `''` | Return `None` early for internal/empty lane IDs |
| `dash_viz_app.py` | Full rewrite — original sent 3.5 MB map JSON through browser on every 200 ms callback (CPU: 106%), used per-lane traces causing 14k+ Plotly objects, zoom reset on every update | Server-side map cache, `Patch()` updates, fixed-size arrow marker, 20 m initial zoom, GT trajectory overlay |

---

### 4. HTTP Bridge — `rlvr/terasim_bridge.py`

Python client to the TeraSim FastAPI REST service (port 8000). Does not import any TeraSim Python package.

```python
class TeraSimBridge:
    def __init__(
        self,
        sim_config_host_dir: str,   # path to rlvr/sim_config/ on host
        gui: bool = False,           # launch sumo-gui via X11 (needs xhost +local:docker)
        fcd_host_dir: str | None,    # bind-mount for FCD output (use /home/..., not /tmp/)
        ...
    )
    def start_episode(self, spawn_states: dict, enable_viz: bool = False) -> None
    def step(self, ego_xy: tuple, ego_yaw_rad: float, ego_speed: float = 0.0) -> dict
    def close(self) -> None
    @property fcd_output_path -> str | None
```

Key implementation notes:
- The Docker container is **auto-started** on the first `start_episode()` call if not already running.
- **`--protected-mode no`** is passed to Redis so the host Python can write the GT trajectory key.
- **`health_check_interval=0`** on the redis client avoids RESP3 handshake incompatibility (redis-py 7.x vs Redis server 6.x).
- Step detection uses `simulation_time > prev_time` polling (not status strings) to avoid race conditions.
- The container is started with port 8050 exposed for the Dash visualizer.

**YAML config files** (`rlvr/sim_config/`):

| File | Purpose |
|---|---|
| `ghost_replay.yaml` | Default — headless, fcd_all output enabled |
| `ghost_replay_gui.yaml` | With `gui_flag: true` for sumo-gui mode |
| `sim.sumocfg` | SUMO config: 0.1 s step, warn-on-collision |
| `background_traffic.rou.xml` | Vehicle type definitions (`car`, `veh_AV`) |
| `maps/shinagawa_odaiba.net.xml` | Generated SUMO network (3148 edges, 3084 junctions) |
| `maps/metadata.json` | `{"av_route_sumo": []}` — avoids TeraSim log noise |

**Known quirk in `ghost_replay.yaml`:**
- `warmup_time_ub: 1` (not `0`) — `rng.integers(0, 0)` raises `ValueError`. Setting to 1 gives `rng.integers(0, 1) = 0` (zero warmup).
- `AV_cfg.route: ["ll_190615"]` — required by the tier4 NADEWithAV; must be a valid edge ID from the net.xml.

---

### 5. Validation Script — `rlvr/scripts/validate_ghost_replay.py`

Phase 1 acceptance test. Drives the AV along `ego_future_map` (80 steps × 0.1 s = 8 s) and asserts:
1. AV stays in the simulation at every step (no collision removal)
2. Final position error < 2 m (coordinate alignment sanity check)

**Result:** PASSED, final error = 0.000 m ✓

```bash
source .venv/bin/activate
python3 rlvr/scripts/validate_ghost_replay.py \
  --npz_path <path>.npz \
  [--gui]                    # sumo-gui window on desktop (needs xhost +local:docker)
  [--viz]                    # Dash map at http://localhost:8050
  [--fcd /home/.../fcd_dir]  # Write FCD output (use /home/..., not /tmp/)
  [--step_delay 0.1]         # Real-time replay speed
```

---

### 6. FCD Offline Replay — `rlvr/scripts/replay_fcd.py`

Visualises a recorded FCD trajectory offline using `SumoNetVis` + matplotlib.

```bash
pip install SumoNetVis
python3 rlvr/scripts/replay_fcd.py --fcd_dir /home/.../fcd_dir
# or
python3 rlvr/scripts/replay_fcd.py --fcd_file /path/to/fcd_all.xml
# Save to MP4:
python3 rlvr/scripts/replay_fcd.py --fcd_dir ... --save replay.mp4
```

---

### 7. Launcher GUI — `rlvr/scripts/launch_gui.py`

Gradio interface for browsing the NPZ sample list and launching ghost replay simulations.
Designed to be extended later with model inference / GRPO panels.

```bash
source .venv/bin/activate
python3 rlvr/scripts/launch_gui.py \
  --npz_list /media/danielsanchez/.../path_list.json
# Opens at http://localhost:7861
```

- Left panel: index navigation (Prev/Next, jump to index), shows current NPZ path + ego info
- Right panel: visualization options (sumo-gui, Dash viewer, FCD recording), step delay slider
- Launch button: streams step-by-step simulation log in real-time via Gradio generator
- **Port 7861** (7860 is reserved for the DPO GUI)

**Note:** sumo-gui and the Dash viewer are independent options:
- sumo-gui → native window on your desktop (requires `xhost +local:docker`)
- Dash viewer → browser at **http://localhost:8050** (open in a second tab after Launch)

---

## What Remains to Be Done (Phase 2+)

### Phase 2: Closed-Loop Model Inference

The ghost replay confirms spatial alignment. Phase 2 runs the **Diffusion Planner model** in the loop
instead of replaying the GT trajectory.

**What needs to be added:**

1. **`rlvr/model_runner.py`** — wrapper around `preference_optimization/model_utils.py::load_model()`
   and the inference pattern from `preference_optimization/utils.py::generate_trajectory()`.
   - Input: `spawn_states` dict + previous step result
   - Output: next ego pose (x, y, yaw_rad) from model prediction

2. **`rlvr/scripts/run_closed_loop.py`** — like `validate_ghost_replay.py` but instead of stepping
   along `ego_future_map`, it calls the model at each step to get the next action.

3. **Reward functions** (verifiable rewards for GRPO):
   - Progress along route
   - Collision penalty (AV disappearing from vehicle list)
   - Comfort (jerk, acceleration limits)
   - Lane keeping

4. **GRPO training loop** — uses the closed-loop runner as the environment, collects rollouts,
   computes GRPO loss, updates model weights.

5. **GUI extension** — add a model panel to `launch_gui.py`:
   - Model checkpoint selector
   - Toggle: GT replay vs model inference
   - Live trajectory display (what the model predicts vs GT)

### Phase 3: Scale and Robustness

- Multiple episodes in parallel (multiple Docker containers)
- Curriculum: start with easy samples (low speed, straight roads), progress to harder scenarios
- Episode termination conditions beyond collision (route completion, time limit)
- NPC adversarial scenarios (NADE conflict generation)

---

## Key File Paths (absolute, on this machine)

| Item | Path |
|---|---|
| RLVR package | `/home/danielsanchez/Diffusion-Planner/rlvr/` |
| TeraSim repo | `/home/danielsanchez/TeraSim/` (tier4 fork, separate git repo) |
| Autoware map | `/home/danielsanchez/autoware_map/shinagawa_odaiba_stable/lanelet2_map.osm` |
| Sample NPZ list | `/media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207/path_list_manual_tokyo_teleport_exit.json` (1731 samples) |
| Python venv | `/home/danielsanchez/Diffusion-Planner/.venv` (Python 3.10) |
| lanelet2 | `/opt/ros/humble/lib/python3.10/site-packages/lanelet2/` |
| autoware_lanelet2_extension_python | `/home/danielsanchez/autoware/install/autoware_lanelet2_extension_python/local/lib/python3.10/dist-packages/` |

## Quick Start (from scratch)

```bash
# 1. Build TeraSim Docker image (one-time, ~5 min)
cd /home/danielsanchez/TeraSim && docker build -t terasim:latest .

# 2. Generate SUMO network (one-time, ~30 s)
cd /home/danielsanchez/Diffusion-Planner
source .venv/bin/activate
python3 rlvr/scripts/convert_lanelet2_to_sumo.py \
  --osm /home/danielsanchez/autoware_map/shinagawa_odaiba_stable/lanelet2_map.osm \
  --output rlvr/sim_config/maps/shinagawa_odaiba.net.xml

# 3. Run ghost replay (headless, fastest)
python3 rlvr/scripts/validate_ghost_replay.py --npz_path <path>.npz

# 4. Run with Dash visualization
python3 rlvr/scripts/validate_ghost_replay.py --npz_path <path>.npz --viz --step_delay 0.1
# Then open http://localhost:8050 in a second browser tab

# 5. Launch GUI browser
python3 rlvr/scripts/launch_gui.py \
  --npz_list /media/danielsanchez/.../path_list_manual_tokyo_teleport_exit.json
# Open http://localhost:7861
```
