# RLVR Integration Design: TeraSim + Diffusion Planner

## 1. Goal and Scope

The long-term goal is to apply reinforcement learning with verifiable rewards (RLVR) using GRPO (or similar on-policy RL) to fine-tune the Diffusion Planner model using a closed-loop driving simulator as the reward source.

This document covers **Phase 1 only**: establishing a reliable, deterministic 1:1 spatial sync between the training data (`.npz` files) and TeraSim, and validating it via a "ghost replay" where the ego is forced along the ground-truth human trajectory while NDE background traffic moves naturally, confirming the simulation is spatially aligned and collision detection works.

GRPO training and reward shaping are explicitly out of scope here. The deliverable for Phase 1 is a working `TeraSimBridge` class and a ghost replay validator script.

---

## 2. Repository Context

**Root**: `/home/danielsanchez/Diffusion-Planner`

Relevant packages:

| Path | Purpose |
|---|---|
| `diffusion_planner/` | Core PyTorch ML package, installed as `pip install -e .` in `.venv` |
| `diffusion_planner_ros/` | ROS 2 Humble node (not needed for RLVR) |
| `preference_optimization/` | DPO pipeline (not needed for RLVR) |
| `ros_scripts/parse_rosbag.py` | Source of truth for how `.npz`/`.json` pairs are generated |
| `rlvr/` | **New directory** — all RLVR code lives here |

The active Python environment is `.venv` at the repo root.

TeraSim is cloned at `/home/danielsanchez/TeraSim`.

---

## 3. TeraSim Architecture

### 3.1 What TeraSim Is

TeraSim is a naturalistic driving simulator from the University of Michigan (mcity), built on SUMO with a Naturalistic Driving Environment (NDE) traffic model layer. Background vehicles are driven by IDM (longitudinal) + MOBIL (lane change), with an optional adversarial ConflictGenerationModel layer (disabled for Phase 1).

Source: `/home/danielsanchez/TeraSim` (cloned from `https://github.com/mcity/TeraSim`, main branch — no releases exist upstream).

### 3.2 Key Packages

| Package | Location | Purpose |
|---|---|---|
| `terasim` | `packages/terasim` | Core `Simulator` class, SUMO lifecycle, TraCI wrapper |
| `terasim-nde-nade` | `packages/terasim-nde-nade` | NDE/NADE environment, vehicle factory, AV handling |
| `terasim-service` | `packages/terasim-service` | FastAPI REST service — the external control interface |

### 3.3 How TeraSim Starts SUMO

TeraSim **always starts SUMO as a subprocess** via `traci.start(sumo_cmd)` inside the container. It does **not** support `--remote-port` or `traci.connect()` from an external process. This is a hard constraint of the architecture.

The correct way to interact with TeraSim from outside Docker is via its **REST API** (port 8000), not raw TraCI.

```
.venv (Diffusion Planner / TeraSimBridge)        Docker container (TeraSim)
  HTTP POST /start_simulation          ◄──8000──► FastAPI → starts SUMO internally
  HTTP POST /simulation_tick/{id}                  → traci.simulationStep()
  HTTP POST /simulation/{id}/agent_command         → traci.vehicle.moveToXY("AV", ...)
  HTTP GET  /simulation/{id}/state                 ← vehicle positions, speeds
```

SUMO startup is driven by a `.sumocfg` config file and a YAML config file passed to `/start_simulation`.

### 3.4 Docker Setup

```bash
cd /home/danielsanchez/TeraSim
docker build -t terasim:latest .
```

Run the container:
```bash
docker run -d \
  --name terasim_rlvr \
  -p 8000:8000 \
  -p 6379:6379 \
  -v /home/danielsanchez/Diffusion-Planner/rlvr/sim_config:/sim_config:ro \
  terasim:latest \
  sh -c "redis-server --daemonize yes && uvicorn terasim_service.__main__:app --host 0.0.0.0 --port 8000"
```

The `/sim_config` volume mount gives TeraSim access to our YAML config, `.sumocfg`, `.net.xml`, and `.rou.xml` files.

**Acceptance test:** `curl http://localhost:8000/docs` returns the FastAPI swagger UI.

### 3.5 REST API — Relevant Endpoints

All requests are JSON over HTTP to `http://localhost:8000`.

**Start simulation (manual stepping mode):**
```
POST /start_simulation?auto_run=false
Body: {"config_file": "/sim_config/ghost_replay.yaml"}
Response: {"simulation_id": "<uuid>", ...}
```

**Advance one step (0.1 s):**
```
POST /simulation_tick/{simulation_id}
```

**Get all agent states:**
```
GET /simulation/{simulation_id}/state
Response: {
  "simulation_time": 1.3,
  "agent_details": {
    "vehicle": {
      "AV": {"x": ..., "y": ..., "sumo_angle": ..., "speed": ...},
      "nde_car_1": {...},
      ...
    }
  }
}
```

**Teleport the ego (AV) to a new position:**
```
POST /simulation/{simulation_id}/agent_command
Body: {
  "agent_id": "AV",
  "agent_type": "vehicle",
  "command_type": "set_state",
  "data": {
    "position": [x, y],
    "sumo_angle": <degrees CW from north>,
    "speed": <m/s>
  }
}
```
Internally this calls `traci.vehicle.moveToXY("AV", "", 0, x, y, angle, keepRoute=2)`.

**Stop simulation:**
```
POST /simulation_control/{simulation_id}
Body: {"command": "stop"}
```

### 3.6 The AV Vehicle

TeraSim hard-codes the ego vehicle ID as `"AV"` (constant `AV_ID = "AV"` in `nade_with_av.py`). The AV:
- Is excluded from adversarial scenario generation
- Is added to the simulation **after** the NDE warmup completes
- With `av_debug_control=False` and `warmup_control.enabled=False`, goes immediately to external control mode
- Is driven step-by-step by the calling code via `/agent_command`

Collision detection: SUMO removes colliding vehicles from the simulation. If `"AV"` disappears from the `/state` response vehicle list, a collision occurred.

### 3.7 Determinism

Determinism is controlled by two values in our YAML config:
- `simulator.parameters.seed: <int>` → passed as `--seed` to SUMO
- `environment.parameters.warmup_time_lb` and `warmup_time_ub` set to the same value to eliminate warmup duration randomness

Same seed + same initial map + same trajectory input = same simulation output.

---

## 4. Data Format Reference

### 4.1 File Pair Per Sample

Every sample consists of two files with matching stems, produced by `ros_scripts/parse_rosbag.py`:

```
<save_root>/<map_name>_<token>.npz   # ego-centric tensor data
<save_root>/<map_name>_<token>.json  # absolute world-frame ego pose at t=0
```

Example from `/media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207/xx1-dpo-npz/2026-02-16/`:
```
or_event_0_t20251203_134141_with_gt_0000000000000031.json
or_event_0_t20251203_134141_with_gt_0000000000000031.npz
```

### 4.2 JSON Metadata (World Frame)

```json
{
  "timestamp": 1234567890123456789,
  "x":  89412.345,
  "y":  42765.123,
  "z":  6.5,
  "qx": 0.0,
  "qy": 0.0,
  "qz": 0.707,
  "qw": 0.707
}
```

The `x`, `y` values are in **MGRS local Cartesian coordinates** (meters). This is the same coordinate system as the `local_x`/`local_y` attributes in the Lanelet2 OSM file and the SUMO network (after map conversion with `projParameter="!"`).

### 4.3 NPZ Keys and Shapes (All Ego-Centric)

All data inside the `.npz` is in the **ego base_link frame at t=0**. The origin is the ego vehicle position at the current timestep.

| Key | Shape | Description |
|---|---|---|
| `ego_agent_past` | `(21, 4)` | Ego history t=-2.0s..t=0s. Each row: `[x, y, cos(yaw), sin(yaw)]`. At the last row (t=0): always `[0, 0, 1, 0]` |
| `ego_agent_future` | `(80, 3)` | Ego GT future t=0.1s..t=8.0s. Each row: `[x, y, yaw_rad]` |
| `ego_current_state` | `(10,)` | `[x, y, cos_yaw, sin_yaw, vx, vy, ax, ay, steer, yaw_rate]` in base_link |
| `neighbor_agents_past` | `(32, 21, 11)` | NPC history. Fields per step: `[x, y, cos_yaw, sin_yaw, vx, vy, w, l, is_veh, is_ped, is_bike]` |
| `neighbor_agents_future` | `(32, 80, 3)` | NPC GT future. Fields: `[x, y, yaw_rad]` |
| `lanes` | `(140, 20, 33)` | Map lane segments in ego frame |
| `route_lanes` | `(25, 20, 33)` | Route lane segments in ego frame |
| `goal_pose` | `(4,)` | `[x, y, cos_yaw, sin_yaw]` in ego frame |
| `ego_shape` | `(3,)` | `[wheelbase, length, width]` in meters |

### 4.4 Coordinate Reconstruction

To get world-frame coordinates from any ego-centric value:

```python
import json
import numpy as np
from scipy.spatial.transform import Rotation

def load_bl2map(json_path: str) -> np.ndarray:
    """Returns the 4x4 base_link-to-map transform matrix from the JSON sidecar."""
    meta = json.load(open(json_path))
    rot = Rotation.from_quat([meta["qx"], meta["qy"], meta["qz"], meta["qw"]])
    bl2map = np.eye(4)
    bl2map[:3, :3] = rot.as_matrix()
    bl2map[:3,  3] = [meta["x"], meta["y"], meta["z"]]
    return bl2map

def ego_centric_to_map(xy_bl: np.ndarray, bl2map: np.ndarray) -> np.ndarray:
    """
    xy_bl: (N, 2) array of [x, y] in base_link frame
    Returns: (N, 2) array of [x, y] in map frame
    """
    ones = np.ones((len(xy_bl), 1))
    pts_h = np.hstack([xy_bl, np.zeros((len(xy_bl), 1)), ones])  # (N, 4) homogeneous
    pts_map = (bl2map @ pts_h.T).T                                # (N, 4)
    return pts_map[:, :2]

def heading_bl_to_map(cos_h: float, sin_h: float, bl2map: np.ndarray) -> float:
    """Convert a heading from base_link frame to map frame yaw angle (radians, CCW from +X)."""
    ego_yaw = np.arctan2(bl2map[1, 0], bl2map[0, 0])
    agent_yaw = np.arctan2(sin_h, cos_h)
    return ego_yaw + agent_yaw
```

### 4.5 SUMO Angle Convention

SUMO uses degrees, measured **clockwise from north (+Y axis)**. ROS/Autoware uses radians, measured **counterclockwise from east (+X axis)**.

```python
def ros_yaw_to_sumo_angle(yaw_rad: float) -> float:
    """Convert ROS yaw (CCW from +X, radians) to SUMO angle (CW from +Y, degrees)."""
    return np.degrees(np.pi / 2 - yaw_rad) % 360
```

### 4.6 Extracting t=0 Spawn States

```python
def extract_spawn_states(npz_path: str, json_path: str) -> dict:
    """
    Returns everything needed to position the AV and NPCs at t=0.
    All positions are in map (MGRS local Cartesian) frame.
    """
    data = np.load(npz_path, allow_pickle=True)
    bl2map = load_bl2map(json_path)
    meta = json.load(open(json_path))

    ego_yaw_map = np.arctan2(bl2map[1, 0], bl2map[0, 0])
    ego = {
        "x": meta["x"],
        "y": meta["y"],
        "yaw_rad": ego_yaw_map,
        "sumo_angle": ros_yaw_to_sumo_angle(ego_yaw_map),
        "vx": float(data["ego_current_state"][4]),
        "length": float(data["ego_shape"][1]),
        "width":  float(data["ego_shape"][2]),
    }

    npc_past = data["neighbor_agents_past"]  # (32, 21, 11)
    npcs = []
    for i in range(32):
        row = npc_past[i, -1]
        if np.all(row == 0):
            continue
        xy_map = ego_centric_to_map(row[:2].reshape(1, 2), bl2map)[0]
        yaw_map = heading_bl_to_map(row[2], row[3], bl2map)
        npcs.append({
            "id": f"npc_{i}",
            "x":          float(xy_map[0]),
            "y":          float(xy_map[1]),
            "yaw_rad":    float(yaw_map),
            "sumo_angle": ros_yaw_to_sumo_angle(float(yaw_map)),
            "vx":         float(np.sqrt(row[4]**2 + row[5]**2)),
            "width":      float(row[6]),
            "length":     float(row[7]),
            "class":      int(np.argmax(row[8:11])),  # 0=veh, 1=ped, 2=bike
        })

    ego_future_bl = data["ego_agent_future"]  # (80, 3): [x, y, yaw_rad]
    ego_future_map_xy = ego_centric_to_map(ego_future_bl[:, :2], bl2map)
    ego_yaw_offset = np.arctan2(bl2map[1, 0], bl2map[0, 0])
    ego_future_map = np.column_stack([
        ego_future_map_xy,
        ego_yaw_offset + ego_future_bl[:, 2]
    ])  # (80, 3): [x, y, yaw_rad] in map frame

    return {
        "ego":             ego,
        "npcs":            npcs,
        "ego_future_map":  ego_future_map,
    }
```

---

## 5. Map Conversion: Lanelet2 to SUMO

### 5.1 The Coordinate Problem

SUMO requires a `.net.xml` road network file. The Autoware Lanelet2 OSM map stores geometry in MGRS local Cartesian coordinates via the `local_x`/`local_y` node attributes (not standard lat/lon). The `projParameter="!"` attribute in the SUMO `<location>` element tells SUMO to treat coordinates as-is, producing a 1:1 match with the MGRS values in our `.json` sidecar files.

### 5.2 Target Map

```
Input:  /home/danielsanchez/autoware_map/shinagawa_odaiba_stable/lanelet2_map.osm
Output: rlvr/sim_config/maps/shinagawa_odaiba.net.xml
```

### 5.3 Conversion Strategy

First try `import lanelet2` (may be available via the ROS installation). If unavailable, fall back to direct XML parsing of `local_x`/`local_y` attributes using `xml.etree.ElementTree`.

Each lanelet becomes one SUMO edge with one lane. Junction nodes at lanelet endpoints. Speed limit from the lanelet `speed_limit` tag (default 50 km/h = 13.9 m/s if absent). The `<location>` element must include `projParameter="!"`.

Script: `rlvr/scripts/convert_lanelet2_to_sumo.py`

**Acceptance:** `sumo-gui -n rlvr/sim_config/maps/shinagawa_odaiba.net.xml` renders a road network.

---

## 6. Simulation Config Files

Three config files are needed in `rlvr/sim_config/` and volume-mounted into the Docker container at `/sim_config/`.

### 6.1 SUMO Config (`sim.sumocfg`)

```xml
<configuration>
  <input>
    <net-file value="/sim_config/maps/shinagawa_odaiba.net.xml"/>
    <route-files value="/sim_config/background_traffic.rou.xml"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="99999"/>
    <step-length value="0.1"/>
  </time>
  <random_number>
    <random value="false"/>
  </random_number>
  <processing>
    <default.action-step-length value="0.1"/>
    <lateral-resolution value="0.25"/>
    <collision.mingap-factor value="0"/>
    <collision.action value="warn"/>
  </processing>
</configuration>
```

### 6.2 Route File (`background_traffic.rou.xml`)

Defines vehicle types only. NDE generates routes dynamically; this file satisfies SUMO's requirement that at least one route file be present.

```xml
<routes>
  <vType id="car" length="4.5" width="1.8" maxSpeed="30" accel="2.6" decel="4.5" sigma="0.5"/>
  <vType id="veh_AV" length="4.5" width="1.8" maxSpeed="30" accel="2.6" decel="4.5" sigma="0.0" color="1,0,0"/>
</routes>
```

### 6.3 TeraSim YAML Config (`ghost_replay.yaml`)

```yaml
simulation_module: "terasim_nde_nade.envs"
simulation_class: "NADEWithAV"

file_paths:
  sumo_net_file: "/sim_config/maps/shinagawa_odaiba.net.xml"
  sumo_config_file: "/sim_config/sim.sumocfg"

environment:
  module: "terasim_nde_nade.envs"
  class: "NADEWithAV"
  parameters:
    vehicle_factory: "terasim_nde_nade.vehicle.nde_vehicle_factory.NDEVehicleFactory"
    info_extractor: "terasim.logger.infoextractor.InfoExtractor"
    log_flag: false
    warmup_time_lb: 0
    warmup_time_ub: 0
    run_time: 99999
    MOBIL_lc_flag: true
    stochastic_acc_flag: false
    drive_rule: "lefthand"   # Japan drives on the left
    AV_cfg:
      type: "veh_AV"
      cache_radius: 150
      control_radius: 50
      warmup_control:
        enabled: false

simulator:
  module: "terasim.simulator"
  class: "Simulator"
  parameters:
    num_tries: 10
    gui_flag: false
    realtime_flag: false
    seed: 42
    sumo_output_file_types:
      - "collision"

output:
  dir: "/tmp/terasim_output"
  name: "ghost_replay"
  nth: "0"
  aggregated_dir: "aggregated"

logging:
  levels:
    - "INFO"
```

---

## 7. TeraSimBridge Implementation

### 7.1 File Location

`rlvr/terasim_bridge.py`

### 7.2 Architecture

`TeraSimBridge` is an HTTP client to the TeraSim REST service running in Docker. It does **not** import any TeraSim Python package or use TraCI directly.

### 7.3 Class Interface

```python
import requests
import subprocess
import time
import numpy as np


class TeraSimBridge:
    def __init__(
        self,
        config_yaml_path: str = "/sim_config/ghost_replay.yaml",
        service_url: str = "http://localhost:8000",
        docker_image: str = "terasim:latest",
        container_name: str = "terasim_rlvr",
        sim_config_host_dir: str = None,  # absolute path on host to rlvr/sim_config/
    ):
        ...

    def start_episode(self, spawn_states: dict) -> None:
        """
        Starts a new simulation episode.
        1. Ensures the Docker container is running (starts it if not).
        2. Posts /start_simulation with auto_run=false.
        3. Waits for TeraSim to complete NDE warmup and add the AV.
        4. Teleports the AV to the spawn_states["ego"] map-frame position.
        """
        ...

    def step(self, ego_xy: tuple[float, float], ego_yaw_rad: float, ego_speed: float = 0.0) -> dict:
        """
        Teleports the AV to the given map-frame position, then advances one 0.1s step.
        Returns:
            {
                "collision": bool,         # True if AV disappeared from vehicle list
                "av_in_sim": bool,         # False if AV was removed (collision or out of bounds)
                "npc_states": list[dict],  # [{id, x, y, yaw_deg, speed}, ...]
                "sim_time": float,
            }
        """
        ...

    def close(self) -> None:
        """Stops the simulation and optionally removes the container."""
        ...

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
```

### 7.4 Implementation Notes

**Container lifecycle:**
```python
def _ensure_container_running(self):
    result = subprocess.run(
        ["docker", "inspect", "--format={{.State.Running}}", self.container_name],
        capture_output=True, text=True
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        subprocess.run([
            "docker", "run", "-d",
            "--name", self.container_name,
            "-p", "8000:8000",
            "-p", "6379:6379",
            "-v", f"{self.sim_config_host_dir}:/sim_config:ro",
            self.docker_image,
            "sh", "-c",
            "redis-server --daemonize yes && uvicorn terasim_service.__main__:app --host 0.0.0.0 --port 8000"
        ], check=True)
        self._wait_for_service()

def _wait_for_service(self, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(f"{self.service_url}/docs", timeout=1)
            return
        except requests.exceptions.ConnectionError:
            time.sleep(1.0)
    raise RuntimeError("TeraSim service did not become ready in time")
```

**Start episode:**
```python
def start_episode(self, spawn_states: dict):
    self._ensure_container_running()

    resp = requests.post(
        f"{self.service_url}/start_simulation",
        params={"auto_run": "false"},
        json={"config_file": self.config_yaml_path},
    )
    resp.raise_for_status()
    self._sim_id = resp.json()["simulation_id"]

    # Wait for NDE warmup and AV spawn (poll /state until "AV" appears)
    self._wait_for_av_spawn()

    # Teleport AV to correct t=0 position
    ego = spawn_states["ego"]
    self._send_agent_command("AV", ego["x"], ego["y"], ego["sumo_angle"], ego["vx"])
```

**Step:**
```python
def step(self, ego_xy, ego_yaw_rad, ego_speed=0.0):
    sumo_angle = ros_yaw_to_sumo_angle(ego_yaw_rad)

    # Teleport ego
    self._send_agent_command("AV", ego_xy[0], ego_xy[1], sumo_angle, ego_speed)

    # Advance simulation
    requests.post(f"{self.service_url}/simulation_tick/{self._sim_id}").raise_for_status()

    # Get state
    state = requests.get(f"{self.service_url}/simulation/{self._sim_id}/state").json()
    vehicles = state["agent_details"].get("vehicle", {})

    av_in_sim = "AV" in vehicles
    npc_states = [
        {"id": k, "x": v["x"], "y": v["y"], "yaw_deg": v["sumo_angle"], "speed": v["speed"]}
        for k, v in vehicles.items() if k != "AV"
    ]

    return {
        "collision": not av_in_sim,
        "av_in_sim": av_in_sim,
        "npc_states": npc_states,
        "sim_time": state.get("simulation_time", 0.0),
    }
```

---

## 8. Ghost Replay Validator

### 8.1 File Location

`rlvr/scripts/validate_ghost_replay.py`

### 8.2 Purpose

Acceptance test for Phase 1. Loads one `.npz`/`.json` pair, runs the ghost replay, and asserts:
1. No collision at any of the 80 steps
2. AV remains in the simulation for all 80 steps
3. AV's final simulated position is within 2 m of the GT final position (coordinate alignment sanity check; 2 m tolerance accounts for SUMO's snapping to nearest lane)

### 8.3 Script Interface

```bash
source .venv/bin/activate
python rlvr/scripts/validate_ghost_replay.py \
  --npz_path /media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207/xx1-dpo-npz/2026-02-16/or_event_0_t20251203_134141_with_gt/or_event_0_t20251203_134141_with_gt_0000000000000031.npz \
  --json_path /media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207/xx1-dpo-npz/2026-02-16/or_event_0_t20251203_134141_with_gt/or_event_0_t20251203_134141_with_gt_0000000000000031.json
```

### 8.4 Implementation Sketch

```python
from rlvr.terasim_bridge import TeraSimBridge
from rlvr.npz_utils import extract_spawn_states

spawn = extract_spawn_states(npz_path, json_path)

bridge_cfg = dict(
    sim_config_host_dir=str(Path(__file__).parents[2] / "rlvr" / "sim_config"),
)

with TeraSimBridge(**bridge_cfg) as sim:
    sim.start_episode(spawn)

    for step_idx, (x, y, yaw_rad) in enumerate(spawn["ego_future_map"]):
        result = sim.step((x, y), yaw_rad)
        assert result["av_in_sim"], f"AV removed from simulation at step {step_idx} (t={step_idx*0.1:.1f}s) — collision or out-of-bounds"

    # Final position check: get last known AV position from state
    final_state = sim._last_state
    av = final_state["agent_details"]["vehicle"]["AV"]
    gt_x, gt_y = spawn["ego_future_map"][-1, :2]
    dist = np.sqrt((av["x"] - gt_x)**2 + (av["y"] - gt_y)**2)
    assert dist < 2.0, f"Final position error too large: {dist:.2f}m (AV at ({av['x']:.1f},{av['y']:.1f}), GT at ({gt_x:.1f},{gt_y:.1f}))"

    print("Ghost replay validation PASSED")
```

---

## 9. File Layout to Create

```
Diffusion-Planner/
└── rlvr/
    ├── __init__.py
    ├── DESIGN.md                             # this document
    ├── npz_utils.py                          # load_bl2map, extract_spawn_states, etc.
    ├── terasim_bridge.py                     # TeraSimBridge HTTP client
    ├── sim_config/
    │   ├── ghost_replay.yaml                 # TeraSim environment config
    │   ├── sim.sumocfg                       # SUMO config
    │   ├── background_traffic.rou.xml        # vehicle type definitions
    │   └── maps/
    │       └── shinagawa_odaiba.net.xml      # generated by convert script
    └── scripts/
        ├── convert_lanelet2_to_sumo.py       # Lanelet2 OSM → SUMO .net.xml
        └── validate_ghost_replay.py          # Phase 1 acceptance test
```

New pip dependency (add to `diffusion_planner/requirements.txt`):
```
requests   # HTTP client for TeraSim REST API (likely already installed)
scipy      # for Rotation (may already be installed)
```

---

## 10. Implementation Order

Execute strictly in this order. Each step has a concrete acceptance criterion before moving to the next.

**Step 1: Build TeraSim Docker image**
- `cd /home/danielsanchez/TeraSim && docker build -t terasim:latest .`
- Acceptance: `docker run --rm terasim:latest sumo --version` prints SUMO version

**Step 2: Verify TeraSim service starts**
- Run the container with port 8000 exposed
- Acceptance: `curl http://localhost:8000/docs` returns 200

**Step 3: Implement `rlvr/npz_utils.py`**
- Implement `load_bl2map`, `ego_centric_to_map`, `heading_bl_to_map`, `ros_yaw_to_sumo_angle`, `extract_spawn_states`
- Test by loading the sample .npz/.json pair from the test data directory
- Acceptance: `ego["x"]` and `ego["y"]` match the JSON `x`/`y` values; first non-zero NPC has plausible map-frame coordinates

**Step 4: Implement `rlvr/scripts/convert_lanelet2_to_sumo.py`**
- Try `import lanelet2` first; fall back to direct XML parsing of `local_x`/`local_y`
- Run on `/home/danielsanchez/autoware_map/shinagawa_odaiba_stable/lanelet2_map.osm`
- Acceptance: `sumo-gui -n rlvr/sim_config/maps/shinagawa_odaiba.net.xml` shows a recognizable road network

**Step 5: Create config files**
- Write `ghost_replay.yaml`, `sim.sumocfg`, `background_traffic.rou.xml` as specified in Section 6
- Acceptance: TeraSim service accepts `/start_simulation` with this config without error

**Step 6: Implement `rlvr/terasim_bridge.py`**
- Implement `__init__`, `start_episode`, `step`, `close`
- Test with a minimal call: start episode, call `step` 10 times, close
- Acceptance: `step()` returns a valid dict with `av_in_sim=True` for 10 steps

**Step 7: Run `rlvr/scripts/validate_ghost_replay.py`**
- Use the sample `.npz`/`.json` pair
- Acceptance: script prints "Ghost replay validation PASSED" with no assertion errors

---

## 11. Known Risks and Open Questions

| Risk | Mitigation |
|---|---|
| TeraSim YAML config format may differ from what's documented here | Inspect working example configs in `/home/danielsanchez/TeraSim/examples/scenarios/` and adjust |
| `lanelet2` Python lib not importable in `.venv` | Use direct XML parsing of `local_x`/`local_y` attributes as fallback |
| SUMO junction connectivity in hand-built `.net.xml` may be wrong | Validate visually with `sumo-gui`; check that lanes connect at intersections |
| NDE warmup time with `warmup_time_lb=0` may cause NDE to behave unexpectedly | Test with a short warmup (e.g., 10 s) if needed; the AV is always added after warmup |
| AV spawn position (from `add_av_safe`) may place AV far from the `.npz` position before we teleport it | The first `agent_command` teleports it immediately; this is expected behavior |
| SUMO's `moveToXY(keepRoute=2)` may fail if AV is too far from any edge | Ensure map coverage is complete; check that ego position falls within the network boundary |
| `drive_rule: "lefthand"` in YAML may not be supported by all TeraSim versions | Verify against `/home/danielsanchez/TeraSim/packages/terasim-nde-nade` source |
| Collision detection via AV disappearing may miss edge cases | Supplement with SUMO collision output file (`sumo_output_file_types: ["collision"]`) |
