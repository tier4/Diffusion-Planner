#!/usr/bin/env python3
"""
TeraSim Ghost Replay Launcher GUI.

Gradio interface for browsing NPZ samples and running ghost replay simulations
with live in-browser visualization.  Scene preview uses the diffusion_planner
visualize_inputs utility (ego-centric frame).  Simulation visualization renders
the live state in MGRS map frame, updating step-by-step in the same window.

Usage:
    source .venv/bin/activate
    python3 rlvr/scripts/launch_gui.py \\
        --npz_list /media/danielsanchez/.../path_list.json

Opens at http://localhost:7861
"""

import argparse
import json
import math
import time
from pathlib import Path

import gradio as gr
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[2]
_SIM_CONFIG_DIR = _REPO_ROOT / "rlvr" / "sim_config"
_DEFAULT_FCD_DIR = str(Path.home() / "terasim_fcd")


# ---------------------------------------------------------------------------
# Lazy imports (avoids slow startup at module load)
# ---------------------------------------------------------------------------
def _get_bridge_and_utils():
    from rlvr.npz_utils import extract_spawn_states
    from rlvr.terasim_bridge import TeraSimBridge
    return extract_spawn_states, TeraSimBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_npz_list(json_path: str) -> list[str]:
    with open(json_path) as f:
        paths = json.load(f)
    if not isinstance(paths, list):
        raise ValueError(f"Expected a JSON list, got {type(paths)}")
    return paths


def _sample_info(npz_path: str) -> str:
    """Return a plain-text summary of the NPZ sample for the info textbox."""
    try:
        extract_spawn_states, _ = _get_bridge_and_utils()
        json_path = npz_path.replace(".npz", ".json")
        spawn = extract_spawn_states(npz_path, json_path)
        ego = spawn["ego"]
        return (
            f"x={ego['x']:.2f}  y={ego['y']:.2f}\n"
            f"yaw={math.degrees(ego['yaw_rad']):.1f}°  "
            f"speed={ego['vx']:.2f} m/s\n"
            f"Active NPCs: {len(spawn['npcs'])}\n"
            f"GT steps: {spawn['ego_future_map'].shape[0]}"
        )
    except Exception as e:
        return f"Could not load: {e}"


def _load_scene_figure(npz_path: str, view_range: int = 80) -> Figure:
    """Load NPZ file and render the driving scene as a matplotlib Figure.

    Renders lanes, route, neighbor agents, and ego vehicle in the ego-centric
    coordinate frame (base_link at t=0) using diffusion_planner.utils.visualize_input.
    Uses load_npz_data to ensure heading conversions (goal_pose, ego_agent_past)
    are applied before visualization.
    """
    import torch
    from diffusion_planner.utils.visualize_input import visualize_inputs
    from preference_optimization.utils import load_npz_data

    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    try:
        data = load_npz_data(npz_path, torch.device("cpu"))

        # visualize_inputs handles tensor→numpy conversion internally
        visualize_inputs(data, ax=ax, view_ranges=[view_range])
        ax.set_title(Path(npz_path).name, fontsize=9)

    except Exception as e:
        ax.text(
            0.5, 0.5, f"Scene preview error:\n{e}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_axis_off()

    return fig


def _extract_scene_geometry(scene_data: dict) -> dict:
    """Pre-extract lane and route boundary segments from NPZ scene data.

    Extracts left/right lane boundaries and route centerlines as lists of (N, 2)
    numpy arrays in the ego-centric frame.  Called once before the simulation
    loop so the geometry can be cheaply redrawn each step.

    Args:
        scene_data: Dict of tensors/arrays as returned by load_npz_data.

    Returns:
        Dict with keys:
          "lane_segs"  — list of (20, 2) arrays, lane left/right boundaries
          "route_segs" — list of (20, 2) arrays, route centerlines
    """
    import torch

    def _to_np(v):
        return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v

    lane_segs = []
    if "lanes" in scene_data:
        lanes = _to_np(scene_data["lanes"])[0]  # (140, 20, 33)
        for i in range(lanes.shape[0]):
            cx, cy = lanes[i, :, 0], lanes[i, :, 1]
            if np.all(cx == 0) and np.all(cy == 0):
                continue
            # Left boundary: centerline + lateral offset (cols 4, 5)
            lx = cx + lanes[i, :, 4]
            ly = cy + lanes[i, :, 5]
            lane_segs.append(np.column_stack([lx, ly]))
            # Right boundary: centerline + lateral offset (cols 6, 7)
            rx = cx + lanes[i, :, 6]
            ry = cy + lanes[i, :, 7]
            lane_segs.append(np.column_stack([rx, ry]))

    route_segs = []
    if "route_lanes" in scene_data:
        route = _to_np(scene_data["route_lanes"])[0]  # (25, 20, 33)
        for i in range(route.shape[0]):
            cx, cy = route[i, :, 0], route[i, :, 1]
            if np.all(cx == 0) and np.all(cy == 0):
                continue
            route_segs.append(np.column_stack([cx, cy]))

    return {"lane_segs": lane_segs, "route_segs": route_segs}


def _filter_npcs_on_lane(
    npcs: list[dict],
    scene_data: dict,
    map2bl: np.ndarray,
    ego_z_map: float,
    max_dist: float = 30.0,
) -> tuple[list[dict], list[dict]]:
    """Split NPCs into on-road and off-road by proximity to lane centerlines.

    Vehicles within max_dist meters of any lane centerline point are on-road
    and safe to spawn in SUMO (keepRoute=0 snaps them to the correct lane).
    Vehicles further away are outside the road network; spawning them causes
    wrong-lane snapping artifacts or SUMO crashes.  Off-road vehicles are
    returned as a separate list for rendering as static visual overlays at
    their exact NPZ positions.

    Args:
        npcs:       List of NPC dicts from extract_spawn_states() (vehicles only).
        scene_data: Dict of tensors/arrays from load_npz_data().
        map2bl:     (4, 4) map→ego-centric transform.
        ego_z_map:  Ego z coordinate in map frame (bl2map[2, 3]).
        max_dist:   Lane proximity threshold in meters.

    Returns:
        (on_lane, off_lane) — two disjoint lists covering all input NPCs.
    """
    import torch

    def _to_np(v):
        return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v

    if "lanes" not in scene_data:
        return npcs, []

    lanes = _to_np(scene_data["lanes"])
    if lanes.ndim == 4:
        lanes = lanes[0]  # (140, 20, 33)

    lane_pts = []
    for i in range(lanes.shape[0]):
        cx, cy = lanes[i, :, 0], lanes[i, :, 1]
        if np.any(cx != 0) or np.any(cy != 0):
            lane_pts.append(np.column_stack([cx, cy]))

    if not lane_pts:
        return npcs, []

    all_lane_pts = np.concatenate(lane_pts, axis=0)  # (N, 2)

    on_lane, off_lane = [], []
    for npc in npcs:
        npc_bl = _map_to_ego(np.array([[npc["x"], npc["y"]]]), map2bl, z_map=ego_z_map)[0]
        min_dist = float(np.min(np.linalg.norm(all_lane_pts - npc_bl, axis=1)))
        if min_dist <= max_dist:
            on_lane.append(npc)
        else:
            off_lane.append(npc)
    return on_lane, off_lane


def _map_to_ego(xy_map: np.ndarray, map2bl: np.ndarray, z_map: float = 0.0) -> np.ndarray:
    """Transform (N, 2) MGRS map-frame positions to ego-centric frame.

    z_map should be set to the ego vehicle's z coordinate in the map frame
    (bl2map[2, 3]).  ego_centric_to_map() transforms ego-frame z=0 points to
    z≈z_ego in the map, so the inverse must receive z=z_ego to recover the
    original ego-centric XY without a ~z_ego * sin(pitch) offset error.

    Args:
        xy_map:  (N, 2) positions in MGRS map frame
        map2bl:  (4, 4) inverse of bl2map — map frame → base_link transform
        z_map:   z coordinate to assume for all points in the map frame
                 (default 0.0, correct only when ego has zero pitch/roll/height)

    Returns:
        (N, 2) positions in ego-centric frame
    """
    n = len(xy_map)
    pts_h = np.column_stack([xy_map, np.full(n, z_map), np.ones(n)])  # (N, 4)
    pts_bl = (map2bl @ pts_h.T).T                                       # (N, 4)
    return pts_bl[:, :2]


def _draw_agent_box(
    ax,
    cx: float, cy: float,
    heading_bl: float,
    length: float, width: float,
    facecolor: str, edgecolor: str,
    alpha: float = 0.75,
    zorder: int = 8,
) -> None:
    """Draw a rotated bounding-box rectangle for a single agent.

    Args:
        ax:          Matplotlib axis.
        cx, cy:      Agent centre in ego-centric frame.
        heading_bl:  Heading in ego-centric frame (radians, CCW from +X).
        length:      Agent length along heading axis (meters).
        width:       Agent width perpendicular to heading (meters).
        facecolor:   Fill colour.
        edgecolor:   Border colour.
        alpha:       Transparency.
        zorder:      Drawing order.
    """
    from matplotlib.patches import Polygon as MplPolygon

    cos_h = math.cos(heading_bl)
    sin_h = math.sin(heading_bl)
    hl, hw = length / 2.0, width / 2.0

    # Four corners in body frame, rotated to world frame
    corners = [
        (cx + hl * cos_h - hw * sin_h, cy + hl * sin_h + hw * cos_h),
        (cx + hl * cos_h + hw * sin_h, cy + hl * sin_h - hw * cos_h),
        (cx - hl * cos_h + hw * sin_h, cy - hl * sin_h - hw * cos_h),
        (cx - hl * cos_h - hw * sin_h, cy - hl * sin_h + hw * cos_h),
    ]
    ax.add_patch(MplPolygon(
        corners, closed=True,
        facecolor=facecolor, edgecolor=edgecolor,
        alpha=alpha, linewidth=0.8, zorder=zorder,
    ))
    # Front indicator line
    ax.plot(
        [cx, cx + (hl * 0.6) * cos_h],
        [cy, cy + (hl * 0.6) * sin_h],
        color=edgecolor, linewidth=1.2, alpha=alpha, zorder=zorder + 1,
    )


def _make_sim_figure(
    scene_geom: dict,
    gt_ego_bl: np.ndarray,
    ego_history_map: list[tuple[float, float, float]],
    npc_current: list[dict],
    vru_current: list[dict],
    ped_map_positions: list[tuple[float, float]],
    step: int,
    total_steps: int,
    map2bl: np.ndarray,
    map_yaw0: float,
    ego_z_map: float = 0.0,
    ego_length: float = 4.5,
    ego_width: float = 2.0,
    static_vehicle_overlay: list[dict] | None = None,
    npc_dim_lookup: dict | None = None,
) -> Figure:
    """Render current simulation state in ego-centric frame with lane overlay.

    Draws the lane/route geometry from the NPZ (static background), the
    ground-truth ego trajectory, the live ego path history, current NPC
    vehicle bounding boxes (blue, from TeraSim), static off-road vehicle
    overlays (grey, from NPZ), and VRU/pedestrian markers (orange)
    — all in the ego-centric frame (base_link at t=0).

    Args:
        scene_geom:            Pre-extracted lane/route segments (from _extract_scene_geometry).
        gt_ego_bl:             (80, 3) GT ego trajectory [x, y, yaw_rad] in ego-centric frame.
        ego_history_map:       List of (x, y, yaw_rad) in MGRS map frame — one per step.
        npc_current:           Vehicle state dicts from sim.step() result (map frame, not AV).
        vru_current:           VRU state dicts from sim.step() result (map frame).
        ped_map_positions:     List of (x, y) map-frame positions from the NPZ for pedestrians.
                               Drawn as static orange dots every frame (not from SUMO state).
        step:                  Current step index (0-based, used for title).
        total_steps:           Total number of GT steps.
        map2bl:                (4, 4) map→ego-centric transform (inverse of bl2map).
        map_yaw0:              Ego heading in map frame at t=0 (radians).
        ego_z_map:             Ego vehicle z coordinate in map frame (bl2map[2, 3]).
                               Passed to _map_to_ego to cancel the z-offset error
                               that arises from ego pitch/roll and non-zero height.
        ego_length:            Ego vehicle length from NPZ ego_shape (meters).
        ego_width:             Ego vehicle width from NPZ ego_shape (meters).
        static_vehicle_overlay: Off-road NPC dicts (not spawned in SUMO).
                               Drawn as grey boxes at exact NPZ t=0 positions.
        npc_dim_lookup:        {npc_id: (length, width)} from NPZ spawn states.
                               Used to draw on-road SUMO vehicles with correct
                               dimensions instead of the car vType defaults.
    """
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # --- Static background: lane boundaries ---
    if scene_geom["lane_segs"]:
        lc = LineCollection(
            scene_geom["lane_segs"], colors="gray", alpha=0.3, linewidths=0.8, zorder=1
        )
        ax.add_collection(lc)

    # --- Static background: route centerlines ---
    for seg in scene_geom["route_segs"]:
        ax.plot(seg[:, 0], seg[:, 1], color="olive", alpha=0.5,
                linewidth=2.0, linestyle="--", zorder=2)

    # --- GT trajectory (ego-centric, from NPZ) ---
    ax.plot(
        gt_ego_bl[:, 0], gt_ego_bl[:, 1],
        color="green", linestyle="--", alpha=0.45, linewidth=1.5,
        label="GT", zorder=3,
    )

    # --- Live ego path history (transform map→ego-centric) ---
    if ego_history_map:
        xy_map = np.array([[p[0], p[1]] for p in ego_history_map])
        xy_bl = _map_to_ego(xy_map, map2bl, z_map=ego_z_map)
        if len(xy_bl) > 1:
            ax.plot(xy_bl[:, 0], xy_bl[:, 1], color="red",
                    linewidth=2.5, alpha=0.85, zorder=6)
        ex, ey = float(xy_bl[-1, 0]), float(xy_bl[-1, 1])
        eyaw_map = ego_history_map[-1][2]
        eyaw_bl = eyaw_map - map_yaw0
        _draw_agent_box(ax, ex, ey, eyaw_bl, ego_length, ego_width,
                        facecolor="red", edgecolor="darkred", alpha=0.85, zorder=10)
    else:
        ex, ey = 0.0, 0.0

    # --- Off-road vehicles: static grey overlays at exact NPZ t=0 positions ---
    # These are not in SUMO and never move.  Drawn before on-road so blue
    # boxes are rendered on top when both lists overlap visually.
    for npc in (static_vehicle_overlay or []):
        nxy = _map_to_ego(np.array([[npc["x"], npc["y"]]]), map2bl, z_map=ego_z_map)[0]
        npc_yaw_bl = math.radians(90.0 - npc.get("sumo_angle", 0.0)) - map_yaw0
        _draw_agent_box(ax, float(nxy[0]), float(nxy[1]), npc_yaw_bl,
                        npc.get("length", 4.5), npc.get("width", 2.0),
                        facecolor="slategray", edgecolor="dimgray",
                        alpha=0.6, zorder=7)

    # --- On-road vehicles from SUMO state (NDE-driven, move over time) ---
    # Dimensions from NPZ lookup: SUMO only reports the car vType defaults.
    _dim_lut = npc_dim_lookup or {}
    for npc in npc_current:
        nxy = _map_to_ego(np.array([[npc["x"], npc["y"]]]), map2bl, z_map=ego_z_map)[0]
        npc_yaw_bl = math.radians(90.0 - npc.get("sumo_angle", 0.0)) - map_yaw0
        length, width = _dim_lut.get(npc["id"], (npc.get("length", 4.5), npc.get("width", 2.0)))
        _draw_agent_box(ax, float(nxy[0]), float(nxy[1]), npc_yaw_bl,
                        length, width,
                        facecolor="royalblue", edgecolor="navy",
                        alpha=0.75, zorder=8)

    # --- VRU agents from SUMO state (cyclists etc.) ---
    for vru in vru_current:
        nxy = _map_to_ego(np.array([[vru["x"], vru["y"]]]), map2bl, z_map=ego_z_map)[0]
        vru_yaw_bl = math.radians(90.0 - vru.get("sumo_angle", 0.0)) - map_yaw0
        length = vru.get("length", 0.5)
        width  = vru.get("width",  0.5)
        if length < 1.0:
            ax.scatter([float(nxy[0])], [float(nxy[1])],
                       c="darkorange", s=60, alpha=0.9,
                       edgecolors="saddlebrown", linewidths=1.0, zorder=9, marker="o")
        else:
            _draw_agent_box(ax, float(nxy[0]), float(nxy[1]), vru_yaw_bl,
                            length, width,
                            facecolor="darkorange", edgecolor="saddlebrown",
                            alpha=0.75, zorder=9)

    # --- Pedestrians: drawn from NPZ spawn positions (not from SUMO) ---
    if ped_map_positions:
        ped_xy = np.array(ped_map_positions)
        ped_bl = _map_to_ego(ped_xy, map2bl, z_map=ego_z_map)
        ax.scatter(ped_bl[:, 0], ped_bl[:, 1],
                   c="darkorange", s=60, alpha=0.9,
                   edgecolors="saddlebrown", linewidths=1.0, zorder=9, marker="o")

    # --- View window: follow current ego, 80 m half-range ---
    # 80 m matches the scene preview and covers the typical NPC range in
    # the dataset (most NPCs are within 80 m of the ego at t=0).
    half = 80.0
    ax.set_xlim(ex - half, ex + half)
    ax.set_ylim(ey - half, ey + half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    t_s = step * 0.1
    total_s = total_steps * 0.1
    n_off = len(static_vehicle_overlay) if static_vehicle_overlay else 0
    ax.set_title(
        f"Step {step}/{total_steps}  ({t_s:.1f}s / {total_s:.1f}s)  "
        f"Veh: {len(npc_current)}  Off-road: {n_off}  Ped: {len(ped_map_positions)}"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("X [m] (ego-centric)")
    ax.set_ylabel("Y [m] (ego-centric)")

    return fig


def _run_simulation(
    npz_path: str,
    use_fcd: bool,
    fcd_dir: str,
    step_delay: float,
):
    """Generator: yields (sim_figure, log_text) while running the ghost replay.

    Called by Gradio on button click.  Streams step-by-step updates to the
    simulation plot and the log textbox.
    """
    import torch
    from preference_optimization.utils import load_npz_data
    from rlvr.npz_utils import load_bl2map

    extract_spawn_states, TeraSimBridge = _get_bridge_and_utils()

    json_path = npz_path.replace(".npz", ".json")
    log = ""
    ego_history: list[tuple[float, float, float]] = []

    def emit(line: str) -> str:
        nonlocal log
        log += line + "\n"
        return log

    yield None, emit(f"NPZ:  {npz_path}")
    yield None, emit(f"JSON: {json_path}")
    yield None, emit("")

    # --- Extract spawn states ---
    yield None, emit("Extracting spawn states…")
    try:
        spawn = extract_spawn_states(npz_path, json_path)
    except Exception as e:
        yield None, emit(f"ERROR: {e}")
        return

    ego = spawn["ego"]
    yield None, emit(
        f"  Ego t=0:  x={ego['x']:.2f}  y={ego['y']:.2f}  "
        f"yaw={math.degrees(ego['yaw_rad']):.1f}°  "
        f"speed={ego['vx']:.2f} m/s"
    )
    yield None, emit(f"  Active NPCs: {len(spawn['npcs'])}")

    # Pedestrian positions in map frame — drawn as static overlay, not in SUMO.
    ped_map_positions = [
        (float(n["x"]), float(n["y"]))
        for n in spawn["npcs"]
        if n.get("class", 0) == 1
    ]
    yield None, emit(f"  Pedestrians (NPZ overlay): {len(ped_map_positions)}")

    # --- Pre-compute static scene geometry (done once, reused every frame) ---
    yield None, emit("Loading scene geometry…")
    try:
        scene_data = load_npz_data(npz_path, torch.device("cpu"))
        scene_geom = _extract_scene_geometry(scene_data)
        bl2map = load_bl2map(json_path)
        map2bl = np.linalg.inv(bl2map)
        map_yaw0 = float(np.arctan2(bl2map[1, 0], bl2map[0, 0]))
        # z coordinate of the ego in the map frame.  ego_centric_to_map() sends
        # z=0 (base_link plane) to z≈z_ego in map; _map_to_ego must receive the
        # same z to invert cleanly and avoid a ~z_ego * sin(pitch) XY offset.
        ego_z_map = float(bl2map[2, 3])
        # Load the raw NPZ once for GT trajectory and ego shape
        raw = np.load(npz_path, allow_pickle=True)
        gt_ego_bl = raw["ego_agent_future"].astype(np.float32)  # (80, 3) [x, y, yaw_rad]
        # ego_shape: [wheel_base, length, width] — indices 1 and 2
        ego_shape = raw["ego_shape"] if "ego_shape" in raw else None
        ego_length = float(ego_shape[1]) if ego_shape is not None and len(ego_shape) > 2 else 4.5
        ego_width  = float(ego_shape[2]) if ego_shape is not None and len(ego_shape) > 2 else 2.0

        # Split vehicles: on-road ones are spawned in SUMO; off-road ones are
        # rendered as static grey overlays at their exact NPZ positions.
        # keepRoute=0 snaps on-road vehicles to the nearest lane (correct
        # behaviour); off-road vehicles would be snapped to the wrong lane,
        # which produces wrong positions and can crash SUMO.
        all_npcs = spawn["npcs"]
        all_vehicles = [n for n in all_npcs if n.get("class", 0) != 1]
        # 5 m threshold: only spawn vehicles that are actually on a road lane.
        # Vehicles in parking lots / on sidewalks are typically >5 m from any
        # lane centerline and would be snapped to a wrong road position by
        # SUMO's keepRoute=0.  They are shown as static grey overlays instead.
        vehicles_on_lane, vehicles_off_lane = _filter_npcs_on_lane(
            all_vehicles, scene_data, map2bl, ego_z_map, max_dist=5.0,
        )
        # NPZ dimension lookup for on-road vehicles: SUMO state only reports
        # the car vType defaults (4.5 m × 1.8 m), not the actual agent shape.
        npc_dim_lookup = {n["id"]: (n["length"], n["width"]) for n in all_vehicles}

        spawn = dict(spawn)
        spawn["npcs"] = vehicles_on_lane
        yield None, emit(
            f"  Vehicles on-road (SUMO): {len(vehicles_on_lane)} / {len(all_vehicles)}  "
            f"off-road (static overlay): {len(vehicles_off_lane)}"
        )
    except Exception as e:
        yield None, emit(f"ERROR loading scene geometry: {e}")
        return

    if use_fcd:
        if fcd_dir.startswith("/tmp"):
            yield None, emit(
                "WARNING: fcd_dir starts with /tmp — Docker bind-mount may not "
                "work on this host. Use a path under /home/ instead."
            )
        yield None, emit(f"FCD output dir: {fcd_dir}")

    yield None, emit("")
    yield None, emit("Starting TeraSim simulation…")

    # Show GT trajectory before simulation starts
    n_steps = len(spawn["ego_future_map"])
    init_fig = _make_sim_figure(
        scene_geom, gt_ego_bl, [], [], [], ped_map_positions,
        0, n_steps, map2bl, map_yaw0, ego_z_map,
        ego_length=ego_length, ego_width=ego_width,
        static_vehicle_overlay=vehicles_off_lane,
        npc_dim_lookup=npc_dim_lookup,
    )
    yield init_fig, emit("  Waiting for episode to start…")

    fcd_host_dir = fcd_dir if use_fcd else None
    sim_fig = init_fig
    try:
        with TeraSimBridge(
            sim_config_host_dir=str(_SIM_CONFIG_DIR),
            gui=False,
            fcd_host_dir=fcd_host_dir,
        ) as sim:
            sim.start_episode(spawn, enable_viz=False)
            yield init_fig, emit("  Episode started.")

            # Seed last-seen table with NPZ t=0 spawn positions so vehicles
            # that SUMO removes (route exhausted, collision) keep rendering at
            # their last known position instead of vanishing from the view.
            npc_last_seen: dict[str, dict] = {
                n["id"]: {
                    "id":         n["id"],
                    "x":          n["x"],
                    "y":          n["y"],
                    "sumo_angle": n["sumo_angle"],
                    "speed":      n["vx"],
                }
                for n in vehicles_on_lane
            }

            for step_idx in range(n_steps):
                x, y, yaw_rad = spawn["ego_future_map"][step_idx]
                result = sim.step((float(x), float(y)), float(yaw_rad))

                # Update last-seen table; vehicles absent from this step's
                # state (removed by SUMO) will retain their previous entry.
                for npc in result["npc_states"]:
                    npc_last_seen[npc["id"]] = npc

                ego_history.append((float(x), float(y), float(yaw_rad)))

                # Debug: at step 0 log raw NPC/VRU positions and their
                # ego-centric transforms so we can see if agents are in-frame.
                if step_idx == 0:
                    npc_states = result["npc_states"]
                    vru_states = result.get("vru_states", [])
                    yield None, emit(
                        f"  [DBG step0] Veh={len(npc_states)}  VRU={len(vru_states)}"
                    )
                    for i, npc in enumerate(npc_states[:5]):
                        nxy = _map_to_ego(
                            np.array([[npc["x"], npc["y"]]]), map2bl, z_map=ego_z_map
                        )[0]
                        yield None, emit(
                            f"    NPC[{i}] map=({npc['x']:.1f},{npc['y']:.1f})"
                            f"  ego=({nxy[0]:.1f},{nxy[1]:.1f})"
                            f"  sumo_angle={npc.get('sumo_angle',0):.1f}°"
                        )
                    for i, vru in enumerate(vru_states[:3]):
                        vxy = _map_to_ego(
                            np.array([[vru["x"], vru["y"]]]), map2bl, z_map=ego_z_map
                        )[0]
                        yield None, emit(
                            f"    VRU[{i}] map=({vru['x']:.1f},{vru['y']:.1f})"
                            f"  ego=({vxy[0]:.1f},{vxy[1]:.1f})"
                            f"  type={vru.get('type','?')}"
                        )
                    # Show ego position in ego-centric (should be near origin at step 0)
                    ego_xy_bl = _map_to_ego(np.array([[x, y]]), map2bl, z_map=ego_z_map)[0]
                    yield None, emit(
                        f"  [DBG step0] ego in ego-centric: ({ego_xy_bl[0]:.2f},{ego_xy_bl[1]:.2f})"
                        f"  view=±80m"
                    )

                if step_delay > 0:
                    time.sleep(step_delay)

                sim_fig = _make_sim_figure(
                    scene_geom, gt_ego_bl, ego_history,
                    list(npc_last_seen.values()), result.get("vru_states", []),
                    ped_map_positions,
                    step_idx + 1, n_steps,
                    map2bl, map_yaw0, ego_z_map,
                    ego_length=ego_length, ego_width=ego_width,
                    static_vehicle_overlay=vehicles_off_lane,
                    npc_dim_lookup=npc_dim_lookup,
                )

                if not result["av_in_sim"]:
                    yield sim_fig, emit(
                        f"\nFAILED: AV removed from simulation at step {step_idx} "
                        f"(t={result['sim_time']:.1f}s) — collision or out-of-bounds."
                    )
                    return

                if (step_idx + 1) % 10 == 0:
                    yield sim_fig, emit(
                        f"  step {step_idx + 1:3d}/{n_steps}  "
                        f"t={result['sim_time']:.1f}s  "
                        f"Veh={len(result['npc_states'])}  "
                        f"VRU={len(result.get('vru_states', []))}"
                    )
                else:
                    yield sim_fig, log

            # --- Final position check ---
            final_state = sim._last_state
            av_state = final_state["agent_details"]["vehicle"]["AV"]
            av_x, av_y = av_state["x"], av_state["y"]
            gt_x, gt_y = (
                float(spawn["ego_future_map"][-1, 0]),
                float(spawn["ego_future_map"][-1, 1]),
            )
            dist = math.sqrt((av_x - gt_x) ** 2 + (av_y - gt_y) ** 2)
            yield sim_fig, emit(
                f"\nFinal position:  sim=({av_x:.2f}, {av_y:.2f})  "
                f"GT=({gt_x:.2f}, {gt_y:.2f})  error={dist:.3f}m"
            )

            if dist >= 2.0:
                yield sim_fig, emit(
                    f"\nFAILED: position error {dist:.3f}m > 2.0m threshold."
                )
                return

            fcd_path = sim.fcd_output_path
            if fcd_path:
                fcd = Path(fcd_path)
                if fcd.exists():
                    yield sim_fig, emit(
                        f"\nFCD written: {fcd_path}  ({fcd.stat().st_size // 1024} KB)"
                    )
                    yield sim_fig, emit(
                        f"Replay:  python3 rlvr/scripts/replay_fcd.py "
                        f"--fcd_file {fcd_path}"
                    )

            yield sim_fig, emit("\n✓  Ghost replay validation PASSED")

    except Exception as e:
        yield None, emit(f"\nERROR: {e}")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_interface(npz_paths: list[str]) -> gr.Blocks:
    total = len(npz_paths)

    state = {"index": 0}

    def _clamp(i: int) -> int:
        return max(0, min(total - 1, i))

    def _load_index(i: int):
        """Update all display components when the sample index changes."""
        i = _clamp(int(i) - 1)  # UI is 1-based
        state["index"] = i
        path = npz_paths[i]
        info = _sample_info(path)
        fig = _load_scene_figure(path)
        # Returns: displayed_index, path_box, info_text, hidden_path, scene_figure
        return i + 1, path, info, path, fig

    def _nav(delta: int, current_displayed: int):
        new = _clamp(int(current_displayed) - 1 + delta)
        return _load_index(new + 1)

    with gr.Blocks(title="TeraSim Ghost Replay") as demo:
        gr.Markdown("# TeraSim Ghost Replay Launcher")
        gr.Markdown(
            f"Loaded **{total}** samples.  "
            "Browse samples to preview the scene, then launch the simulation."
        )

        # Hidden state: resolved NPZ path passed to simulation
        npz_path_state = gr.Textbox(visible=False)

        # ── Top row: browser + scene preview ────────────────────────────────
        with gr.Row():
            # Left column: navigation controls
            with gr.Column(scale=1):
                gr.Markdown("### NPZ Browser")

                with gr.Row():
                    index_input = gr.Number(
                        label="Sample index",
                        value=1,
                        minimum=1,
                        maximum=total,
                        precision=0,
                    )
                    gr.Markdown(f"/ {total}")

                with gr.Row():
                    btn_prev = gr.Button("◄ Prev", size="sm")
                    btn_next = gr.Button("Next ►", size="sm")

                current_path_box = gr.Textbox(
                    label="Current NPZ path",
                    value=npz_paths[0],
                    interactive=False,
                    lines=2,
                )
                sample_info_md = gr.Textbox(
                    label="Sample info",
                    value="Loading…",
                    interactive=False,
                    lines=5,
                )

            # Right column: scene preview
            with gr.Column(scale=2):
                gr.Markdown("### Scene Preview (ego-centric frame)")
                scene_plot = gr.Plot(label="Scene at t=0")

        # ── Simulation options ───────────────────────────────────────────────
        with gr.Row():
            step_delay = gr.Slider(
                label="Step delay (s) — 0 = as fast as possible, 0.1 = real-time",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.1,
            )
            with gr.Column():
                use_fcd = gr.Checkbox(
                    label="Record FCD output",
                    value=False,
                    info="Write SUMO FCD trajectory XML to disk for offline replay.",
                )
                fcd_dir = gr.Textbox(
                    label="FCD output directory",
                    value=_DEFAULT_FCD_DIR,
                    placeholder="/home/user/terasim_fcd",
                    interactive=True,
                )

        launch_btn = gr.Button(
            "🚀  Launch Simulation with Visualization",
            variant="primary",
            size="lg",
        )

        # ── Simulation output: live plot + log ──────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Live Simulation (MGRS map frame)")
                sim_plot = gr.Plot(label="Simulation state")
            with gr.Column(scale=1):
                gr.Markdown("### Output Log")
                output_log = gr.Textbox(
                    label="Simulation log",
                    lines=20,
                    max_lines=40,
                    interactive=False,
                )

        # ── Event wiring ─────────────────────────────────────────────────────
        _browse_outputs = [
            index_input, current_path_box, sample_info_md,
            npz_path_state, scene_plot,
        ]

        index_input.submit(fn=_load_index, inputs=[index_input], outputs=_browse_outputs)
        btn_prev.click(fn=lambda i: _nav(-1, i), inputs=[index_input], outputs=_browse_outputs)
        btn_next.click(fn=lambda i: _nav(+1, i), inputs=[index_input], outputs=_browse_outputs)

        # Load first sample on page open
        demo.load(fn=lambda: _load_index(1), outputs=_browse_outputs)

        launch_btn.click(
            fn=_run_simulation,
            inputs=[npz_path_state, use_fcd, fcd_dir, step_delay],
            outputs=[sim_plot, output_log],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TeraSim Ghost Replay Launcher GUI."
    )
    parser.add_argument(
        "--npz_list",
        required=True,
        help="JSON file containing a list of .npz paths.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Gradio server port (default 7861; 7860 is used by the DPO GUI).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    args = parser.parse_args()

    print(f"Loading NPZ list from {args.npz_list}…")
    npz_paths = _load_npz_list(args.npz_list)
    print(f"  {len(npz_paths)} samples loaded.")

    demo = build_interface(npz_paths)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
