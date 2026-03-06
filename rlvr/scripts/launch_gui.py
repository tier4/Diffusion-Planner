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
import os
import random
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

_CSS = """
.compact-status textarea { font-size: 0.85rem !important; }
"""


# ---------------------------------------------------------------------------
# Lazy imports (avoids slow startup at module load)
# ---------------------------------------------------------------------------
def _get_bridge_and_utils():
    from rlvr.npz_utils import extract_spawn_states
    from rlvr.terasim_bridge import TeraSimBridge
    return extract_spawn_states, TeraSimBridge


# ---------------------------------------------------------------------------
# Top-level mutable state (shared across Gradio callbacks)
# ---------------------------------------------------------------------------
model_state = {
    "model":             None,   # loaded nn.Module
    "model_args":        None,   # Config object from load_model()
    "device":            None,   # torch.device
    "trajectories_ego":  None,   # np.ndarray (N, 80, 4) [x,y,cos,sin] ego frame
    "trajectories_map":  None,   # np.ndarray (N, 80, 3) [x_map,y_map,yaw_map]
    "traj_labels":       None,   # list[str]
}


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


def _load_scene_figure(
    npz_path: str,
    trajectories_ego: np.ndarray | None = None,
    view_range: int = 80,
) -> Figure:
    """Load NPZ file and render the driving scene as a matplotlib Figure.

    Renders lanes, route, neighbor agents, and ego vehicle in the ego-centric
    coordinate frame (base_link at t=0) using diffusion_planner.utils.visualize_input.
    Uses load_npz_data to ensure heading conversions (goal_pose, ego_agent_past)
    are applied before visualization.

    If trajectories_ego is provided, overlays N colored trajectory lines on the
    scene preview so the user can review candidates before launching simulation.

    Args:
        npz_path:         Path to the .npz sample file.
        trajectories_ego: Optional (N, 80, 4) array [x, y, cos, sin] of model
                          trajectories in ego-centric frame.
        view_range:       Axis half-range in meters.
    """
    import matplotlib.pyplot as plt
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

    if trajectories_ego is not None and len(trajectories_ego) > 0:
        cmap = plt.cm.get_cmap("tab10")
        labels = model_state.get("traj_labels") or [
            f"traj_{i}" for i in range(len(trajectories_ego))
        ]
        for i, traj in enumerate(trajectories_ego):
            ax.plot(
                traj[:, 0], traj[:, 1],
                color=cmap(i % 10), linewidth=1.8, alpha=0.75,
                label=labels[i],
            )
        ax.legend(fontsize=7, loc="upper right")

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


def _convert_trajectories_to_map(
    trajectories_ego: np.ndarray,  # (N, 80, 4) [x, y, cos, sin]
    bl2map: np.ndarray,            # (4, 4)
) -> np.ndarray:                   # (N, 80, 3) [x_map, y_map, yaw_rad_map]
    """Convert N ego-centric trajectories to MGRS map frame."""
    from rlvr.npz_utils import ego_centric_to_map, heading_bl_to_map
    N, T, _ = trajectories_ego.shape
    out = np.zeros((N, T, 3), dtype=np.float64)
    for i, traj in enumerate(trajectories_ego):
        xy_map = ego_centric_to_map(traj[:, :2], bl2map)   # (T, 2)
        out[i, :, :2] = xy_map
        for t in range(T):
            out[i, t, 2] = heading_bl_to_map(
                float(traj[t, 2]), float(traj[t, 3]), bl2map
            )
    return out


def _load_model(model_path: str) -> str:
    """Load a Diffusion Planner checkpoint into module-level model_state."""
    import torch
    from preference_optimization.model_utils import load_model

    if not model_path or not os.path.isfile(model_path):
        return "Model file not found"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from pathlib import Path
        model, model_args = load_model(Path(model_path), device)
        model.eval()
        model_state["model"]      = model
        model_state["model_args"] = model_args
        model_state["device"]     = device
        return f"Loaded on {device}"
    except Exception as e:
        return f"Error: {e}"


def _generate_trajectories(
    npz_path, n_samples, include_det, noise_min, noise_max,
    eg, uc, ucs, urf, urfs, ulk, ulks, ucf, ucfs, ua, uas, ai, ap, gs,
):
    """Generate N model trajectories for the current scene and update model_state."""
    import torch
    from diffusion_planner.model.guidance.composer import GuidanceComposer
    from guidance_gui.generate_samples import generate_samples
    from guidance_gui.guidance_ui import make_guidance_set_config
    from preference_optimization.utils import load_npz_data
    from rlvr.npz_utils import load_bl2map

    if model_state["model"] is None:
        return gr.update(), "No model loaded — click 'Load Model' first."

    device     = model_state["device"]
    model      = model_state["model"]
    model_args = model_state["model_args"]

    # Load and normalize observation (same pattern as preference_optimization/utils.py)
    data = load_npz_data(npz_path, device)
    data = model_args.observation_normalizer(data)

    # Build guidance composer
    guidance_cfg = make_guidance_set_config(
        eg, uc, ucs, urf, urfs, ulk, ulks, ucf, ucfs, ua, uas, ai, ap, gs
    )
    composer = GuidanceComposer(guidance_cfg) if guidance_cfg else None

    trajectories, labels = [], []

    if include_det:
        traj = generate_samples(
            model, model_args, data,
            noise_scale=0.0, n_samples=1,
            composer=None, device=device,
        )
        trajectories.append(traj[0])
        labels.append("deterministic")

    remaining = int(n_samples) - (1 if include_det else 0)
    for i in range(max(remaining, 0)):
        noise = (
            random.uniform(float(noise_min), float(noise_max))
            if float(noise_max) > 0 else 0.0
        )
        traj = generate_samples(
            model, model_args, data,
            noise_scale=noise, n_samples=1,
            composer=composer, device=device,
        )
        trajectories.append(traj[0])
        guidance_tag = "+guidance" if composer else ""
        labels.append(f"sample_{i+1} s={noise:.2f}{guidance_tag}")

    if not trajectories:
        return gr.update(), "No trajectories generated (n_samples=0)."

    model_state["trajectories_ego"] = np.stack(trajectories)   # (N, 80, 4)
    model_state["traj_labels"]      = labels

    json_path = npz_path.replace(".npz", ".json")
    bl2map = load_bl2map(json_path)
    model_state["trajectories_map"] = _convert_trajectories_to_map(
        model_state["trajectories_ego"], bl2map
    )

    fig = _load_scene_figure(npz_path, trajectories_ego=model_state["trajectories_ego"])
    return fig, f"Generated {len(labels)}: {', '.join(labels)}"


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



def _run_multi_traj(
    npz_path: str,
    step_delay: float,
):
    """Generator: yields (sim_figure, log_text, results_table) for multi-trajectory mode.

    Runs each model trajectory through an independent TeraSim episode (NPCs react
    adversarially to each), collects per-step safety states, computes metrics, and
    yields a ranked results table at the end.
    """
    import torch
    from preference_optimization.utils import load_npz_data
    from rlvr.npz_utils import load_bl2map
    from rlvr.trajectory_evaluator import (
        StepState,
        compute_score,
        finalize_metrics,
        metrics_to_dataframe,
        rank_trajectories,
    )

    extract_spawn_states, TeraSimBridge = _get_bridge_and_utils()

    json_path = npz_path.replace(".npz", ".json")
    log = ""

    def emit(line: str) -> str:
        nonlocal log
        log += line + "\n"
        return log

    yield None, emit("=== Multi-Trajectory Evaluation ==="), None
    yield None, emit(f"NPZ:  {npz_path}"), None

    # --- Load scene geometry once ---
    yield None, emit("Loading scene geometry…"), None
    try:
        scene_data = load_npz_data(npz_path, torch.device("cpu"))
        scene_geom = _extract_scene_geometry(scene_data)
        bl2map     = load_bl2map(json_path)
        map2bl     = np.linalg.inv(bl2map)
        map_yaw0   = float(np.arctan2(bl2map[1, 0], bl2map[0, 0]))
        ego_z_map  = float(bl2map[2, 3])

        raw        = np.load(npz_path, allow_pickle=True)
        gt_ego_bl  = raw["ego_agent_future"].astype(np.float32)  # (80, 3) ego-centric
        ego_shape  = raw["ego_shape"] if "ego_shape" in raw else None
        ego_length = float(ego_shape[1]) if ego_shape is not None and len(ego_shape) > 2 else 4.5
        ego_width  = float(ego_shape[2]) if ego_shape is not None and len(ego_shape) > 2 else 2.0

        spawn = extract_spawn_states(npz_path, json_path)
        gt_traj_map = spawn["ego_future_map"]  # (80, 3) [x_map, y_map, yaw_rad_map]

        # Pedestrian overlay positions (not spawned in SUMO, drawn statically)
        ped_map_positions = [
            (float(n["x"]), float(n["y"]))
            for n in spawn["npcs"]
            if n.get("class", 0) == 1
        ]

        # On-road filtering: only spawn vehicles that SUMO can snap to a lane
        all_vehicles = [n for n in spawn["npcs"] if n.get("class", 0) != 1]
        vehicles_on_lane, vehicles_off_lane = _filter_npcs_on_lane(
            all_vehicles, scene_data, map2bl, ego_z_map, max_dist=5.0,
        )
        npc_dim_lookup = {n["id"]: (n["length"], n["width"]) for n in all_vehicles}

        spawn = dict(spawn)
        spawn["npcs"] = vehicles_on_lane

        yield None, emit(
            f"  Vehicles on-road: {len(vehicles_on_lane)}/{len(all_vehicles)}  "
            f"off-road overlay: {len(vehicles_off_lane)}  "
            f"peds: {len(ped_map_positions)}"
        ), None
    except Exception as e:
        yield None, emit(f"ERROR loading scene: {e}"), None
        return

    trajectories = model_state["trajectories_map"]    # (N, 80, 3)
    labels       = model_state["traj_labels"]          # list[str]
    N            = len(trajectories)
    yield None, emit(f"Trajectories to evaluate: {N}"), None

    all_metrics = []

    try:
        with TeraSimBridge(sim_config_host_dir=str(_SIM_CONFIG_DIR)) as sim:
            for traj_idx, (traj_map, label) in enumerate(zip(trajectories, labels)):
                yield None, emit(f"\n[{traj_idx+1}/{N}] Starting: {label}"), None

                try:
                    sim.start_episode(spawn, enable_viz=False)
                    yield None, emit(f"  Episode started."), None

                    step_states: list[StepState] = []
                    ego_history_map: list[tuple] = []
                    episode_had_collision = False

                    for step_i, (x, y, yaw_rad) in enumerate(traj_map):
                        result     = sim.step((float(x), float(y)), float(yaw_rad))
                        full_state = sim._last_state
                        av_in_sim  = result["av_in_sim"]

                        all_vehicles_state = full_state["agent_details"].get("vehicle", {})
                        all_vrus_state     = full_state["agent_details"].get("vru", {})

                        vehicle_states_dict = {
                            k: v for k, v in all_vehicles_state.items() if k != "AV"
                        }
                        vru_states_dict = dict(all_vrus_state)

                        if step_states:
                            prev_xy = step_states[-1].ego_xy_map
                            ego_speed = float(np.linalg.norm([
                                (x - prev_xy[0]) / 0.1,
                                (y - prev_xy[1]) / 0.1,
                            ]))
                        else:
                            ego_speed = 0.0

                        av_state = all_vehicles_state.get("AV", {})
                        step_states.append(StepState(
                            step=step_i,
                            ego_xy_map=(float(x), float(y)),
                            ego_speed=ego_speed,
                            av_lane_id=av_state.get("lane_id", ""),
                            av_lateral_lane_pos=float(av_state.get("lateral_lane_pos", 0.0)),
                            av_lane_width=float(av_state.get("lane_width", 0.0)),
                            av_width=ego_width,
                            vehicle_states=vehicle_states_dict,
                            vru_states=vru_states_dict,
                            av_in_sim=av_in_sim,
                        ))

                        ego_history_map.append((float(x), float(y), float(yaw_rad)))

                        npc_list = [{"id": k, **v} for k, v in vehicle_states_dict.items()]
                        vru_list = [{"id": k, **v} for k, v in vru_states_dict.items()]

                        sim_fig = _make_sim_figure(
                            scene_geom, gt_ego_bl, ego_history_map,
                            npc_list, vru_list, ped_map_positions,
                            step_i + 1, 80,
                            map2bl, map_yaw0, ego_z_map,
                            ego_length=ego_length, ego_width=ego_width,
                            static_vehicle_overlay=vehicles_off_lane,
                            npc_dim_lookup=npc_dim_lookup,
                        )

                        if float(step_delay) > 0:
                            time.sleep(float(step_delay))

                        if not av_in_sim:
                            episode_had_collision = True
                            yield sim_fig, emit(
                                f"  [{traj_idx+1}/{N}] COLLISION at step {step_i+1}"
                            ), None
                            break

                        if (step_i + 1) % 10 == 0:
                            yield sim_fig, emit(
                                f"  step {step_i+1:3d}/80  "
                                f"Veh={len(npc_list)}  VRU={len(vru_list)}"
                            ), None
                        else:
                            yield sim_fig, log, None

                    # Compute metrics before closing — step_states are valid
                    # even when the episode ended in a collision.
                    m = finalize_metrics(step_states, traj_map, gt_traj_map)
                    m.label = label
                    m.score = compute_score(m)
                    all_metrics.append(m)
                    yield None, emit(
                        f"  Done: score={m.score:.3f}  collision={m.collision}  "
                        f"clearance={m.min_clearance_m:.1f}m  FDE={m.fde_from_gt_m:.1f}m"
                    ), None

                    try:
                        sim.close()
                    except Exception:
                        pass

                except Exception as ep_err:
                    # Episode failed mid-run (e.g. step timeout after collision).
                    # If steps were collected, inject a terminal collision state so
                    # finalize_metrics correctly attributes collision=True, then
                    # score the trajectory normally.
                    if step_states:
                        last = step_states[-1]
                        step_states.append(StepState(
                            step=last.step + 1,
                            ego_xy_map=last.ego_xy_map,
                            ego_speed=0.0,
                            av_in_sim=False,
                        ))
                        m = finalize_metrics(step_states, traj_map, gt_traj_map)
                        m.label = label
                        m.score = compute_score(m)
                        all_metrics.append(m)
                        yield None, emit(
                            f"  [{traj_idx+1}/{N}] COLLISION (step timeout) — "
                            f"score={m.score:.3f}  ({ep_err})"
                        ), None
                    else:
                        yield None, emit(
                            f"  [{traj_idx+1}/{N}] EPISODE FAILED (no data): {ep_err}"
                        ), None
                    try:
                        sim.close()
                    except Exception:
                        pass
                    sim._ensure_container_running(force_restart=True)
                    yield None, emit("  Container restarted. Continuing…"), None

    except Exception as e:
        yield None, emit(f"\nFATAL ERROR: {e}"), None
        return

    ranked  = rank_trajectories(all_metrics)
    df_data = metrics_to_dataframe(ranked)
    yield None, emit(
        f"\nAll {N} trajectories evaluated.\nBest: {ranked[0].label}  "
        f"score={ranked[0].score:.3f}"
    ), df_data


def _run_simulation(
    npz_path: str,
    use_fcd: bool,
    fcd_dir: str,
    step_delay: float,
):
    """Dispatcher: runs model trajectories through TeraSim.

    Requires model trajectories to have been generated first via 'Generate
    Trajectories'.  Yields (sim_figure, log_text, results_table) tuples.
    """
    if model_state["trajectories_map"] is None:
        yield None, (
            "No model trajectories available.\n"
            "Load a model and click 'Generate Trajectories' before launching simulation."
        ), None
        return
    yield from _run_multi_traj(npz_path, step_delay)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_interface(npz_paths: list[str], model_path_default: str = "") -> gr.Blocks:
    from guidance_gui.guidance_ui import build_guidance_panel

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

    with gr.Blocks(title="TeraSim Multi-Traj Evaluator", css=_CSS) as demo:
        gr.Markdown("## TeraSim — Multi-Trajectory Evaluator")

        # Hidden state: resolved NPZ path passed to simulation
        npz_path_state = gr.Textbox(visible=False)

        # ── Top row: browser + scene preview ────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("**NPZ Browser** — " + f"{total} samples")
                with gr.Row(equal_height=True):
                    btn_prev    = gr.Button("◄", size="sm", min_width=40)
                    index_input = gr.Number(value=1, minimum=1, maximum=total,
                                            precision=0, label="", show_label=False,
                                            min_width=70)
                    btn_next    = gr.Button("►", size="sm", min_width=40)
                current_path_box = gr.Textbox(
                    label="Path", value=npz_paths[0],
                    interactive=False, lines=2, max_lines=3,
                )
                sample_info_md = gr.Textbox(
                    label="Info", value="Loading…",
                    interactive=False, lines=4, max_lines=6,
                )

            with gr.Column(scale=3):
                scene_plot = gr.Plot(label="Scene preview (ego-centric, t=0)",
                                     show_label=True)

        # ── Model & Trajectory Generation ───────────────────────────────────
        with gr.Accordion("Model & Trajectory Generation", open=True):
            # Row 1: model path + load
            with gr.Row(equal_height=True):
                model_path_tb  = gr.Textbox(
                    label="Model checkpoint (.pth)", value=model_path_default,
                    scale=4, lines=1,
                )
                load_model_btn = gr.Button("Load", variant="secondary",
                                           scale=1, min_width=80)
                model_status_lb = gr.Textbox(
                    value="no model loaded", label="Status",
                    interactive=False, scale=2, lines=1, max_lines=2,
                )

            # Row 2: sampling knobs
            with gr.Row(equal_height=True):
                n_samples_sl   = gr.Slider(1, 8, value=4, step=1,
                                           label="N trajectories", scale=2)
                include_det_cb = gr.Checkbox(value=True,
                                             label="Include deterministic", scale=1)
                noise_min_sl   = gr.Slider(0.0, 5.0, value=0.5, step=0.1,
                                           label="Noise min", scale=2)
                noise_max_sl   = gr.Slider(0.0, 5.0, value=3.0, step=0.1,
                                           label="Noise max", scale=2)

            # Guidance — collapsed by default to avoid scroll marathon
            with gr.Accordion("Guidance options", open=False):
                panel = build_guidance_panel()

            # Row 3: generate
            with gr.Row(equal_height=True):
                generate_btn  = gr.Button("Generate Trajectories",
                                          variant="primary", scale=2)
                gen_status_lb = gr.Textbox(
                    value="", label="Generation status",
                    interactive=False, scale=3, lines=1, max_lines=3,
                )

        # ── Simulation controls (compact single row) ─────────────────────────
        with gr.Row(equal_height=True):
            step_delay_sl = gr.Slider(
                label="Step delay (s) — 0 = max speed, 0.1 = real-time",
                minimum=0.0, maximum=1.0, step=0.05, value=0.1, scale=3,
            )
            use_fcd = gr.Checkbox(
                label="Record FCD", value=False, scale=1,
                info="Save SUMO FCD trajectory XML for offline replay.",
            )
            fcd_dir = gr.Textbox(
                label="FCD directory", value=_DEFAULT_FCD_DIR,
                interactive=True, scale=2, lines=1,
            )
            launch_btn = gr.Button("▶ Launch Simulation",
                                   variant="primary", scale=1, min_width=160)

        # ── Simulation output: live plot + log ──────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                sim_plot = gr.Plot(label="Live simulation (ego-centric frame)")
            with gr.Column(scale=1):
                output_log = gr.Textbox(
                    label="Log", lines=20, max_lines=50, interactive=False,
                )

        # ── Results table ────────────────────────────────────────────────────
        results_table = gr.Dataframe(
            headers=[
                "Rank", "Label", "Score", "Collision", "Progress%", "Dist(m)",
                "MinClear(m)", "MinTTC(s)", "NearMiss", "OffRoad%", "Jerk", "FDE(m)",
            ],
            label="Trajectory Ranking",
            wrap=True,
        )

        # ── Event wiring ─────────────────────────────────────────────────────
        _browse_outputs = [
            index_input, current_path_box, sample_info_md,
            npz_path_state, scene_plot,
        ]

        index_input.submit(fn=_load_index, inputs=[index_input], outputs=_browse_outputs)
        btn_prev.click(fn=lambda i: _nav(-1, i), inputs=[index_input], outputs=_browse_outputs)
        btn_next.click(fn=lambda i: _nav(+1, i), inputs=[index_input], outputs=_browse_outputs)

        def _on_page_load():
            *browse, fig = _load_index(1)
            # Report pre-loaded model status so the UI reflects startup load
            dev = model_state["device"]
            status = f"loaded on {dev}" if dev is not None else "no model loaded"
            return (*browse, fig, status)

        # Load first sample and sync model status on page open
        demo.load(
            fn=_on_page_load,
            outputs=[*_browse_outputs, model_status_lb],
        )

        # Load model
        load_model_btn.click(
            fn=_load_model,
            inputs=[model_path_tb],
            outputs=[model_status_lb],
        )

        # Generate trajectories → update scene preview with overlay
        generate_btn.click(
            fn=_generate_trajectories,
            inputs=[
                npz_path_state,
                n_samples_sl, include_det_cb, noise_min_sl, noise_max_sl,
                *panel.inputs,   # 14 guidance values
            ],
            outputs=[scene_plot, gen_status_lb],
        )

        launch_btn.click(
            fn=_run_simulation,
            inputs=[npz_path_state, use_fcd, fcd_dir, step_delay_sl],
            outputs=[sim_plot, output_log, results_table],
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
        "--model_path",
        default="",
        help="Optional path to a Diffusion Planner .pth checkpoint to pre-load.",
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

    # Ensure TeraSim Docker container is running before the GUI opens.
    print("Checking TeraSim Docker container…")
    from rlvr.terasim_bridge import TeraSimBridge
    TeraSimBridge(sim_config_host_dir=str(_SIM_CONFIG_DIR))._ensure_container_running()
    print("  TeraSim container ready.")

    # Pre-load model if provided on the command line.
    if args.model_path:
        print(f"Loading model from {args.model_path}…")
        status = _load_model(args.model_path)
        print(f"  {status}")

    demo = build_interface(npz_paths, model_path_default=args.model_path)
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
