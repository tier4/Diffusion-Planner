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


def _load_scene_figure(npz_path: str, view_range: int = 60) -> Figure:
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


def _map_to_ego(xy_map: np.ndarray, map2bl: np.ndarray) -> np.ndarray:
    """Transform (N, 2) MGRS map-frame positions to ego-centric frame.

    Args:
        xy_map:  (N, 2) positions in MGRS map frame
        map2bl:  (4, 4) inverse of bl2map — map frame → base_link transform

    Returns:
        (N, 2) positions in ego-centric frame
    """
    n = len(xy_map)
    pts_h = np.column_stack([xy_map, np.zeros(n), np.ones(n)])  # (N, 4)
    pts_bl = (map2bl @ pts_h.T).T                                # (N, 4)
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
    step: int,
    total_steps: int,
    map2bl: np.ndarray,
    map_yaw0: float,
) -> Figure:
    """Render current simulation state in ego-centric frame with lane overlay.

    Draws the lane/route geometry from the NPZ (static background), the
    ground-truth ego trajectory, the live ego path history, current NPC
    vehicle bounding boxes, and VRU (pedestrian/cyclist) markers — all in
    the ego-centric frame (base_link at t=0).

    Args:
        scene_geom:       Pre-extracted lane/route segments (from _extract_scene_geometry).
        gt_ego_bl:        (80, 3) GT ego trajectory [x, y, yaw_rad] in ego-centric frame.
        ego_history_map:  List of (x, y, yaw_rad) in MGRS map frame — one per completed step.
        npc_current:      Vehicle state dicts from sim.step() result (map frame, not AV).
        vru_current:      VRU state dicts from sim.step() result (map frame).
        step:             Current step index (0-based, used for title).
        total_steps:      Total number of GT steps.
        map2bl:           (4, 4) map→ego-centric transform (inverse of bl2map).
        map_yaw0:         Ego heading in map frame at t=0 (radians).
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
        xy_bl = _map_to_ego(xy_map, map2bl)
        if len(xy_bl) > 1:
            ax.plot(xy_bl[:, 0], xy_bl[:, 1], color="red",
                    linewidth=2.5, alpha=0.85, zorder=6)
        ex, ey = float(xy_bl[-1, 0]), float(xy_bl[-1, 1])
        eyaw_map = ego_history_map[-1][2]
        eyaw_bl = eyaw_map - map_yaw0
        _draw_agent_box(ax, ex, ey, eyaw_bl, 4.5, 2.0,
                        facecolor="red", edgecolor="darkred", alpha=0.85, zorder=10)
    else:
        ex, ey = 0.0, 0.0

    # --- NPC vehicles (transform map→ego-centric, draw bounding boxes) ---
    for npc in npc_current:
        nxy = _map_to_ego(np.array([[npc["x"], npc["y"]]]), map2bl)[0]
        npc_yaw_bl = math.radians(90.0 - npc.get("sumo_angle", 0.0)) - map_yaw0
        _draw_agent_box(ax, float(nxy[0]), float(nxy[1]), npc_yaw_bl,
                        npc.get("length", 4.5), npc.get("width", 2.0),
                        facecolor="royalblue", edgecolor="navy", alpha=0.75, zorder=8)

    # --- VRU agents: pedestrians / cyclists ---
    for vru in vru_current:
        nxy = _map_to_ego(np.array([[vru["x"], vru["y"]]]), map2bl)[0]
        vru_yaw_bl = math.radians(90.0 - vru.get("sumo_angle", 0.0)) - map_yaw0
        # Pedestrians are very small — use a circle marker; cyclists slightly larger box
        length = vru.get("length", 0.5)
        width  = vru.get("width",  0.5)
        if length < 1.0:
            # Pedestrian — circle
            ax.scatter([float(nxy[0])], [float(nxy[1])],
                       c="darkorange", s=60, alpha=0.9,
                       edgecolors="saddlebrown", linewidths=1.0, zorder=9, marker="o")
        else:
            # Cyclist / small vehicle
            _draw_agent_box(ax, float(nxy[0]), float(nxy[1]), vru_yaw_bl,
                            length, width,
                            facecolor="darkorange", edgecolor="saddlebrown",
                            alpha=0.75, zorder=9)

    # --- View window: follow current ego, 40 m half-range ---
    half = 40.0
    ax.set_xlim(ex - half, ex + half)
    ax.set_ylim(ey - half, ey + half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    t_s = step * 0.1
    total_s = total_steps * 0.1
    ax.set_title(
        f"Step {step}/{total_steps}  ({t_s:.1f}s / {total_s:.1f}s)  "
        f"Veh: {len(npc_current)}  VRU: {len(vru_current)}"
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

    # --- Pre-compute static scene geometry (done once, reused every frame) ---
    yield None, emit("Loading scene geometry…")
    try:
        scene_data = load_npz_data(npz_path, torch.device("cpu"))
        scene_geom = _extract_scene_geometry(scene_data)
        bl2map = load_bl2map(json_path)
        map2bl = np.linalg.inv(bl2map)
        map_yaw0 = float(np.arctan2(bl2map[1, 0], bl2map[0, 0]))
        # GT trajectory in ego-centric frame (raw yaw, same as NPZ ego_agent_future)
        raw = np.load(npz_path, allow_pickle=True)
        gt_ego_bl = raw["ego_agent_future"].astype(np.float32)  # (80, 3) [x, y, yaw_rad]
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
    init_fig = _make_sim_figure(scene_geom, gt_ego_bl, [], [], [], 0, n_steps, map2bl, map_yaw0)
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

            for step_idx in range(n_steps):
                x, y, yaw_rad = spawn["ego_future_map"][step_idx]
                result = sim.step((float(x), float(y)), float(yaw_rad))

                ego_history.append((float(x), float(y), float(yaw_rad)))

                if step_delay > 0:
                    time.sleep(step_delay)

                sim_fig = _make_sim_figure(
                    scene_geom, gt_ego_bl, ego_history,
                    result["npc_states"], result.get("vru_states", []),
                    step_idx + 1, n_steps,
                    map2bl, map_yaw0,
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
