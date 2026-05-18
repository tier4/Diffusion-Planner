"""Scene Branch Editor — interactive GUI for placing static obstacles and branching scenes.

Launch:
    source .venv/bin/activate
    python -m scenario_generation.tools.scene_branch_editor \
        --npz_dir /path/to/replay_npz_dir \
        [--model_path /path/to/model.pth] \
        [--reward_config /path/to/reward_config.json] \
        [--port 7870]
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch
from matplotlib.patches import Rectangle

from scenario_generation.npz_loader import from_npz
from scenario_generation.scene_context import AgentType, SceneContext
from scenario_generation.tools.scene_tree import (
    BranchNode,
    ObstaclePlacement,
    SceneTree,
)
from scenario_generation.visualize import (
    _EGO_COLOR,
    _LANE_COLOR,
    _ROUTE_COLOR,
    _agent_color,
    draw_agent_box,
    draw_lanes,
    draw_road_borders,
    draw_route,
    draw_stop_lines,
    draw_trajectory,
)

_PLACED_COLOR = "#ff8800"
_PLACED_SELECTED_COLOR = "#ff2200"
_DET_COLOR = "#0088ff"
_GUIDED_COLORS = ["#ff00aa", "#aa00ff", "#00ccaa", "#ffaa00"]
_GT_COLOR = "#22bb22"
_VIEW_HALF_DEFAULT = 50.0

ALL_GUIDANCE_NAMES = [
    "centerline_following",
    "route_centerline_following",
    "speed",
    "lane_keeping",
    "road_border",
    "route_following",
    "collision",
    "anchor_following",
    "lateral",
    "longitudinal",
]


class _ModelCache:
    """Lazy model loader — loads once on first use."""

    def __init__(self, model_path: str | None):
        self._model_path = model_path
        self._model = None
        self._model_args = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def available(self) -> bool:
        return self._model_path is not None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if self._model_path is None:
            raise RuntimeError("No model path provided")
        from preference_optimization.model_utils import load_model
        self._model, self._model_args = load_model(
            Path(self._model_path), self._device,
        )
        self._model.eval()

    @torch.no_grad()
    def predict_det(self, npz_path: str, obstacles: list | None = None) -> np.ndarray | None:
        """Run deterministic inference. Returns (80, 4) [x,y,cos_h,sin_h] or None."""
        self._ensure_loaded()
        from guidance_gui.generate_samples import generate_samples
        data = self._load_npz(npz_path, obstacles=obstacles)
        trajs = generate_samples(
            self._model, self._model_args, data,
            noise_scale=0.0, n_samples=1, composer=None,
            device=self._device,
        )
        return trajs[0]  # (80, 4)

    @torch.no_grad()
    def predict_guided(
        self, npz_path: str, guidance_cfgs: list[tuple[str, float]],
        noise_scale: float = 1.0, n_samples: int = 1,
        obstacles: list | None = None,
    ) -> np.ndarray | None:
        """Run guided inference. Returns (n_samples, 80, 4)."""
        self._ensure_loaded()
        from diffusion_planner.model.guidance.composer import GuidanceComposer
        from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
        from guidance_gui.generate_samples import generate_samples

        data = self._load_npz(npz_path, obstacles=obstacles)

        # Compute DET trajectory first — needed as reference_trajectory for
        # lateral/longitudinal guidance (same pattern as trajectory_ranker_gui)
        det_raw = generate_samples(
            self._model, self._model_args, data,
            noise_scale=0.0, n_samples=1, composer=None, device=self._device,
        )
        det_traj_tensor = torch.from_numpy(det_raw[0]).unsqueeze(0).to(self._device)
        data["reference_trajectory"] = det_traj_tensor  # [1, 80, 4]

        # Extract ego speed for speed guidance params
        ego_state = data["ego_current_state"][0]
        speed = float(ego_state[4].item()) if ego_state.shape[0] > 4 else 5.0

        fns = []
        for name, scale in guidance_cfgs:
            params = {}
            if name == "speed":
                params["v_high"] = speed * 1.2
                params["v_low"] = max(0.0, speed * 0.5)
            fns.append(GuidanceConfig(name=name, enabled=True, scale=scale, params=params))

        if not fns:
            return det_raw

        set_cfg = GuidanceSetConfig(functions=fns, global_scale=1.0)
        composer = GuidanceComposer(set_cfg)
        return generate_samples(
            self._model, self._model_args, data,
            noise_scale=noise_scale, n_samples=n_samples,
            composer=composer, device=self._device,
        )

    def _load_npz(self, npz_path: str, obstacles: list | None = None) -> dict[str, torch.Tensor]:
        from preference_optimization.utils import load_npz_data
        data = load_npz_data(npz_path, self._device)
        pnn = self._model_args.predicted_neighbor_num
        # Inject obstacles BEFORE normalization so they're in the same space
        if obstacles:
            data = _inject_obstacles_into_tensors(data, obstacles, self._device)
        if "neighbor_agents_past" in data and data["neighbor_agents_past"].shape[1] > pnn:
            data["neighbor_agents_past"] = data["neighbor_agents_past"][:, :pnn]
        if "neighbor_agents_future" in data and data["neighbor_agents_future"].shape[1] > pnn:
            data["neighbor_agents_future"] = data["neighbor_agents_future"][:, :pnn]
        # Pad fields to match normalizer expected dims (psim NPZs may have fewer channels)
        norm_dict = self._model_args.observation_normalizer._normalization_dict
        for k, v in norm_dict.items():
            if k in data and isinstance(data[k], torch.Tensor):
                expected_dim = v["mean"].shape[-1]
                actual_dim = data[k].shape[-1]
                if actual_dim < expected_dim:
                    pad = torch.zeros(
                        *data[k].shape[:-1], expected_dim - actual_dim,
                        dtype=data[k].dtype, device=data[k].device,
                    )
                    data[k] = torch.cat([data[k], pad], dim=-1)
        # Ensure float32 for all tensors (psim NPZs sometimes load as float64)
        for k in data:
            if isinstance(data[k], torch.Tensor) and data[k].dtype == torch.float64:
                data[k] = data[k].float()
        data = self._model_args.observation_normalizer(data)
        return data


def _inject_obstacles_into_tensors(
    data: dict[str, torch.Tensor],
    obstacles: list,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Inject placed obstacles into the neighbor_agents_past tensor.

    Each obstacle becomes a stationary neighbor at its (x, y, yaw) with zero
    velocity and vehicle type. Inserted at the front (closest neighbors) and
    existing neighbors are shifted back.
    """
    if not obstacles:
        return data
    nap = data["neighbor_agents_past"]  # [1, N, 31, 11]
    B, N, T, F = nap.shape

    # Sort by distance from ego (at origin) to match the distance-sorted convention
    obstacles = sorted(obstacles, key=lambda o: math.hypot(o.x, o.y))

    new_rows = []
    for obs in obstacles:
        cos_h = math.cos(obs.yaw_rad)
        sin_h = math.sin(obs.yaw_rad)
        # 11 features: [x, y, cos_h, sin_h, vx, vy, width, length, vehicle, ped, bike]
        feat = torch.tensor(
            [obs.x, obs.y, cos_h, sin_h, 0.0, 0.0, obs.width, obs.length, 1.0, 0.0, 0.0],
            dtype=torch.float32, device=device,
        )
        row = feat.unsqueeze(0).expand(T, -1).clone()  # [31, 11] same at every timestep
        # Zero out older history beyond history_steps (matching C++ fill/age behavior)
        hist = getattr(obs, "history_steps", 30)
        n_valid = min(hist + 1, T)  # +1 for current frame
        if n_valid < T:
            row[:T - n_valid] = 0.0
        new_rows.append(row.unsqueeze(0).unsqueeze(0))  # [1, 1, 31, 11]

    if new_rows:
        new_block = torch.cat(new_rows, dim=1)  # [1, n_obs, 31, 11]
        # Prepend obstacles, truncate to N total
        nap_new = torch.cat([new_block, nap], dim=1)[:, :N, :, :]
        data = dict(data)
        data["neighbor_agents_past"] = nap_new
        if "neighbor_agents_future" in data:
            naf = data["neighbor_agents_future"]
            _, _, Tf, Ff = naf.shape
            zero_fut = torch.zeros(1, len(obstacles), Tf, Ff, dtype=naf.dtype, device=device)
            naf_new = torch.cat([zero_fut, naf], dim=1)[:, :N, :, :]
            data["neighbor_agents_future"] = naf_new

    return data


def _traj_cos_sin_to_xyh(traj: np.ndarray) -> np.ndarray:
    """Convert (T, 4) [x,y,cos_h,sin_h] to (T, 3) [x,y,heading_rad]."""
    heading = np.arctan2(traj[:, 3], traj[:, 2])
    return np.column_stack([traj[:, :2], heading])


def _extract_border_polylines(scene: SceneContext) -> list[np.ndarray]:
    """Extract road border polylines from line_strings (channel 3 = border flag)."""
    ls = scene.map_data.line_strings
    polylines = []
    if ls.shape[-1] < 4:
        return polylines
    for i in range(ls.shape[0]):
        line = ls[i]
        if np.abs(line[:, :2]).sum() < 1e-6:
            continue
        valid = (line[:, 3] > 0.5) & (np.abs(line[:, :2]).sum(axis=1) > 0.01)
        if valid.sum() >= 2:
            polylines.append(line[valid, :2].copy())
    return polylines


def render_scene_at_step(
    scene: SceneContext,
    obstacles: list[ObstaclePlacement] | None = None,
    selected_obstacle: str | None = None,
    view_half: float = _VIEW_HALF_DEFAULT,
    step_idx: int = 0,
    total_steps: int = 1,
    figsize: tuple[float, float] = (10, 10),
    gt_traj: np.ndarray | None = None,
    det_traj: np.ndarray | None = None,
    guided_trajs: list[np.ndarray] | None = None,
    show_rb_dist: bool = False,
    show_nb_dist: bool = False,
    hide_neighbors: bool = False,
    map_border_polylines: list[np.ndarray] | None = None,
    ego_world_pose: np.ndarray | None = None,
) -> matplotlib.figure.Figure:
    """Render a scene with placed obstacles overlaid, matching replay sim style."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor("#f8f8f8")

    # Lane network
    draw_lanes(ax, scene.map_data, alpha=0.5)

    # Road borders (red) — from NPZ if available
    draw_road_borders(ax, scene.map_data)

    # Road borders from lanelet2 map (world frame → ego frame via canonical transform)
    _ego_frame_borders = None
    if map_border_polylines and ego_world_pose is not None:
        from matplotlib.collections import LineCollection as _LC
        from scenario_generation.transforms import _rotation_matrix, transform_positions
        ex_w, ey_w, eyaw_w = ego_world_pose
        ego_xy_w = np.array([ex_w, ey_w], dtype=np.float64)
        R_w = _rotation_matrix(eyaw_w)
        ego_segs = []
        _ego_frame_borders = []
        for pl in map_border_polylines:
            if pl.shape[0] < 2:
                continue
            pts_ego = transform_positions(pl.astype(np.float64), R_w, ego_xy_w)
            if np.abs(pts_ego).max() > view_half * 2:
                in_view = ((np.abs(pts_ego[:, 0]) < view_half * 1.5) &
                           (np.abs(pts_ego[:, 1]) < view_half * 1.5))
                if in_view.sum() < 2:
                    continue
            ego_segs.append(pts_ego)
            _ego_frame_borders.append(pts_ego)
        if ego_segs:
            ax.add_collection(_LC(ego_segs, colors="red", linewidths=2.5, alpha=0.7, zorder=4))

    # Stop lines
    draw_stop_lines(ax, scene.map_data)

    # Ego route
    ego = scene.ego_agent
    if ego is not None and ego.route_lanes is not None:
        draw_route(ax, ego.route_lanes, color=_ROUTE_COLOR, alpha=0.5, lw=2.5)

    # Traffic light colored overlay on lanes + route_lanes
    from matplotlib.collections import LineCollection
    tl_hex = {0: "#22bb22", 1: "#ddaa00", 2: "#dd2222"}  # green, yellow, red
    tl_segments: dict[str, list[np.ndarray]] = {}
    _ego_rl = ego.route_lanes if ego is not None else None
    for lane_tensor in [scene.map_data.lanes, _ego_rl]:
        if lane_tensor is None:
            continue
        for i in range(lane_tensor.shape[0]):
            lane = lane_tensor[i]
            pts = lane[:, :2]
            if np.abs(pts).sum() < 1e-6:
                continue
            tl_onehot = lane[0, 8:13]
            if tl_onehot.sum() < 0.5:
                continue
            ch = int(np.argmax(tl_onehot))
            if ch >= 3:  # WHITE=3, NONE=4
                continue
            hex_color = tl_hex.get(ch)
            if hex_color is None:
                continue
            valid = np.abs(pts).sum(axis=1) > 0.1
            if valid.sum() < 2:
                continue
            tl_segments.setdefault(hex_color, []).append(pts[valid])
    for hex_color, segs in tl_segments.items():
        ax.add_collection(LineCollection(
            segs, colors=hex_color, linewidths=2.5, alpha=0.85, zorder=4,
        ))

    # Static objects from map
    so = scene.map_data.static_objects
    for i in range(so.shape[0]):
        if np.abs(so[i, :2]).sum() < 1e-6:
            continue
        x, y = so[i, 0], so[i, 1]
        cos_h, sin_h = so[i, 2], so[i, 3]
        w, l = so[i, 4], so[i, 5]
        if w < 0.1 or l < 0.1:
            continue
        heading = math.atan2(sin_h, cos_h)
        draw_agent_box(ax, x, y, heading, l, w, "#999999", alpha=0.4, lw=0.5, zorder=5)

    # Agents (ego + neighbors)
    nb_idx = 0
    for agent in scene.agents:
        is_ego = agent.id == scene.ego_agent_id
        if not is_ego and hide_neighbors:
            continue
        pos = agent.current_position
        heading = agent.current_heading

        is_placed = agent.id.startswith("placed_")
        if is_ego:
            color = _EGO_COLOR
        elif is_placed:
            color = _PLACED_COLOR
        else:
            color = _agent_color(agent.agent_type, nb_idx)
            nb_idx += 1

        # Past trail
        past = agent.past_trajectory
        valid = np.abs(past[:, :2]).sum(axis=1) > 1e-6
        if valid.sum() > 1:
            ax.plot(past[valid, 0], past[valid, 1], "--", color=color,
                    lw=0.9, alpha=0.5, zorder=7)

        # Bounding box
        if is_placed:
            rear_ovh = (agent.length - agent.length * 0.65) / 2
            t_rot = mtransforms.Affine2D().rotate(heading).translate(pos[0], pos[1]) + ax.transData
            rect = Rectangle(
                (-rear_ovh, -agent.width / 2), agent.length, agent.width,
                lw=2.5, ec=color, fc=color, alpha=0.3,
                linestyle="--", zorder=25, transform=t_rot,
            )
            ax.add_patch(rect)
        else:
            draw_agent_box(
                ax, pos[0], pos[1], heading, agent.length, agent.width,
                color, alpha=0.85 if is_ego else 0.55,
                lw=2 if is_ego else 1, zorder=20 if is_ego else 15,
            )

        # Heading arrow
        arrow_len = max(agent.length, 2.5)
        ax.annotate(
            "", xy=(pos[0] + arrow_len * math.cos(heading),
                    pos[1] + arrow_len * math.sin(heading)),
            xytext=(pos[0], pos[1]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=12),
            zorder=21 if is_ego else 16,
        )

        # Speed label
        speed = float(np.linalg.norm(agent.current_velocity))
        ax.annotate(
            f"{agent.id} ({speed:.1f} m/s)", (pos[0], pos[1]),
            fontsize=7, color=color, ha="center", va="bottom",
            xytext=(0, 6), textcoords="offset points", zorder=22,
        )

        # Goal
        if agent.goal_pose is not None and is_ego:
            gx, gy = agent.goal_pose[0], agent.goal_pose[1]
            ax.plot(gx, gy, "*", color=color, ms=14, zorder=25,
                    markeredgecolor="black", markeredgewidth=0.8)

    # Placed obstacles (distinctive orange dashed outline)
    if obstacles:
        for obs in obstacles:
            is_selected = (selected_obstacle == obs.label)
            color = _PLACED_SELECTED_COLOR if is_selected else _PLACED_COLOR
            lw = 3.0 if is_selected else 2.0
            yaw = obs.yaw_rad

            # Draw OBB with dashed outline
            rear_overhang = (obs.length - obs.length * 0.65) / 2
            t_rot = mtransforms.Affine2D().rotate(yaw).translate(obs.x, obs.y) + ax.transData
            rect = Rectangle(
                (-rear_overhang, -obs.width / 2), obs.length, obs.width,
                lw=lw, ec=color, fc=color, alpha=0.3,
                linestyle="--", zorder=30, transform=t_rot,
            )
            ax.add_patch(rect)

            # Heading arrow
            arrow_len = max(obs.length, 2.0)
            ax.annotate(
                "", xy=(obs.x + arrow_len * math.cos(yaw),
                        obs.y + arrow_len * math.sin(yaw)),
                xytext=(obs.x, obs.y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=14),
                zorder=31,
            )

            # Label
            ax.annotate(
                f"[{obs.label}]", (obs.x, obs.y),
                fontsize=8, fontweight="bold", color=color,
                ha="center", va="bottom",
                xytext=(0, 10), textcoords="offset points", zorder=32,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85, lw=1),
            )

    # Reward overlays: road border distance + neighbor distance
    if (show_rb_dist or show_nb_dist) and ego is not None:
        ego_pos = ego.current_position
        ego_h = ego.current_heading

        if show_rb_dist:
            from scenario_generation.replay import _nearest_border_point
            border_polylines = _extract_border_polylines(scene)
            # Fall back to map-derived borders (already in ego frame)
            if not border_polylines and _ego_frame_borders:
                border_polylines = _ego_frame_borders
            bp = _nearest_border_point(ego_pos, border_polylines)
            if bp is not None:
                dist = float(np.linalg.norm(bp - ego_pos))
                color = "#dd2222" if dist < 0.5 else "#ff8800" if dist < 1.0 else "#22bb22"
                ax.plot([ego_pos[0], bp[0]], [ego_pos[1], bp[1]],
                        "-", color=color, lw=2.5, alpha=0.9, zorder=35)
                ax.plot(bp[0], bp[1], "o", color=color, ms=6, zorder=36)
                mid_x, mid_y = (ego_pos[0] + bp[0]) / 2, (ego_pos[1] + bp[1]) / 2
                ax.annotate(
                    f"RB {dist:.2f}m", (mid_x, mid_y),
                    fontsize=8, fontweight="bold", color=color,
                    ha="center", va="bottom", zorder=37,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
                )

        if show_nb_dist:
            from scenario_generation.gui.lanelet_scene_builder import _obb_corners
            from rlvr.reward import _closest_points_between_rects
            import torch as _torch
            ego_corners = _obb_corners(
                float(ego_pos[0]), float(ego_pos[1]), ego_h,
                float(ego.length), float(ego.width),
            )
            nb_agents = [a for a in scene.agents if a.id != scene.ego_agent_id]
            if nb_agents:
                nb_corners_list = [
                    _obb_corners(
                        float(a.current_position[0]), float(a.current_position[1]),
                        float(a.current_heading), float(a.length), float(a.width),
                    )
                    for a in nb_agents
                ]
                n = len(nb_agents)
                r1 = _torch.from_numpy(
                    np.broadcast_to(ego_corners, (n, 4, 2)).copy().astype(np.float32)
                )
                r2 = _torch.from_numpy(np.stack(nb_corners_list).astype(np.float32))
                pt_e, pt_n = _closest_points_between_rects(r1, r2)
                dists = (pt_e - pt_n).norm(dim=-1)
                for i in range(min(n, 5)):
                    d = float(dists[i].item())
                    if d > 10.0:
                        continue
                    pe = pt_e[i].numpy()
                    pn = pt_n[i].numpy()
                    color = "#dd2222" if d < 0.5 else "#ff8800" if d < 1.5 else "#22bb22"
                    ax.plot([pe[0], pn[0]], [pe[1], pn[1]],
                            "-", color=color, lw=2.0, alpha=0.8, zorder=35)
                    mid_x, mid_y = (pe[0] + pn[0]) / 2, (pe[1] + pn[1]) / 2
                    ax.annotate(
                        f"{d:.2f}m", (mid_x, mid_y),
                        fontsize=7, color=color, ha="center", va="bottom", zorder=37,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8),
                    )

    # Trajectory overlays
    ego_len = ego.length if ego else 4.5
    ego_wid = ego.width if ego else 1.8
    if gt_traj is not None and gt_traj.shape[0] > 1:
        draw_trajectory(ax, gt_traj, _GT_COLOR, label="GT", lw=2.0, zorder=25,
                        show_footprints=True, length=ego_len, width=ego_wid)
    if det_traj is not None and det_traj.shape[0] > 1:
        draw_trajectory(ax, det_traj, _DET_COLOR, label="DET", lw=2.0, zorder=26,
                        show_footprints=True, length=ego_len, width=ego_wid)
    if guided_trajs:
        for i, gt in enumerate(guided_trajs):
            if gt is not None and gt.shape[0] > 1:
                color = _GUIDED_COLORS[i % len(_GUIDED_COLORS)]
                draw_trajectory(ax, gt, color, label=f"Guided #{i+1}", lw=1.5, zorder=27,
                                show_footprints=False, length=ego_len, width=ego_wid)

    # Viewport: center on ego
    if ego is not None:
        ex, ey = ego.current_position
        ax.set_xlim(ex - view_half, ex + view_half)
        ax.set_ylim(ey - view_half, ey + view_half)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Legend for trajectory overlays
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

    ax.set_title(f"Step {step_idx} / {total_steps - 1}", fontsize=10)

    fig.tight_layout()
    return fig


def _fig_to_pil(fig: matplotlib.figure.Figure):
    """Convert matplotlib Figure to PIL Image."""
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _transform_point_between_steps(
    seq: list[str], from_step: int, to_step: int, x: float, y: float, yaw_rad: float,
) -> tuple[float, float, float]:
    """Transform a point from one timestep's ego frame to another's.

    Uses the ego displacement chain from ego_agent_past[-2] at each step.
    Works for both forward (to > from) and backward (to < from) transforms.
    Returns (x', y', yaw') in to_step's ego frame.
    """
    if from_step == to_step:
        return x, y, yaw_rad

    # Chain displacements step-by-step
    # Each step's ego_agent_past[-2] gives the previous ego position in the
    # current step's frame. The displacement from prev→current in current's
    # frame is (-past[-2][0], -past[-2][1]), yaw change = -past[-2][2].
    cum_x, cum_y, cum_yaw = 0.0, 0.0, 0.0

    if to_step > from_step:
        # Forward: chain from from_step+1 to to_step
        for s in range(from_step + 1, min(to_step + 1, len(seq))):
            past = np.load(seq[s])["ego_agent_past"]
            prev = past[-2]
            dx_local, dy_local, dyaw = -prev[0], -prev[1], -prev[2]
            cum_yaw += dyaw
            c, sn = math.cos(cum_yaw), math.sin(cum_yaw)
            cum_x += c * dx_local - sn * dy_local
            cum_y += sn * dx_local + c * dy_local
        # Point was at (x, y) in from_step frame. In to_step frame:
        # First express (x, y) relative to ego's new position
        rx, ry = x - cum_x, y - cum_y
        c, sn = math.cos(-cum_yaw), math.sin(-cum_yaw)
        new_x = c * rx - sn * ry
        new_y = sn * rx + c * ry
        new_yaw = yaw_rad - cum_yaw
    else:
        # Backward: chain from to_step+1 to from_step (then invert)
        for s in range(to_step + 1, min(from_step + 1, len(seq))):
            past = np.load(seq[s])["ego_agent_past"]
            prev = past[-2]
            dx_local, dy_local, dyaw = -prev[0], -prev[1], -prev[2]
            cum_yaw += dyaw
            c, sn = math.cos(cum_yaw), math.sin(cum_yaw)
            cum_x += c * dx_local - sn * dy_local
            cum_y += sn * dx_local + c * dy_local
        # cum describes to_step→from_step. We need to apply the forward
        # transform: the point at (x, y) in from_step frame maps to
        # to_step frame by rotating by cum_yaw and translating.
        c, sn = math.cos(cum_yaw), math.sin(cum_yaw)
        new_x = c * x - sn * y + cum_x
        new_y = sn * x + c * y + cum_y
        new_yaw = yaw_rad + cum_yaw

    return new_x, new_y, new_yaw


def _recover_ego_world_pose(seq: list[str], step: int) -> np.ndarray | None:
    """Recover ego map-frame pose at a given step.

    First tries the sidecar JSON (psim NPZs have one per step with x, y, qz, qw).
    Falls back to None if unavailable.
    """
    if not seq or step >= len(seq):
        return None
    npz_path = Path(seq[min(step, len(seq) - 1)])
    json_path = npz_path.with_suffix(".json")
    if json_path.exists():
        try:
            import json
            with open(json_path) as f:
                d = json.load(f)
            x, y = d["x"], d["y"]
            qz, qw = d.get("qz", 0.0), d.get("qw", 1.0)
            from scenario_generation.transforms import yaw_from_quat
            qx, qy = d.get("qx", 0.0), d.get("qy", 0.0)
            yaw = yaw_from_quat(qx, qy, qz, qw)
            return np.array([x, y, yaw], dtype=np.float64)
        except (KeyError, json.JSONDecodeError):
            pass
    return None


def _reconstruct_gt_from_sequence(seq: list[str], current_step: int, max_future: int = 80) -> np.ndarray | None:
    """Reconstruct GT ego future from subsequent NPZ files in the sequence.

    Each NPZ stores ego at origin. The future ego positions at step+1..step+T
    are the ego_agent_past[-1] of those NPZs, but in THEIR ego frame (origin).
    To get positions in the CURRENT step's ego frame, we chain the relative
    displacements from each step's ego_agent_past (the last two entries give
    the per-step delta).
    """
    n_future = min(max_future, len(seq) - current_step - 1)
    if n_future < 2:
        return None

    # Load current step to get the ego's world-frame anchor
    cur = np.load(seq[current_step])
    cur_past = cur["ego_agent_past"]  # (31, 3) [x, y, yaw] in current ego frame
    # ego is at origin: cur_past[-1] = [0, 0, 0]

    # For each future step, load the ego_agent_past and extract the
    # position of the CURRENT step's ego in that step's frame.
    # Actually simpler: each step's ego_agent_past[-1] = [0,0,0] (ego at origin).
    # But the ego_agent_past[-2] tells us where the ego was 1 step ago.
    # So future_step's past[-1] is at origin, and we need to express that
    # in the current step's frame.
    #
    # The cleanest way: accumulate displacements. At each step k, the ego
    # moved from past[-2] to past[-1]=[0,0,0]. The displacement in step k's
    # frame is -past[-2]. We rotate this into the current frame.

    gt_points = []
    cumulative_x, cumulative_y = 0.0, 0.0
    cumulative_yaw = 0.0

    for i in range(1, n_future + 1):
        future_idx = current_step + i
        if future_idx >= len(seq):
            break
        fut = np.load(seq[future_idx])
        fut_past = fut["ego_agent_past"]  # (31, 3)
        # Displacement from prev to current in this step's ego frame
        prev_in_cur = fut_past[-2]  # where ego was 1 step ago, in this step's frame
        # The ego moved from prev_in_cur to [0,0,0]
        dx_local = -prev_in_cur[0]
        dy_local = -prev_in_cur[1]
        dyaw = -prev_in_cur[2]

        # Update yaw first, then rotate displacement into step-0 frame
        cumulative_yaw += dyaw
        cos_a = math.cos(cumulative_yaw)
        sin_a = math.sin(cumulative_yaw)
        cumulative_x += cos_a * dx_local - sin_a * dy_local
        cumulative_y += sin_a * dx_local + cos_a * dy_local

        gt_points.append([cumulative_x, cumulative_y, cumulative_yaw])

    if len(gt_points) < 2:
        return None
    return np.array(gt_points, dtype=np.float32)


def build_interface(tree: SceneTree, model_cache: _ModelCache | None = None,
                    map_borders: list[np.ndarray] | None = None,
                    map_builder=None):
    """Build the Gradio interface for the scene branch editor."""

    with gr.Blocks(title="Scene Branch Editor") as demo:
        # ── State ──
        tree_state = gr.State(value=tree)
        selected_obstacle_state = gr.State(value=None)
        det_traj_state = gr.State(value=None)      # cached (80, 3) or None
        guided_trajs_state = gr.State(value=None)   # cached list[(80, 3)] or None

        gr.Markdown("# Scene Branch Editor")

        with gr.Row():
            # ═══════ LEFT PANEL ═══════
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Navigation")
                with gr.Row():
                    load_dir_input = gr.Textbox(
                        label="NPZ Directory", value=tree.base_npz_dir,
                        scale=3, interactive=True,
                    )
                    load_dir_btn = gr.Button("Load", size="sm", scale=1)

                load_tree_input = gr.Textbox(label="Tree JSON", placeholder="/path/to/scene_tree.json")
                with gr.Row():
                    load_tree_btn = gr.Button("Load Tree", size="sm")
                    save_tree_btn = gr.Button("Save Tree", size="sm")

                gr.Markdown("### Timeline")
                step_slider = gr.Slider(
                    minimum=0,
                    maximum=max(0, len(tree.get_npz_sequence("root")) - 1),
                    value=0, step=1, label="Step (drag)",
                )
                with gr.Row():
                    btn_first = gr.Button("|<", size="sm", min_width=40)
                    btn_prev = gr.Button("<", size="sm", min_width=40)
                    btn_next = gr.Button(">", size="sm", min_width=40)
                    btn_last = gr.Button(">|", size="sm", min_width=40)
                with gr.Row():
                    step_jump_input = gr.Number(
                        label="Jump to step", value=0, precision=0, scale=2,
                    )
                    step_jump_btn = gr.Button("Go", size="sm", scale=1)
                step_info = gr.Markdown("Step 0 / 0")

                gr.Markdown("### Obstacle Placement")
                with gr.Row():
                    obs_x = gr.Number(label="X (m)", value=10.0, precision=1)
                    obs_y = gr.Number(label="Y (m)", value=0.0, precision=1)
                obs_yaw = gr.Slider(
                    minimum=-180, maximum=180, value=0, step=5,
                    label="Yaw (deg)",
                )
                with gr.Row():
                    obs_length = gr.Slider(
                        minimum=1.0, maximum=15.0, value=4.5, step=0.1,
                        label="Length (m)",
                    )
                    obs_width = gr.Slider(
                        minimum=0.5, maximum=3.0, value=1.8, step=0.1,
                        label="Width (m)",
                    )
                obs_history = gr.Slider(
                    minimum=0, maximum=30, value=30, step=1,
                    label="History steps (0=just appeared, 30=full)",
                )
                with gr.Row():
                    place_btn = gr.Button("Place Obstacle", variant="primary")
                    preview_btn = gr.Button("Preview", variant="secondary")

                gr.Markdown("### View")
                view_half = gr.Slider(
                    minimum=10, maximum=200, value=50, step=5,
                    label="View radius (m)",
                )

                gr.Markdown("### Crop")
                with gr.Row():
                    crop_start = gr.Number(label="Start", value=0, precision=0)
                    crop_end = gr.Number(label="End", value=0, precision=0)
                with gr.Row():
                    crop_btn = gr.Button("Apply Crop", size="sm")
                    crop_clear_btn = gr.Button("Clear Crop", size="sm")

            # ═══════ CENTER PANEL ═══════
            with gr.Column(scale=3, min_width=600):
                scene_image = gr.Image(
                    label="Scene View",
                    type="pil",
                    interactive=False,
                    height=600,
                )

                # Timeline playback
                with gr.Row():
                    btn_play = gr.Button("Play ▶", size="sm", min_width=60)
                    btn_stop = gr.Button("Stop ■", size="sm", min_width=60, variant="stop")
                    play_fps = gr.Slider(minimum=1, maximum=30, value=10, step=1,
                                         label="FPS", scale=1)

                _has_model = model_cache is not None and model_cache.available

                # Trajectory overlay controls — below canvas
                with gr.Row():
                    show_gt = gr.Checkbox(label="Show GT", value=True, scale=1)
                    show_det = gr.Checkbox(label="Show DET", value=False,
                                           interactive=_has_model, scale=1)
                    show_guided = gr.Checkbox(label="Show Guided", value=False,
                                              interactive=_has_model, scale=1)
                    hide_neighbors = gr.Checkbox(label="Hide Neighbors", value=False, scale=1)
                    show_rb_dist = gr.Checkbox(label="Road Border", value=False, scale=1)
                    show_nb_dist = gr.Checkbox(label="Neighbor Dist", value=False, scale=1)
                    if not _has_model:
                        gr.Markdown("*No model — pass `--model_path`*", scale=2)

                with gr.Accordion("Guidance Controls", open=False):
                    guidance_toggles = {}
                    guidance_scales = {}
                    with gr.Row():
                        for gname in ALL_GUIDANCE_NAMES[:5]:
                            with gr.Column(min_width=100):
                                guidance_toggles[gname] = gr.Checkbox(
                                    label=gname.replace("_following", "").replace("_", " ").title(),
                                    value=False, interactive=_has_model,
                                )
                                guidance_scales[gname] = gr.Slider(
                                    minimum=0.0, maximum=10.0, value=2.0, step=0.5,
                                    show_label=False, interactive=_has_model,
                                )
                    with gr.Row():
                        for gname in ALL_GUIDANCE_NAMES[5:]:
                            _min = -10.0 if gname == "lateral" else 0.0
                            with gr.Column(min_width=100):
                                guidance_toggles[gname] = gr.Checkbox(
                                    label=gname.replace("_following", "").replace("_", " ").title(),
                                    value=False, interactive=_has_model,
                                )
                                guidance_scales[gname] = gr.Slider(
                                    minimum=_min, maximum=10.0, value=2.0, step=0.5,
                                    show_label=False, interactive=_has_model,
                                )
                    with gr.Row():
                        guided_noise = gr.Slider(
                            minimum=0.0, maximum=5.0, value=0.0, step=0.1,
                            label="Noise", interactive=_has_model, scale=2,
                        )
                        guided_k = gr.Slider(
                            minimum=1, maximum=8, value=1, step=1,
                            label="K", interactive=_has_model, scale=1,
                        )
                        generate_guided_btn = gr.Button(
                            "Generate", variant="primary",
                            interactive=_has_model, scale=1,
                        )

                # Simulate controls — horizontal row
                with gr.Row():
                    sim_steps = gr.Number(label="Sim steps", value=80, precision=0,
                                          scale=1, min_width=80)
                    sim_mode = gr.Dropdown(
                        choices=["perfect", "mpc"], value="perfect",
                        label="Mode", scale=1, min_width=80,
                    )
                    sim_use_guidance = gr.Checkbox(
                        label="Apply guidance", value=False,
                        interactive=_has_model, scale=1,
                    )
                    sim_btn = gr.Button("Simulate", variant="primary", scale=1,
                                        interactive=_has_model)
                sim_status = gr.Markdown("")

            # ═══════ RIGHT PANEL ═══════
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Branch Tree")
                branch_dropdown = gr.Dropdown(
                    choices=list(tree.branches.keys()),
                    value=tree.active_branch,
                    label="Active Branch",
                    interactive=True,
                )
                branch_info = gr.Markdown(_branch_info_md(tree, tree.active_branch))
                with gr.Row():
                    fork_btn = gr.Button("Fork Here", size="sm", variant="primary")
                    delete_branch_btn = gr.Button("Delete Branch", size="sm", variant="stop")

                gr.Markdown("### Modifications")
                mods_display = gr.Markdown(
                    _modifications_md(tree, tree.active_branch),
                )
                obs_select = gr.Dropdown(
                    choices=[], value=None,
                    label="Select obstacle", interactive=True,
                    allow_custom_value=False,
                )
                with gr.Row():
                    remove_obs_btn = gr.Button("Remove", size="sm", variant="stop")
                gr.Markdown("#### Edit selected")
                with gr.Row():
                    edit_x = gr.Number(label="X", value=0, precision=1)
                    edit_y = gr.Number(label="Y", value=0, precision=1)
                with gr.Row():
                    edit_yaw = gr.Slider(minimum=-180, maximum=180, value=0, step=5, label="Yaw (deg)")
                with gr.Row():
                    edit_length = gr.Slider(minimum=1.0, maximum=15.0, value=4.5, step=0.1, label="Len")
                    edit_width = gr.Slider(minimum=0.5, maximum=3.0, value=1.8, step=0.1, label="Wid")
                edit_history = gr.Slider(minimum=0, maximum=30, value=30, step=1, label="History")
                apply_edit_btn = gr.Button("Apply Edit", size="sm", variant="primary")

                gr.Markdown("### Export")
                export_dir = gr.Textbox(label="Output Dir", placeholder="/path/to/export")
                export_btn = gr.Button("Export NPZs", variant="secondary")
                export_status = gr.Markdown("")

                save_status = gr.Markdown("")

        # ── Callbacks ──

        def _render(tree: SceneTree, step: int, view_r: float,
                    selected_obs: str | None,
                    preview_placement: ObstaclePlacement | None = None,
                    show_gt_val: bool = True,
                    det_traj: np.ndarray | None = None,
                    guided_trajs: list[np.ndarray] | None = None,
                    rb_dist: bool = False, nb_dist: bool = False,
                    hide_nb: bool = False):
            """Core render function: load NPZ at step, draw scene + obstacles."""
            branch = tree.branches[tree.active_branch]
            seq = tree.get_npz_sequence(tree.active_branch)
            if not seq:
                return _empty_image("No NPZ files found"), f"Step {step} / 0"

            step = max(0, min(step, len(seq) - 1))
            scene = from_npz(seq[step])

            # Apply ego shape override if NPZ didn't have it
            if tree.ego_shape:
                ego = scene.ego_agent
                if ego is not None:
                    wb, ln, wd = tree.ego_shape
                    ego.wheelbase = wb
                    ego.length = ln
                    ego.width = wd

            # Don't draw obstacle overlays on resimulated branches — they're
            # already baked into the NPZ as regular neighbors.
            if branch.npz_dir is not None:
                obstacles_at_step = []
            else:
                obstacles = tree.get_all_obstacles(tree.active_branch)
                obstacles_at_step = []
                for o in obstacles:
                    if o.timestep > step:
                        continue
                    if o.timestep != step:
                        # Transform from placement frame to current view frame
                        nx, ny, nyaw = _transform_point_between_steps(
                            seq, o.timestep, step, o.x, o.y, o.yaw_rad,
                        )
                        obstacles_at_step.append(ObstaclePlacement(
                            label=o.label, timestep=o.timestep,
                            x=nx, y=ny, yaw_deg=math.degrees(nyaw),
                            length=o.length, width=o.width,
                            history_steps=o.history_steps,
                        ))
                    else:
                        obstacles_at_step.append(o)

            if preview_placement is not None:
                obstacles_at_step = obstacles_at_step + [preview_placement]

            # GT future: first try NPZ field, then reconstruct from future steps
            gt_traj_render = None
            if show_gt_val:
                ego = scene.ego_agent
                if ego is not None and ego.future_trajectory is not None:
                    gt = ego.future_trajectory
                    if np.abs(gt).sum() > 1e-6:
                        gt_traj_render = gt
                # Reconstruct from future NPZ steps if NPZ field is zeros
                if gt_traj_render is None and len(seq) > step + 1:
                    gt_traj_render = _reconstruct_gt_from_sequence(seq, step, max_future=80)

            # Ego world pose for map border transform
            ego_wp = _recover_ego_world_pose(seq, step) if map_borders else None

            fig = render_scene_at_step(
                scene, obstacles_at_step, selected_obs,
                view_half=view_r, step_idx=step, total_steps=len(seq),
                gt_traj=gt_traj_render,
                det_traj=det_traj,
                guided_trajs=guided_trajs,
                show_rb_dist=rb_dist, show_nb_dist=nb_dist,
                hide_neighbors=hide_nb,
                map_border_polylines=map_borders, ego_world_pose=ego_wp,
            )
            img = _fig_to_pil(fig)
            info = f"Step **{step}** / **{len(seq) - 1}** | Branch: `{tree.active_branch}`"
            if branch.fork_timestep is not None:
                info += f" | Forked from parent step {branch.fork_timestep}"
            if branch.crop_range:
                info += f" | Crop: [{branch.crop_range[0]}, {branch.crop_range[1]}]"
            return img, info

        def _safe_step(step):
            if step is None:
                return 0
            try:
                return max(0, int(step))
            except (TypeError, ValueError):
                return 0

        def _get_npz_path(tree, step):
            seq = tree.get_npz_sequence(tree.active_branch)
            if not seq:
                return None
            step = max(0, min(_safe_step(step), len(seq) - 1))
            return seq[step]

        def _predict_det_with_obs(tree, step):
            npz_path = _get_npz_path(tree, step)
            if not npz_path:
                return None
            obs = _get_obstacles_at_step(tree, _safe_step(step))
            raw = model_cache.predict_det(npz_path, obstacles=obs or None)
            return _traj_cos_sin_to_xyh(raw)

        def on_render(tree, step, view_r, selected_obs, gt_on, det_on, guided_on,
                      hide_nb, rb_on, nb_on, det_cache, guided_cache):
            det_traj = None
            if det_on and model_cache and model_cache.available:
                det_traj = _predict_det_with_obs(tree, step)
                det_cache = det_traj
            elif det_on and det_cache is not None:
                det_traj = det_cache
            else:
                det_cache = None

            guided_list = guided_cache if guided_on else None

            img, info = _render(tree, _safe_step(step), view_r, selected_obs,
                                show_gt_val=gt_on, det_traj=det_traj,
                                guided_trajs=guided_list,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb)
            return img, info, det_cache, guided_cache

        def on_step_change(tree, step, view_r, selected_obs, gt_on, det_on, hide_nb, rb_on, nb_on,
                           guided_on, *g_args):
            s = _safe_step(step)
            det_traj, guided = _recompute_trajs(tree, s, det_on, guided_on, g_args or None)
            img, info = _render(tree, s, view_r, selected_obs,
                                show_gt_val=gt_on, det_traj=det_traj, guided_trajs=guided,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb)
            return img, info, s, det_traj, guided

        def _on_nav_impl(direction, tree, step, view_r, selected_obs, gt_on, det_on,
                         hide_nb, rb_on, nb_on, guided_on, *g_args):
            seq = tree.get_npz_sequence(tree.active_branch)
            max_s = max(0, len(seq) - 1) if seq else 0
            if direction == "first":
                s = 0
            elif direction == "prev":
                s = max(0, _safe_step(step) - 1)
            elif direction == "next":
                s = min(max_s, _safe_step(step) + 1)
            else:
                s = max_s
            det_traj, guided = _recompute_trajs(tree, s, det_on, guided_on, g_args or None)
            img, info = _render(tree, s, view_r, selected_obs,
                                show_gt_val=gt_on, det_traj=det_traj, guided_trajs=guided,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb)
            return img, info, s, det_traj, guided

        def on_preview(tree, step, view_r, selected_obs, x, y, yaw, length, width, gt_on, det_cache, guided_cache):
            preview = ObstaclePlacement(
                label="(preview)", timestep=_safe_step(step),
                x=round(float(x), 1), y=round(float(y), 1),
                yaw_deg=round(float(yaw) / 5) * 5,
                length=float(length), width=float(width),
            )
            img, info = _render(tree, _safe_step(step), view_r, selected_obs, preview,
                                show_gt_val=gt_on, det_traj=det_cache,
                                guided_trajs=guided_cache)
            return img, info

        def _obs_choices(tree):
            branch = tree.branches[tree.active_branch]
            return [o.label for o in branch.modifications]

        def _find_obs(tree, label):
            if not label:
                return None
            for o in tree.get_all_obstacles(tree.active_branch):
                if o.label == label:
                    return o
            return None

        def _get_obstacles_at_step(tree, step):
            obstacles = tree.get_all_obstacles(tree.active_branch)
            return [o for o in obstacles if o.timestep <= step]

        def _recompute_trajs(tree, step, det_on, guided_on=False,
                             guidance_args_tuple=None):
            """Recompute DET and guided trajectories if toggled on.

            When guided_on and guidance_args_tuple is provided, regenerates
            guided trajectories with the current guidance config.
            """
            det_traj = None
            guided = None
            if not (model_cache and model_cache.available):
                return det_traj, guided
            obs = _get_obstacles_at_step(tree, _safe_step(step))
            npz_path = _get_npz_path(tree, step)
            if not npz_path:
                return det_traj, guided
            if det_on:
                raw = model_cache.predict_det(npz_path, obstacles=obs or None)
                det_traj = _traj_cos_sin_to_xyh(raw)
            if guided_on and guidance_args_tuple:
                cfgs = []
                for gi, gname in enumerate(ALL_GUIDANCE_NAMES):
                    enabled = guidance_args_tuple[gi * 2]
                    scale = guidance_args_tuple[gi * 2 + 1]
                    if enabled:
                        cfgs.append((gname, float(scale)))
                noise = float(guidance_args_tuple[-2])
                k = int(guidance_args_tuple[-1])
                raw_g = model_cache.predict_guided(
                    npz_path, cfgs, noise_scale=noise, n_samples=max(1, k),
                    obstacles=obs or None,
                )
                guided = [_traj_cos_sin_to_xyh(raw_g[j]) for j in range(raw_g.shape[0])]
            return det_traj, guided

        def on_place(tree, step, view_r, x, y, yaw, length, width, history,
                     gt_on, det_on, guided_on, hide_nb, rb_on, nb_on, *g_args):
            label = tree.next_obstacle_label(tree.active_branch)
            placement = ObstaclePlacement(
                label=label, timestep=_safe_step(step),
                x=float(x), y=float(y), yaw_deg=float(yaw),
                length=float(length), width=float(width),
                history_steps=int(history),
            )
            tree.add_obstacle(tree.active_branch, placement)
            s = _safe_step(step)
            det_traj, guided = _recompute_trajs(tree, s, det_on, guided_on, g_args or None)
            img, info = _render(tree, s, view_r, label, show_gt_val=gt_on,
                                det_traj=det_traj, guided_trajs=guided,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb)
            mods = _modifications_md(tree, tree.active_branch)
            choices = _obs_choices(tree)
            return (tree, img, info, mods, label, det_traj, guided,
                    gr.update(choices=choices, value=label))

        def on_select_obstacle(tree, label, step, view_r, gt_on, det_on, det_cache, guided_cache):
            obs = _find_obs(tree, label)
            s = _safe_step(step)
            img, info = _render(tree, s, view_r, label, show_gt_val=gt_on,
                                det_traj=det_cache, guided_trajs=guided_cache)
            if obs:
                return (img, info, label,
                        obs.x, obs.y, obs.yaw_deg, obs.length, obs.width,
                        getattr(obs, "history_steps", 30))
            return (img, info, label,
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update())

        def on_remove_obstacle(tree, label, step, view_r, gt_on, det_on,
                               guided_on, hide_nb, rb_on, nb_on, *g_args):
            if label:
                tree.remove_obstacle(tree.active_branch, label.strip())
            s = _safe_step(step)
            det_traj, guided = _recompute_trajs(tree, s, det_on, guided_on, g_args or None)
            img, info = _render(tree, s, view_r, None, show_gt_val=gt_on,
                                det_traj=det_traj, guided_trajs=guided,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb)
            mods = _modifications_md(tree, tree.active_branch)
            choices = _obs_choices(tree)
            return (tree, img, info, mods, None,
                    gr.update(choices=choices, value=None), det_traj, guided)

        def on_apply_edit(tree, label, step, view_r, gt_on, det_on, guided_on,
                          hide_nb, rb_on, nb_on, x, y, yaw, length, width, history, *g_args):
            if not label:
                return (tree, gr.update(), "No obstacle selected",
                        gr.update(), gr.update(), gr.update(), gr.update())
            branch = tree.branches[tree.active_branch]
            for i, o in enumerate(branch.modifications):
                if o.label == label:
                    branch.modifications[i] = ObstaclePlacement(
                        label=label, timestep=o.timestep,
                        x=round(float(x), 1), y=round(float(y), 1),
                        yaw_deg=round(float(yaw) / 5) * 5,
                        length=float(length), width=float(width),
                        history_steps=int(history),
                    )
                    break
            s = _safe_step(step)
            det_traj, guided = _recompute_trajs(tree, s, det_on, guided_on, g_args or None)
            img, info = _render(tree, s, view_r, label, show_gt_val=gt_on,
                                det_traj=det_traj, guided_trajs=guided,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb)
            mods = _modifications_md(tree, tree.active_branch)
            return tree, img, info, mods, label, det_traj, guided

        def on_generate_guided(tree, step, gt_on, view_r, selected_obs,
                               noise, k, det_on, det_cache, *guidance_args):
            if model_cache is None or not model_cache.available:
                return gr.update(), "No model loaded", det_cache, None

            # Parse guidance toggles/scales from *guidance_args
            cfgs = []
            for i, gname in enumerate(ALL_GUIDANCE_NAMES):
                enabled = guidance_args[i * 2]
                scale = guidance_args[i * 2 + 1]
                if enabled:
                    cfgs.append((gname, float(scale)))

            npz_path = _get_npz_path(tree, step)
            if npz_path is None:
                return gr.update(), "No NPZ at this step", det_cache, None

            obs = _get_obstacles_at_step(tree, _safe_step(step))

            # DET trajectory (recompute if toggled on)
            det_traj = None
            if det_on:
                if det_cache is not None:
                    det_traj = det_cache
                else:
                    raw = model_cache.predict_det(npz_path, obstacles=obs or None)
                    det_traj = _traj_cos_sin_to_xyh(raw)
                    det_cache = det_traj

            # Guided trajectories
            raw_guided = model_cache.predict_guided(
                npz_path, cfgs, noise_scale=float(noise), n_samples=int(k),
                obstacles=obs or None,
            )
            guided_list = [_traj_cos_sin_to_xyh(raw_guided[i]) for i in range(raw_guided.shape[0])]

            img, info = _render(tree, _safe_step(step), view_r, selected_obs,
                                show_gt_val=gt_on, det_traj=det_traj,
                                guided_trajs=guided_list)
            return img, info, det_cache, guided_list

        def on_branch_change(tree, branch_id, step, view_r, selected_obs, gt_on):
            if branch_id not in tree.branches:
                return (tree, gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), None, None)
            if tree.active_branch == branch_id:
                return (tree, gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update())
            tree.active_branch = branch_id
            seq = tree.get_npz_sequence(branch_id)
            max_step = max(0, len(seq) - 1)
            img, info = _render(tree, 0, view_r, None, show_gt_val=gt_on)
            b_info = _branch_info_md(tree, branch_id)
            mods = _modifications_md(tree, branch_id)
            return (tree, img, info, b_info, mods,
                    gr.update(maximum=max_step, value=0), None, None, None)

        def on_fork(tree, step, view_r, gt_on):
            new_id = tree.fork_branch(tree.active_branch, _safe_step(step))
            tree.active_branch = new_id
            choices = list(tree.branches.keys())
            seq = tree.get_npz_sequence(new_id)
            max_step = max(0, len(seq) - 1)
            img, info = _render(tree, 0, view_r, None, show_gt_val=gt_on)
            b_info = _branch_info_md(tree, new_id)
            mods = _modifications_md(tree, new_id)
            return (tree, img, info, b_info, mods,
                    gr.update(choices=choices, value=new_id),
                    gr.update(maximum=max_step, value=0), None, None, None)

        def on_delete_branch(tree, view_r, gt_on):
            if tree.active_branch == "root":
                return (tree, gr.update(), "Cannot delete root", gr.update(),
                        gr.update(), gr.update(), gr.update(), None, None, None)
            tree.delete_branch(tree.active_branch)
            tree.active_branch = "root"
            choices = list(tree.branches.keys())
            seq = tree.get_npz_sequence("root")
            max_step = max(0, len(seq) - 1)
            img, info = _render(tree, 0, view_r, None, show_gt_val=gt_on)
            b_info = _branch_info_md(tree, "root")
            mods = _modifications_md(tree, "root")
            return (tree, img, info, b_info, mods,
                    gr.update(choices=choices, value="root"),
                    gr.update(maximum=max_step, value=0), None, None, None)

        def on_load_dir(tree, npz_dir, view_r, gt_on):
            new_tree = SceneTree.create_from_npz_dir(npz_dir)
            seq = new_tree.get_npz_sequence("root")
            max_step = max(0, len(seq) - 1)
            img, info = _render(new_tree, 0, view_r, None, show_gt_val=gt_on)
            choices = list(new_tree.branches.keys())
            b_info = _branch_info_md(new_tree, "root")
            mods = _modifications_md(new_tree, "root")
            return (new_tree, img, info, b_info, mods,
                    gr.update(choices=choices, value="root"),
                    gr.update(maximum=max_step, value=0), None, None, None)

        def on_load_tree(path, view_r, gt_on):
            loaded = SceneTree.load(path)
            seq = loaded.get_npz_sequence(loaded.active_branch)
            max_step = max(0, len(seq) - 1)
            img, info = _render(loaded, 0, view_r, None, show_gt_val=gt_on)
            choices = list(loaded.branches.keys())
            b_info = _branch_info_md(loaded, loaded.active_branch)
            mods = _modifications_md(loaded, loaded.active_branch)
            return (loaded, img, info, b_info, mods,
                    gr.update(choices=choices, value=loaded.active_branch),
                    gr.update(maximum=max_step, value=0), None, None, None)

        def on_save_tree(tree, path):
            if not path:
                return "No path specified"
            tree.save(path)
            return f"Saved to `{path}`"

        def on_crop(tree, step, view_r, start, end, selected_obs, gt_on):
            tree.set_crop(tree.active_branch, int(start), int(end))
            seq = tree.get_npz_sequence(tree.active_branch)
            max_step = max(0, len(seq) - 1)
            s = min(_safe_step(step), max_step)
            img, info = _render(tree, s, view_r, selected_obs, show_gt_val=gt_on)
            b_info = _branch_info_md(tree, tree.active_branch)
            return tree, img, info, b_info, gr.update(maximum=max_step, value=s)

        def on_crop_clear(tree, step, view_r, selected_obs, gt_on):
            tree.clear_crop(tree.active_branch)
            seq = tree.get_npz_sequence(tree.active_branch)
            max_step = max(0, len(seq) - 1)
            s = min(_safe_step(step), max_step)
            img, info = _render(tree, s, view_r, selected_obs, show_gt_val=gt_on)
            b_info = _branch_info_md(tree, tree.active_branch)
            return tree, img, info, b_info, gr.update(maximum=max_step, value=s)

        # ── Wire up events ──
        # Guidance toggle+scale inputs for recomputation, plus noise and K at end
        _g_inputs = ([v for gname in ALL_GUIDANCE_NAMES
                       for v in (guidance_toggles[gname], guidance_scales[gname])]
                     + [guided_noise, guided_k])
        _overlay_inputs = [show_guided, hide_neighbors, show_rb_dist, show_nb_dist]

        nav_inputs = ([tree_state, step_slider, view_half, selected_obstacle_state,
                       show_gt, show_det, hide_neighbors, show_rb_dist, show_nb_dist,
                       show_guided] + _g_inputs)
        nav_outputs = [scene_image, step_info, step_slider, det_traj_state, guided_trajs_state]

        step_slider.release(
            on_step_change, nav_inputs, nav_outputs,
        )

        def on_step_jump(tree, jump_val, view_r, selected_obs, gt_on, det_on,
                         hide_nb, rb_on, nb_on, guided_on, *g_args):
            s = _safe_step(jump_val)
            seq = tree.get_npz_sequence(tree.active_branch)
            s = min(s, max(0, len(seq) - 1)) if seq else 0
            det_traj, guided = _recompute_trajs(tree, s, det_on, guided_on, g_args or None)
            img, info = _render(tree, s, view_r, selected_obs,
                                show_gt_val=gt_on, det_traj=det_traj, guided_trajs=guided,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb)
            return img, info, s, det_traj, guided

        step_jump_btn.click(
            on_step_jump,
            [tree_state, step_jump_input, view_half, selected_obstacle_state,
             show_gt, show_det, hide_neighbors, show_rb_dist, show_nb_dist,
             show_guided] + _g_inputs,
            nav_outputs,
        )
        _render_trigger_inputs = [tree_state, step_slider, view_half, selected_obstacle_state,
                                   show_gt, show_det, show_guided,
                                   hide_neighbors, show_rb_dist, show_nb_dist,
                                   det_traj_state, guided_trajs_state]
        _render_trigger_outputs = [scene_image, step_info, det_traj_state, guided_trajs_state]

        for _trigger in [view_half, show_gt, show_det, show_guided,
                         hide_neighbors, show_rb_dist, show_nb_dist]:
            _trigger.change(on_render, _render_trigger_inputs, _render_trigger_outputs)

        for direction, btn in [("first", btn_first), ("prev", btn_prev),
                                ("next", btn_next), ("last", btn_last)]:
            btn.click(
                lambda *args, d=direction: _on_nav_impl(d, *args),
                nav_inputs, nav_outputs,
            )

        preview_btn.click(
            on_preview,
            [tree_state, step_slider, view_half, selected_obstacle_state,
             obs_x, obs_y, obs_yaw, obs_length, obs_width, show_gt,
             det_traj_state, guided_trajs_state],
            [scene_image, step_info],
        )

        place_btn.click(
            on_place,
            [tree_state, step_slider, view_half,
             obs_x, obs_y, obs_yaw, obs_length, obs_width, obs_history,
             show_gt, show_det] + _overlay_inputs + _g_inputs,
            [tree_state, scene_image, step_info, mods_display,
             selected_obstacle_state, det_traj_state, guided_trajs_state,
             obs_select],
        )

        obs_select.change(
            on_select_obstacle,
            [tree_state, obs_select, step_slider, view_half, show_gt,
             show_det, det_traj_state, guided_trajs_state],
            [scene_image, step_info, selected_obstacle_state,
             edit_x, edit_y, edit_yaw, edit_length, edit_width, edit_history],
        )

        remove_obs_btn.click(
            on_remove_obstacle,
            [tree_state, obs_select, step_slider, view_half,
             show_gt, show_det] + _overlay_inputs + _g_inputs,
            [tree_state, scene_image, step_info, mods_display,
             selected_obstacle_state, obs_select, det_traj_state, guided_trajs_state],
        )

        apply_edit_btn.click(
            on_apply_edit,
            [tree_state, obs_select, step_slider, view_half,
             show_gt, show_det] + _overlay_inputs +
            [edit_x, edit_y, edit_yaw, edit_length, edit_width, edit_history] + _g_inputs,
            [tree_state, scene_image, step_info, mods_display, selected_obstacle_state,
             det_traj_state, guided_trajs_state],
        )

        # Guidance generation button
        guidance_btn_inputs = ([tree_state, step_slider, show_gt, view_half,
                                selected_obstacle_state, guided_noise, guided_k,
                                show_det, det_traj_state]
                               + [v for gname in ALL_GUIDANCE_NAMES
                                  for v in (guidance_toggles[gname], guidance_scales[gname])])
        generate_guided_btn.click(
            on_generate_guided,
            guidance_btn_inputs,
            [scene_image, step_info, det_traj_state, guided_trajs_state],
        )

        _branch_switch_outputs = [tree_state, scene_image, step_info, branch_info, mods_display,
                                  step_slider, selected_obstacle_state, det_traj_state, guided_trajs_state]
        _full_switch_outputs = [tree_state, scene_image, step_info, branch_info, mods_display,
                                branch_dropdown, step_slider, selected_obstacle_state,
                                det_traj_state, guided_trajs_state]

        branch_dropdown.change(
            on_branch_change,
            [tree_state, branch_dropdown, step_slider, view_half, selected_obstacle_state, show_gt],
            _branch_switch_outputs,
        )

        fork_btn.click(
            on_fork, [tree_state, step_slider, view_half, show_gt],
            _full_switch_outputs,
        )

        delete_branch_btn.click(
            on_delete_branch, [tree_state, view_half, show_gt],
            _full_switch_outputs,
        )

        load_dir_btn.click(
            on_load_dir, [tree_state, load_dir_input, view_half, show_gt],
            _full_switch_outputs,
        )

        load_tree_btn.click(
            on_load_tree, [load_tree_input, view_half, show_gt],
            _full_switch_outputs,
        )

        save_tree_btn.click(
            on_save_tree, [tree_state, load_tree_input],
            [save_status],
        )

        crop_btn.click(
            on_crop,
            [tree_state, step_slider, view_half, crop_start, crop_end, selected_obstacle_state, show_gt],
            [tree_state, scene_image, step_info, branch_info, step_slider],
        )

        crop_clear_btn.click(
            on_crop_clear,
            [tree_state, step_slider, view_half, selected_obstacle_state, show_gt],
            [tree_state, scene_image, step_info, branch_info, step_slider],
        )

        # Simulate N steps — closed-loop forward simulation
        def on_simulate(tree, step, n_steps, advance_mode, use_guidance,
                        gt_on, view_r, *guidance_args, progress=gr.Progress()):
            if model_cache is None or not model_cache.available:
                return (tree, gr.update(), "No model loaded — pass `--model_path`",
                        gr.update(), gr.update(), gr.update(), gr.update(), None, None)

            branch = tree.branches[tree.active_branch]
            # Get source scene from parent (not from any previous resim output)
            saved_npz_dir = branch.npz_dir
            branch.npz_dir = None  # temporarily clear so get_npz_sequence uses parent
            seq = tree.get_npz_sequence(tree.active_branch)
            branch.npz_dir = saved_npz_dir  # restore
            if not seq:
                return (tree, gr.update(), "No NPZ sequence",
                        gr.update(), gr.update(), gr.update(), gr.update(), None, None)

            s = _safe_step(step)
            n = max(1, int(n_steps))
            npz_path = seq[min(s, len(seq) - 1)]

            # Create clean output dir for this branch's resim (remove stale files)
            out_dir = Path(tree.base_npz_dir).parent / f"branch_{tree.active_branch}_resim"
            if out_dir.exists():
                for old_f in out_dir.glob("*.npz"):
                    old_f.unlink()
            out_dir.mkdir(parents=True, exist_ok=True)

            progress(0, desc="Loading model...")
            model_cache._ensure_loaded()

            from scenario_generation.npz_loader import from_npz as _from_npz
            from scenario_generation.tensor_converter import MapTensorCache, dump_step_npz
            from scenario_generation.simulate import (
                _predict_batch,
                advance_scene,
                advance_scene_mpc,
            )

            scene = _from_npz(npz_path)

            # Inject placed obstacles as stationary agents
            obstacles = tree.get_all_obstacles(tree.active_branch)
            obs_at_step = [o for o in obstacles if o.timestep <= s]
            from scenario_generation.scene_context import Agent, AgentType
            for obs in obs_at_step:
                T_PAST = 31
                history = np.tile([obs.x, obs.y, obs.yaw_rad], (T_PAST, 1)).astype(np.float32)
                velocities = np.zeros((T_PAST, 2), dtype=np.float32)
                h = getattr(obs, "history_steps", 30)
                agent = Agent(
                    id=f"placed_{obs.label}",
                    agent_type=AgentType.VEHICLE,
                    length=obs.length, width=obs.width,
                    wheelbase=obs.length * 0.65,
                    past_trajectory=history,
                    past_velocities=velocities,
                    age_steps=min(h, T_PAST - 1),
                )
                scene.agents.append(agent)

            # If we have the map builder, rebuild line_strings with proper
            # border flags (psim NPZs lack them) so the model sees road borders
            _has_builder = map_builder is not None
            if _has_builder:
                # Get initial ego world pose from sidecar JSON
                _init_ego_wp = _recover_ego_world_pose(seq, min(s, len(seq) - 1))

            map_cache = MapTensorCache(scene.map_data)
            device = model_cache._device
            model = model_cache._model
            model_args = model_cache._model_args
            if advance_mode == "mpc":
                _mpc_trackers: dict = {}
                def advance_fn(sc, pr):
                    advance_scene_mpc(sc, pr, _mpc_trackers)
            else:
                advance_fn = advance_scene

            placed_ids = {f"placed_{o.label}" for o in obs_at_step}

            # Optionally apply guidance during simulation
            _orig_guidance_fn = model.decoder._guidance_fn
            _orig_guidance_scale = model.decoder._guidance_scale
            if use_guidance and guidance_args:
                from diffusion_planner.model.guidance.composer import GuidanceComposer
                from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig
                fns = []
                for gi, gname in enumerate(ALL_GUIDANCE_NAMES):
                    enabled = guidance_args[gi * 2]
                    scale = guidance_args[gi * 2 + 1]
                    if enabled:
                        params = {}
                        if gname == "speed":
                            ego = scene.ego_agent
                            if ego is not None:
                                spd = float(np.linalg.norm(ego.current_velocity))
                                params["v_high"] = spd * 1.2
                                params["v_low"] = max(0.0, spd * 0.5)
                        fns.append(GuidanceConfig(name=gname, enabled=True,
                                                   scale=float(scale), params=params))
                if fns:
                    set_cfg = GuidanceSetConfig(functions=fns, global_scale=1.0)
                    composer = GuidanceComposer(set_cfg)
                    model.decoder._guidance_fn = composer
                    model.decoder._guidance_scale = 1.0

            # Track ego world pose for sidecar JSONs
            import json as _json
            if _has_builder and _init_ego_wp is not None:
                _ego_world = _init_ego_wp.copy()
            else:
                _ego_world = None

            progress(0, desc=f"Simulating 0/{n} steps...")
            try:
                for i in range(n):
                    # Refresh map data from the lanelet2 builder (same as replay.py
                    # line 1824-1832). This gives the model correct lanes, road
                    # borders, stop lines, and polygons in the current ego frame.
                    if _has_builder and _ego_world is not None:
                        ego_xy_w = _ego_world[:2].astype(np.float32)
                        ll_ids = list(map_builder.closest_lanelets(ego_xy_w, 140))
                        scene.map_data = map_builder._build_map_data(
                            ll_ids, center_xy=ego_xy_w)
                        map_cache = MapTensorCache(scene.map_data)

                    ids_to_predict = [a.id for a in scene.agents if a.id not in placed_ids]
                    preds = _predict_batch(model, model_args, scene, ids_to_predict, device,
                                           map_cache=map_cache)

                    # Save ego prediction into NPZ as ego_agent_future
                    ego_id = scene.ego_agent_id
                    data = dump_step_npz(scene, map_cache, future_len=80)
                    if ego_id in preds:
                        ego_pred = preds[ego_id]
                        heading = np.arctan2(ego_pred[:, 3], ego_pred[:, 2])
                        data["ego_agent_future"] = np.column_stack(
                            [ego_pred[:, :2], heading]
                        ).astype(np.float32)

                    # Save sidecar JSON with ego world pose
                    if _ego_world is not None:
                        yaw_w = float(_ego_world[2])
                        sidecar = {
                            "x": float(_ego_world[0]), "y": float(_ego_world[1]),
                            "qz": math.sin(yaw_w / 2), "qw": math.cos(yaw_w / 2),
                        }
                        (out_dir / f"replay_step_{i:04d}.json").write_text(_json.dumps(sidecar))

                    advance_fn(scene, preds)
                    np.savez(out_dir / f"replay_step_{i:04d}.npz", **data)

                    # Update ego world pose: world = init_world + R(init_yaw) @ scene_pos
                    if _ego_world is not None and _init_ego_wp is not None:
                        ego_new = scene.ego_agent
                        iy = float(_init_ego_wp[2])
                        ci, si = math.cos(iy), math.sin(iy)
                        ep = ego_new.current_position
                        _ego_world[0] = _init_ego_wp[0] + ci * ep[0] - si * ep[1]
                        _ego_world[1] = _init_ego_wp[1] + si * ep[0] + ci * ep[1]
                        _ego_world[2] = iy + float(ego_new.current_heading)
                    progress((i + 1) / n, desc=f"Simulating {i+1}/{n} steps...")
            finally:
                model.decoder._guidance_fn = _orig_guidance_fn
                model.decoder._guidance_scale = _orig_guidance_scale

            # Update branch with resim output
            branch.npz_dir = str(out_dir)
            branch.resim_steps = n
            branch.resim_advance_mode = advance_mode
            branch.resim_model_path = model_cache._model_path

            new_seq = tree.get_npz_sequence(tree.active_branch)
            max_step = max(0, len(new_seq) - 1)
            img, info = _render(tree, 0, view_r, None, show_gt_val=gt_on)
            b_info = _branch_info_md(tree, tree.active_branch)
            mods = _modifications_md(tree, tree.active_branch)
            status = f"Simulated **{n}** steps ({advance_mode}). Output: `{out_dir}`"
            return (tree, img, status, b_info, mods,
                    gr.update(maximum=max_step, value=0), info, None, None)

        _sim_inputs = ([tree_state, step_slider, sim_steps, sim_mode, sim_use_guidance,
                        show_gt, view_half]
                       + [v for gname in ALL_GUIDANCE_NAMES
                          for v in (guidance_toggles[gname], guidance_scales[gname])])
        sim_btn.click(
            on_simulate,
            _sim_inputs,
            [tree_state, scene_image, sim_status, branch_info, mods_display,
             step_slider, step_info, det_traj_state, guided_trajs_state],
        )

        # Play button — pre-renders frames as PIL images for smooth playback
        def on_play(tree, step, view_r, gt_on, hide_nb, rb_on, nb_on, fps):
            import time
            seq = tree.get_npz_sequence(tree.active_branch)
            if not seq:
                return
            s = _safe_step(step)
            max_s = len(seq) - 1
            interval = 1.0 / max(1, int(fps))
            branch = tree.branches[tree.active_branch]
            is_resimulated = branch.npz_dir is not None
            raw_obstacles = tree.get_all_obstacles(tree.active_branch) if not is_resimulated else []
            while s <= max_s:
                t0 = time.monotonic()
                scene = from_npz(seq[s])
                if tree.ego_shape:
                    ego = scene.ego_agent
                    if ego:
                        ego.wheelbase, ego.length, ego.width = tree.ego_shape
                obs_at_step = []
                for o in raw_obstacles:
                    if o.timestep > s:
                        continue
                    if o.timestep != s:
                        nx, ny, nyaw = _transform_point_between_steps(
                            seq, o.timestep, s, o.x, o.y, o.yaw_rad,
                        )
                        obs_at_step.append(ObstaclePlacement(
                            label=o.label, timestep=o.timestep,
                            x=nx, y=ny, yaw_deg=math.degrees(nyaw),
                            length=o.length, width=o.width,
                            history_steps=o.history_steps,
                        ))
                    else:
                        obs_at_step.append(o)
                gt_traj_r = None
                if gt_on:
                    ego = scene.ego_agent
                    if ego and ego.future_trajectory is not None and np.abs(ego.future_trajectory).sum() > 1e-6:
                        gt_traj_r = ego.future_trajectory
                    if gt_traj_r is None and len(seq) > s + 1:
                        gt_traj_r = _reconstruct_gt_from_sequence(seq, s, max_future=80)
                ego_wp = _recover_ego_world_pose(seq, s) if map_borders else None
                fig = render_scene_at_step(
                    scene, obs_at_step, None,
                    view_half=view_r, step_idx=s, total_steps=len(seq),
                    gt_traj=gt_traj_r,
                    show_rb_dist=rb_on, show_nb_dist=nb_on,
                    hide_neighbors=hide_nb,
                    map_border_polylines=map_borders, ego_world_pose=ego_wp,
                )
                img = _fig_to_pil(fig)
                info = f"Step **{s}** / **{max_s}** | Branch: `{tree.active_branch}` | ▶ Playing"
                yield img, info, s
                elapsed = time.monotonic() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                s += 1

        _play_event = btn_play.click(
            on_play,
            [tree_state, step_slider, view_half, show_gt,
             hide_neighbors, show_rb_dist, show_nb_dist, play_fps],
            [scene_image, step_info, step_slider],
        )
        btn_stop.click(None, None, None, cancels=[_play_event])

        # Initial render
        demo.load(on_render, _render_trigger_inputs, _render_trigger_outputs)

    return demo


# ── Markdown helpers ──


def _branch_info_md(tree: SceneTree, branch_id: str) -> str:
    branch = tree.branches.get(branch_id)
    if branch is None:
        return "Branch not found"
    seq = tree.get_npz_sequence(branch_id)
    lines = [
        f"**Branch:** `{branch_id}`",
        f"**Parent:** `{branch.parent_id or 'none'}`",
        f"**Fork step:** {branch.fork_timestep if branch.fork_timestep is not None else 'N/A'}",
        f"**Steps:** {len(seq)}",
        f"**Crop:** {branch.crop_range or 'none'}",
        f"**Obstacles:** {len(branch.modifications)}",
        f"**Children:** {', '.join(tree.get_children(branch_id)) or 'none'}",
    ]
    if branch.resim_steps is not None:
        lines.append(f"**Resim:** {branch.resim_steps} steps ({branch.resim_advance_mode})")
    return "\n\n".join(lines)


def _modifications_md(tree: SceneTree, branch_id: str) -> str:
    branch = tree.branches.get(branch_id)
    if branch is None:
        return ""
    if not branch.modifications:
        return "*No obstacles placed in this branch.*"
    lines = ["| Label | Step | X | Y | Yaw | Size |",
             "|-------|------|---|---|-----|------|"]
    for o in branch.modifications:
        lines.append(
            f"| `{o.label}` | {o.timestep} | {o.x:.1f} | {o.y:.1f} "
            f"| {o.yaw_deg:.0f}° | {o.length}×{o.width} |"
        )
    inherited = [m for m in tree.get_all_obstacles(branch_id)
                 if m not in branch.modifications]
    if inherited:
        lines.append("")
        lines.append("**Inherited:**")
        for o in inherited:
            lines.append(f"- `{o.label}` @ step {o.timestep} ({o.x:.1f}, {o.y:.1f})")
    return "\n".join(lines)


def _empty_image(text: str = "No scene loaded"):
    """Create a placeholder image."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor("#f0f0f0")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=16, color="#888")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    img = _fig_to_pil(fig)
    return img


def main():
    parser = argparse.ArgumentParser(description="Scene Branch Editor")
    parser.add_argument("--npz_dir", type=str, required=True,
                        help="Path to replay NPZ directory")
    parser.add_argument("--tree_json", type=str, default=None,
                        help="Load existing scene tree JSON")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint (for inference)")
    parser.add_argument("--reward_config", type=str, default=None,
                        help="Path to reward config JSON (for overlays)")
    parser.add_argument("--ego_shape", type=str, default=None,
                        help="Ego wheelbase,length,width (e.g. '4.76,7.24,2.29' for a bus)")
    parser.add_argument("--map_path", type=str, default=None,
                        help="Path to lanelet2 .osm map (for road border overlays)")
    parser.add_argument("--port", type=int, default=7870)
    args = parser.parse_args()

    ego_shape_override = None
    if args.ego_shape:
        parts = [float(x) for x in args.ego_shape.split(",")]
        if len(parts) == 3:
            ego_shape_override = tuple(parts)

    if args.tree_json:
        tree = SceneTree.load(args.tree_json)
    else:
        tree = SceneTree.create_from_npz_dir(args.npz_dir)

    if ego_shape_override:
        tree.ego_shape = ego_shape_override

    mc = _ModelCache(args.model_path) if args.model_path else None

    # Load road border polylines from lanelet2 map if provided
    map_border_polylines = None
    builder = None
    if args.map_path:
        try:
            from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
            builder = LaneletSceneBuilder(args.map_path)
            map_border_polylines = builder.road_border_polylines()
            print(f"Loaded {len(map_border_polylines)} road border polylines from map")
        except Exception as e:
            print(f"Warning: could not load map borders: {e}")

    demo = build_interface(tree, model_cache=mc, map_borders=map_border_polylines,
                           map_builder=builder)
    demo.launch(server_name="0.0.0.0", server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
