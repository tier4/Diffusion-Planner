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
_PLACED_MOVING_COLOR = "#cc44ff"
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
    def predict_det(self, npz_path: str, obstacles: list | None = None,
                    zero_neighbors: bool = False,
                    ego_shape_override: tuple[float, ...] | None = None,
                    return_neighbor_preds: bool = False,
                    ) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
        """Run deterministic inference.

        Returns (80, 4) [x,y,cos_h,sin_h] or, when return_neighbor_preds=True,
        a tuple of (ego (80,4), neighbors (N,80,4)).
        """
        self._ensure_loaded()
        data = self._load_npz(npz_path, obstacles=obstacles,
                              zero_neighbors=zero_neighbors,
                              ego_shape_override=ego_shape_override)

        P = 1 + self._model_args.predicted_neighbor_num
        future_len = self._model_args.future_len
        ego_current = data["ego_current_state"][:, :4]
        neighbors_current = data["neighbor_agents_past"][:, :P - 1, -1, :4]
        current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)
        xT = current_states[:, :, None, :].expand(-1, -1, future_len + 1, -1).clone()
        data["sampled_trajectories"] = xT

        _, decoder_output = self._model(data)
        pred = decoder_output["prediction"]  # (1, P, T, 4)
        ego_traj = pred[0, 0].cpu().numpy()
        if return_neighbor_preds:
            nb_preds = pred[0, 1:].cpu().numpy()  # (P-1, T, 4)
            return ego_traj, nb_preds
        return ego_traj

    @torch.no_grad()
    def predict_guided(
        self, npz_path: str, guidance_cfgs: list[tuple[str, float]],
        noise_scale: float = 1.0, n_samples: int = 1,
        obstacles: list | None = None,
        zero_neighbors: bool = False,
        ego_shape_override: tuple[float, ...] | None = None,
        anchor_index: int = 0,
        anchor_path: str | None = None,
    ) -> np.ndarray | None:
        """Run guided inference. Returns (n_samples, 80, 4)."""
        self._ensure_loaded()
        from diffusion_planner.model.guidance.composer import GuidanceComposer
        from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig

        from guidance_gui.generate_samples import generate_samples

        data = self._load_npz(npz_path, obstacles=obstacles,
                              zero_neighbors=zero_neighbors,
                              ego_shape_override=ego_shape_override)

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
            if name == "anchor_following" and anchor_path:
                params["prototypes_path"] = anchor_path
                params["anchor_index"] = int(anchor_index)
            fns.append(GuidanceConfig(name=name, enabled=True, scale=scale, params=params))

        composer = None
        if fns:
            set_cfg = GuidanceSetConfig(functions=fns, global_scale=1.0)
            composer = GuidanceComposer(set_cfg)

        if not fns and noise_scale == 0.0:
            return det_raw

        return generate_samples(
            self._model, self._model_args, data,
            noise_scale=noise_scale, n_samples=n_samples,
            composer=composer, device=self._device,
        )

    def _load_npz(self, npz_path: str, obstacles: list | None = None,
                  zero_neighbors: bool = False,
                  ego_shape_override: tuple[float, ...] | None = None,
                  ) -> dict[str, torch.Tensor]:
        from preference_optimization.utils import load_npz_data
        data = load_npz_data(npz_path, self._device,
                             ego_shape_override=ego_shape_override)
        pnn = self._model_args.predicted_neighbor_num
        if zero_neighbors:
            for k in ("neighbor_agents_past", "neighbor_agents_future"):
                if k in data:
                    data[k] = torch.zeros_like(data[k])
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

    DT = 0.1
    new_rows = []
    for obs in obstacles:
        cos_h = math.cos(obs.yaw_rad)
        sin_h = math.sin(obs.yaw_rad)
        spd = getattr(obs, "speed", 0.0) if getattr(obs, "is_moving", False) else 0.0
        vx = spd * cos_h
        vy = spd * sin_h
        row = torch.zeros(T, F, dtype=torch.float32, device=device)
        for t in range(T):
            backward = (T - 1 - t) * spd * DT
            row[t, 0] = obs.x - backward * cos_h
            row[t, 1] = obs.y - backward * sin_h
            row[t, 2] = cos_h
            row[t, 3] = sin_h
            row[t, 4] = vx
            row[t, 5] = vy
            row[t, 6] = obs.width
            row[t, 7] = obs.length
            row[t, 8] = 1.0  # vehicle
        hist = getattr(obs, "history_steps", 30)
        n_valid = min(hist + 1, T)
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


def _obb_corners_from_placement(obs: ObstaclePlacement) -> np.ndarray:
    from scenario_generation.gui.lanelet_scene_builder import _obb_corners
    return _obb_corners(obs.x, obs.y, obs.yaw_rad, obs.length, obs.width)


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
    show_rb_dist: bool = True,
    show_nb_dist: bool = True,
    hide_neighbors: bool = False,
    dim_neighbors: bool = False,
    map_border_polylines: list[np.ndarray] | None = None,
    ego_world_pose: np.ndarray | None = None,
    show_traj_rb: bool = False,
    show_traj_nb: bool = False,
    nb_pred_trajs: np.ndarray | None = None,
) -> matplotlib.figure.Figure:
    """Render a scene with placed obstacles overlaid, matching replay sim style."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor("#f8f8f8")

    # Lane network
    draw_lanes(ax, scene.map_data, alpha=0.7)

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
        is_placed = agent.id.startswith("placed_")
        is_neighbor = not is_ego and not is_placed
        if is_neighbor and hide_neighbors:
            continue
        pos = agent.current_position
        heading = agent.current_heading

        # Dim neighbors: light grey, low alpha (still visible for context)
        _dimmed = is_neighbor and dim_neighbors
        if is_ego:
            color = _EGO_COLOR
        elif is_placed:
            color = _PLACED_COLOR
        elif _dimmed:
            color = "#bbbbbb"
        else:
            color = _agent_color(agent.agent_type, nb_idx)
        if is_neighbor:
            nb_idx += 1

        _alpha_box = 0.2 if _dimmed else (0.85 if is_ego else 0.55)
        _alpha_trail = 0.15 if _dimmed else 0.5

        # Past trail
        past = agent.past_trajectory
        valid = np.abs(past[:, :2]).sum(axis=1) > 1e-6
        if valid.sum() > 1:
            ax.plot(past[valid, 0], past[valid, 1], "--", color=color,
                    lw=0.9, alpha=_alpha_trail, zorder=7)

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
                color, alpha=_alpha_box,
                lw=2 if is_ego else (0.5 if _dimmed else 1),
                zorder=20 if is_ego else 15,
            )

        # Heading arrow (skip for dimmed neighbors)
        if _dimmed:
            continue
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

    # Placed obstacles (orange=static, blue=moving, red=selected)
    if obstacles:
        for obs in obstacles:
            is_selected = (selected_obstacle == obs.label)
            _is_mov = getattr(obs, "is_moving", False)
            if is_selected:
                color = _PLACED_SELECTED_COLOR
            elif _is_mov:
                color = _PLACED_MOVING_COLOR
            else:
                color = _PLACED_COLOR
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

            # Heading arrow (longer for moving, proportional to speed)
            _spd = getattr(obs, "speed", 0.0) if _is_mov else 0.0
            arrow_len = max(obs.length, 2.0) + _spd * 0.3
            ax.annotate(
                "", xy=(obs.x + arrow_len * math.cos(yaw),
                        obs.y + arrow_len * math.sin(yaw)),
                xytext=(obs.x, obs.y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=14),
                zorder=31,
            )

            # Label (include speed for moving obstacles)
            _lbl = f"[{obs.label}]"
            if _is_mov and _spd > 0:
                _lbl += f" {_spd:.1f} m/s"
            ax.annotate(
                _lbl, (obs.x, obs.y),
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
            import torch as _t_rb
            from rlvr.reward import compute_road_border_penalty, RewardConfig
            ego_traj_1 = _t_rb.tensor([[[
                float(ego_pos[0]), float(ego_pos[1]),
                float(np.cos(ego_h)), float(np.sin(ego_h)),
            ]]], dtype=_t_rb.float32)
            ego_shape_t = _t_rb.tensor([
                float(ego.wheelbase), float(ego.length), float(ego.width),
            ], dtype=_t_rb.float32)
            data_dict = {}
            if hasattr(scene, "map_data") and scene.map_data is not None:
                md = scene.map_data
                if hasattr(md, "line_strings") and md.line_strings is not None:
                    data_dict["line_strings"] = _t_rb.from_numpy(
                        md.line_strings.astype(np.float32)
                    )
            if data_dict:
                from rlvr.reward import _point_to_segments_dist
                _, _, _, _, _, per_t_min = compute_road_border_penalty(
                    ego_traj_1, ego_shape_t, data_dict, RewardConfig(),
                )
                dist = float(per_t_min[0, 0].item())
                if dist < 50.0:
                    color = "#dd2222" if dist < 0.5 else "#ff8800" if dist < 1.0 else "#22bb22"
                    # Find closest ego perimeter point and border segment point
                    ls_t = data_dict["line_strings"]
                    if ls_t.dim() == 3:
                        ls_t = ls_t.unsqueeze(0)
                    border_flag = ls_t[0, :, :, 3] if ls_t.shape[-1] >= 4 else None
                    if border_flag is not None:
                        border_xy = ls_t[0, :, :, :2]
                        is_border = border_flag > 0.5
                        has_coords = border_xy.norm(dim=-1) > 1e-3
                        valid = is_border & has_coords
                        valid_pair = valid[:, :-1] & valid[:, 1:]
                        idx = _t_rb.where(valid_pair.reshape(-1))[0]
                        if idx.shape[0] > 0:
                            seg_p1 = border_xy[:, :-1].reshape(-1, 2)[idx]
                            seg_p2 = border_xy[:, 1:].reshape(-1, 2)[idx]
                            wb_f = float(ego.wheelbase)
                            ln_f = float(ego.length)
                            wd_f = float(ego.width)
                            ro_f = (ln_f - wb_f) / 2
                            half_l = ln_f / 2
                            corners = [
                                (-ro_f, -wd_f/2), (-ro_f, 0.0), (-ro_f, wd_f/2),
                                (ln_f - ro_f, -wd_f/2), (ln_f - ro_f, 0.0), (ln_f - ro_f, wd_f/2),
                                (half_l - ro_f, -wd_f/2), (half_l - ro_f, wd_f/2),
                            ]
                            cos_h = float(np.cos(ego_h))
                            sin_h = float(np.sin(ego_h))
                            perim_pts = []
                            for lx, ly in corners:
                                wx = ego_pos[0] + cos_h * lx - sin_h * ly
                                wy = ego_pos[1] + sin_h * lx + cos_h * ly
                                perim_pts.append([wx, wy])
                            perim_t = _t_rb.tensor(perim_pts, dtype=_t_rb.float32)
                            d_mat = _point_to_segments_dist(perim_t, seg_p1, seg_p2)
                            min_per_pt = d_mat.min(dim=1)
                            best_pt_idx = min_per_pt.values.argmin().item()
                            best_seg_idx = min_per_pt.indices[best_pt_idx].item()
                            ep = perim_pts[best_pt_idx]
                            s1 = seg_p1[best_seg_idx].numpy()
                            s2 = seg_p2[best_seg_idx].numpy()
                            p = np.array(ep, dtype=np.float64)
                            d_seg = s2 - s1
                            t_val = max(0.0, min(1.0, float(np.dot(p - s1, d_seg) / max(np.dot(d_seg, d_seg), 1e-12))))
                            bp = s1 + t_val * d_seg
                            ax.plot([ep[0], bp[0]], [ep[1], bp[1]],
                                    "-", color=color, lw=2.5, alpha=0.9, zorder=35)
                            ax.plot(bp[0], bp[1], "o", color=color, ms=6, zorder=36)
                    mid_x = float(ego_pos[0])
                    mid_y = float(ego_pos[1]) + 1.5
                    ax.annotate(
                        f"RB {dist:.2f}m", (mid_x, mid_y),
                        fontsize=8, fontweight="bold", color=color,
                        ha="center", va="bottom", zorder=37,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
                    )

        if show_nb_dist:
            import torch as _torch

            from rlvr.reward import _closest_points_between_rects
            from scenario_generation.gui.lanelet_scene_builder import _obb_corners
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

    # Neighbor predicted trajectories (thin lines, one per neighbor)
    if nb_pred_trajs is not None:
        _nb_colors = ["#ee6644", "#44aa88", "#8866cc", "#ccaa22", "#44aadd"]
        nb_agents = [a for a in scene.agents if a.id != scene.ego_agent_id]
        for ni in range(min(nb_pred_trajs.shape[0], len(nb_agents))):
            traj_4 = nb_pred_trajs[ni]  # (T, 4) [x, y, cos_h, sin_h]
            if np.abs(traj_4[:, :2]).sum() < 1e-4:
                continue
            heading = np.arctan2(traj_4[:, 3], traj_4[:, 2])
            traj_xyh = np.column_stack([traj_4[:, :2], heading])
            nc = _nb_colors[ni % len(_nb_colors)]
            draw_trajectory(ax, traj_xyh, nc, label=None, lw=0.8, zorder=14,
                            show_footprints=False, length=2.0, width=1.0)

    # Worst-case RB/NB distance along predicted trajectories
    if (show_traj_rb or show_traj_nb) and ego is not None:
        border_polylines_for_traj = _extract_border_polylines(scene)
        if not border_polylines_for_traj and _ego_frame_borders:
            border_polylines_for_traj = _ego_frame_borders

        nb_agents = [a for a in scene.agents if a.id != scene.ego_agent_id]
        _placed_obs_corners = []
        for obs in (obstacles or []):
            _placed_obs_corners.append(_obb_corners_from_placement(obs))

        _trajs_to_check: list[tuple[np.ndarray, str, str]] = []
        if det_traj is not None and det_traj.shape[0] > 1:
            _trajs_to_check.append((det_traj, _DET_COLOR, "DET"))
        if guided_trajs:
            for gi, gtr in enumerate(guided_trajs):
                if gtr is not None and gtr.shape[0] > 1:
                    _trajs_to_check.append(
                        (gtr, _GUIDED_COLORS[gi % len(_GUIDED_COLORS)], f"G#{gi+1}"))

        for traj_pts, traj_color, traj_label in _trajs_to_check:
            # Sample every 5th point for speed
            idxs = list(range(0, traj_pts.shape[0], 5))
            if (traj_pts.shape[0] - 1) not in idxs:
                idxs.append(traj_pts.shape[0] - 1)

            if show_traj_rb:
                import torch as _t_trb
                from rlvr.reward import compute_road_border_penalty, RewardConfig
                _trb_data = {}
                if hasattr(scene, "map_data") and scene.map_data is not None:
                    md = scene.map_data
                    if hasattr(md, "line_strings") and md.line_strings is not None:
                        _trb_data["line_strings"] = _t_trb.from_numpy(
                            md.line_strings.astype(np.float32))
                if _trb_data:
                    T_tr = traj_pts.shape[0]
                    ego_traj_t = _t_trb.zeros(1, T_tr, 4)
                    ego_traj_t[0, :, 0] = _t_trb.from_numpy(traj_pts[:, 0].astype(np.float32))
                    ego_traj_t[0, :, 1] = _t_trb.from_numpy(traj_pts[:, 1].astype(np.float32))
                    ego_traj_t[0, :, 2] = _t_trb.from_numpy(np.cos(traj_pts[:, 2]).astype(np.float32))
                    ego_traj_t[0, :, 3] = _t_trb.from_numpy(np.sin(traj_pts[:, 2]).astype(np.float32))
                    ego_shape_t = _t_trb.tensor([
                        float(ego.wheelbase), float(ego.length), float(ego.width),
                    ], dtype=_t_trb.float32)
                    _, _, _, _, _, per_t_min = compute_road_border_penalty(
                        ego_traj_t, ego_shape_t, _trb_data, RewardConfig())
                    per_t = per_t_min[0]  # (T,)
                    worst_rb_idx = int(per_t.argmin().item())
                    worst_rb_dist = float(per_t[worst_rb_idx].item())
                    if worst_rb_dist < 50.0:
                        wx = float(traj_pts[worst_rb_idx, 0])
                        wy = float(traj_pts[worst_rb_idx, 1])
                        wh = float(traj_pts[worst_rb_idx, 2])
                        dc = "#dd2222" if worst_rb_dist < 0.5 else "#ff8800" if worst_rb_dist < 1.0 else "#22bb22"
                        draw_agent_box(ax, wx, wy, wh, ego_len, ego_wid, traj_color,
                                       alpha=0.4, lw=2.0, zorder=38)
                        from rlvr.reward import _point_to_segments_dist as _ptsd_trb
                        ls_trb = _trb_data["line_strings"]
                        if ls_trb.dim() == 3:
                            ls_trb = ls_trb.unsqueeze(0)
                        bf = ls_trb[0, :, :, 3] if ls_trb.shape[-1] >= 4 else None
                        if bf is not None:
                            bxy = ls_trb[0, :, :, :2]
                            vld = (bf > 0.5) & (bxy.norm(dim=-1) > 1e-3)
                            vpair = vld[:, :-1] & vld[:, 1:]
                            vidx = _t_trb.where(vpair.reshape(-1))[0]
                            if vidx.shape[0] > 0:
                                sp1 = bxy[:, :-1].reshape(-1, 2)[vidx]
                                sp2 = bxy[:, 1:].reshape(-1, 2)[vidx]
                                wb_t = float(ego.wheelbase)
                                ro_t = (ego_len - wb_t) / 2
                                cos_wh = float(np.cos(wh))
                                sin_wh = float(np.sin(wh))
                                half_lt = ego_len / 2
                                corners_t = [
                                    (-ro_t, -ego_wid/2), (-ro_t, 0.0), (-ro_t, ego_wid/2),
                                    (ego_len - ro_t, -ego_wid/2), (ego_len - ro_t, 0.0), (ego_len - ro_t, ego_wid/2),
                                    (half_lt - ro_t, -ego_wid/2), (half_lt - ro_t, ego_wid/2),
                                ]
                                pp = []
                                for lx, ly in corners_t:
                                    pp.append([wx + cos_wh*lx - sin_wh*ly,
                                               wy + sin_wh*lx + cos_wh*ly])
                                pp_t = _t_trb.tensor(pp, dtype=_t_trb.float32)
                                dm = _ptsd_trb(pp_t, sp1, sp2)
                                mpp = dm.min(dim=1)
                                bpi = mpp.values.argmin().item()
                                bsi = mpp.indices[bpi].item()
                                ep_t = pp[bpi]
                                s1_t = sp1[bsi].numpy()
                                s2_t = sp2[bsi].numpy()
                                d_s = s2_t - s1_t
                                tv = max(0.0, min(1.0, float(np.dot(np.array(ep_t) - s1_t, d_s) / max(np.dot(d_s, d_s), 1e-12))))
                                bp_t = s1_t + tv * d_s
                                ax.plot([ep_t[0], bp_t[0]], [ep_t[1], bp_t[1]],
                                        "-", color=dc, lw=2.5, alpha=0.9, zorder=39)
                                ax.plot(bp_t[0], bp_t[1], "o", color=dc, ms=5, zorder=39)
                        ax.annotate(
                            f"{traj_label} RB {worst_rb_dist:.2f}m @t{worst_rb_idx}",
                            (wx, wy), fontsize=7, fontweight="bold", color=dc,
                            ha="center", va="top", xytext=(0, -8),
                            textcoords="offset points", zorder=40,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=traj_color, alpha=0.85),
                    )

            _all_nb_corners = _placed_obs_corners[:]
            if show_traj_nb:
                import torch as _torch

                from rlvr.reward import _closest_points_between_rects
                from scenario_generation.gui.lanelet_scene_builder import _obb_corners
                for a in nb_agents:
                    _all_nb_corners.append(_obb_corners(
                        float(a.current_position[0]), float(a.current_position[1]),
                        float(a.current_heading), float(a.length), float(a.width),
                    ))
            if show_traj_nb and _all_nb_corners:
                n_nb = len(_all_nb_corners)
                r2 = _torch.from_numpy(np.stack(_all_nb_corners).astype(np.float32))
                worst_nb_dist, worst_nb_idx = float("inf"), 0
                worst_nb_pe, worst_nb_pn = None, None
                for ti in idxs:
                    px, py = float(traj_pts[ti, 0]), float(traj_pts[ti, 1])
                    ph = float(traj_pts[ti, 2])
                    ego_c = _obb_corners(px, py, ph, ego_len, ego_wid)
                    r1 = _torch.from_numpy(
                        np.broadcast_to(ego_c, (n_nb, 4, 2)).copy().astype(np.float32))
                    pe, pn = _closest_points_between_rects(r1, r2)
                    d_min = float((pe - pn).norm(dim=-1).min().item())
                    if d_min < worst_nb_dist:
                        worst_nb_dist = d_min
                        worst_nb_idx = ti
                        min_i = int((pe - pn).norm(dim=-1).argmin().item())
                        worst_nb_pe = pe[min_i].numpy()
                        worst_nb_pn = pn[min_i].numpy()
                if worst_nb_pe is not None and worst_nb_dist < 20.0:
                    wx, wy = float(traj_pts[worst_nb_idx, 0]), float(traj_pts[worst_nb_idx, 1])
                    wh = float(traj_pts[worst_nb_idx, 2])
                    dc = "#dd2222" if worst_nb_dist < 0.5 else "#ff8800" if worst_nb_dist < 1.5 else "#22bb22"
                    draw_agent_box(ax, wx, wy, wh, ego_len, ego_wid, traj_color,
                                   alpha=0.4, lw=2.0, zorder=38)
                    ax.plot([worst_nb_pe[0], worst_nb_pn[0]], [worst_nb_pe[1], worst_nb_pn[1]],
                            "-", color=dc, lw=2.5, alpha=0.9, zorder=39)
                    ax.annotate(
                        f"{traj_label} NB {worst_nb_dist:.2f}m @t{worst_nb_idx}",
                        (wx, wy), fontsize=7, fontweight="bold", color=dc,
                        ha="center", va="bottom", xytext=(0, 8),
                        textcoords="offset points", zorder=40,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=traj_color, alpha=0.85),
                    )

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
            with np.load(seq[s]) as _npz:
                past = _npz["ego_agent_past"]
            prev = past[-2]
            dx_local, dy_local, dyaw = -prev[0], -prev[1], -prev[2]
            c, sn = math.cos(cum_yaw), math.sin(cum_yaw)
            cum_x += c * dx_local - sn * dy_local
            cum_y += sn * dx_local + c * dy_local
            cum_yaw += dyaw
        # Point was at (x, y) in from_step frame. In to_step frame:
        rx, ry = x - cum_x, y - cum_y
        c, sn = math.cos(-cum_yaw), math.sin(-cum_yaw)
        new_x = c * rx - sn * ry
        new_y = sn * rx + c * ry
        new_yaw = yaw_rad - cum_yaw
    else:
        # Backward: chain from to_step+1 to from_step (then invert)
        for s in range(to_step + 1, min(from_step + 1, len(seq))):
            with np.load(seq[s]) as _npz:
                past = _npz["ego_agent_past"]
            prev = past[-2]
            dx_local, dy_local, dyaw = -prev[0], -prev[1], -prev[2]
            c, sn = math.cos(cum_yaw), math.sin(cum_yaw)
            cum_x += c * dx_local - sn * dy_local
            cum_y += sn * dx_local + c * dy_local
            cum_yaw += dyaw
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
    with np.load(seq[current_step]) as _cur:
        cur_past = _cur["ego_agent_past"].copy()  # (31, 3) [x, y, yaw]
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
        with np.load(seq[future_idx]) as _fut:
            fut_past = _fut["ego_agent_past"].copy()  # (31, 3)
        # Displacement from prev to current in this step's ego frame
        prev_in_cur = fut_past[-2]  # where ego was 1 step ago, in this step's frame
        # The ego moved from prev_in_cur to [0,0,0]
        dx_local = -prev_in_cur[0]
        dy_local = -prev_in_cur[1]
        dyaw = -prev_in_cur[2]

        # Rotate displacement into accumulated frame, then update yaw
        cos_a = math.cos(cumulative_yaw)
        sin_a = math.sin(cumulative_yaw)
        cumulative_x += cos_a * dx_local - sin_a * dy_local
        cumulative_y += sin_a * dx_local + cos_a * dy_local
        cumulative_yaw += dyaw

        gt_points.append([cumulative_x, cumulative_y, cumulative_yaw])

    if len(gt_points) < 2:
        return None
    return np.array(gt_points, dtype=np.float32)


def _build_moving_agent(
    obs: ObstaclePlacement,
    map_builder,
    ego_wp_arr: np.ndarray | None,
) -> "Agent":
    """Build an Agent with plausible motion history for a moving obstacle."""
    from scenario_generation.scene_context import Agent, AgentType

    T_PAST = 31
    DT = 0.1
    aid = f"placed_{obs.label}"
    yaw = obs.yaw_rad
    spd = obs.speed

    # Try map-based history if route is available
    if (map_builder is not None
            and obs.route_lanelet_ids
            and ego_wp_arr is not None):
        ci, si = math.cos(ego_wp_arr[2]), math.sin(ego_wp_arr[2])
        wx = ego_wp_arr[0] + ci * obs.x - si * obs.y
        wy = ego_wp_arr[1] + si * obs.x + ci * obs.y
        wyaw = ego_wp_arr[2] + yaw
        history_world, _ = map_builder.generate_history(
            np.array([wx, wy], dtype=np.float32), wyaw, spd,
            obs.route_lanelet_ids[0], n_steps=T_PAST, dt=DT,
        )
        # Transform world-frame history to ego-frame (sim-start frame)
        from scenario_generation.transforms import _rotation_matrix, transform_positions
        R_w = _rotation_matrix(ego_wp_arr[2])
        ego_xy = np.array(ego_wp_arr[:2], dtype=np.float64)
        hist_xy = transform_positions(
            history_world[:, :2].astype(np.float64), R_w, ego_xy,
        ).astype(np.float32)
        hist_h = history_world[:, 2] - ego_wp_arr[2]
        history = np.column_stack([hist_xy, hist_h]).astype(np.float32)

        # Route tensors (world-frame -> ego-frame via tensor converter)
        route_lanes, route_sl, route_hsl = map_builder._route_to_33dim(obs.route_lanelet_ids)
        # Transform route_lanes centerline points to ego frame
        for seg_i in range(route_lanes.shape[0]):
            pts = route_lanes[seg_i, :, :2]
            valid = np.abs(pts).sum(axis=1) > 0.01
            if valid.any():
                route_lanes[seg_i, valid, :2] = transform_positions(
                    pts[valid].astype(np.float64), R_w, ego_xy,
                ).astype(np.float32)

        goal_pose_ego = None
        if obs.goal_pose is not None:
            gx, gy, gh = obs.goal_pose
            g_ego = transform_positions(
                np.array([[gx, gy]], dtype=np.float64), R_w, ego_xy,
            ).astype(np.float32)[0]
            goal_pose_ego = np.array([g_ego[0], g_ego[1], gh - ego_wp_arr[2]],
                                     dtype=np.float32)
    else:
        # Straight-line fallback
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        history = np.zeros((T_PAST, 3), dtype=np.float32)
        for t in range(T_PAST):
            backward = (T_PAST - 1 - t) * spd * DT
            history[t, 0] = obs.x - backward * cos_y
            history[t, 1] = obs.y - backward * sin_y
            history[t, 2] = yaw
        route_lanes = None
        route_sl = None
        route_hsl = None
        goal_pose_ego = None

    # Derive velocities from history
    velocities = np.zeros((T_PAST, 2), dtype=np.float32)
    if T_PAST >= 2:
        diffs = np.diff(history[:, :2], axis=0) / DT
        velocities[1:] = diffs

    agent = Agent(
        id=aid,
        agent_type=AgentType.VEHICLE,
        length=obs.length, width=obs.width,
        wheelbase=obs.length * 0.65,
        past_trajectory=history,
        past_velocities=velocities,
        age_steps=T_PAST - 1,
        route_lanes=route_lanes,
        route_speed_limit=route_sl,
        route_has_speed_limit=route_hsl,
        goal_pose=goal_pose_ego,
        route_lanelet_ids=obs.route_lanelet_ids,
    )
    return agent


def _generate_neighbor_reference(
    agent: "Agent",
    map_builder,
    ego_wp_arr: np.ndarray | None,
    n_steps: int,
    dt: float = 0.1,
) -> np.ndarray:
    """Generate an open-loop reference trajectory for a moving neighbor.

    Returns (n_steps, 3) [x, y, yaw] in the sim ego frame.
    """
    pos = agent.current_position
    heading = agent.current_heading
    vel = agent.current_velocity
    speed = float(np.linalg.norm(vel))

    if (map_builder is not None
            and agent.route_lanelet_ids
            and ego_wp_arr is not None):
        from scenario_generation.transforms import _rotation_matrix, transform_positions
        ci, si = math.cos(ego_wp_arr[2]), math.sin(ego_wp_arr[2])
        wx = ego_wp_arr[0] + ci * pos[0] - si * pos[1]
        wy = ego_wp_arr[1] + si * pos[0] + ci * pos[1]

        # Build forward centerline polyline from route
        cl_pts = []
        for ll_id in agent.route_lanelet_ids:
            if ll_id in map_builder._cache:
                cl_pts.append(map_builder._cache[ll_id].raw_centerline)
        if cl_pts:
            polyline = np.concatenate(cl_pts, axis=0)
            # Project current world position onto polyline
            diffs = polyline - np.array([wx, wy])
            dists = np.linalg.norm(diffs, axis=1)
            nearest_idx = int(np.argmin(dists))

            # Walk forward from nearest_idx, sampling at speed * dt intervals
            seg_diffs = np.diff(polyline[nearest_idx:], axis=0)
            seg_lens = np.linalg.norm(seg_diffs, axis=1)
            arc = np.concatenate([[0.0], np.cumsum(seg_lens)])

            ref_world = np.zeros((n_steps, 3), dtype=np.float32)
            for step in range(n_steps):
                fwd_dist = (step + 1) * speed * dt
                seg_i = np.searchsorted(arc, fwd_dist) - 1
                seg_i = max(0, min(seg_i, len(arc) - 2))
                seg_len = max(arc[seg_i + 1] - arc[seg_i], 1e-6)
                frac = (fwd_dist - arc[seg_i]) / seg_len
                frac = max(0.0, min(1.0, frac))
                pt = polyline[nearest_idx + seg_i] + frac * seg_diffs[min(seg_i, len(seg_diffs) - 1)]
                if seg_i < len(seg_diffs):
                    h = math.atan2(seg_diffs[seg_i, 1], seg_diffs[seg_i, 0])
                else:
                    h = heading + ego_wp_arr[2]
                ref_world[step] = [pt[0], pt[1], h]

            # Transform to ego frame
            R_w = _rotation_matrix(ego_wp_arr[2])
            ego_xy = np.array(ego_wp_arr[:2], dtype=np.float64)
            ref_ego_xy = transform_positions(
                ref_world[:, :2].astype(np.float64), R_w, ego_xy,
            ).astype(np.float32)
            ref_ego_h = ref_world[:, 2] - ego_wp_arr[2]
            return np.column_stack([ref_ego_xy, ref_ego_h]).astype(np.float32)

    # Straight-line fallback
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    ref = np.zeros((n_steps, 3), dtype=np.float32)
    for step in range(n_steps):
        d = (step + 1) * speed * dt
        ref[step] = [pos[0] + d * cos_h, pos[1] + d * sin_h, heading]
    return ref


def build_interface(tree: SceneTree, model_cache: _ModelCache | None = None,
                    map_borders: list[np.ndarray] | None = None,
                    map_builder=None, reward_config=None):
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
                    obs_is_moving = gr.Checkbox(label="Moving", value=False)
                    obs_speed = gr.Number(
                        label="Speed (m/s)", value=5.0, precision=1,
                        visible=False, min_width=100,
                    )
                obs_route_info = gr.Markdown("")
                with gr.Row():
                    place_btn = gr.Button("Place Obstacle", variant="primary")
                    preview_btn = gr.Button("Preview", variant="secondary")

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

                # Timeline playback + view
                with gr.Row():
                    btn_play = gr.Button("Play ▶", size="sm", min_width=60)
                    btn_stop = gr.Button("Stop ■", size="sm", min_width=60, variant="stop")
                    play_fps = gr.Slider(minimum=1, maximum=30, value=10, step=1,
                                         label="FPS", scale=1)
                    view_half = gr.Slider(minimum=10, maximum=200, value=50, step=5,
                                          label="View radius (m)", scale=1)

                _has_model = model_cache is not None and model_cache.available

                # Trajectory overlay controls — below canvas
                with gr.Row():
                    show_gt = gr.Checkbox(label="Show GT", value=True, scale=1)
                    show_det = gr.Checkbox(label="Show DET", value=False,
                                           interactive=_has_model, scale=1)
                    show_guided = gr.Checkbox(label="Show Guided", value=False,
                                              interactive=_has_model, scale=1)
                    hide_neighbors = gr.Checkbox(label="Dim/Zero Neighbors", value=False, scale=1)
                    show_rb_dist = gr.Checkbox(label="Road Border", value=True, scale=1)
                    show_nb_dist = gr.Checkbox(label="Neighbor Dist", value=True, scale=1)
                with gr.Row():
                    show_traj_rb = gr.Checkbox(label="Traj RB Worst", value=False, scale=1)
                    show_traj_nb = gr.Checkbox(label="Traj NB Worst", value=False, scale=1)
                    show_nb_preds = gr.Checkbox(label="NB Preds", value=False,
                                                interactive=_has_model, scale=1)
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
                    _default_proto = str(Path(__file__).resolve().parent.parent.parent
                                         / "guidance_gui" / "prototypes_k16.npy")
                    with gr.Accordion("Anchor Prototypes", open=False):
                        with gr.Row():
                            anchor_index_sl = gr.Slider(
                                minimum=0, maximum=15, value=0, step=1,
                                label="Anchor Index", interactive=_has_model, scale=1,
                            )
                            anchor_path_tb = gr.Textbox(
                                value=_default_proto,
                                label="Prototypes Path", interactive=_has_model, scale=2,
                            )
                        from guidance_gui.visualization import render_prototype_gallery
                        _init_gallery = render_prototype_gallery(_default_proto) or []
                        anchor_gallery = gr.Gallery(
                            value=_init_gallery,
                            columns=8, rows=2, height=220,
                            allow_preview=False,
                            selected_index=0 if _init_gallery else None,
                            label="Click to select anchor",
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
                    sim_ego_mode = gr.Dropdown(
                        choices=["closed-loop", "open-loop"],
                        value="closed-loop", label="Ego",
                        interactive=_has_model, scale=1, min_width=100,
                    )
                    sim_neighbor_mode = gr.Dropdown(
                        choices=["closed-loop", "open-loop"],
                        value="closed-loop", label="Neighbors",
                        interactive=_has_model, scale=1, min_width=100,
                    )
                    sim_btn = gr.Button("Simulate", variant="primary", scale=1,
                                        interactive=_has_model)
                sim_status = gr.Markdown("")

                # Export & RSFT save — horizontal layout
                with gr.Accordion("Export / Save for RSFT", open=False):
                    with gr.Row():
                        export_dir = gr.Textbox(label="Export Dir", placeholder="/path/to/export",
                                                scale=3)
                        export_btn = gr.Button("Export NPZs", variant="secondary", scale=1)
                    export_status = gr.Markdown("")
                    with gr.Row():
                        rsft_dir = gr.Textbox(label="RSFT Dir", placeholder="/path/to/rsft_curated",
                                              scale=3)
                        rsft_save_btn = gr.Button("Save Scene + Guided Traj", variant="primary",
                                                   interactive=_has_model, scale=1)
                    rsft_status = gr.Markdown("")

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
                with gr.Row():
                    edit_is_moving = gr.Checkbox(label="Moving", value=False)
                    edit_speed = gr.Number(label="Spd (m/s)", value=0.0, precision=1,
                                           visible=False, min_width=80)
                apply_edit_btn = gr.Button("Apply Edit", size="sm", variant="primary")

                save_status = gr.Markdown("")

        # ── Callbacks ──

        def _render(tree: SceneTree, step: int, view_r: float,
                    selected_obs: str | None,
                    preview_placement: ObstaclePlacement | None = None,
                    show_gt_val: bool = True,
                    det_traj: np.ndarray | None = None,
                    guided_trajs: list[np.ndarray] | None = None,
                    rb_dist: bool = True, nb_dist: bool = True,
                    hide_nb: bool = False,
                    traj_rb: bool = False, traj_nb: bool = False,
                    nb_pred_trajs: np.ndarray | None = None):
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
                            is_moving=o.is_moving, speed=o.speed,
                            route_lanelet_ids=o.route_lanelet_ids,
                            goal_pose=o.goal_pose,
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

            # Ego world pose for map border transform + line_strings refresh
            ego_wp = _recover_ego_world_pose(seq, step) if (map_borders or map_builder) else None

            # Refresh line_strings from map if source NPZ lacks border flags
            if (scene.map_data is not None
                    and scene.map_data.line_strings is not None
                    and scene.map_data.line_strings.shape[-1] < 4
                    and map_builder is not None and ego_wp is not None):
                from scenario_generation.simulate import _refresh_line_strings
                _refresh_line_strings(
                    scene, map_builder,
                    np.array(ego_wp[:2], dtype=np.float64),
                    np.array(ego_wp, dtype=np.float64),
                )

            fig = render_scene_at_step(
                scene, obstacles_at_step, selected_obs,
                view_half=view_r, step_idx=step, total_steps=len(seq),
                gt_traj=gt_traj_render,
                det_traj=det_traj,
                guided_trajs=guided_trajs,
                show_rb_dist=rb_dist, show_nb_dist=nb_dist,
                dim_neighbors=hide_nb,
                map_border_polylines=map_borders, ego_world_pose=ego_wp,
                show_traj_rb=traj_rb, show_traj_nb=traj_nb,
                nb_pred_trajs=nb_pred_trajs,
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

        def _predict_det_with_obs(tree, step, zero_neighbors=False,
                                  return_nb_preds=False):
            npz_path = _get_npz_path(tree, step)
            if not npz_path:
                return (None, None) if return_nb_preds else None
            obs = _get_obstacles_at_step(tree, _safe_step(step))
            result = model_cache.predict_det(
                npz_path, obstacles=obs or None,
                zero_neighbors=zero_neighbors,
                ego_shape_override=tree.ego_shape,
                return_neighbor_preds=return_nb_preds,
            )
            if return_nb_preds:
                ego_raw, nb_raw = result
                return _traj_cos_sin_to_xyh(ego_raw), nb_raw
            return _traj_cos_sin_to_xyh(result)

        def on_render(tree, step, view_r, selected_obs, gt_on, det_on, guided_on,
                      hide_nb, rb_on, nb_on, traj_rb_on, traj_nb_on,
                      det_cache, guided_cache, nb_preds_on):
            det_traj = None
            _nb_preds = None
            if det_on and model_cache and model_cache.available:
                if nb_preds_on:
                    det_traj, _nb_preds = _predict_det_with_obs(
                        tree, step, zero_neighbors=hide_nb, return_nb_preds=True)
                else:
                    det_traj = _predict_det_with_obs(tree, step,
                                                     zero_neighbors=hide_nb)
                det_cache = det_traj
            elif det_on and det_cache is not None:
                det_traj = det_cache
            else:
                det_cache = None

            guided_list = guided_cache if guided_on else None

            img, info = _render(tree, _safe_step(step), view_r, selected_obs,
                                show_gt_val=gt_on, det_traj=det_traj,
                                guided_trajs=guided_list,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb,
                                traj_rb=traj_rb_on, traj_nb=traj_nb_on,
                                nb_pred_trajs=_nb_preds)
            return img, info, det_cache, guided_cache
        def on_step_change(tree, step, view_r, selected_obs, gt_on, det_on, hide_nb, rb_on, nb_on,
                           traj_rb_on, traj_nb_on, guided_on, prev_guided_cache, *g_args):
            s = _safe_step(step)
            det_traj, guided = _recompute_trajs(tree, s, det_on, guided_on, g_args or None,
                                                prev_guided=prev_guided_cache,
                                                zero_neighbors=hide_nb)
            img, info = _render(tree, s, view_r, selected_obs,
                                show_gt_val=gt_on, det_traj=det_traj, guided_trajs=guided,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb,
                                traj_rb=traj_rb_on, traj_nb=traj_nb_on)
            return img, info, s, det_traj, guided
        def _on_nav_impl(direction, tree, step, view_r, selected_obs, gt_on, det_on,
                         hide_nb, rb_on, nb_on, traj_rb_on, traj_nb_on,
                         guided_on, prev_guided_cache, *g_args):
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
            det_traj, guided = _recompute_trajs(tree, s, det_on, guided_on, g_args or None,
                                                prev_guided=prev_guided_cache,
                                                zero_neighbors=hide_nb)
            img, info = _render(tree, s, view_r, selected_obs,
                                show_gt_val=gt_on, det_traj=det_traj, guided_trajs=guided,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb,
                                traj_rb=traj_rb_on, traj_nb=traj_nb_on)
            return img, info, s, det_traj, guided
        def on_preview(tree, step, view_r, selected_obs, x, y, yaw, length, width,
                       gt_on, det_cache, guided_cache,
                       rb_on, nb_on, hide_nb, traj_rb_on, traj_nb_on):
            preview = ObstaclePlacement(
                label="(preview)", timestep=_safe_step(step),
                x=round(float(x), 1), y=round(float(y), 1),
                yaw_deg=round(float(yaw) / 5) * 5,
                length=float(length), width=float(width),
            )
            img, info = _render(tree, _safe_step(step), view_r, selected_obs, preview,
                                show_gt_val=gt_on, det_traj=det_cache,
                                guided_trajs=guided_cache,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb,
                                traj_rb=traj_rb_on, traj_nb=traj_nb_on)
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
            branch = tree.branches[tree.active_branch]
            if branch.npz_dir is not None:
                return []
            obstacles = tree.get_all_obstacles(tree.active_branch)
            seq = tree.get_npz_sequence(tree.active_branch)
            result = []
            for o in obstacles:
                if o.timestep > step:
                    continue
                if o.timestep != step and seq:
                    nx, ny, nyaw = _transform_point_between_steps(
                        seq, o.timestep, step, o.x, o.y, o.yaw_rad,
                    )
                    result.append(ObstaclePlacement(
                        label=o.label, timestep=o.timestep,
                        x=nx, y=ny, yaw_deg=math.degrees(nyaw),
                        length=o.length, width=o.width,
                        history_steps=o.history_steps,
                        is_moving=o.is_moving, speed=o.speed,
                        route_lanelet_ids=o.route_lanelet_ids,
                        goal_pose=o.goal_pose,
                    ))
                else:
                    result.append(o)
            return result

        def _recompute_trajs(tree, step, det_on, guided_on=False,
                             guidance_args_tuple=None, prev_guided=None,
                             zero_neighbors=False):
            """Recompute DET and guided trajectories if toggled on.

            When guided_on and guidance_args_tuple is provided, regenerates
            guided trajectories with the current guidance config. If no
            guidances are enabled, preserves prev_guided.
            """
            det_traj = None
            guided = prev_guided if guided_on else None
            if not (model_cache and model_cache.available):
                return det_traj, guided
            obs = _get_obstacles_at_step(tree, _safe_step(step))
            npz_path = _get_npz_path(tree, step)
            if not npz_path:
                return det_traj, guided
            if det_on:
                raw = model_cache.predict_det(npz_path, obstacles=obs or None,
                                              zero_neighbors=zero_neighbors,
                                              ego_shape_override=tree.ego_shape)
                det_traj = _traj_cos_sin_to_xyh(raw)
            if guided_on and guidance_args_tuple:
                cfgs = []
                for gi, gname in enumerate(ALL_GUIDANCE_NAMES):
                    enabled = guidance_args_tuple[gi * 2]
                    scale = guidance_args_tuple[gi * 2 + 1]
                    if enabled:
                        cfgs.append((gname, float(scale)))
                if cfgs:
                    noise = float(guidance_args_tuple[-4])
                    k = int(guidance_args_tuple[-3])
                    a_idx = int(guidance_args_tuple[-2])
                    a_path = str(guidance_args_tuple[-1])
                    raw_g = model_cache.predict_guided(
                        npz_path, cfgs, noise_scale=noise, n_samples=max(1, k),
                        zero_neighbors=zero_neighbors,
                        ego_shape_override=tree.ego_shape,
                        anchor_index=a_idx, anchor_path=a_path,
                    )
                    guided = [_traj_cos_sin_to_xyh(raw_g[j]) for j in range(raw_g.shape[0])]
            return det_traj, guided

        def on_place(tree, step, view_r, x, y, yaw, length, width, history,
                     is_moving, speed_val,
                     gt_on, det_on, guided_on, hide_nb, rb_on, nb_on,
                     traj_rb_on, traj_nb_on, *g_args):
            label = tree.next_obstacle_label(tree.active_branch)
            s = _safe_step(step)
            _moving = bool(is_moving)
            _speed = max(0.0, float(speed_val)) if _moving else 0.0

            route_ids = None
            goal = None
            route_info_text = ""
            if _moving and map_builder is not None:
                seq = tree.get_npz_sequence(tree.active_branch)
                ego_wp = _recover_ego_world_pose(seq, s) if seq else None
                if ego_wp is not None:
                    yaw_rad = math.radians(float(yaw))
                    ci, si = math.cos(ego_wp[2]), math.sin(ego_wp[2])
                    wx = ego_wp[0] + ci * float(x) - si * float(y)
                    wy = ego_wp[1] + si * float(x) + ci * float(y)
                    wyaw = ego_wp[2] + yaw_rad
                    ll_id = map_builder.snap_to_nearest_ll(
                        np.array([wx, wy]), heading_rad=wyaw,
                    )
                    if ll_id is not None:
                        route_ids = map_builder.find_route(ll_id, min_length_m=150.0)
                        goal_arr = map_builder._route_goal(route_ids)
                        goal = (float(goal_arr[0]), float(goal_arr[1]), float(goal_arr[2]))
                        route_info_text = f"Route: {len(route_ids)} lanelets"
                    else:
                        route_info_text = "Could not snap to lanelet"
                else:
                    route_info_text = "No ego world pose available"
            elif _moving:
                route_info_text = "No map -- straight-line mode"

            placement = ObstaclePlacement(
                label=label, timestep=s,
                x=float(x), y=float(y), yaw_deg=float(yaw),
                length=float(length), width=float(width),
                history_steps=int(history),
                is_moving=_moving, speed=_speed,
                route_lanelet_ids=route_ids, goal_pose=goal,
            )
            tree.add_obstacle(tree.active_branch, placement)
            det_traj, guided = _recompute_trajs(tree, s, det_on, guided_on, g_args or None,
                                                zero_neighbors=hide_nb)
            img, info = _render(tree, s, view_r, label, show_gt_val=gt_on,
                                det_traj=det_traj, guided_trajs=guided,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb,
                                traj_rb=traj_rb_on, traj_nb=traj_nb_on)
            mods = _modifications_md(tree, tree.active_branch)
            choices = _obs_choices(tree)
            return (tree, img, info, mods, label, det_traj, guided,
                    gr.update(choices=choices, value=label), route_info_text)

        def on_select_obstacle(tree, label, step, view_r, gt_on, det_on, det_cache, guided_cache,
                               rb_on, nb_on, hide_nb, traj_rb_on, traj_nb_on):
            obs = _find_obs(tree, label)
            s = _safe_step(step)
            img, info = _render(tree, s, view_r, label, show_gt_val=gt_on,
                                det_traj=det_cache, guided_trajs=guided_cache,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb,
                                traj_rb=traj_rb_on, traj_nb=traj_nb_on)
            if obs:
                _mov = getattr(obs, "is_moving", False)
                _spd = getattr(obs, "speed", 0.0)
                return (img, info, label,
                        obs.x, obs.y, obs.yaw_deg, obs.length, obs.width,
                        getattr(obs, "history_steps", 30),
                        _mov, _spd, gr.update(visible=_mov))
            return (img, info, label,
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update())

        def on_remove_obstacle(tree, label, step, view_r, gt_on, det_on,
                               guided_on, hide_nb, rb_on, nb_on,
                               traj_rb_on, traj_nb_on, *g_args):
            if label:
                tree.remove_obstacle(tree.active_branch, label.strip())
            s = _safe_step(step)
            det_traj, guided = _recompute_trajs(tree, s, det_on, guided_on, g_args or None,
                                                zero_neighbors=hide_nb)
            img, info = _render(tree, s, view_r, None, show_gt_val=gt_on,
                                det_traj=det_traj, guided_trajs=guided,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb,
                                traj_rb=traj_rb_on, traj_nb=traj_nb_on)
            mods = _modifications_md(tree, tree.active_branch)
            choices = _obs_choices(tree)
            return (tree, img, info, mods, None,
                    gr.update(choices=choices, value=None), det_traj, guided)

        def on_apply_edit(tree, label, step, view_r, gt_on, det_on, guided_on,
                          hide_nb, rb_on, nb_on, traj_rb_on, traj_nb_on,
                          x, y, yaw, length, width, history,
                          ed_is_moving, ed_speed, *g_args):
            if not label:
                return (tree, gr.update(), "No obstacle selected",
                        gr.update(), gr.update(), gr.update(), gr.update())
            # Search current branch first, then walk ancestors to find the obstacle
            _found = False
            bid = tree.active_branch
            while bid is not None and not _found:
                br = tree.branches.get(bid)
                if br is None:
                    break
                for i, o in enumerate(br.modifications):
                    if o.label == label:
                        _moving = bool(ed_is_moving)
                        _speed = max(0.0, float(ed_speed)) if _moving else 0.0
                        _route = o.route_lanelet_ids
                        _goal = o.goal_pose
                        # Recompute route when toggling static->moving
                        if _moving and not o.is_moving and map_builder is not None:
                            s = _safe_step(step)
                            seq = tree.get_npz_sequence(tree.active_branch)
                            ego_wp = _recover_ego_world_pose(seq, s) if seq else None
                            if ego_wp is not None:
                                _yaw_r = math.radians(round(float(yaw) / 5) * 5)
                                ci, si = math.cos(ego_wp[2]), math.sin(ego_wp[2])
                                _rx = round(float(x), 1)
                                _ry = round(float(y), 1)
                                wx = ego_wp[0] + ci * _rx - si * _ry
                                wy = ego_wp[1] + si * _rx + ci * _ry
                                wyaw = ego_wp[2] + _yaw_r
                                ll_id = map_builder.snap_to_nearest_ll(
                                    np.array([wx, wy]), heading_rad=wyaw,
                                )
                                if ll_id is not None:
                                    _route = map_builder.find_route(ll_id, min_length_m=150.0)
                                    g_arr = map_builder._route_goal(_route)
                                    _goal = (float(g_arr[0]), float(g_arr[1]), float(g_arr[2]))
                        br.modifications[i] = ObstaclePlacement(
                            label=label, timestep=o.timestep,
                            x=round(float(x), 1), y=round(float(y), 1),
                            yaw_deg=round(float(yaw) / 5) * 5,
                            length=float(length), width=float(width),
                            history_steps=int(history),
                            is_moving=_moving,
                            speed=_speed,
                            route_lanelet_ids=_route,
                            goal_pose=_goal,
                        )
                        _found = True
                        break
                bid = br.parent_id
            s = _safe_step(step)
            det_traj, guided = _recompute_trajs(tree, s, det_on, guided_on, g_args or None,
                                                zero_neighbors=hide_nb)
            img, info = _render(tree, s, view_r, label, show_gt_val=gt_on,
                                det_traj=det_traj, guided_trajs=guided,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb,
                                traj_rb=traj_rb_on, traj_nb=traj_nb_on)
            mods = _modifications_md(tree, tree.active_branch)
            return tree, img, info, mods, label, det_traj, guided

        def on_generate_guided(tree, step, gt_on, view_r, selected_obs,
                               noise, k, det_on, det_cache, hide_nb,
                               rb_on, nb_on, traj_rb_on, traj_nb_on,
                               anchor_idx, anchor_proto_path,
                               *guidance_args):
            if model_cache is None or not model_cache.available:
                return gr.update(), "No model loaded", det_cache, None

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

            det_traj = None
            if det_on:
                if det_cache is not None:
                    det_traj = det_cache
                else:
                    raw = model_cache.predict_det(npz_path, obstacles=obs or None,
                                                  zero_neighbors=hide_nb,
                                                  ego_shape_override=tree.ego_shape)
                    det_traj = _traj_cos_sin_to_xyh(raw)
                    det_cache = det_traj

            raw_guided = model_cache.predict_guided(
                npz_path, cfgs, noise_scale=float(noise), n_samples=int(k),
                obstacles=obs or None, zero_neighbors=hide_nb,
                ego_shape_override=tree.ego_shape,
                anchor_index=int(anchor_idx), anchor_path=str(anchor_proto_path),
            )
            guided_list = [_traj_cos_sin_to_xyh(raw_guided[i]) for i in range(raw_guided.shape[0])]

            img, info = _render(tree, _safe_step(step), view_r, selected_obs,
                                show_gt_val=gt_on, det_traj=det_traj,
                                guided_trajs=guided_list,
                                rb_dist=rb_on, nb_dist=nb_on,
                                hide_nb=hide_nb,
                                traj_rb=traj_rb_on, traj_nb=traj_nb_on)
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
            tree.set_crop(tree.active_branch, _safe_step(start), _safe_step(end))
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
                     + [guided_noise, guided_k, anchor_index_sl, anchor_path_tb])
        _overlay_inputs = [show_guided, hide_neighbors, show_rb_dist, show_nb_dist,
                           show_traj_rb, show_traj_nb]

        nav_inputs = ([tree_state, step_slider, view_half, selected_obstacle_state,
                       show_gt, show_det, hide_neighbors, show_rb_dist, show_nb_dist,
                       show_traj_rb, show_traj_nb,
                       show_guided, guided_trajs_state] + _g_inputs)
        nav_outputs = [scene_image, step_info, step_slider, det_traj_state, guided_trajs_state]

        step_slider.release(
            on_step_change, nav_inputs, nav_outputs,
        )

        def on_step_jump(tree, jump_val, view_r, selected_obs, gt_on, det_on,
                         hide_nb, rb_on, nb_on, traj_rb_on, traj_nb_on,
                         guided_on, prev_guided_cache, *g_args):
            s = _safe_step(jump_val)
            seq = tree.get_npz_sequence(tree.active_branch)
            s = min(s, max(0, len(seq) - 1)) if seq else 0
            det_traj, guided = _recompute_trajs(tree, s, det_on, guided_on, g_args or None,
                                                prev_guided=prev_guided_cache,
                                                zero_neighbors=hide_nb)
            img, info = _render(tree, s, view_r, selected_obs,
                                show_gt_val=gt_on, det_traj=det_traj, guided_trajs=guided,
                                rb_dist=rb_on, nb_dist=nb_on, hide_nb=hide_nb,
                                traj_rb=traj_rb_on, traj_nb=traj_nb_on)
            return img, info, s, det_traj, guided
        step_jump_btn.click(
            on_step_jump,
            [tree_state, step_jump_input, view_half, selected_obstacle_state,
             show_gt, show_det, hide_neighbors, show_rb_dist, show_nb_dist,
             show_traj_rb, show_traj_nb,
             show_guided, guided_trajs_state] + _g_inputs,
            nav_outputs,
        )
        _render_trigger_inputs = [tree_state, step_slider, view_half, selected_obstacle_state,
                                   show_gt, show_det, show_guided,
                                   hide_neighbors, show_rb_dist, show_nb_dist,
                                   show_traj_rb, show_traj_nb,
                                   det_traj_state, guided_trajs_state, show_nb_preds]
        _render_trigger_outputs = [scene_image, step_info, det_traj_state, guided_trajs_state]

        for _trigger in [view_half, show_gt, show_det, show_guided,
                         hide_neighbors, show_rb_dist, show_nb_dist,
                         show_traj_rb, show_traj_nb, show_nb_preds]:
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
             det_traj_state, guided_trajs_state,
             show_rb_dist, show_nb_dist, hide_neighbors, show_traj_rb, show_traj_nb],
            [scene_image, step_info],
        )

        # Toggle speed field visibility (no queue to avoid blocking)
        obs_is_moving.change(
            lambda v: gr.update(visible=v),
            [obs_is_moving], [obs_speed],
            queue=False,
        )
        edit_is_moving.change(
            lambda v: gr.update(visible=v),
            [edit_is_moving], [edit_speed],
            queue=False,
        )

        place_btn.click(
            on_place,
            [tree_state, step_slider, view_half,
             obs_x, obs_y, obs_yaw, obs_length, obs_width, obs_history,
             obs_is_moving, obs_speed,
             show_gt, show_det] + _overlay_inputs + _g_inputs,
            [tree_state, scene_image, step_info, mods_display,
             selected_obstacle_state, det_traj_state, guided_trajs_state,
             obs_select, obs_route_info],
        )

        obs_select.change(
            on_select_obstacle,
            [tree_state, obs_select, step_slider, view_half, show_gt,
             show_det, det_traj_state, guided_trajs_state,
             show_rb_dist, show_nb_dist, hide_neighbors, show_traj_rb, show_traj_nb],
            [scene_image, step_info, selected_obstacle_state,
             edit_x, edit_y, edit_yaw, edit_length, edit_width, edit_history,
             edit_is_moving, edit_speed, edit_speed],
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
            [edit_x, edit_y, edit_yaw, edit_length, edit_width, edit_history,
             edit_is_moving, edit_speed] + _g_inputs,
            [tree_state, scene_image, step_info, mods_display, selected_obstacle_state,
             det_traj_state, guided_trajs_state],
        )

        # Guidance generation button
        guidance_btn_inputs = ([tree_state, step_slider, show_gt, view_half,
                                selected_obstacle_state, guided_noise, guided_k,
                                show_det, det_traj_state, hide_neighbors,
                                show_rb_dist, show_nb_dist, show_traj_rb, show_traj_nb,
                                anchor_index_sl, anchor_path_tb]
                               + [v for gname in ALL_GUIDANCE_NAMES
                                  for v in (guidance_toggles[gname], guidance_scales[gname])])
        generate_guided_btn.click(
            on_generate_guided,
            guidance_btn_inputs,
            [scene_image, step_info, det_traj_state, guided_trajs_state],
        )

        # Anchor gallery: click selects index, path change reloads gallery
        def _on_anchor_select(evt: gr.SelectData):
            return int(evt.index)

        anchor_gallery.select(_on_anchor_select, None, anchor_index_sl)

        def _on_anchor_path_change(path):
            from guidance_gui.visualization import render_prototype_gallery as _rpg
            imgs = _rpg(path) or []
            k = len(imgs)
            return (gr.update(value=imgs),
                    gr.update(maximum=max(0, k - 1), value=0))

        anchor_path_tb.change(
            _on_anchor_path_change, [anchor_path_tb],
            [anchor_gallery, anchor_index_sl],
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

        # Export NPZs — copy branch sequence to output dir with sequential naming
        def on_export(tree, out_dir):
            import json as _json
            import shutil
            if not out_dir or not out_dir.strip():
                return "Specify an output directory"
            out = Path(out_dir.strip())
            seq = tree.get_npz_sequence(tree.active_branch)
            if not seq:
                return "No NPZ files to export"
            out.mkdir(parents=True, exist_ok=True)
            exported = []
            for i, src in enumerate(seq):
                dst = out / f"scene_{i:06d}.npz"
                shutil.copy2(src, dst)
                exported.append(str(dst))
            scene_list = out / "scene_list.json"
            with open(scene_list, "w") as f:
                _json.dump(exported, f, indent=2)
            return (f"Exported **{len(exported)}** NPZs to `{out}`\n\n"
                    f"Scene list: `{scene_list}`")

        export_btn.click(on_export, [tree_state, export_dir], [export_status])

        # Save for RSFT: bake guided trajectory into ego_agent_future
        def on_rsft_save(tree, step, out_dir, guided_cache):
            import json as _json
            if not out_dir or not out_dir.strip():
                return "Specify an RSFT output directory"
            if not guided_cache or len(guided_cache) == 0:
                return "Generate a guided trajectory first (Show Guided → Generate)"

            s = _safe_step(step)
            npz_path = _get_npz_path(tree, s)
            if not npz_path:
                return "No NPZ at current step"

            # guided_cache[0] is (80, 3) [x, y, heading_rad]
            traj_xyh = np.array(guided_cache[0]).astype(np.float32)

            # Convert to (T, 4) [x, y, cos, sin] for reward scoring
            traj_4col = torch.from_numpy(np.column_stack([
                traj_xyh[:, :2],
                np.cos(traj_xyh[:, 2]),
                np.sin(traj_xyh[:, 2]),
            ]).astype(np.float32)).unsqueeze(0)

            from preference_optimization.utils import load_npz_data as _load_npz
            from rlvr.reward import RewardConfig as _RC
            from rlvr.reward import compute_reward_batch as _crb
            scene_data = _load_npz(npz_path, torch.device("cpu"),
                                   ego_shape_override=tree.ego_shape)
            obs_at_step = _get_obstacles_at_step(tree, s)
            if obs_at_step:
                scene_data = _inject_obstacles_into_tensors(
                    scene_data, obs_at_step, torch.device("cpu"))
            # Ensure line_strings have border flags (channel 3+) for RB scoring.
            # Rebuild from lanelet2 map if the NPZ lacks them.
            ls_check = scene_data.get("line_strings")
            _has_rb = (ls_check is not None and ls_check.shape[-1] >= 4)
            if not _has_rb:
                if map_builder is None:
                    return ("**ERROR** — line_strings lack road border flags "
                            "and no --map_path provided. Pass --map_path to enable RB scoring.")
                ego_wp = _recover_ego_world_pose(
                    tree.get_npz_sequence(tree.active_branch), s)
                if ego_wp is None:
                    return ("**ERROR** — cannot recover ego world pose "
                            "(no sidecar JSON). Cannot rebuild line_strings for RB scoring.")
                from scenario_generation.npz_loader import from_npz as _fnpz
                from scenario_generation.simulate import _refresh_line_strings
                _tmp_scene = _fnpz(npz_path)
                _origin = np.array(ego_wp, dtype=np.float64)
                _refresh_line_strings(_tmp_scene, map_builder,
                                     _origin[:2], _origin)
                ls_t = torch.from_numpy(
                    _tmp_scene.map_data.line_strings).unsqueeze(0).float()
                scene_data["line_strings"] = ls_t
            rc = reward_config if reward_config is not None else _RC()
            if reward_config is None:
                rc.rb_gate_enabled = True
                rc.enable_lane_departure = True
            rewards = _crb(traj_4col, scene_data, rc)
            r = rewards[0]

            violations = []
            if r.rb_crossing:
                violations.append(f"Road border crossing (min dist {r.rb_min_dist:.2f}m)")
            if r.lane_crossing:
                violations.append("Lane departure")
            if r.kinematic_violated:
                violations.append("Kinematic infeasibility")
            if r.collision_step is not None:
                violations.append(f"Collision at timestep {r.collision_step}")
            if r.static_crossing:
                violations.append(f"Static obstacle crossing (min dist {r.sc_min_dist:.2f}m)")

            if violations:
                return ("**REJECTED** — trajectory violates reward gates:\n\n"
                        + "\n".join(f"- {v}" for v in violations)
                        + f"\n\nTotal reward: {r.total:.1f}")

            # Passed all gates — save
            out = Path(out_dir.strip())
            out.mkdir(parents=True, exist_ok=True)

            existing = list(out.glob("scene_*.npz"))
            if existing:
                nums = []
                for p in existing:
                    try:
                        nums.append(int(p.stem.split("_")[-1]))
                    except ValueError:
                        pass
                idx = max(nums) + 1 if nums else 0
            else:
                idx = 0

            # Start from the raw NPZ (pre-load_npz_data) to avoid
            # double heading_to_cos_sin conversion on ego_agent_past
            # and goal_pose. Then overlay fields that were rebuilt/modified.
            with np.load(npz_path) as raw:
                npz_data = {k: raw[k].astype(np.float32)
                            if raw[k].dtype == np.float64 else raw[k]
                            for k in raw.files}
            # Overlay obstacle-injected neighbors from scene_data
            if obs_at_step:
                for k in ("neighbor_agents_past", "neighbor_agents_future"):
                    if k in scene_data and isinstance(scene_data[k], torch.Tensor):
                        npz_data[k] = scene_data[k].squeeze(0).cpu().numpy()
            # Overlay rebuilt line_strings (4-col with border flags)
            if "line_strings" in scene_data:
                ls = scene_data["line_strings"]
                if isinstance(ls, torch.Tensor):
                    ls = ls.squeeze(0).cpu().numpy()
                if ls.shape[-1] >= 4:
                    npz_data["line_strings"] = ls.astype(np.float32)
            # Rebuild polygons from map (3-col with type) if source is 2-col
            if map_builder is not None and npz_data.get("polygons") is not None:
                if npz_data["polygons"].shape[-1] < 3:
                    ego_wp = _recover_ego_world_pose(
                        tree.get_npz_sequence(tree.active_branch), s)
                    if ego_wp is not None:
                        from scenario_generation.transforms import (
                            _rotation_matrix, transform_positions,
                        )
                        poly_world = map_builder.build_polygons_tensor(
                            np.array(ego_wp[:2], dtype=np.float32))
                        R_init = _rotation_matrix(float(ego_wp[2]) if len(ego_wp) > 2 else 0.0)
                        init_xy = np.array(ego_wp[:2], dtype=np.float64)
                        for pi in range(poly_world.shape[0]):
                            pts = poly_world[pi, :, :2]
                            valid = np.abs(pts).sum(axis=1) > 0.1
                            if valid.any():
                                poly_world[pi, valid, :2] = transform_positions(
                                    pts[valid].astype(np.float64), R_init, init_xy,
                                ).astype(np.float32)
                        npz_data["polygons"] = poly_world.astype(np.float32)
            npz_data["ego_agent_future"] = traj_xyh
            if tree.ego_shape:
                npz_data["ego_shape"] = np.array(list(tree.ego_shape), dtype=np.float32)
            # Sanity: crash if critical fields are missing
            for req in ("ego_agent_past", "neighbor_agents_past",
                        "lanes", "line_strings", "ego_shape",
                        "ego_current_state"):
                if req not in npz_data:
                    return f"**ERROR** — saved NPZ would be missing `{req}`. Fix upstream."
            dst = out / f"scene_{idx:04d}.npz"
            np.savez(dst, **npz_data)

            # Save sidecar JSON with ego world pose for future map rebuilds
            ego_wp = _recover_ego_world_pose(
                tree.get_npz_sequence(tree.active_branch), s)
            if ego_wp is not None:
                import math as _math
                yaw = float(ego_wp[2]) if len(ego_wp) > 2 else 0.0
                sidecar = dst.with_suffix(".json")
                with open(sidecar, "w") as _sf:
                    _json.dump({"x": float(ego_wp[0]), "y": float(ego_wp[1]),
                                "qx": 0.0, "qy": 0.0,
                                "qz": _math.sin(yaw / 2),
                                "qw": _math.cos(yaw / 2)}, _sf)

            scene_list_path = out / "scene_list.json"
            if scene_list_path.exists():
                with open(scene_list_path) as f:
                    scenes = _json.load(f)
            else:
                scenes = []
            scenes.append(str(dst))
            with open(scene_list_path, "w") as f:
                _json.dump(scenes, f, indent=2)

            return (f"**SAVED** scene **#{idx}** to `{dst}`\n\n"
                    f"Gates: RB={r.rb_min_dist:.2f}m, CL={r.centerline:.2f}, "
                    f"reward={r.total:.1f}\n\n"
                    f"Total: {len(scenes)} scenes in `{scene_list_path}`")

        rsft_save_btn.click(
            on_rsft_save,
            [tree_state, step_slider, rsft_dir, guided_trajs_state],
            [rsft_status],
        )

        # Simulate N steps — unified loop supporting independent ego/neighbor loop modes
        def on_simulate(tree, step, n_steps, advance_mode, use_guidance,
                        gt_on, view_r, hide_nb, ego_mode, neighbor_mode,
                        guided_cache, det_cache,
                        *guidance_args, progress=gr.Progress()):
            if model_cache is None or not model_cache.available:
                return (tree, gr.update(), "No model loaded -- pass `--model_path`",
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), None, None)

            s = _safe_step(step)

            new_id = tree.fork_branch(tree.active_branch, s)
            tree.active_branch = new_id
            branch = tree.branches[new_id]

            seq = tree.get_npz_sequence(branch.parent_id)
            if not seq:
                return (tree, gr.update(), "No NPZ sequence", gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(),
                        None, None)

            n = max(1, int(n_steps))
            npz_path = seq[min(s, len(seq) - 1)]

            out_dir = Path(tree.base_npz_dir).parent / f"branch_{new_id}_resim"
            if out_dir.exists():
                for old_f in out_dir.glob("*.npz"):
                    old_f.unlink()
            out_dir.mkdir(parents=True, exist_ok=True)

            progress(0, desc="Loading model...")
            model_cache._ensure_loaded()

            from copy import deepcopy

            from scenario_generation.mpc_tracker import PerfectTracker
            from scenario_generation.npz_loader import from_npz as _from_npz
            from scenario_generation.simulate import (
                _advance_agent,
                _predict_batch,
                advance_scene_mpc,
                _refresh_line_strings,
            )
            from scenario_generation.tensor_converter import MapTensorCache, dump_step_npz

            scene = _from_npz(npz_path)

            # ── Collect obstacle placements (deep walk ignores npz_dir boundaries) ──
            # This ensures moving-neighbor metadata survives across resim forks.
            all_obstacles = tree.get_all_obstacles_deep(tree.active_branch)
            obs_at_step = []
            for o in all_obstacles:
                if o.timestep > s:
                    continue
                if o.timestep != s and seq:
                    nx, ny, nyaw = _transform_point_between_steps(
                        seq, o.timestep, s, o.x, o.y, o.yaw_rad,
                    )
                    obs_at_step.append(ObstaclePlacement(
                        label=o.label, timestep=o.timestep,
                        x=nx, y=ny, yaw_deg=math.degrees(nyaw),
                        length=o.length, width=o.width,
                        history_steps=o.history_steps,
                        is_moving=o.is_moving, speed=o.speed,
                        route_lanelet_ids=o.route_lanelet_ids,
                        goal_pose=o.goal_pose,
                    ))
                else:
                    obs_at_step.append(o)

            ego_wp = _recover_ego_world_pose(seq, min(s, len(seq) - 1))
            ego_wp_arr = np.array([ego_wp[0], ego_wp[1], ego_wp[2]]) if ego_wp is not None else None

            from scenario_generation.scene_context import Agent, AgentType
            moving_ids: set[str] = set()
            static_ids: set[str] = set()

            # If starting from a previous resim, placed agents lost their IDs
            # in the NPZ round-trip (placed_X -> neighbor_N). Read the saved
            # ID mapping to identify and remove them before re-injection.
            import json as _json_placed
            _placed_map_path = Path(npz_path).parent / "_placed_ids.json"
            if _placed_map_path.exists():
                _saved_map = _json_placed.loads(_placed_map_path.read_text())
                # _saved_map: {neighbor_index: placed_id}
                # Remove the NPZ neighbors that are actually placed agents
                _npz_ids_to_remove = set()
                for _ni_str, _pid in _saved_map.items():
                    _nb_id = f"neighbor_{_ni_str}"
                    _npz_ids_to_remove.add(_nb_id)
                scene.agents = [a for a in scene.agents
                                if a.id not in _npz_ids_to_remove]

            for obs in obs_at_step:
                aid = f"placed_{obs.label}"
                T_PAST = 31
                _is_mov = obs.is_moving and obs.speed > 0

                # Check if agent still exists by exact ID (only on first sim)
                existing = next((a for a in scene.agents if a.id == aid), None)
                if existing is not None:
                    if _is_mov:
                        moving_ids.add(aid)
                        existing.route_lanelet_ids = obs.route_lanelet_ids
                    else:
                        static_ids.add(aid)
                    continue

                if _is_mov:
                    moving_ids.add(aid)
                    agent = _build_moving_agent(
                        obs, map_builder, ego_wp_arr,
                    )
                else:
                    static_ids.add(aid)
                    history = np.tile([obs.x, obs.y, obs.yaw_rad], (T_PAST, 1)).astype(np.float32)
                    velocities = np.zeros((T_PAST, 2), dtype=np.float32)
                    h = getattr(obs, "history_steps", 30)
                    agent = Agent(
                        id=aid,
                        agent_type=AgentType.VEHICLE,
                        length=obs.length, width=obs.width,
                        wheelbase=obs.length * 0.65,
                        past_trajectory=history,
                        past_velocities=velocities,
                        age_steps=min(h, T_PAST - 1),
                    )
                scene.agents.append(agent)

            placed_ids = static_ids | moving_ids

            # ── Guidance setup ──
            model = model_cache._model
            model_args = model_cache._model_args
            _orig_guidance_fn = model.decoder._guidance_fn
            _orig_guidance_scale = model.decoder._guidance_scale
            if use_guidance and guidance_args:
                from diffusion_planner.model.guidance.composer import GuidanceComposer
                from diffusion_planner.model.guidance.config import (
                    GuidanceConfig,
                    GuidanceSetConfig,
                )
                _sim_anchor_idx = int(guidance_args[-2]) if len(guidance_args) >= 2 else 0
                _sim_anchor_path = str(guidance_args[-1]) if len(guidance_args) > 1 else ""
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
                        if gname == "anchor_following" and _sim_anchor_path:
                            params["prototypes_path"] = _sim_anchor_path
                            params["anchor_index"] = _sim_anchor_idx
                        fns.append(GuidanceConfig(name=gname, enabled=True,
                                                   scale=float(scale), params=params))
                if fns:
                    set_cfg = GuidanceSetConfig(functions=fns, global_scale=1.0)
                    composer = GuidanceComposer(set_cfg)
                    model.decoder._guidance_fn = composer
                    model.decoder._guidance_scale = 1.0

            # ── Ego open-loop plan ──
            ego_ol = (ego_mode == "open-loop")
            ego_plan = None
            if ego_ol:
                if guided_cache and len(guided_cache) > 0:
                    traj_xyh = np.array(guided_cache[0])
                    ego_plan = np.column_stack([
                        traj_xyh[:, :2],
                        np.cos(traj_xyh[:, 2]),
                        np.sin(traj_xyh[:, 2]),
                    ]).astype(np.float32)
                if ego_plan is None and det_cache is not None:
                    traj_xyh = np.array(det_cache)
                    ego_plan = np.column_stack([
                        traj_xyh[:, :2],
                        np.cos(traj_xyh[:, 2]),
                        np.sin(traj_xyh[:, 2]),
                    ]).astype(np.float32)
                if ego_plan is None:
                    return (tree, gr.update(),
                            "Ego open-loop requires a DET or guided trajectory -- "
                            "toggle Show DET or generate guided first",
                            gr.update(), gr.update(), gr.update(), gr.update(),
                            gr.update(), None, None)
                n = min(n, ego_plan.shape[0])

            # ── Neighbor open-loop references ──
            nb_ol = (neighbor_mode == "open-loop")
            neighbor_refs: dict[str, np.ndarray] = {}
            if nb_ol and moving_ids:
                for nid in moving_ids:
                    agent = scene.get_agent(nid)
                    neighbor_refs[nid] = _generate_neighbor_reference(
                        agent, map_builder, ego_wp_arr, n,
                    )

            # ── Determine which IDs need model prediction ──
            ids_to_predict: list[str] = []
            ego_id = scene.ego_agent_id
            if not ego_ol:
                ids_to_predict.append(ego_id)
            if not nb_ol:
                ids_to_predict.extend(sorted(moving_ids))

            scene_sim = deepcopy(scene)
            ego_id = scene_sim.ego_agent_id

            # Save placed agent ID mapping so subsequent resims can identify
            # them after the NPZ round-trip strips their IDs.
            # The tensor converter sorts neighbors by distance from ego, so
            # record which distance-rank each placed agent lands at.
            import json as _json_placed
            _ego_pos = scene_sim.get_agent(ego_id).current_position
            _nb_agents = [(a, math.hypot(a.current_position[0] - _ego_pos[0],
                                          a.current_position[1] - _ego_pos[1]))
                          for a in scene_sim.agents if a.id != ego_id]
            _nb_agents.sort(key=lambda x: x[1])
            _placed_map = {}
            for _rank, (_a, _) in enumerate(_nb_agents):
                if _a.id in placed_ids:
                    _placed_map[str(_rank)] = _a.id
            (out_dir / "_placed_ids.json").write_text(_json_placed.dumps(_placed_map))

            # Map refresh setup
            if map_builder is not None and ego_wp_arr is not None:
                _refresh_line_strings(
                    scene_sim, map_builder, ego_wp_arr[:2], ego_wp_arr,
                )
            map_cache_sim = MapTensorCache(scene_sim.map_data)
            _init_yaw = float(ego_wp_arr[2]) if ego_wp_arr is not None else 0.0

            trackers: dict = {}

            try:
                for t in range(n):
                    progress((t + 1) / n, f"Sim step {t+1}/{n}")

                    # Map refresh every 5 steps
                    if (map_builder is not None and ego_wp_arr is not None
                            and t > 0 and t % 5 == 0):
                        ep = scene_sim.get_agent(ego_id).current_position
                        eh = scene_sim.get_agent(ego_id).current_heading
                        ci, si = math.cos(_init_yaw), math.sin(_init_yaw)
                        cur_wx = ego_wp_arr[0] + ci * ep[0] - si * ep[1]
                        cur_wy = ego_wp_arr[1] + si * ep[0] + ci * ep[1]
                        _refresh_line_strings(
                            scene_sim, map_builder,
                            np.array([cur_wx, cur_wy], dtype=np.float64),
                            ego_wp_arr,
                        )
                        map_cache_sim = MapTensorCache(scene_sim.map_data)

                    # Model prediction for closed-loop agents
                    preds: dict[str, np.ndarray] = {}
                    if ids_to_predict:
                        _live_ids = [aid for aid in ids_to_predict
                                     if scene_sim.get_agent(aid) is not None]
                        if _live_ids:
                            if hide_nb:
                                _keep = {ego_id} | placed_ids
                                _saved = scene_sim.agents[:]
                                scene_sim.agents = [a for a in scene_sim.agents
                                                    if a.id in _keep]
                                preds = _predict_batch(
                                    model, model_args, scene_sim, _live_ids,
                                    str(model_cache._device), map_cache=map_cache_sim,
                                )
                                scene_sim.agents = _saved
                            else:
                                preds = _predict_batch(
                                    model, model_args, scene_sim, _live_ids,
                                    str(model_cache._device), map_cache=map_cache_sim,
                                )

                    # Dump NPZ
                    npz_data = dump_step_npz(
                        scene_sim, map_cache_sim,
                        future_len=model_args.future_len,
                    )
                    npz_data["ego_agent_future"] = np.zeros(
                        (model_args.future_len, 3), dtype=np.float32)
                    if ego_wp_arr is not None:
                        import json as _json_sim
                        ep = scene_sim.get_agent(ego_id).current_position
                        eh = scene_sim.get_agent(ego_id).current_heading
                        ci, si = math.cos(_init_yaw), math.sin(_init_yaw)
                        wx = ego_wp_arr[0] + ci * ep[0] - si * ep[1]
                        wy = ego_wp_arr[1] + si * ep[0] + ci * ep[1]
                        wyaw = _init_yaw + eh
                        sidecar = {"x": float(wx), "y": float(wy),
                                   "qz": math.sin(wyaw / 2), "qw": math.cos(wyaw / 2),
                                   "qx": 0.0, "qy": 0.0}
                        (out_dir / f"replay_step_{t:04d}.json").write_text(
                            _json_sim.dumps(sidecar))
                    np.savez(out_dir / f"replay_step_{t:04d}.npz", **npz_data)

                    if t >= n - 1:
                        break

                    # ── Advance ego ──
                    if ego_ol:
                        step_pred = ego_plan[t]
                        new_heading = float(np.arctan2(step_pred[3], step_pred[2]))
                        new_pos = np.array([float(step_pred[0]), float(step_pred[1]),
                                            new_heading], dtype=np.float32)
                        _advance_agent(scene_sim.get_agent(ego_id), new_pos)
                    elif ego_id in preds:
                        advance_scene_mpc(
                            scene_sim, {ego_id: preds[ego_id]}, trackers,
                            tracker_type=advance_mode,
                        )

                    # ── Advance moving neighbors ──
                    if not nb_ol and moving_ids:
                        nb_preds = {nid: preds[nid] for nid in moving_ids if nid in preds}
                        if nb_preds:
                            advance_scene_mpc(
                                scene_sim, nb_preds, trackers,
                                tracker_type="perfect",
                            )
                    elif nb_ol and moving_ids:
                        for nid in moving_ids:
                            agent = scene_sim.get_agent(nid)
                            if agent is None:
                                continue
                            ref = neighbor_refs.get(nid)
                            if ref is None or t >= len(ref):
                                continue
                            if nid not in trackers:
                                trackers[nid] = PerfectTracker(dt=0.1)
                            vel = agent.current_velocity
                            speed = float(np.linalg.norm(vel))
                            x0 = np.array([
                                float(agent.current_position[0]),
                                float(agent.current_position[1]),
                                float(agent.current_heading),
                                speed,
                            ], dtype=np.float64)
                            new_pos, new_speed = trackers[nid].track(x0, ref[t:])
                            _advance_agent(agent, new_pos, dt=0.1,
                                           new_speed=float(new_speed))
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
            choices = list(tree.branches.keys())
            _modes = f"ego={ego_mode}, nb={neighbor_mode}"
            status = (f"Simulated **{n}** steps ({advance_mode}, {_modes}) "
                      f"on branch `{new_id}`. Output: `{out_dir}`")
            return (tree, img, status, b_info, mods,
                    gr.update(choices=choices, value=new_id),
                    gr.update(maximum=max_step, value=0), info, None, None)

        _sim_inputs = ([tree_state, step_slider, sim_steps, sim_mode, sim_use_guidance,
                        show_gt, view_half, hide_neighbors, sim_ego_mode, sim_neighbor_mode,
                        guided_trajs_state, det_traj_state]
                       + [v for gname in ALL_GUIDANCE_NAMES
                          for v in (guidance_toggles[gname], guidance_scales[gname])]
                       + [anchor_index_sl, anchor_path_tb])
        sim_btn.click(
            on_simulate,
            _sim_inputs,
            [tree_state, scene_image, sim_status, branch_info, mods_display,
             branch_dropdown, step_slider, step_info, det_traj_state, guided_trajs_state],
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
                            is_moving=o.is_moving, speed=o.speed,
                            route_lanelet_ids=o.route_lanelet_ids,
                            goal_pose=o.goal_pose,
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
                ego_wp = _recover_ego_world_pose(seq, s) if (map_borders or map_builder) else None
                if (scene.map_data is not None
                        and scene.map_data.line_strings is not None
                        and scene.map_data.line_strings.shape[-1] < 4
                        and map_builder is not None and ego_wp is not None):
                    from scenario_generation.simulate import _refresh_line_strings as _rls2
                    _rls2(scene, map_builder,
                          np.array(ego_wp[:2], dtype=np.float64),
                          np.array(ego_wp, dtype=np.float64))
                fig = render_scene_at_step(
                    scene, obs_at_step, None,
                    view_half=view_r, step_idx=s, total_steps=len(seq),
                    gt_traj=gt_traj_r,
                    show_rb_dist=rb_on, show_nb_dist=nb_on,
                    dim_neighbors=hide_nb,
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
    lines = ["| Label | Step | X | Y | Yaw | Size | Type |",
             "|-------|------|---|---|-----|------|------|"]
    for o in branch.modifications:
        _type = f"{o.speed:.1f} m/s" if o.is_moving else "static"
        lines.append(
            f"| `{o.label}` | {o.timestep} | {o.x:.1f} | {o.y:.1f} "
            f"| {o.yaw_deg:.0f} | {o.length}x{o.width} | {_type} |"
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
    elif ego_shape_override:
        tree = SceneTree.create_from_npz_dir_with_shape(
            args.npz_dir, ego_shape_override)
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

    reward_cfg = None
    if args.reward_config:
        from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
        reward_cfg = load_reward_config(args.reward_config)
        print(f"Loaded reward config from {args.reward_config}")

    demo = build_interface(tree, model_cache=mc, map_borders=map_border_polylines,
                           map_builder=builder, reward_config=reward_cfg)
    demo.launch(server_name="0.0.0.0", server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
