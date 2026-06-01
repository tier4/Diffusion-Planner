"""Central scene-rendering library (gradio-free).

Canonical renderer shared by the Scene Branch Editor GUI and standalone viz
tools. ``render_scene_at_step`` draws the full scene (lanes, road borders,
route, traffic-light overlay, neighbor OBBs, ego box, trajectory overlays with
rear-axle-correct footprints, and RB / neighbor clearance lines). Import this
module instead of reimplementing scene rendering in individual tools.
"""
from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import Rectangle

from scenario_generation.scene_context import AgentType, SceneContext
from scenario_generation.tools.scene_tree import ObstaclePlacement
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

# --- render-only color/view constants (moved from scene_branch_editor) ---
_PLACED_COLOR = "#ff8800"
_PLACED_MOVING_COLOR = "#cc44ff"
_PLACED_SELECTED_COLOR = "#ff2200"
_DET_COLOR = "#0088ff"
_GUIDED_COLORS = ["#ff00aa", "#aa00ff", "#00ccaa", "#ffaa00"]
_GT_COLOR = "#22bb22"
_VIEW_HALF_DEFAULT = 50.0


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
            rear_ovh = agent.length / 2  # placed obstacle is a neighbor (centroid-referenced)
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
                # Ego pose is the rear axle (base_link); offset the box forward.
                # Neighbors are centroid-referenced — leave wheelbase=None.
                wheelbase=(float(ego.wheelbase)
                           if is_ego and ego is not None else None),
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

            # Draw OBB with dashed outline (placed obstacle = neighbor, centroid-referenced)
            rear_overhang = obs.length / 2
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
                float(ego.wheelbase),  # ego pose is rear axle — offset forward
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
    # Ego pose is the rear axle (base_link); footprints must offset forward.
    ego_wb = float(ego.wheelbase) if ego else None
    if gt_traj is not None and gt_traj.shape[0] > 1:
        draw_trajectory(ax, gt_traj, _GT_COLOR, label="GT", lw=2.0, zorder=25,
                        show_footprints=True, length=ego_len, width=ego_wid,
                        wheelbase=ego_wb)
    if det_traj is not None and det_traj.shape[0] > 1:
        draw_trajectory(ax, det_traj, _DET_COLOR, label="DET", lw=2.0, zorder=26,
                        show_footprints=True, length=ego_len, width=ego_wid,
                        wheelbase=ego_wb)
    if guided_trajs:
        for i, gt in enumerate(guided_trajs):
            if gt is not None and gt.shape[0] > 1:
                color = _GUIDED_COLORS[i % len(_GUIDED_COLORS)]
                draw_trajectory(ax, gt, color, label=f"Guided #{i+1}", lw=1.5, zorder=27,
                                show_footprints=False, length=ego_len, width=ego_wid,
                                wheelbase=ego_wb)

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
                                       alpha=0.4, lw=2.0, zorder=38,
                                       wheelbase=float(ego.wheelbase))
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
                    ego_c = _obb_corners(px, py, ph, ego_len, ego_wid,
                                         float(ego.wheelbase))
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
                                   alpha=0.4, lw=2.0, zorder=38,
                                   wheelbase=float(ego.wheelbase))
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


