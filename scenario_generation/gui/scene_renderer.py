"""Render a generated SceneContext as a matplotlib figure.

Provides both standalone scene rendering (with map background from lanelet data)
and agent info extraction for the GUI sidebar.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from scenario_generation.scene_context import AgentType, SceneContext

# Colors (same palette as visualize.py)
_EGO_COLOR = "#3366cc"
_NEIGHBOR_COLORS = [
    "#e67300", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]
_UNFOCUSED_COLOR = "#aaaaaa"
_LANE_COLOR = "#bbbbbb"


def _agent_color(agent_type: AgentType, neighbor_idx: int) -> str:
    if agent_type == AgentType.PEDESTRIAN:
        return "#ff6699"
    if agent_type == AgentType.BICYCLE:
        return "#66ccff"
    return _NEIGHBOR_COLORS[neighbor_idx % len(_NEIGHBOR_COLORS)]


def _draw_box(ax, x, y, heading, length, width, color, alpha=0.8, lw=1.5, zorder=10):
    rear_overhang = (length - length * 0.65) / 2.0
    t_rot = mtransforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
    rect = Rectangle(
        (-rear_overhang, -width / 2), length, width,
        lw=lw, ec=color, fc=color, alpha=alpha, zorder=zorder, transform=t_rot,
    )
    ax.add_patch(rect)


def _draw_heading_arrow(ax, x, y, heading, length, color, lw=2.0, zorder=11):
    arrow_len = max(length, 2.5)
    dx = arrow_len * math.cos(heading)
    dy = arrow_len * math.sin(heading)
    ax.annotate(
        "", xy=(x + dx, y + dy), xytext=(x, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=12),
        zorder=zorder,
    )


def _draw_goal_marker(ax, gx, gy, color, size=12, zorder=15, label=None):
    ax.plot(gx, gy, "X", color=color, ms=size, zorder=zorder,
            markeredgecolor="black", markeredgewidth=0.8)
    if label:
        ax.annotate(
            label, (gx, gy), fontsize=6, color=color,
            ha="center", va="bottom", zorder=zorder + 1,
            xytext=(0, 8), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.7, lw=0.5),
        )


def _draw_footprint_history(ax, past_traj, length, width, color, alpha=0.12, zorder=5):
    valid = np.abs(past_traj[:, :2]).sum(axis=1) > 1e-6
    valid_indices = np.where(valid)[0]
    for idx in valid_indices[::5]:
        _draw_box(
            ax, past_traj[idx, 0], past_traj[idx, 1], past_traj[idx, 2],
            length, width, color, alpha=alpha, lw=0.3, zorder=zorder,
        )


def _draw_route(ax, route_lanes, color, alpha=0.4, lw=2.0, zorder=3):
    if route_lanes is None:
        return
    for i in range(route_lanes.shape[0]):
        lane = route_lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        pts = lane[:, :2]
        valid = np.abs(pts).sum(axis=1) > 0.1
        if valid.sum() < 2:
            continue
        ax.plot(pts[valid, 0], pts[valid, 1], "-", color=color, alpha=alpha, lw=lw, zorder=zorder)


def _draw_past_trail(ax, past_traj, color, alpha=0.4, lw=1.0, zorder=4):
    valid = np.abs(past_traj[:, :2]).sum(axis=1) > 1e-6
    if valid.sum() < 2:
        return
    pts = past_traj[valid]
    ax.plot(pts[:, 0], pts[:, 1], "--", color=color, lw=lw, alpha=alpha, zorder=zorder)


def _draw_lanes(ax, map_data, alpha=0.5):
    """Draw lane centerlines and boundaries from MapData."""
    lanes = map_data.lanes
    segments_cl = []
    segments_lb = []
    segments_rb = []

    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        pts = lane[:, :2]
        valid = np.abs(pts).sum(axis=1) > 0.1
        if valid.sum() < 2:
            continue
        segments_cl.append(pts[valid])
        if lane.shape[1] > 7:
            lb = (pts + lane[:, 4:6])[valid]
            rb = (pts + lane[:, 6:8])[valid]
            segments_lb.append(lb)
            segments_rb.append(rb)

    if segments_cl:
        lc = LineCollection(segments_cl, colors=_LANE_COLOR, linewidths=0.6, alpha=alpha * 0.5)
        ax.add_collection(lc)
    for segs in (segments_lb, segments_rb):
        if segs:
            lc = LineCollection(segs, colors=_LANE_COLOR, linewidths=0.8, alpha=alpha)
            ax.add_collection(lc)


def render_scene_figure(
    scene: SceneContext,
    focus_agent_id: str | None = None,
    figsize: tuple[float, float] = (12, 10),
) -> Figure:
    """Render the full scene as a matplotlib Figure.

    Args:
        scene: The generated SceneContext.
        focus_agent_id: If set, show full detail for this agent, minimal for others.
            None or "all" shows all agents with moderate detail.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object (can be passed directly to gr.Plot).
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor("#f8f8f8")

    # Draw lane map background
    _draw_lanes(ax, scene.map_data)

    show_all = focus_agent_id is None or focus_agent_id == "all"

    nb_idx = 0
    for agent in scene.agents:
        is_ego = (agent.id == scene.ego_agent_id)
        is_focused = show_all or (agent.id == focus_agent_id)

        if is_ego:
            color = _EGO_COLOR
        else:
            color = _agent_color(agent.agent_type, nb_idx)
            nb_idx += 1

        pos = agent.current_position
        heading = agent.current_heading

        if is_focused:
            _draw_route(ax, agent.route_lanes, color=color, alpha=0.5, lw=2.5, zorder=3)
            _draw_footprint_history(
                ax, agent.past_trajectory, agent.length, agent.width, color,
                alpha=0.15, zorder=5,
            )
            _draw_past_trail(ax, agent.past_trajectory, color, alpha=0.5, lw=1.2, zorder=4)
            _draw_box(
                ax, pos[0], pos[1], heading, agent.length, agent.width,
                color, alpha=0.8 if is_ego else 0.6, lw=2.5 if is_ego else 2.0, zorder=10,
            )
            _draw_heading_arrow(ax, pos[0], pos[1], heading, agent.length, color, lw=2.5, zorder=11)

            if agent.goal_pose is not None:
                _draw_goal_marker(
                    ax, agent.goal_pose[0], agent.goal_pose[1],
                    color, size=14, label=f"{agent.id} goal", zorder=15,
                )

            speed = float(np.linalg.norm(agent.current_velocity))
            ax.annotate(
                f"{agent.id} ({speed:.1f} m/s)", (pos[0], pos[1]),
                fontsize=7, color=color, ha="center", va="bottom", zorder=12,
                xytext=(0, 8), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8, lw=0.5),
            )
        else:
            _draw_past_trail(ax, agent.past_trajectory, _UNFOCUSED_COLOR, alpha=0.2, lw=0.6)
            _draw_box(
                ax, pos[0], pos[1], heading, agent.length, agent.width,
                _UNFOCUSED_COLOR, alpha=0.3, lw=0.8, zorder=8,
            )
            _draw_heading_arrow(
                ax, pos[0], pos[1], heading, agent.length, _UNFOCUSED_COLOR, lw=1.0, zorder=9,
            )
            if agent.goal_pose is not None:
                ax.plot(
                    agent.goal_pose[0], agent.goal_pose[1], "x",
                    color=_UNFOCUSED_COLOR, ms=6, zorder=8, mew=1.0,
                )

    # Auto-zoom to scene extent
    all_pts = []
    for agent in scene.agents:
        valid = np.abs(agent.past_trajectory[:, :2]).sum(axis=1) > 1e-6
        if valid.sum() > 0:
            all_pts.append(agent.past_trajectory[valid, :2])
        if agent.goal_pose is not None:
            all_pts.append(agent.goal_pose[:2].reshape(1, 2))
    if all_pts:
        pts = np.vstack(all_pts)
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        half = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))) * 0.6 + 20
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Legend
    from matplotlib.lines import Line2D
    handles = []
    nb_idx = 0
    for agent in scene.agents:
        is_ego = (agent.id == scene.ego_agent_id)
        if is_ego:
            color = _EGO_COLOR
        else:
            color = _agent_color(agent.agent_type, nb_idx)
            nb_idx += 1
        handles.append(Line2D([0], [0], color=color, lw=2, label=agent.id))
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper left")

    fig.tight_layout()
    return fig


def get_agent_info(scene: SceneContext, agent_id: str) -> str:
    """Return a markdown string with agent details."""
    try:
        agent = scene.get_agent(agent_id)
    except KeyError:
        return "No agent selected"

    pos = agent.current_position
    heading_deg = math.degrees(agent.current_heading)
    speed = float(np.linalg.norm(agent.current_velocity))

    lines = [
        f"**{agent.id}** ({agent.agent_type.value})",
        f"- Position: ({pos[0]:.1f}, {pos[1]:.1f})",
        f"- Heading: {heading_deg:.1f} deg",
        f"- Speed: {speed:.1f} m/s",
        f"- Size: {agent.length:.2f} x {agent.width:.2f} m",
    ]

    if agent.goal_pose is not None:
        gx, gy = agent.goal_pose[0], agent.goal_pose[1]
        dist = float(np.linalg.norm(pos - agent.goal_pose[:2]))
        lines.append(f"- Goal: ({gx:.1f}, {gy:.1f}), dist={dist:.0f}m")

    if agent.route_lanes is not None:
        n_segs = int((np.abs(agent.route_lanes[:, :, :2]).sum(axis=(1, 2)) > 1e-6).sum())
        lines.append(f"- Route: {n_segs} lane segments")

    return "\n".join(lines)
