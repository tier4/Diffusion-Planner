"""Render a generated SceneContext as a canvas figure.

Provides both standalone scene rendering (with map background from lanelet data)
and agent info extraction for the GUI sidebar.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from scenario_generation.canvas import Figure, Panel, nice_ticks, oriented_box
from scenario_generation.scene_context import AgentType, SceneContext

# Colors (same palette as visualize.py)
_EGO_COLOR = "#3366cc"
_NEIGHBOR_COLORS = [
    "#e67300",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]
_UNFOCUSED_COLOR = "#aaaaaa"
_LANE_COLOR = "#bbbbbb"

View = Callable[[np.ndarray], np.ndarray]


def _identity(xy: np.ndarray) -> np.ndarray:
    """Data points unchanged."""
    return np.asarray(xy, dtype=np.float64)


def _agent_color(agent_type: AgentType, neighbor_idx: int) -> str:
    if agent_type == AgentType.PEDESTRIAN:
        return "#ff6699"
    if agent_type == AgentType.BICYCLE:
        return "#66ccff"
    return _NEIGHBOR_COLORS[neighbor_idx % len(_NEIGHBOR_COLORS)]


def _draw_box(
    ax: Panel, x, y, heading, length, width, color, alpha=0.8, lw=1.5, zorder=10, view=_identity
):
    rear_overhang = (length - length * 0.65) / 2.0
    corners = oriented_box(float(x), float(y), float(heading), length, width, rear_overhang)
    ax.polygons([view(corners)], fill=color, edge=color, width=lw, alpha=alpha, z=zorder)


def _draw_heading_arrow(ax: Panel, x, y, heading, length, color, lw=2.0, zorder=11, view=_identity):
    arrow_len = max(length, 2.5)
    dx = arrow_len * math.cos(heading)
    dy = arrow_len * math.sin(heading)
    ax.arrows(view(np.array([x, y])), view(np.array([x + dx, y + dy])), color, lw, 9, z=zorder)


def _draw_goal_marker(ax: Panel, gx, gy, color, size=12, zorder=15, label=None, view=_identity):
    goal = view(np.array([gx, gy]))
    ax.markers(goal, "X", size, color, edge="black", edge_width=0.8, z=zorder)
    if label:
        ax.text(
            tuple(goal),
            label,
            6,
            color,
            ha="center",
            va="bottom",
            offset=(0, 8),
            box={"face": "#ffffff", "edge": color, "alpha": 0.7, "pad": 0.15},
            z=zorder + 1,
        )


def _draw_footprint_history(
    ax: Panel, past_traj, length, width, color, alpha=0.12, zorder=5, view=_identity
):
    valid = np.abs(past_traj[:, :2]).sum(axis=1) > 1e-6
    valid_indices = np.where(valid)[0]
    for idx in valid_indices[::5]:
        _draw_box(
            ax,
            past_traj[idx, 0],
            past_traj[idx, 1],
            past_traj[idx, 2],
            length,
            width,
            color,
            alpha=alpha,
            lw=0.3,
            zorder=zorder,
            view=view,
        )


def _draw_route(ax: Panel, route_lanes, color, alpha=0.4, lw=2.0, zorder=3, view=_identity):
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
        ax.lines([view(pts[valid])], color, width=lw, alpha=alpha, z=zorder)


def _draw_past_trail(ax: Panel, past_traj, color, alpha=0.4, lw=1.0, zorder=4, view=_identity):
    valid = np.abs(past_traj[:, :2]).sum(axis=1) > 1e-6
    if valid.sum() < 2:
        return
    pts = past_traj[valid]
    ax.lines([view(pts[:, :2])], color, width=lw, alpha=alpha, dash="dash", z=zorder)


def _draw_lanes(ax: Panel, map_data, alpha=0.5, view=_identity):
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
        segments_cl.append(view(pts[valid]))
        if lane.shape[1] > 7:
            lb = (pts + lane[:, 4:6])[valid]
            rb = (pts + lane[:, 6:8])[valid]
            segments_lb.append(view(lb))
            segments_rb.append(view(rb))

    ax.lines(segments_cl, _LANE_COLOR, width=0.6, alpha=alpha * 0.5)
    for segs in (segments_lb, segments_rb):
        ax.lines(segs, _LANE_COLOR, width=0.8, alpha=alpha)


def _scene_limits(scene: SceneContext, focus_agent_id, rotation, zoom):
    """The ``(center, half)`` auto-zoom of the scene, or ``None`` when it has no points."""
    all_pts = []
    for agent in scene.agents:
        valid = np.abs(agent.past_trajectory[:, :2]).sum(axis=1) > 1e-6
        if valid.sum() > 0:
            all_pts.append(agent.past_trajectory[valid, :2])
        if agent.goal_pose is not None:
            all_pts.append(agent.goal_pose[:2].reshape(1, 2))
    # Include lane centerline extent
    lanes = scene.map_data.lanes
    for i in range(lanes.shape[0]):
        pts_l = lanes[i, :, :2]
        valid_l = np.abs(pts_l).sum(axis=1) > 0.1
        if valid_l.sum() > 0:
            all_pts.append(pts_l[valid_l])
    if not all_pts:
        return None
    pts = np.vstack(all_pts)
    # Zoom center: focused agent if set, otherwise scene center
    focus_center = None
    if focus_agent_id and focus_agent_id != "all":
        try:
            fa = scene.get_agent(focus_agent_id)
            focus_center = fa.current_position
        except KeyError:
            pass

    if abs(rotation) > 0.01:
        cos_r, sin_r = math.cos(-rotation), math.sin(-rotation)
        cx_w, cy_w = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        dx, dy = pts[:, 0] - cx_w, pts[:, 1] - cy_w
        rx = dx * cos_r - dy * sin_r
        ry = dx * sin_r + dy * cos_r
        half = max(float(rx.max() - rx.min()), float(ry.max() - ry.min())) * 0.6 + 20
    else:
        cx_w, cy_w = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        half = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))) * 0.6 + 20

    # Apply zoom (>1 = zoom in = smaller half)
    safe_zoom = max(0.1, float(zoom))
    half = half / safe_zoom

    if focus_center is not None:
        cx_w, cy_w = float(focus_center[0]), float(focus_center[1])
    return (cx_w, cy_w), half


def _rotation_view(center: tuple[float, float], rotation: float) -> View:
    """Rotate data points by ``-rotation`` around ``center``."""
    cos_r, sin_r = math.cos(-rotation), math.sin(-rotation)
    cx, cy = center

    def view(xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        dx, dy = xy[..., 0] - cx, xy[..., 1] - cy
        return np.stack([cx + dx * cos_r - dy * sin_r, cy + dx * sin_r + dy * cos_r], -1)

    return view


def render_scene_figure(
    scene: SceneContext,
    focus_agent_id: str | None = None,
    rotation: float = 0.0,
    zoom: float = 1.0,
    figsize: tuple[float, float] = (12, 10),
) -> Figure:
    """Render the full scene as a canvas Figure.

    Args:
        scene: The generated SceneContext.
        focus_agent_id: If set, show full detail for this agent, minimal for others.
            None or "all" shows all agents with moderate detail.
        rotation: View rotation in radians (matches the map canvas rotation).
        figsize: Figure size in inches.

    Returns:
        Canvas Figure; ``.image()`` gives the PIL image for a ``gr.Image``.
    """
    fig = Figure(figsize, 100.0, facecolor="#f8f8f8")
    ax = fig.panel((0.07, 0.06, 0.91, 0.92), equal=True)

    limits = _scene_limits(scene, focus_agent_id, rotation, zoom)
    rotated = abs(rotation) > 0.01
    view = _identity
    if limits is not None:
        (cx_w, cy_w), half = limits
        ax.set_xlim(cx_w - half, cx_w + half)
        ax.set_ylim(cy_w - half, cy_w + half)
        if rotated:
            view = _rotation_view((cx_w, cy_w), rotation)
            margin = abs(math.sin(rotation)) + abs(math.cos(rotation))
            ax.set_xlim(cx_w - half * margin, cx_w + half * margin)
            ax.set_ylim(cy_w - half * margin, cy_w + half * margin)
    elif rotated:
        view = _rotation_view((0.0, 0.0), rotation)

    # Draw lane map background
    _draw_lanes(ax, scene.map_data, view=view)

    show_all = focus_agent_id is None or focus_agent_id == "all"

    nb_idx = 0
    for agent in scene.agents:
        is_ego = agent.id == scene.ego_agent_id
        is_focused = show_all or (agent.id == focus_agent_id)

        if is_ego:
            color = _EGO_COLOR
        else:
            color = _agent_color(agent.agent_type, nb_idx)
            nb_idx += 1

        pos = agent.current_position
        heading = agent.current_heading

        if is_focused:
            _draw_route(ax, agent.route_lanes, color=color, alpha=0.5, lw=2.5, zorder=3, view=view)
            _draw_footprint_history(
                ax,
                agent.past_trajectory,
                agent.length,
                agent.width,
                color,
                alpha=0.15,
                zorder=5,
                view=view,
            )
            _draw_past_trail(
                ax, agent.past_trajectory, color, alpha=0.5, lw=1.2, zorder=4, view=view
            )
            _draw_box(
                ax,
                pos[0],
                pos[1],
                heading,
                agent.length,
                agent.width,
                color,
                alpha=0.8 if is_ego else 0.6,
                lw=2.5 if is_ego else 2.0,
                zorder=10,
                view=view,
            )
            _draw_heading_arrow(
                ax, pos[0], pos[1], heading, agent.length, color, lw=2.5, zorder=11, view=view
            )

            if agent.goal_pose is not None:
                _draw_goal_marker(
                    ax,
                    agent.goal_pose[0],
                    agent.goal_pose[1],
                    color,
                    size=14,
                    label=f"{agent.id} goal",
                    zorder=15,
                    view=view,
                )

            speed = float(np.linalg.norm(agent.current_velocity))
            ax.text(
                tuple(view(np.array(pos[:2]))),
                f"{agent.id} ({speed:.1f} m/s)",
                7,
                color,
                ha="center",
                va="bottom",
                offset=(0, 8),
                box={"face": "#ffffff", "edge": color, "alpha": 0.8, "pad": 0.2},
                z=12,
            )
        else:
            _draw_past_trail(
                ax, agent.past_trajectory, _UNFOCUSED_COLOR, alpha=0.2, lw=0.6, view=view
            )
            _draw_box(
                ax,
                pos[0],
                pos[1],
                heading,
                agent.length,
                agent.width,
                _UNFOCUSED_COLOR,
                alpha=0.3,
                lw=0.8,
                zorder=8,
                view=view,
            )
            _draw_heading_arrow(
                ax,
                pos[0],
                pos[1],
                heading,
                agent.length,
                _UNFOCUSED_COLOR,
                lw=1.0,
                zorder=9,
                view=view,
            )
            if agent.goal_pose is not None:
                ax.markers(
                    view(np.array(agent.goal_pose[:2])),
                    "x",
                    6,
                    _UNFOCUSED_COLOR,
                    edge_width=1.0,
                    z=8,
                )

    ax.grid(0.15)

    if rotated:
        if ax.xlim is not None:
            xticks, yticks = nice_ticks(*ax.xlim), nice_ticks(*ax.ylim)
            ax.xticks(xticks, [""] * len(xticks))
            ax.yticks(yticks, [""] * len(yticks))
    else:
        ax.xlabel("X (m)")
        ax.ylabel("Y (m)")

    # Legend
    nb_idx = 0
    for agent in scene.agents:
        is_ego = agent.id == scene.ego_agent_id
        if is_ego:
            color = _EGO_COLOR
        else:
            color = _agent_color(agent.agent_type, nb_idx)
            nb_idx += 1
        ax.lines([], color, width=2, label=agent.id)
    ax.legend("upper left")

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
