"""Visualize a SceneContext: agents, lanes, trajectories, goal.

Usage:
    # From NPZ file
    python -m scenario_generation.visualize /path/to/scene.npz

    # Multiple scenes in a grid
    python -m scenario_generation.visualize /path/to/dir/*.npz --cols 3

    # Highlight a specific agent as ego
    python -m scenario_generation.visualize scene.npz --ego neighbor_0

    # Save to file instead of showing
    python -m scenario_generation.visualize scene.npz -o scene.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import Rectangle

from scenario_generation.npz_loader import from_npz
from scenario_generation.scene_context import AgentType, SceneContext

# Colors
_EGO_COLOR = "#3366cc"
_NEIGHBOR_COLORS = ["#e67300", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
_PED_COLOR = "#ff6699"
_BIKE_COLOR = "#66ccff"
_LANE_COLOR = "#888888"
_ROUTE_COLOR = "#4488ff"
_GT_COLOR = "#22bb22"


def _agent_color(agent_type: AgentType, idx: int) -> str:
    if agent_type == AgentType.PEDESTRIAN:
        return _PED_COLOR
    if agent_type == AgentType.BICYCLE:
        return _BIKE_COLOR
    return _NEIGHBOR_COLORS[idx % len(_NEIGHBOR_COLORS)]


def draw_lanes(ax, map_data, alpha=0.5):
    """Draw lane centerlines and boundaries."""
    lanes = map_data.lanes
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        pts = lane[:, :2]
        valid = np.abs(pts).sum(axis=1) > 0.1
        if valid.sum() < 2:
            continue
        # Centerline
        ax.plot(pts[valid, 0], pts[valid, 1], "-", color=_LANE_COLOR, alpha=alpha * 0.8, lw=0.8)
        # Boundaries (offsets from centerline)
        if lane.shape[1] > 7:
            lb = lane[:, 4:6]
            rb = lane[:, 6:8]
            ax.plot((pts + lb)[valid, 0], (pts + lb)[valid, 1], "-", color=_LANE_COLOR, alpha=alpha, lw=1.0)
            ax.plot((pts + rb)[valid, 0], (pts + rb)[valid, 1], "-", color=_LANE_COLOR, alpha=alpha, lw=1.0)


def draw_road_borders(ax, map_data):
    """Draw road borders in red (line_strings with border flag in channel 3)."""
    ls = map_data.line_strings
    if ls.shape[-1] < 4:
        return
    for i in range(ls.shape[0]):
        line = ls[i]
        if np.abs(line[:, :2]).sum() < 1e-6:
            continue
        valid = (line[:, 3] > 0.5) & (np.abs(line[:, :2]).sum(axis=1) > 0.01)
        if valid.sum() > 1:
            ax.plot(line[valid, 0], line[valid, 1], color="red", lw=2.5, alpha=0.7, zorder=4)


def draw_stop_lines(ax, map_data):
    """Draw stop lines in yellow (line_strings with stop flag in channel 2)."""
    ls = map_data.line_strings
    if ls.shape[-1] < 3:
        return
    for i in range(ls.shape[0]):
        line = ls[i]
        if np.abs(line[:, :2]).sum() < 1e-6:
            continue
        valid = (line[:, 2] > 0.5) & (np.abs(line[:, :2]).sum(axis=1) > 0.01)
        if valid.sum() > 1:
            ax.plot(line[valid, 0], line[valid, 1], color="#ddaa00", lw=2.0, alpha=0.8, zorder=4)


def draw_route(ax, route_lanes, color=None, alpha=0.4, lw=1.5):
    """Draw route lane centerlines."""
    if route_lanes is None:
        return
    c = color or _ROUTE_COLOR
    for i in range(route_lanes.shape[0]):
        lane = route_lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        pts = lane[:, :2]
        valid = np.abs(pts).sum(axis=1) > 0.1
        if valid.sum() < 2:
            continue
        ax.plot(pts[valid, 0], pts[valid, 1], "-", color=c, alpha=alpha, lw=lw)


def draw_agent_box(ax, x, y, heading, length, width, color, alpha=0.8, lw=1.5, zorder=10,
                   wheelbase: float | None = None):
    """Draw an oriented bounding box for an agent.

    When ``wheelbase`` is provided, (x, y) is treated as rear-axle midpoint
    (ego convention). When None, (x, y) is the bbox centroid (neighbor
    convention from the perception pipeline).
    """
    if wheelbase is not None:
        rear_overhang = (length - wheelbase) / 2
    else:
        rear_overhang = length / 2
    t_rot = mtransforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
    rect = Rectangle(
        (-rear_overhang, -width / 2), length, width,
        lw=lw, ec=color, fc=color, alpha=alpha, zorder=zorder, transform=t_rot,
    )
    ax.add_patch(rect)


def draw_trajectory(ax, traj, color, label=None, lw=2, zorder=10, show_footprints=False,
                    length=4.0, width=1.8):
    """Draw a trajectory line with optional footprints.

    Args:
        traj: (T, 3) [x, y, heading_rad] in scene frame.
    """
    ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=lw, alpha=0.6, zorder=zorder)
    ax.plot(traj[::3, 0], traj[::3, 1], "o", color=color, ms=2.5, alpha=0.8,
            mew=0, zorder=zorder + 1, label=label)

    if show_footprints:
        for ts in range(5, len(traj), 10):
            draw_agent_box(ax, traj[ts, 0], traj[ts, 1], traj[ts, 2],
                           length, width, color, alpha=0.12, lw=0.3, zorder=zorder - 1)
        if len(traj) > 1:
            draw_agent_box(ax, traj[-1, 0], traj[-1, 1], traj[-1, 2],
                           length, width, color, alpha=0.35, lw=1.0, zorder=zorder - 1)


def draw_scene(ax, scene: SceneContext, ego_id: str | None = None):
    """Draw a full scene on a matplotlib axes.

    Args:
        ax: Matplotlib axes.
        scene: The SceneContext to visualize.
        ego_id: Agent to highlight as ego. Defaults to scene.ego_agent_id.
    """
    if ego_id is None:
        ego_id = scene.ego_agent_id

    # Lanes, road borders, stop lines
    draw_lanes(ax, scene.map_data)
    draw_road_borders(ax, scene.map_data)
    draw_stop_lines(ax, scene.map_data)

    # Static objects
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

    # Agents
    nb_idx = 0
    for agent in scene.agents:
        is_ego = (agent.id == ego_id)
        pos = agent.current_position
        heading = agent.current_heading

        if is_ego:
            color = _EGO_COLOR
            label = f"ego ({agent.id})"
            zorder = 20
        else:
            color = _agent_color(agent.agent_type, nb_idx)
            label = f"{agent.id} ({agent.agent_type.value})"
            zorder = 15
            nb_idx += 1

        # Current bounding box
        draw_agent_box(ax, pos[0], pos[1], heading, agent.length, agent.width,
                       color, alpha=0.8 if is_ego else 0.5, lw=2 if is_ego else 1, zorder=zorder,
                       wheelbase=agent.wheelbase if is_ego else None)

        # Past trajectory
        past = agent.past_trajectory
        valid = np.abs(past[:, :2]).sum(axis=1) > 1e-6
        if valid.sum() > 1:
            ax.plot(past[valid, 0], past[valid, 1], "--", color=color, lw=0.8,
                    alpha=0.4, zorder=zorder - 2)

        # Heading arrow
        arrow_len = max(agent.length, 2.0)
        dx = arrow_len * math.cos(heading)
        dy = arrow_len * math.sin(heading)
        ax.annotate("", xy=(pos[0] + dx, pos[1] + dy), xytext=(pos[0], pos[1]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                     zorder=zorder + 1)

        # Future trajectory (GT) and final GT pose
        if agent.future_trajectory is not None:
            gt = agent.future_trajectory
            gt_label = "GT future" if is_ego else None
            draw_trajectory(ax, gt, _GT_COLOR, label=gt_label,
                            lw=1.5 if is_ego else 0.8, zorder=zorder - 1,
                            show_footprints=is_ego,
                            length=agent.length, width=agent.width)
            # Ghost box at final GT pose
            draw_agent_box(ax, gt[-1, 0], gt[-1, 1], gt[-1, 2],
                           agent.length, agent.width, color,
                           alpha=0.25, lw=1.0, zorder=zorder - 1)

        # Route
        if agent.route_lanes is not None:
            draw_route(ax, agent.route_lanes, color=color, alpha=0.3 if is_ego else 0.2)

        # Goal
        if agent.goal_pose is not None:
            gx, gy = agent.goal_pose[0], agent.goal_pose[1]
            marker = "*" if is_ego else "D"
            ms = 15 if is_ego else 10
            ax.plot(gx, gy, marker, color=color, ms=ms, zorder=zorder + 3,
                    markeredgecolor="black", markeredgewidth=0.8)
            ax.annotate(f"{agent.id} goal", (gx, gy), fontsize=5, color=color,
                        ha="center", va="bottom", zorder=zorder + 4,
                        xytext=(0, 8), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.7, lw=0.5))

        # Label
        ax.annotate(agent.id, (pos[0], pos[1]), fontsize=6, color=color,
                    ha="center", va="bottom", zorder=zorder + 2,
                    xytext=(0, 5), textcoords="offset points")

    # Auto-zoom
    all_pts = []
    for agent in scene.agents:
        valid = np.abs(agent.past_trajectory[:, :2]).sum(axis=1) > 1e-6
        if valid.sum() > 0:
            all_pts.append(agent.past_trajectory[valid, :2])
        if agent.future_trajectory is not None:
            all_pts.append(agent.future_trajectory[:, :2])
        if agent.goal_pose is not None:
            all_pts.append(agent.goal_pose[:2].reshape(1, 2))
    if all_pts:
        pts = np.vstack(all_pts)
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        half = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 0.6 + 10
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=6, loc="upper left")


def visualize_scene(scene: SceneContext, ego_id: str | None = None,
                    title: str | None = None, save_path: str | None = None):
    """Visualize a single scene. Shows interactively or saves to file."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    draw_scene(ax, scene, ego_id)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_scenes_grid(scenes: list[tuple[SceneContext, str]], cols: int = 3,
                          save_path: str | None = None):
    """Visualize multiple scenes in a grid."""
    n = len(scenes)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    axes_flat = axes.flatten()

    for i, (scene, title) in enumerate(scenes):
        draw_scene(axes_flat[i], scene)
        axes_flat[i].set_title(title, fontsize=9)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize SceneContext from NPZ files")
    parser.add_argument("npz_paths", nargs="+", type=Path, help="NPZ file(s) to visualize")
    parser.add_argument("--ego", type=str, default=None, help="Agent ID to highlight as ego")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Save to file instead of showing")
    parser.add_argument("--cols", type=int, default=3, help="Columns in grid layout")
    args = parser.parse_args()

    if len(args.npz_paths) == 1:
        scene = from_npz(args.npz_paths[0])
        title = args.npz_paths[0].stem
        visualize_scene(scene, ego_id=args.ego, title=title,
                        save_path=str(args.output) if args.output else None)
    else:
        scenes = []
        for p in args.npz_paths:
            scene = from_npz(p)
            scenes.append((scene, p.stem))
        visualize_scenes_grid(scenes, cols=args.cols,
                              save_path=str(args.output) if args.output else None)


if __name__ == "__main__":
    main()
