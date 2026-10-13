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

import numpy as np
from PIL import Image

from scenario_generation.canvas import Panel, oriented_box, scene_panel
from scenario_generation.npz_loader import from_npz
from scenario_generation.scene_context import AgentType, SceneContext

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


def _lane_points(lanes: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(lane, valid point mask)`` for every lane with at least two valid centerline points."""
    rows = []
    for lane in lanes:
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        valid = np.abs(lane[:, :2]).sum(axis=1) > 0.1
        if valid.sum() >= 2:
            rows.append((lane, valid))
    return rows


def draw_lanes(canvas: Panel, map_data, alpha=0.5):
    """Draw lane centerlines and boundaries."""
    rows = _lane_points(map_data.lanes)
    canvas.lines([lane[valid, :2] for lane, valid in rows], _LANE_COLOR, 0.8, alpha * 0.8)
    bounds = [
        (lane[:, :2] + lane[:, offset : offset + 2])[valid]
        for lane, valid in rows
        if lane.shape[1] > 7
        for offset in (4, 6)
    ]
    canvas.lines(bounds, _LANE_COLOR, 1.0, alpha)


def _flagged_lines(map_data, channel: int) -> list[np.ndarray]:
    """``line_strings`` points whose flag ``channel`` is set, one polyline per line string."""
    ls = map_data.line_strings
    if ls.shape[-1] <= channel:
        return []
    lines = []
    for line in ls:
        if np.abs(line[:, :2]).sum() < 1e-6:
            continue
        valid = (line[:, channel] > 0.5) & (np.abs(line[:, :2]).sum(axis=1) > 0.01)
        if valid.sum() > 1:
            lines.append(line[valid, :2])
    return lines


def draw_road_borders(canvas: Panel, map_data):
    """Draw road borders in red (line_strings with border flag in channel 3)."""
    canvas.lines(_flagged_lines(map_data, 3), "#ff0000", 2.5, 0.7, z=4)


def draw_stop_lines(canvas: Panel, map_data):
    """Draw stop lines in yellow (line_strings with stop flag in channel 2)."""
    canvas.lines(_flagged_lines(map_data, 2), "#ddaa00", 2.0, 0.8, z=4)


def draw_route(canvas: Panel, route_lanes, color=None, alpha=0.4, lw=1.5):
    """Draw route lane centerlines."""
    if route_lanes is None:
        return
    rows = _lane_points(route_lanes)
    canvas.lines([lane[valid, :2] for lane, valid in rows], color or _ROUTE_COLOR, lw, alpha)


def draw_agent_box(
    canvas: Panel,
    x,
    y,
    heading,
    length,
    width,
    color,
    alpha=0.8,
    lw=1.5,
    zorder=10,
    wheelbase: float | None = None,
    filled: bool = True,
):
    """Draw an oriented bounding box for an agent.

    When ``wheelbase`` is provided, (x, y) is treated as rear-axle midpoint
    (ego convention). When None, (x, y) is the bbox centroid (neighbor
    convention from the perception pipeline).

    ``filled=False`` draws an outline-only (hollow) box: face is transparent,
    ``alpha`` applies to the edge only.
    """
    rear_overhang = (length - wheelbase) / 2 if wheelbase is not None else length / 2
    corners = oriented_box(float(x), float(y), float(heading), length, width, rear_overhang)
    canvas.polygons(
        [corners], fill=color if filled else None, edge=color, width=lw, alpha=alpha, z=zorder
    )


def draw_trajectory(
    canvas: Panel,
    traj,
    color,
    label=None,
    lw=2,
    zorder=10,
    show_footprints=False,
    length=4.0,
    width=1.8,
    wheelbase: float | None = None,
):
    """Draw a trajectory line with optional footprints.

    Args:
        traj: (T, 3) [x, y, heading_rad] in scene frame.
        wheelbase: when provided, footprint boxes treat each (x, y) as the
            rear-axle midpoint and offset forward (ego convention). When None,
            (x, y) is the bbox centroid (neighbor convention).
    """
    canvas.lines([traj[:, :2]], color, lw, 0.6, z=zorder)
    canvas.markers(traj[::3, :2], "o", 2.5, color, alpha=0.8, z=zorder + 1, label=label)

    if show_footprints:
        for ts in range(5, len(traj), 10):
            draw_agent_box(
                canvas,
                traj[ts, 0],
                traj[ts, 1],
                traj[ts, 2],
                length,
                width,
                color,
                alpha=0.12,
                lw=0.3,
                zorder=zorder - 1,
                wheelbase=wheelbase,
            )
        if len(traj) > 1:
            draw_agent_box(
                canvas,
                traj[-1, 0],
                traj[-1, 1],
                traj[-1, 2],
                length,
                width,
                color,
                alpha=0.35,
                lw=1.0,
                zorder=zorder - 1,
                wheelbase=wheelbase,
            )


def scene_view(scene: SceneContext) -> tuple[tuple[float, float], tuple[float, float]]:
    """Square ``(xlim, ylim)`` around every agent's history, future and goal."""
    all_pts = []
    for agent in scene.agents:
        valid = np.abs(agent.past_trajectory[:, :2]).sum(axis=1) > 1e-6
        if valid.sum() > 0:
            all_pts.append(agent.past_trajectory[valid, :2])
        if agent.future_trajectory is not None:
            all_pts.append(agent.future_trajectory[:, :2])
        if agent.goal_pose is not None:
            all_pts.append(agent.goal_pose[:2].reshape(1, 2))
    if not all_pts:
        return (-50.0, 50.0), (-50.0, 50.0)
    pts = np.vstack(all_pts)
    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    half = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))) * 0.6 + 10
    return (cx - half, cx + half), (cy - half, cy + half)


def scene_canvas(scene: SceneContext, size_in: float = 12.0, dpi: float = 150.0) -> Panel:
    """A square panel over :func:`scene_view` with a light grid."""
    xlim, ylim = scene_view(scene)
    canvas = scene_panel(
        xlim, ylim, size_in=(size_in, size_in), dpi=dpi, margins=(0.06, 0.97, 0.05, 0.95)
    )
    canvas.grid(0.15)
    return canvas


def draw_scene(canvas: Panel, scene: SceneContext, ego_id: str | None = None):
    """Draw a full scene on a canvas.

    Args:
        canvas: The canvas, usually from :func:`scene_canvas`.
        scene: The SceneContext to visualize.
        ego_id: Agent to highlight as ego. Defaults to scene.ego_agent_id.
    """
    if ego_id is None:
        ego_id = scene.ego_agent_id

    draw_lanes(canvas, scene.map_data)
    draw_road_borders(canvas, scene.map_data)
    draw_stop_lines(canvas, scene.map_data)

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
        draw_agent_box(canvas, x, y, heading, l, w, "#999999", alpha=0.4, lw=0.5, zorder=5)

    nb_idx = 0
    for agent in scene.agents:
        is_ego = agent.id == ego_id
        pos = agent.current_position
        heading = agent.current_heading

        if is_ego:
            color = _EGO_COLOR
            zorder = 20
        else:
            color = _agent_color(agent.agent_type, nb_idx)
            zorder = 15
            nb_idx += 1

        draw_agent_box(
            canvas,
            pos[0],
            pos[1],
            heading,
            agent.length,
            agent.width,
            color,
            alpha=0.8 if is_ego else 0.5,
            lw=2 if is_ego else 1,
            zorder=zorder,
            wheelbase=agent.wheelbase if is_ego else None,
        )

        past = agent.past_trajectory
        valid = np.abs(past[:, :2]).sum(axis=1) > 1e-6
        if valid.sum() > 1:
            canvas.lines([past[valid, :2]], color, 0.8, 0.4, dash="dash", z=zorder - 2)

        arrow_len = max(agent.length, 2.0)
        head = (pos[0] + arrow_len * math.cos(heading), pos[1] + arrow_len * math.sin(heading))
        canvas.arrows(np.array(pos[:2]), np.array(head), color, width=1.5, z=zorder + 1)

        if agent.future_trajectory is not None:
            gt = agent.future_trajectory
            draw_trajectory(
                canvas,
                gt,
                _GT_COLOR,
                label="GT future" if is_ego else None,
                lw=1.5 if is_ego else 0.8,
                zorder=zorder - 1,
                show_footprints=is_ego,
                length=agent.length,
                width=agent.width,
            )
            draw_agent_box(
                canvas,
                gt[-1, 0],
                gt[-1, 1],
                gt[-1, 2],
                agent.length,
                agent.width,
                color,
                alpha=0.25,
                lw=1.0,
                zorder=zorder - 1,
            )

        if agent.route_lanes is not None:
            draw_route(canvas, agent.route_lanes, color=color, alpha=0.3 if is_ego else 0.2)

        if agent.goal_pose is not None:
            goal = np.array(agent.goal_pose[:2], dtype=np.float64)
            canvas.markers(
                goal,
                "*" if is_ego else "D",
                15 if is_ego else 10,
                color,
                edge="#000000",
                edge_width=0.8,
                z=zorder + 3,
            )
            canvas.text(
                tuple(goal),
                f"{agent.id} goal",
                5,
                color,
                ha="center",
                va="bottom",
                offset=(0, 8),
                box={"face": "#ffffff", "edge": color, "alpha": 0.7, "pad": 0.15},
                z=zorder + 4,
            )

        canvas.text(
            (pos[0], pos[1]),
            agent.id,
            6,
            color,
            ha="center",
            va="bottom",
            offset=(0, 5),
            z=zorder + 2,
        )

    canvas.legend("upper left")


def scene_image(
    scene: SceneContext, ego_id: str | None = None, title: str | None = None
) -> Image.Image:
    """One scene drawn on its own canvas."""
    canvas = scene_canvas(scene)
    draw_scene(canvas, scene, ego_id)
    if title:
        canvas.title(title, 10)
    return canvas.figure.image()


def visualize_scene(
    scene: SceneContext,
    ego_id: str | None = None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Visualize a single scene. Shows it in the image viewer or saves to file."""
    image = scene_image(scene, ego_id, title)
    if save_path:
        image.save(save_path, compress_level=1)
    else:
        image.show()


def visualize_scenes_grid(
    scenes: list[tuple[SceneContext, str]], cols: int = 3, save_path: str | None = None
):
    """Visualize multiple scenes in a grid."""
    images = [scene_image(scene, title=title) for scene, title in scenes]
    rows = (len(images) + cols - 1) // cols
    width, height = images[0].size
    grid = Image.new("RGB", (width * cols, height * rows), (255, 255, 255))
    for i, image in enumerate(images):
        grid.paste(image, ((i % cols) * width, (i // cols) * height))
    if save_path:
        grid.save(save_path, compress_level=1)
        print(f"Saved: {save_path}")
    else:
        grid.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize SceneContext from NPZ files")
    parser.add_argument("npz_paths", nargs="+", type=Path, help="NPZ file(s) to visualize")
    parser.add_argument("--ego", type=str, default=None, help="Agent ID to highlight as ego")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Save to file instead of showing"
    )
    parser.add_argument("--cols", type=int, default=3, help="Columns in grid layout")
    args = parser.parse_args()

    if len(args.npz_paths) == 1:
        scene = from_npz(args.npz_paths[0])
        title = args.npz_paths[0].stem
        visualize_scene(
            scene, ego_id=args.ego, title=title, save_path=str(args.output) if args.output else None
        )
    else:
        scenes = []
        for p in args.npz_paths:
            scene = from_npz(p)
            scenes.append((scene, p.stem))
        visualize_scenes_grid(
            scenes, cols=args.cols, save_path=str(args.output) if args.output else None
        )


if __name__ == "__main__":
    main()
