"""Visualization utilities for the Guidance Playground.

Provides standalone plotting functions that accept trajectory arrays directly,
adapted from the plotting methods in preference_optimization/annotation_gui.py.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from scenario_generation.canvas import Figure, Panel, oriented_box
from scenario_generation.colormaps import qualitative

# ---------------------------------------------------------------------------
# Colour palette for N independent samples
# ---------------------------------------------------------------------------


def sample_color(i: int) -> str:
    return qualitative("tab10", i)


# ---------------------------------------------------------------------------
# Kinematic helpers (identical logic to annotation_gui.py helpers)
# ---------------------------------------------------------------------------


def _calculate_velocities(trajectory: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
    """Compute per-step speed (km/h) from trajectory positions.

    Args:
        trajectory: (T, 4) [x, y, cos, sin] ego-centric trajectory.
        ego_state: (10,) ego_current_state (first two elements are x, y).

    Returns:
        (T,) speed array in km/h.
    """
    positions = np.vstack([ego_state[:2], trajectory[:, :2]])  # (T+1, 2)
    diffs = np.diff(positions, axis=0)  # (T, 2)
    speed_ms = np.sqrt((diffs**2).sum(axis=1)) / 0.1  # m/s at dt=0.1s
    return speed_ms * 3.6  # km/h


def _calculate_accelerations(velocities_kmh: np.ndarray) -> np.ndarray:
    """Compute per-step longitudinal acceleration (m/s²) from speed profile.

    Args:
        velocities_kmh: (T,) speed in km/h.

    Returns:
        (T,) acceleration in m/s², padded with 0 at the end.
    """
    vel_ms = velocities_kmh / 3.6
    acc = np.diff(vel_ms) / 0.1
    return np.append(acc, 0.0)


def _calculate_curvature(trajectory: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
    """Compute per-step path curvature (1/m) from heading changes.

    Args:
        trajectory: (T, 4) [x, y, cos, sin].
        ego_state: (10,) ego_current_state (indices 2,3 = cos, sin of heading).

    Returns:
        (T,) curvature in 1/m.
    """
    cos_vals = np.concatenate([[ego_state[2]], trajectory[:, 2]])
    sin_vals = np.concatenate([[ego_state[3]], trajectory[:, 3]])
    headings = np.arctan2(sin_vals, cos_vals)

    positions = np.vstack([ego_state[:2], trajectory[:, :2]])
    diffs = np.diff(positions, axis=0)
    arc_lengths = np.sqrt((diffs**2).sum(axis=1))

    heading_diffs = np.diff(np.unwrap(headings))

    curvatures = np.zeros(len(heading_diffs))
    valid = arc_lengths > 1e-6
    curvatures[valid] = heading_diffs[valid] / arc_lengths[valid]
    return curvatures


def _calculate_lateral_acceleration(
    curvatures: np.ndarray, velocities_kmh: np.ndarray
) -> np.ndarray:
    vel_ms = velocities_kmh / 3.6
    return vel_ms**2 * np.abs(curvatures)


def _gt_velocities(ego_future: np.ndarray, ego_state: np.ndarray) -> np.ndarray | None:
    """Ground-truth speed from ego_agent_future (80, 3) = [x, y, yaw_rad]."""
    valid = ~((ego_future[:, 0] == 0) & (ego_future[:, 1] == 0))
    if not np.any(valid):
        return None
    positions = np.vstack([ego_state[:2], ego_future[:, :2]])
    diffs = np.diff(positions, axis=0)
    return np.sqrt((diffs**2).sum(axis=1)) / 0.1 * 3.6


def _gt_curvature(ego_future: np.ndarray, ego_state: np.ndarray) -> np.ndarray | None:
    """Ground-truth curvature from ego_agent_future (80, 3) = [x, y, yaw_rad]."""
    valid = ~((ego_future[:, 0] == 0) & (ego_future[:, 1] == 0))
    if not np.any(valid):
        return None
    cos_ego = ego_state[2]
    sin_ego = ego_state[3]
    headings = np.concatenate([[np.arctan2(sin_ego, cos_ego)], ego_future[:, 2]])
    positions = np.vstack([ego_state[:2], ego_future[:, :2]])
    diffs = np.diff(positions, axis=0)
    arc_lengths = np.sqrt((diffs**2).sum(axis=1))
    heading_diffs = np.diff(np.unwrap(headings))
    curvatures = np.zeros(len(heading_diffs))
    valid_mask = arc_lengths > 1e-6
    curvatures[valid_mask] = heading_diffs[valid_mask] / arc_lengths[valid_mask]
    return curvatures


def _draw_vehicle_footprint(
    ax: Panel,
    x: float,
    y: float,
    heading: float,
    ego_shape: np.ndarray,
    color: str,
    alpha: float = 0.5,
) -> None:
    """Draw a rotated rectangle representing the vehicle footprint."""
    wheel_base = float(ego_shape[0])
    length = float(ego_shape[1])
    width = float(ego_shape[2])

    # Offset rear axle → vehicle centre
    cx = x + (wheel_base / 2) * np.cos(heading)
    cy = y + (wheel_base / 2) * np.sin(heading)

    corners = oriented_box(cx, cy, heading, length, width, length / 2)
    ax.polygons([corners], fill=color, edge=color, width=1.5, alpha=alpha)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def plot_trajectory(
    samples: np.ndarray,
    data: dict,
    anchor: np.ndarray | None = None,
    anchor_enabled: bool = False,
    time_step: int | None = None,
    view_range: int = 60,
) -> Image.Image:
    """Main trajectory plot showing N samples, GT dashed, optional anchor dotted.

    Args:
        samples: (N, T, 4) array of ego trajectories [x, y, cos, sin].
        data: Observation dict with tensors (on any device); used for map viz.
        anchor: (M, 2) prototype polyline xy to show as dotted orange line, or None.
        anchor_enabled: Whether anchor guidance is active (controls visibility).
        time_step: If set, draws vehicle footprints at that step.
        view_range: Half-range for axis limits in metres.

    Returns:
        PIL Image of the figure.
    """
    from diffusion_planner.utils.visualize_input import visualize_inputs

    fig = Figure((10, 10), 100)
    ax = fig.panel(equal=True)

    # Determine plot centre from first sample trajectory midpoint
    ref_traj = samples[0]
    center_x = (ref_traj[0, 0] + ref_traj[-1, 0]) / 2
    center_y = (ref_traj[0, 1] + ref_traj[-1, 1]) / 2

    # Background map
    data_cpu = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in data.items()}
    visualize_inputs(data_cpu, save_path=None, ax=ax, view_ranges=[120])

    # N samples
    N = samples.shape[0]
    for i in range(N):
        traj = samples[i]
        color = sample_color(i)
        ax.lines([traj[:, :2]], color, width=2.5, alpha=0.8, label=f"Sample {i}")
        if time_step is not None and 0 <= time_step < len(traj):
            ego_shape = data_cpu.get("ego_shape")
            if ego_shape is not None:
                ego_shape_np = ego_shape.numpy()[0] if hasattr(ego_shape, "numpy") else ego_shape[0]
                x, y = traj[time_step, 0], traj[time_step, 1]
                heading = np.arctan2(traj[time_step, 3], traj[time_step, 2])
                _draw_vehicle_footprint(ax, x, y, heading, ego_shape_np, color=sample_color(i))
            ax.markers(
                traj[time_step, :2], "o", size=np.sqrt(50), color=color, edge="#000000", z=10
            )

    # Ground truth
    if "ego_agent_future" in data_cpu:
        ego_future = data_cpu["ego_agent_future"]
        if hasattr(ego_future, "numpy"):
            ego_future = ego_future.numpy()
        ego_future = np.array(ego_future).reshape(-1, 3)  # (80, 3)
        valid = ~((ego_future[:, 0] == 0) & (ego_future[:, 1] == 0))
        if np.any(valid):
            ax.lines(
                [ego_future[valid, :2]],
                "#808080",
                width=2,
                alpha=0.7,
                dash="dash",
                label="GT",
            )

    # Anchor prototype
    if anchor_enabled and anchor is not None:
        T = min(anchor.shape[0], samples.shape[1])
        ax.lines(
            [anchor[:T, :2]],
            "#ffa500",
            width=2,
            alpha=0.85,
            dash="dot",
            label="Anchor prototype",
        )

    ax.legend("upper left")
    ax.title("Trajectory Samples")
    half = view_range / 2
    ax.set_xlim(center_x - half, center_x + half)
    ax.set_ylim(center_y - half, center_y + half)

    return fig.image()


def _profile_panels(fig: Figure) -> tuple[Panel, Panel]:
    """Two stacked chart panels filling a profile figure."""
    (top,), (bottom,) = fig.grid(2, 1, margins=(0.16, 0.96, 0.08, 0.95), hspace=0.3)
    return top, bottom


def plot_velocity(
    samples: np.ndarray,
    data: dict,
) -> Image.Image:
    """Speed and longitudinal acceleration profiles for all N samples + GT.

    Args:
        samples: (N, T, 4) ego trajectories.
        data: Observation dict (needs ego_current_state, optionally ego_agent_future).

    Returns:
        PIL Image with two charts (speed top, acceleration bottom).
    """
    fig = Figure((6, 7), 100)
    ax_vel, ax_acc = _profile_panels(fig)

    data_cpu = {k: v.cpu().numpy() if hasattr(v, "cpu") else v for k, v in data.items()}
    ego_state = np.array(data_cpu["ego_current_state"]).reshape(-1)  # (10,)

    for i, traj in enumerate(samples):
        color = sample_color(i)
        vel = _calculate_velocities(traj, ego_state)
        acc = _calculate_accelerations(vel)
        t = np.arange(len(vel))
        ax_vel.lines([np.stack([t, vel], 1)], color, width=1.8, alpha=0.75, label=f"S{i}")
        ax_acc.lines([np.stack([t, acc], 1)], color, width=1.8, alpha=0.75)

    # Ground truth
    if "ego_agent_future" in data_cpu:
        ego_future = np.array(data_cpu["ego_agent_future"]).reshape(-1, 3)
        gt_vel = _gt_velocities(ego_future, ego_state)
        if gt_vel is not None:
            ax_vel.lines(
                [np.stack([np.arange(len(gt_vel)), gt_vel], 1)],
                "#000000",
                width=2,
                alpha=0.7,
                dash="dash",
                label="GT",
            )

    ax_vel.ylabel("Speed (km/h)")
    ax_vel.set_ylim(0, 80)
    ax_vel.title("Speed")
    ax_vel.legend("upper right")
    ax_vel.grid(0.3)

    ax_acc.ylabel("Accel (m/s²)")
    ax_acc.xlabel("Time step")
    ax_acc.set_ylim(-3, 3)
    ax_acc.title("Longitudinal Acceleration")
    ax_acc.grid(0.3)
    ax_acc.hline(0, "#808080", width=0.5, dash="dash")

    return fig.image()


def plot_lateral_curvature(
    samples: np.ndarray,
    data: dict,
) -> Image.Image:
    """Lateral acceleration and curvature profiles for all N samples + GT.

    Args:
        samples: (N, T, 4) ego trajectories.
        data: Observation dict.

    Returns:
        PIL Image with two charts.
    """
    fig = Figure((6, 7), 100)
    ax_lat, ax_curv = _profile_panels(fig)

    data_cpu = {k: v.cpu().numpy() if hasattr(v, "cpu") else v for k, v in data.items()}
    ego_state = np.array(data_cpu["ego_current_state"]).reshape(-1)

    for i, traj in enumerate(samples):
        color = sample_color(i)
        vel = _calculate_velocities(traj, ego_state)
        curv = _calculate_curvature(traj, ego_state)
        lat_acc = _calculate_lateral_acceleration(curv, vel)
        t = np.arange(len(curv))
        ax_lat.lines([np.stack([t, lat_acc], 1)], color, width=1.8, alpha=0.75, label=f"S{i}")
        ax_curv.lines([np.stack([t, curv], 1)], color, width=1.8, alpha=0.75)

    # Ground truth
    if "ego_agent_future" in data_cpu:
        ego_future = np.array(data_cpu["ego_agent_future"]).reshape(-1, 3)
        gt_vel = _gt_velocities(ego_future, ego_state)
        gt_curv = _gt_curvature(ego_future, ego_state)
        if gt_vel is not None and gt_curv is not None:
            gt_lat = _calculate_lateral_acceleration(gt_curv, gt_vel)
            ax_lat.lines(
                [np.stack([np.arange(len(gt_lat)), gt_lat], 1)],
                "#000000",
                width=2,
                alpha=0.7,
                dash="dash",
                label="GT",
            )
            ax_curv.lines(
                [np.stack([np.arange(len(gt_curv)), gt_curv], 1)],
                "#000000",
                width=2,
                alpha=0.7,
                dash="dash",
            )

    ax_lat.ylabel("Lat. accel (m/s²)")
    ax_lat.set_ylim(0, 8)
    ax_lat.title("Lateral Acceleration")
    ax_lat.hline(3.0, "#800080", width=1, dash="dot", alpha=0.5)
    ax_lat.legend("upper right")
    ax_lat.grid(0.3)

    ax_curv.ylabel("Curvature (1/m)")
    ax_curv.xlabel("Time step")
    ax_curv.set_ylim(-0.2, 0.2)
    ax_curv.title("Path Curvature")
    ax_curv.grid(0.3)
    ax_curv.hline(0, "#808080", width=0.5, dash="dash")

    return fig.image()


def render_prototype_thumbnail(
    all_protos: np.ndarray, index: int, count: int, selected: bool = False
) -> Image.Image:
    """Render one prototype thumbnail with all others shown in grey for context.

    Args:
        all_protos: (K, 80, 2) all prototype trajectories in ego-centric metres.
        index: Index of the prototype to highlight in blue.
        count: Number of training samples assigned to this cluster.
        selected: When True, draws a prominent orange border to indicate the
            active anchor selection.

    Returns:
        PIL Image for use in gr.Gallery.
    """
    fig = Figure((2, 2), 72)
    ax = fig.panel((0.03, 0.03, 0.94, 0.82), equal=True)
    ax.lines(
        [proto[:, :2] for i, proto in enumerate(all_protos) if i != index],
        "#cccccc",
        width=0.8,
        alpha=0.6,
    )
    ax.lines([all_protos[index, :, :2]], "#4169e1", width=2.2, z=5)
    ax.markers(np.zeros(2), "o", size=np.sqrt(20), color="#ff0000", z=6)
    ax.title(f"#{index}  n={count}", 8, weight="bold" if selected else "normal")
    ax.axis_off()
    img = fig.image()
    if selected:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        border = 5
        for b in range(border):
            draw.rectangle([b, b, w - 1 - b, h - 1 - b], outline=(255, 140, 0))
    return img


def render_prototype_gallery(
    prototypes_path: str, selected_index: int = -1
) -> list[tuple[Image.Image, str]] | None:
    """Load a prototypes .npy file and return a gallery-ready list.

    Args:
        prototypes_path: Path to prototypes .npy file of shape (K, 80, 2).
        selected_index: Index of the currently selected anchor prototype.
            Pass -1 (default) for no selection highlight.

    Returns:
        List of (PIL Image, label) tuples for gr.Gallery, or None if path invalid.
    """
    import os

    if not prototypes_path or not os.path.exists(prototypes_path):
        return None
    protos = np.load(prototypes_path)  # (K, 80, 2)
    K = protos.shape[0]
    counts_path = prototypes_path.replace(".npy", "_counts.npy")
    counts = np.load(counts_path) if os.path.exists(counts_path) else np.ones(K, dtype=int)
    return [
        (
            render_prototype_thumbnail(protos, i, int(counts[i]), selected=(i == selected_index)),
            f"#{i} (n={int(counts[i])})",
        )
        for i in range(K)
    ]


def compute_stats(
    samples: np.ndarray,
    gt_trajectory: np.ndarray | None,
    anchor: np.ndarray | None,
) -> dict[str, float]:
    """Compute summary statistics over N samples.

    Args:
        samples: (N, T, 4) trajectories.
        gt_trajectory: (T, 3) ground-truth [x, y, yaw] or None.
        anchor: (M, 2) anchor prototype polyline xy or None (any M).

    Returns:
        Dict with keys: min_ade_gt, max_ade_gt, ade_anchor, spread_fde_std.
        Values are NaN when the reference trajectory is not available.
    """
    N, T, _ = samples.shape
    result: dict[str, float] = {
        "min_ade_gt": float("nan"),
        "max_ade_gt": float("nan"),
        "ade_anchor": float("nan"),
        "spread_fde_std": float("nan"),
    }

    if gt_trajectory is not None:
        gt_xy = gt_trajectory[:T, :2]
        ades = []
        for traj in samples:
            ade = float(np.mean(np.linalg.norm(traj[:, :2] - gt_xy, axis=1)))
            ades.append(ade)
        result["min_ade_gt"] = float(np.min(ades))
        result["max_ade_gt"] = float(np.max(ades))

    if anchor is not None:
        # The anchor is an arc-length path polyline (any point count), not
        # per-timestep waypoints: measure mean distance from each sample point
        # to its NEAREST polyline segment instead of index-aligned L2.
        a, b = anchor[:-1, :2], anchor[1:, :2]
        ab = b - a
        ab_sq = np.maximum((ab * ab).sum(-1), 1e-8)
        dists = []
        for traj in samples:
            ap = traj[:, None, :2] - a  # (T, S-1, 2)
            t_proj = np.clip((ap * ab).sum(-1) / ab_sq, 0.0, 1.0)
            closest = a + t_proj[..., None] * ab
            d = np.linalg.norm(traj[:, None, :2] - closest, axis=-1).min(axis=-1)
            dists.append(float(d.mean()))
        result["ade_anchor"] = float(np.mean(dists))

    # Spread: std of final-position distances across samples
    final_positions = samples[:, -1, :2]  # (N, 2)
    if N > 1:
        result["spread_fde_std"] = float(
            np.std(np.linalg.norm(final_positions - final_positions.mean(axis=0), axis=1))
        )
    else:
        result["spread_fde_std"] = 0.0

    return result
