"""Visualization utilities for the Guidance Playground.

Provides standalone plotting functions that accept trajectory arrays directly,
adapted from the plotting methods in preference_optimization/annotation_gui.py.
"""

from __future__ import annotations

import io

import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Colour palette for N independent samples
# ---------------------------------------------------------------------------

_SAMPLE_CMAP = cm.get_cmap("tab10")

def sample_color(i: int) -> tuple:
    return _SAMPLE_CMAP(i % 10)


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
    diffs = np.diff(positions, axis=0)                          # (T, 2)
    speed_ms = np.sqrt((diffs ** 2).sum(axis=1)) / 0.1         # m/s at dt=0.1s
    return speed_ms * 3.6                                       # km/h


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
    arc_lengths = np.sqrt((diffs ** 2).sum(axis=1))

    heading_diffs = np.diff(np.unwrap(headings))

    curvatures = np.zeros(len(heading_diffs))
    valid = arc_lengths > 1e-6
    curvatures[valid] = heading_diffs[valid] / arc_lengths[valid]
    return curvatures


def _calculate_lateral_acceleration(curvatures: np.ndarray, velocities_kmh: np.ndarray) -> np.ndarray:
    vel_ms = velocities_kmh / 3.6
    return vel_ms ** 2 * np.abs(curvatures)


def _gt_velocities(ego_future: np.ndarray, ego_state: np.ndarray) -> np.ndarray | None:
    """Ground-truth speed from ego_agent_future (80, 3) = [x, y, yaw_rad]."""
    valid = ~((ego_future[:, 0] == 0) & (ego_future[:, 1] == 0))
    if not np.any(valid):
        return None
    positions = np.vstack([ego_state[:2], ego_future[:, :2]])
    diffs = np.diff(positions, axis=0)
    return np.sqrt((diffs ** 2).sum(axis=1)) / 0.1 * 3.6


def _gt_curvature(ego_future: np.ndarray, ego_state: np.ndarray) -> np.ndarray | None:
    """Ground-truth curvature from ego_agent_future (80, 3) = [x, y, yaw_rad]."""
    valid = ~((ego_future[:, 0] == 0) & (ego_future[:, 1] == 0))
    if not np.any(valid):
        return None
    cos_ego = ego_state[2]
    sin_ego = ego_state[3]
    headings = np.concatenate(
        [[np.arctan2(sin_ego, cos_ego)], ego_future[:, 2]]
    )
    positions = np.vstack([ego_state[:2], ego_future[:, :2]])
    diffs = np.diff(positions, axis=0)
    arc_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    heading_diffs = np.diff(np.unwrap(headings))
    curvatures = np.zeros(len(heading_diffs))
    valid_mask = arc_lengths > 1e-6
    curvatures[valid_mask] = heading_diffs[valid_mask] / arc_lengths[valid_mask]
    return curvatures


def _draw_vehicle_footprint(ax, x: float, y: float, heading: float,
                             ego_shape: np.ndarray, color: str, alpha: float = 0.5) -> None:
    """Draw a rotated rectangle representing the vehicle footprint."""
    wheel_base = float(ego_shape[0])
    length = float(ego_shape[1])
    width = float(ego_shape[2])

    # Offset rear axle → vehicle centre
    cx = x + (wheel_base / 2) * np.cos(heading)
    cy = y + (wheel_base / 2) * np.sin(heading)

    rect = patches.Rectangle(
        (-length / 2, -width / 2), length, width,
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=alpha
    )
    transform = (
        patches.mpatches.mpl.transforms.Affine2D()
        .rotate(heading)
        .translate(cx, cy)
        + ax.transData
    )
    rect.set_transform(transform)
    ax.add_patch(rect)


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
) -> Figure:
    """Main trajectory plot showing N samples, GT dashed, optional anchor dotted.

    Args:
        samples: (N, T, 4) array of ego trajectories [x, y, cos, sin].
        data: Observation dict with tensors (on any device); used for map viz.
        anchor: (80, 2) prototype xy to show as dotted orange line, or None.
        anchor_enabled: Whether anchor guidance is active (controls visibility).
        time_step: If set, draws vehicle footprints at that step.
        view_range: Half-range for axis limits in metres.

    Returns:
        matplotlib Figure.
    """
    from diffusion_planner.utils.visualize_input import visualize_inputs

    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

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
        ax.plot(
            traj[:, 0], traj[:, 1],
            color=color, linewidth=2.5, alpha=0.8,
            label=f"Sample {i}"
        )
        if time_step is not None and 0 <= time_step < len(traj):
            ego_shape = data_cpu.get("ego_shape")
            if ego_shape is not None:
                ego_shape_np = (
                    ego_shape.numpy()[0] if hasattr(ego_shape, "numpy") else ego_shape[0]
                )
                x, y = traj[time_step, 0], traj[time_step, 1]
                heading = np.arctan2(traj[time_step, 3], traj[time_step, 2])
                _draw_vehicle_footprint(ax, x, y, heading, ego_shape_np, color=sample_color(i))
            ax.scatter([traj[time_step, 0]], [traj[time_step, 1]],
                       color=color, s=50, zorder=10, edgecolors="black")

    # Ground truth
    if "ego_agent_future" in data_cpu:
        ego_future = data_cpu["ego_agent_future"]
        if hasattr(ego_future, "numpy"):
            ego_future = ego_future.numpy()
        ego_future = np.array(ego_future).reshape(-1, 3)  # (80, 3)
        valid = ~((ego_future[:, 0] == 0) & (ego_future[:, 1] == 0))
        if np.any(valid):
            ax.plot(
                ego_future[valid, 0], ego_future[valid, 1],
                color="gray", linewidth=2, linestyle="--", alpha=0.7,
                label="GT"
            )

    # Anchor prototype
    if anchor_enabled and anchor is not None:
        T = min(anchor.shape[0], samples.shape[1])
        ax.plot(
            anchor[:T, 0], anchor[:T, 1],
            color="orange", linewidth=2, linestyle=":", alpha=0.85,
            label="Anchor prototype"
        )

    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Trajectory Samples")
    half = view_range / 2
    ax.set_xlim(center_x - half, center_x + half)
    ax.set_ylim(center_y - half, center_y + half)
    ax.set_aspect("equal")

    return fig


def plot_velocity(
    samples: np.ndarray,
    data: dict,
) -> Figure:
    """Speed and longitudinal acceleration profiles for all N samples + GT.

    Args:
        samples: (N, T, 4) ego trajectories.
        data: Observation dict (needs ego_current_state, optionally ego_agent_future).

    Returns:
        matplotlib Figure with two subplots (speed top, acceleration bottom).
    """
    fig = Figure(figsize=(6, 7))
    ax_vel = fig.add_subplot(211)
    ax_acc = fig.add_subplot(212)

    data_cpu = {k: v.cpu().numpy() if hasattr(v, "cpu") else v for k, v in data.items()}
    ego_state = np.array(data_cpu["ego_current_state"]).reshape(-1)  # (10,)

    for i, traj in enumerate(samples):
        color = sample_color(i)
        vel = _calculate_velocities(traj, ego_state)
        acc = _calculate_accelerations(vel)
        t = np.arange(len(vel))
        ax_vel.plot(t, vel, color=color, linewidth=1.8, alpha=0.75, label=f"S{i}")
        ax_acc.plot(t, acc, color=color, linewidth=1.8, alpha=0.75)

    # Ground truth
    if "ego_agent_future" in data_cpu:
        ego_future = np.array(data_cpu["ego_agent_future"]).reshape(-1, 3)
        gt_vel = _gt_velocities(ego_future, ego_state)
        if gt_vel is not None:
            ax_vel.plot(np.arange(len(gt_vel)), gt_vel,
                        color="black", linewidth=2, linestyle="--", alpha=0.7, label="GT")

    ax_vel.set_ylabel("Speed (km/h)")
    ax_vel.set_ylim(0, 80)
    ax_vel.set_title("Speed")
    ax_vel.legend(loc="upper right", fontsize=7)
    ax_vel.grid(True, alpha=0.3)

    ax_acc.set_ylabel("Accel (m/s²)")
    ax_acc.set_xlabel("Time step")
    ax_acc.set_ylim(-3, 3)
    ax_acc.set_title("Longitudinal Acceleration")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    return fig


def plot_lateral_curvature(
    samples: np.ndarray,
    data: dict,
) -> Figure:
    """Lateral acceleration and curvature profiles for all N samples + GT.

    Args:
        samples: (N, T, 4) ego trajectories.
        data: Observation dict.

    Returns:
        matplotlib Figure with two subplots.
    """
    fig = Figure(figsize=(6, 7))
    ax_lat = fig.add_subplot(211)
    ax_curv = fig.add_subplot(212)

    data_cpu = {k: v.cpu().numpy() if hasattr(v, "cpu") else v for k, v in data.items()}
    ego_state = np.array(data_cpu["ego_current_state"]).reshape(-1)

    for i, traj in enumerate(samples):
        color = sample_color(i)
        vel = _calculate_velocities(traj, ego_state)
        curv = _calculate_curvature(traj, ego_state)
        lat_acc = _calculate_lateral_acceleration(curv, vel)
        t = np.arange(len(curv))
        ax_lat.plot(t, lat_acc, color=color, linewidth=1.8, alpha=0.75, label=f"S{i}")
        ax_curv.plot(t, curv, color=color, linewidth=1.8, alpha=0.75)

    # Ground truth
    if "ego_agent_future" in data_cpu:
        ego_future = np.array(data_cpu["ego_agent_future"]).reshape(-1, 3)
        gt_vel = _gt_velocities(ego_future, ego_state)
        gt_curv = _gt_curvature(ego_future, ego_state)
        if gt_vel is not None and gt_curv is not None:
            gt_lat = _calculate_lateral_acceleration(gt_curv, gt_vel)
            ax_lat.plot(np.arange(len(gt_lat)), gt_lat,
                        color="black", linewidth=2, linestyle="--", alpha=0.7, label="GT")
            ax_curv.plot(np.arange(len(gt_curv)), gt_curv,
                         color="black", linewidth=2, linestyle="--", alpha=0.7)

    ax_lat.set_ylabel("Lat. accel (m/s²)")
    ax_lat.set_ylim(0, 8)
    ax_lat.set_title("Lateral Acceleration")
    ax_lat.axhline(y=3.0, color="purple", linestyle=":", linewidth=1, alpha=0.5)
    ax_lat.legend(loc="upper right", fontsize=7)
    ax_lat.grid(True, alpha=0.3)

    ax_curv.set_ylabel("Curvature (1/m)")
    ax_curv.set_xlabel("Time step")
    ax_curv.set_ylim(-0.2, 0.2)
    ax_curv.set_title("Path Curvature")
    ax_curv.grid(True, alpha=0.3)
    ax_curv.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    return fig


def _fig_to_pil(fig: Figure) -> Image.Image:
    """Convert a matplotlib Figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


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
    fig = Figure(figsize=(2, 2))
    ax = fig.add_subplot(111)
    for i, proto in enumerate(all_protos):
        if i != index:
            ax.plot(proto[:, 0], proto[:, 1], color="#cccccc", linewidth=0.8, alpha=0.6)
    ax.plot(all_protos[index, :, 0], all_protos[index, :, 1],
            color="royalblue", linewidth=2.2, zorder=5)
    ax.scatter([0], [0], c="red", s=20, zorder=6)
    ax.set_title(f"#{index}  n={count}", fontsize=8, fontweight="bold" if selected else "normal")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    img = _fig_to_pil(fig)
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
    protos = np.load(prototypes_path)           # (K, 80, 2)
    K = protos.shape[0]
    counts_path = prototypes_path.replace(".npy", "_counts.npy")
    counts = np.load(counts_path) if os.path.exists(counts_path) else np.ones(K, dtype=int)
    return [
        (render_prototype_thumbnail(protos, i, int(counts[i]), selected=(i == selected_index)),
         f"#{i} (n={int(counts[i])})")
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
        anchor: (T, 2) anchor prototype xy or None.

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
        anchor_xy = anchor[:T, :2]
        ades_anc = []
        for traj in samples:
            ade = float(np.mean(np.linalg.norm(traj[:, :2] - anchor_xy, axis=1)))
            ades_anc.append(ade)
        result["ade_anchor"] = float(np.mean(ades_anc))

    # Spread: std of final-position distances across samples
    final_positions = samples[:, -1, :2]  # (N, 2)
    if N > 1:
        result["spread_fde_std"] = float(
            np.std(np.linalg.norm(final_positions - final_positions.mean(axis=0), axis=1))
        )
    else:
        result["spread_fde_std"] = 0.0

    return result
