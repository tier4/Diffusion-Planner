"""Render example scenes that show each measured attention trend concretely.

Aggregate statistics say a trend exists; they do not show what it looks like.
For each trend this finds the paired frame where the two models differ most and
draws both models' attention over that scene side by side, then finds a
closed-loop scene where one model stalls and the other does not.

Drawing follows ``rlvr.autoresearch.visualize_scenes``: matplotlib, 8x8 inch
panels, grey lane boundaries, red road borders, a dark blue ego box, pose
oriented footprint rectangles, and the two models distinguished blue against
orange.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import argcomplete
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from tqdm import tqdm  # noqa: E402

from diffusion_planner.analysis import (  # noqa: E402
    SceneTokenLayout,
    annotate_records,
    capture_fusion_attention,
    ego_query_attention,
    rollout,
    token_records,
)
from diffusion_planner.data import PlannerDataset  # noqa: E402
from diffusion_planner.data.dimensions import (  # noqa: E402
    TRAJECTORY_DIM,
    TRAJECTORY_LENGTH,
)
from diffusion_planner.data.transforms import (  # noqa: E402
    PlannerDataNormalizer,
    PlannerUnknownLabelAugmentation,
)
from diffusion_planner.models.diffusion_planner import DiffusionPlanner  # noqa: E402
from diffusion_planner.utils.checkpoint import load_model  # noqa: E402
from diffusion_planner.visualizer.schema import (  # noqa: E402
    EgoIndex,
)

# Palette taken from rlvr.autoresearch.visualize_scenes.
LANE_COLOR = "#bbb"
BORDER_COLOR = "red"
EGO_COLOR = "#3366cc"
MODEL_COLORS = ("#3366cc", "orange")


def _numpy_frame(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
        for key, value in frame.items()
    }


def _records_for(
    model: DiffusionPlanner,
    frame: dict[str, Any],
    normalizer: PlannerDataNormalizer,
    layout: SceneTokenLayout,
    device: torch.device,
    layer: int | str = "mean",
) -> list[dict[str, Any]]:
    normalized = normalizer({key: np.asarray(v) for key, v in frame.items()})
    data = {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in normalized.items()
    }
    with capture_fusion_attention(model) as capture, torch.no_grad():
        model.scene_encoder(data)
    attention = (
        ego_query_attention(capture, layout, layer=layer)[0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    return annotate_records(token_records(frame, attention, layout), frame)


def _head_attention(
    model: DiffusionPlanner,
    frame: dict[str, Any],
    normalizer: PlannerDataNormalizer,
    layout: SceneTokenLayout,
    device: torch.device,
    head: int,
) -> np.ndarray:
    """Ego-query attention for one head, averaged over layers."""
    normalized = normalizer({key: np.asarray(v) for key, v in frame.items()})
    data = {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in normalized.items()
    }
    with capture_fusion_attention(model) as capture, torch.no_grad():
        model.scene_encoder(data)
    return (
        ego_query_attention(capture, layout, head=head)[0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def _shares(records: list[dict[str, Any]]) -> dict[str, float]:
    total = sum(r["attention"] for r in records) or 1.0
    neighbors = sum(r["attention"] for r in records if r["block"] == "neighbors") or 1.0
    lanes = sum(r["attention"] for r in records if r["block"] == "lanes") or 1.0
    spatial = [r for r in records if r["distance_m"] is not None]
    spatial_total = sum(r["attention"] for r in spatial) or 1.0
    return {
        "neighbors": neighbors / total,
        "route": sum(r["attention"] for r in records if r["block"] == "route_lanes")
        / total,
        "ahead": sum(
            r["attention"]
            for r in records
            if r["block"] == "neighbors" and r.get("bearing") == "ahead"
        )
        / neighbors,
        "near": sum(r["attention"] for r in spatial if r["distance_m"] < 20.0)
        / spatial_total,
        "pedestrian": sum(
            r["attention"]
            for r in records
            if r["block"] == "neighbors" and r.get("agent_class") == "pedestrian"
        )
        / neighbors,
        "marked_lane": sum(
            r["attention"]
            for r in records
            if r["block"] == "lanes" and r.get("lane_marking")
        )
        / lanes,
        "behind": sum(
            r["attention"]
            for r in records
            if r["block"] == "neighbors" and r.get("bearing") == "behind"
        )
        / neighbors,
        "far": sum(r["attention"] for r in spatial if r["distance_m"] >= 50.0)
        / spatial_total,
        "bicycle": sum(
            r["attention"]
            for r in records
            if r["block"] == "neighbors" and r.get("agent_class") == "bicycle"
        )
        / neighbors,
        "borders": sum(r["attention"] for r in records if r["block"] == "road_borders")
        / total,
        "concentration": sum(
            sorted((r["attention"] for r in records), reverse=True)[:10]
        )
        / total,
    }


def _box(
    ax: Axes,
    x: float,
    y: float,
    cos_yaw: float,
    sin_yaw: float,
    length: float,
    width: float,
    rear_offset: float,
    *,
    face: str,
    edge: str,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> None:
    """A pose-oriented box anchored ``rear_offset`` behind the reference point.

    Verbatim from ``visualize_scenes``: ego boxes are rear-axle referenced with
    ``ro = (length - wheelbase) / 2``, neighbours are centroid referenced with
    ``nro = length / 2``.
    """
    norm = float(np.hypot(cos_yaw, sin_yaw))
    if norm < 1e-3:
        return
    heading = float(np.arctan2(sin_yaw / norm, cos_yaw / norm))
    transform = mtransforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
    ax.add_patch(
        Rectangle(
            (-rear_offset, -width / 2),
            length,
            width,
            lw=linewidth,
            ec=edge,
            fc=face,
            alpha=alpha,
            zorder=zorder,
            transform=transform,
        )
    )


def _draw_scene(
    ax: Axes,
    frame: dict[str, Any],
    traj: np.ndarray | None = None,
    label: str = "",
    colour: str = EGO_COLOR,
    *,
    show_gt: bool = True,
) -> None:
    """Port of ``rlvr.autoresearch.visualize_scenes.draw_scene``.

    Every style constant is the original's. Only the field access changes,
    because this schema splits what the old npz fused: lane boundary offsets sit
    at columns 2:6 of a six-wide lane rather than 4:8 of an eight-wide one, road
    borders are their own tensor rather than flagged line-strings, and neighbour
    extents live in ``agent_shape`` rather than inside the pose tensor.
    """
    shape = np.asarray(frame["ego_shape"], dtype=np.float32)
    wheelbase, ego_length, ego_width = (
        float(shape[0]),
        float(shape[1]),
        float(shape[2]),
    )
    rear_offset = (ego_length - wheelbase) / 2

    lanes = np.asarray(frame["lanes"], dtype=np.float32)
    for lane in lanes:
        points = lane[:, :2]
        if np.abs(points).sum() < 1e-6:
            continue
        left, right = lane[:, 2:4], lane[:, 4:6]
        valid = np.abs(points).sum(axis=1) > 0.1
        if valid.sum() > 1:
            ax.plot(
                (points + left)[valid, 0],
                (points + left)[valid, 1],
                "-",
                color=LANE_COLOR,
                alpha=0.5,
                lw=0.7,
            )
            ax.plot(
                (points + right)[valid, 0],
                (points + right)[valid, 1],
                "-",
                color=LANE_COLOR,
                alpha=0.5,
                lw=0.7,
            )

    for line in np.asarray(frame["road_borders"], dtype=np.float32):
        valid = np.abs(line).sum(axis=1) > 0.01
        if valid.sum() > 1:
            ax.plot(
                line[valid, 0],
                line[valid, 1],
                color=BORDER_COLOR,
                lw=3,
                alpha=0.7,
                zorder=4,
            )

    if show_gt and "ego_agent_future" in frame:
        gt = np.asarray(frame["ego_agent_future"], dtype=np.float32)
        ax.plot(gt[:, 0], gt[:, 1], "g-", lw=2, alpha=0.5, zorder=5)
        ax.plot(
            gt[::3, 0],
            gt[::3, 1],
            "go",
            ms=3,
            alpha=0.7,
            mew=0,
            zorder=6,
            label="GT",
        )

    _box(
        ax,
        0.0,
        0.0,
        1.0,
        0.0,
        ego_length,
        ego_width,
        rear_offset,
        face=EGO_COLOR,
        edge="black",
        alpha=0.9,
        linewidth=2,
        zorder=20,
    )

    poses = np.asarray(frame["neighbor_agents_past"], dtype=np.float32)
    sizes = np.asarray(frame["agent_shape"], dtype=np.float32)
    for index in range(poses.shape[0]):
        pose = poses[index]
        if np.abs(pose).sum() < 1e-6:
            continue
        width, length = float(sizes[index, 0]), float(sizes[index, 1])
        if length < 0.1 or width < 0.1:
            continue
        _box(
            ax,
            float(pose[-1, 0]),
            float(pose[-1, 1]),
            float(pose[-1, 2]),
            float(pose[-1, 3]),
            length,
            width,
            length / 2,
            face="#ff8844",
            edge="#cc4400",
            alpha=0.7,
            linewidth=1.5,
            zorder=15,
        )

    if traj is None:
        return

    path_length = float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum())
    ax.plot(traj[:, 0], traj[:, 1], "-", color=colour, lw=2, alpha=0.5, zorder=10)
    ax.plot(
        traj[::3, 0],
        traj[::3, 1],
        "o",
        color=colour,
        ms=3.5,
        alpha=0.9,
        mew=0,
        zorder=11,
        label=f"{label} ({path_length:.1f}m)",
    )
    for step in range(5, len(traj), 10):
        _box(
            ax,
            float(traj[step, 0]),
            float(traj[step, 1]),
            float(traj[step, 2]),
            float(traj[step, 3]),
            ego_length,
            ego_width,
            rear_offset,
            face=colour,
            edge=colour,
            alpha=0.15,
            linewidth=0.5,
            zorder=8,
        )
    _box(
        ax,
        float(traj[-1, 0]),
        float(traj[-1, 1]),
        float(traj[-1, 2]),
        float(traj[-1, 3]),
        ego_length,
        ego_width,
        rear_offset,
        face=colour,
        edge=colour,
        alpha=0.4,
        linewidth=1.5,
        zorder=9,
    )


def _predict_ego(
    model: DiffusionPlanner,
    frame: dict[str, Any],
    normalizer: PlannerDataNormalizer,
    device: torch.device,
    *,
    seed: int = 0,
) -> np.ndarray:
    """The model's sampled ego trajectory, denormalized, as (T, 4)."""
    normalized = normalizer({key: np.asarray(v) for key, v in frame.items()})
    data = {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in normalized.items()
    }
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        (
            1,
            int(normalized["neighbor_agents_past"].shape[0]) + 1,
            TRAJECTORY_LENGTH,
            TRAJECTORY_DIM,
        ),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    with torch.inference_mode():
        prediction, _ = model.sample(data, initial_noise=noise, num_steps=10)
    trajectories = prediction[0].detach().float().cpu().numpy()
    return normalizer.denormalize_trajectory(trajectories).astype(np.float32)[0]


def _legend_handles(show_gt: bool) -> list[Any]:
    """Explicit key: the scene is assembled from primitives, not labelled series."""
    handles: list[Any] = [
        Patch(facecolor=EGO_COLOR, edgecolor="black", label="ego"),
        Patch(
            facecolor="#ff8844", edgecolor="#cc4400", alpha=0.7, label="other agents"
        ),
        Line2D([], [], color=BORDER_COLOR, lw=3, alpha=0.7, label="road border"),
        Line2D([], [], color=LANE_COLOR, lw=1.2, alpha=0.8, label="lane boundary"),
    ]
    if show_gt:
        handles.append(
            Line2D(
                [],
                [],
                color="g",
                lw=2,
                marker="o",
                ms=3,
                alpha=0.7,
                label="ego ground truth",
            )
        )
    handles.append(
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            ms=9,
            markerfacecolor="#d1495b",
            markeredgecolor="black",
            markeredgewidth=0.4,
            label="attention (area and colour ∝ share)",
        )
    )
    return handles


def _annotate_top(
    ax: Axes, records: list[dict[str, Any]], blocks: set[str] | None, top: int
) -> None:
    """Print the share on the most-attended tokens; labelling all is unreadable."""
    chosen = sorted(
        (
            r
            for r in records
            if r["x_m"] is not None and (blocks is None or r["block"] in blocks)
        ),
        key=lambda r: -r["attention_pct"],
    )[:top]
    for record in chosen:
        ax.annotate(
            f"{record['attention_pct']:.2f}%",
            (record["x_m"], record["y_m"]),
            textcoords="offset points",
            xytext=(0, 13),
            ha="center",
            fontsize=6.5,
            zorder=25,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )


def _attention_layer(
    ax: Axes, records: list[dict[str, Any]], blocks: set[str] | None, scale: float
) -> Any:
    """Attention markers on top of the scene, on a scale shared across panels."""
    chosen = [
        r
        for r in records
        if r["x_m"] is not None and (blocks is None or r["block"] in blocks)
    ]
    if not chosen:
        return None
    shares = np.array([r["attention_pct"] for r in chosen], dtype=float)
    return ax.scatter(
        [r["x_m"] for r in chosen],
        [r["y_m"] for r in chosen],
        s=20.0 + 460.0 * shares / max(scale, 1e-9),
        c=shares,
        cmap="inferno",
        vmin=0.0,
        vmax=scale,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.4,
        zorder=15,
    )


def _extent(
    frame: dict[str, Any],
    groups: list[list[dict[str, Any]]],
    blocks: set[str] | None,
    reach: float = 120.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bounds that contain everything actually drawn.

    Driven by the plotted attention markers rather than a symmetric heuristic:
    a fixed reach silently clips the very tokens a figure exists to show.
    """
    points = [np.zeros((1, 2), dtype=np.float32)]
    poses = np.asarray(frame["neighbor_agents_past"], dtype=np.float32)
    occupied = np.abs(poses).sum(axis=(1, 2)) > 0.0
    if occupied.any():
        points.append(poses[occupied][:, -1, :2])
    marked = [
        (r["x_m"], r["y_m"])
        for group in groups
        for r in group
        if r["x_m"] is not None and (blocks is None or r["block"] in blocks)
    ]
    if marked:
        points.append(np.asarray(marked, dtype=np.float32))
    stacked = np.vstack(points)
    stacked = stacked[np.abs(stacked).max(axis=1) <= reach]
    if stacked.size == 0:
        stacked = np.zeros((1, 2), dtype=np.float32)
    pad = 8.0
    x_lo, x_hi = float(stacked[:, 0].min()) - pad, float(stacked[:, 0].max()) + pad
    y_lo, y_hi = float(stacked[:, 1].min()) - pad, float(stacked[:, 1].max()) + pad
    # With equal aspect, a wide flat scene renders as a sliver and a tall one as a
    # column. Expand whichever axis is short until the panel is between 1:1 and
    # 2.5:1, so every figure in the set reads at a similar scale.
    width, height = x_hi - x_lo, y_hi - y_lo
    ratio = 2.5
    if width > ratio * height:
        centre = 0.5 * (y_lo + y_hi)
        half = 0.5 * width / ratio
        y_lo, y_hi = centre - half, centre + half
    elif height > width:
        centre = 0.5 * (x_lo + x_hi)
        half = 0.5 * height
        x_lo, x_hi = centre - half, centre + half
    return (x_lo, x_hi), (y_lo, y_hi)


def _shared_scale(groups: list[list[dict[str, Any]]], blocks: set[str] | None) -> float:
    return max(
        (
            r["attention_pct"]
            for group in groups
            for r in group
            if r["x_m"] is not None and (blocks is None or r["block"] in blocks)
        ),
        default=1.0,
    )


def _figsize(
    bounds: tuple[tuple[float, float], tuple[float, float]],
    columns: int,
    rows: int,
    panel_width: float = 8.0,
) -> tuple[float, float]:
    """Figure size matching the data aspect, so equal-aspect panels fill the canvas."""
    (x_lo, x_hi), (y_lo, y_hi) = bounds
    ratio = (y_hi - y_lo) / max(x_hi - x_lo, 1e-6)
    panel_height = min(max(panel_width * ratio, 3.2), panel_width)
    return (panel_width * columns, panel_height * rows + 1.3)


def _finish(
    ax: Axes,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    title: str,
    legend: bool = True,
    show_gt: bool = True,
) -> None:
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    if legend:
        # draw_scene labels the ground truth "GT" and each trajectory with its
        # path length; keep the trajectory entries and let the proxy name the GT.
        existing, names = ax.get_legend_handles_labels()
        extra = [h for h, name in zip(existing, names, strict=True) if name != "GT"]
        ax.legend(
            handles=[*_legend_handles(show_gt), *extra],
            fontsize=7,
            loc="upper left",
            framealpha=0.85,
        )
    ax.set_title(title, fontsize=9)


def _comparison(
    frame: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
    trajectories: dict[str, np.ndarray],
    labels: tuple[str, str],
    suptitle: str,
    subtitles: tuple[str, str],
    blocks: set[str] | None,
    path: Path,
) -> None:
    scale = _shared_scale([records[label] for label in labels], blocks)
    bounds = _extent(frame, [records[label] for label in labels], blocks)
    figure, axes = plt.subplots(1, 2, figsize=_figsize(bounds, 2, 1))
    handle = None
    for ax, label, subtitle, colour in zip(
        axes, labels, subtitles, MODEL_COLORS, strict=True
    ):
        _draw_scene(ax, frame, trajectories.get(label), label, colour)
        handle = _attention_layer(ax, records[label], blocks, scale) or handle
        _annotate_top(ax, records[label], blocks, top=8)
        _finish(ax, bounds, subtitle)
        ax.set_xlabel("x (m)", fontsize=8)
    axes[0].set_ylabel("y (m)", fontsize=8)
    if handle is not None:
        bar = figure.colorbar(handle, ax=axes, fraction=0.025, pad=0.02)
        bar.set_label("attention (% of ego-query attention)", fontsize=8)
    figure.suptitle(suptitle, fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def _grid_2x2(
    frames: list[list[dict[str, Any]]],
    records: list[list[list[dict[str, Any]]]],
    titles: list[list[str]],
    suptitle: str,
    path: Path,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    """Two rows of two panels sharing one attention scale."""
    scale = _shared_scale([r for row in records for r in row], None)
    figure, axes = plt.subplots(2, 2, figsize=_figsize(bounds, 2, 2))
    handle = None
    for row in range(2):
        for column in range(2):
            ax = axes[row][column]
            _draw_scene(ax, frames[row][column], show_gt=False)
            handle = _attention_layer(ax, records[row][column], None, scale) or handle
            _annotate_top(ax, records[row][column], None, top=6)
            _finish(
                ax,
                bounds,
                titles[row][column],
                legend=row == 0 and column == 0,
                show_gt=False,
            )
    if handle is not None:
        bar = figure.colorbar(handle, ax=axes, fraction=0.025, pad=0.02)
        bar.set_label("attention (% of ego-query attention)", fontsize=8)
    figure.suptitle(suptitle, fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def _filmstrip(
    frames: dict[str, list[dict[str, Any]]],
    speeds: dict[str, list[float]],
    labels: tuple[str, str],
    steps: list[int],
    ideal: float,
    path: Path,
) -> None:
    columns = len(steps)
    figure = plt.figure(figsize=(8 * columns, 8 * 2 + 5))
    grid = figure.add_gridspec(3, columns, height_ratios=[4, 4, 1.5], hspace=0.16)
    for row, label in enumerate(labels):
        for column, step in enumerate(steps):
            ax = figure.add_subplot(grid[row, column])
            _draw_scene(ax, frames[label][step], show_gt=False)
            ax.set_xlim(-25.0, 60.0)
            ax.set_ylim(-30.0, 30.0)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.15)
            if row == 0 and column == 0:
                ax.legend(
                    handles=_legend_handles(False),
                    fontsize=7,
                    loc="upper left",
                    framealpha=0.85,
                )
            ax.set_title(f"{label} · rollout step {step}", fontsize=9)
    ax = figure.add_subplot(grid[2, :])
    for colour, label in zip(MODEL_COLORS, labels, strict=True):
        ax.plot(
            range(1, len(speeds[label]) + 1),
            speeds[label],
            color=colour,
            lw=2,
            alpha=0.85,
            label=label,
        )
    ax.axhline(ideal, color="red", ls="--", lw=1.5, alpha=0.7, label="recorded speed")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("metres per step")
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8, loc="upper right")
    figure.suptitle(
        "Closed loop: a scene where one model stalls and the other does not",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(figure)


TRENDS: tuple[tuple[str, str, set[str] | None, str, int], ...] = (
    (
        "ahead",
        "Agent attention swings forward",
        {"neighbors"},
        "share of agent attention ahead of the ego",
        1,
    ),
    (
        "neighbors",
        "Attention moves off the agents and onto the map",
        None,
        "share of all attention on agents",
        -1,
    ),
    (
        "route",
        "Route attention rises",
        None,
        "share of all attention on route lanes",
        1,
    ),
    (
        "near",
        "Attention lands closer to the ego",
        None,
        "share of spatial attention within 20 m",
        1,
    ),
    (
        "pedestrian",
        "Within agents, attention shifts toward pedestrians",
        {"neighbors"},
        "share of agent attention on pedestrians",
        1,
    ),
    (
        "marked_lane",
        "Marked lane boundaries preferred over virtual ones",
        {"lanes"},
        "share of lane attention on marked-boundary lanes",
        1,
    ),
    (
        "behind",
        "The 3-class model looks behind itself more",
        {"neighbors"},
        "share of agent attention behind the ego",
        -1,
    ),
    (
        "far",
        "The 3-class model reaches further out",
        None,
        "share of spatial attention beyond 50 m",
        -1,
    ),
    (
        "bicycle",
        "Bicycles lose attention share",
        {"neighbors"},
        "share of agent attention on bicycles",
        -1,
    ),
    (
        "borders",
        "Road borders gain attention",
        None,
        "share of all attention on road borders",
        1,
    ),
    (
        "concentration",
        "The 3-class model concentrates on fewer tokens",
        None,
        "share held by the ten most-attended tokens",
        -1,
    ),
)


def main() -> None:
    """Find and render one illustrative scene per trend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--search-frames", type=int, default=240)
    parser.add_argument("--search-scenes", type=int, default=120)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PlannerDataset(str(args.parquet), file_capacity=8)
    generator = np.random.default_rng(args.seed)
    labels = (args.label_a, args.label_b)
    models = {
        args.label_a: load_model(args.model_a, DiffusionPlanner).to(device).eval(),
        args.label_b: load_model(args.model_b, DiffusionPlanner).to(device).eval(),
    }
    widen = PlannerUnknownLabelAugmentation()
    normalizer = PlannerDataNormalizer()
    layout = SceneTokenLayout.from_dimensions()
    plots = args.out_dir / "plots"

    candidates = sorted(
        generator.choice(
            len(dataset), size=min(args.search_frames, len(dataset)), replace=False
        ).tolist()
    )
    best: dict[str, tuple[float, int]] = {}
    for index in tqdm(candidates, desc="searching frames", unit="frame"):
        frame = dict(widen(_numpy_frame(dataset[index])))
        if float(np.abs(frame["neighbor_agents_past"]).sum()) == 0.0:
            continue
        shares = {
            label: _shares(_records_for(model, frame, normalizer, layout, device))
            for label, model in models.items()
        }
        for key, _title, _blocks, _caption, sign in TRENDS:
            delta = sign * (shares[labels[1]][key] - shares[labels[0]][key])
            if key not in best or delta > best[key][0]:
                best[key] = (delta, index)

    for key, title, blocks, caption, _sign in TRENDS:
        if key not in best:
            continue
        delta, index = best[key]
        frame = dict(widen(_numpy_frame(dataset[index])))
        records = {
            label: _records_for(model, frame, normalizer, layout, device)
            for label, model in models.items()
        }
        trajectories = {
            label: _predict_ego(model, frame, normalizer, device)
            for label, model in models.items()
        }
        measured = {label: _shares(records[label])[key] for label in labels}
        subtitles = (
            f"{labels[0]} — {caption}: {measured[labels[0]]:.1%}",
            f"{labels[1]} — {caption}: {measured[labels[1]]:.1%}",
        )
        _comparison(
            frame,
            records,
            trajectories,
            labels,
            f"{title} — frame {index}",
            subtitles,
            blocks,
            plots / f"example_{key}.png",
        )
        print(f"{key}: frame {index}, gap {delta:+.4f}")

    if "route" in best:
        index = best["route"][1]
        frame = dict(widen(_numpy_frame(dataset[index])))
        layered = {
            (label, layer): _records_for(
                model, frame, normalizer, layout, device, layer=layer
            )
            for label, model in models.items()
            for layer in (0, 3)
        }
        titles = []
        for label in labels:
            row = []
            for layer in (0, 3):
                entries = layered[(label, layer)]
                total = sum(r["attention"] for r in entries) or 1.0
                share = (
                    sum(r["attention"] for r in entries if r["block"] == "route_lanes")
                    / total
                )
                row.append(f"{label} · fusion layer {layer} · route share {share:.1%}")
            titles.append(row)
        _grid_2x2(
            [[frame, frame], [frame, frame]],
            [
                [layered[(labels[0], 0)], layered[(labels[0], 3)]],
                [layered[(labels[1], 0)], layered[(labels[1], 3)]],
            ],
            titles,
            "Route attention is reorganised by depth, not uniformly raised",
            plots / "example_layers.png",
            _extent(frame, list(layered.values()), None),
        )
        print(f"layers: frame {index}")

    # Per-head view: the layer-and-head average hides that heads specialise.
    if "route" in best:
        index = best["route"][1]
        frame = dict(widen(_numpy_frame(dataset[index])))
        model = models[labels[1]]
        heads = (0, 3, 6, 9)
        by_head = {
            head: annotate_records(
                token_records(
                    frame,
                    _head_attention(model, frame, normalizer, layout, device, head),
                    layout,
                ),
                frame,
            )
            for head in heads
        }
        titles = []
        rows = []
        frames_grid = []
        for pair in ((heads[0], heads[1]), (heads[2], heads[3])):
            titles.append([f"{labels[1]} · attention head {head}" for head in pair])
            rows.append([by_head[pair[0]], by_head[pair[1]]])
            frames_grid.append([frame, frame])
        _grid_2x2(
            frames_grid,
            rows,
            titles,
            f"Individual attention heads specialise — {labels[1]}, fusion layer average",
            plots / "example_heads.png",
            _extent(frame, list(by_head.values()), None),
        )
        print(f"heads: frame {index}")

    scenes = sorted(
        generator.choice(
            len(dataset), size=min(args.search_scenes, len(dataset)), replace=False
        ).tolist()
    )
    chosen = None
    best_gap = 0.0
    for index in tqdm(scenes, desc="searching rollouts", unit="scene"):
        start = dict(widen(_numpy_frame(dataset[index])))
        speed = float(start["ego_agent_past"][-1, EgoIndex.VELOCITY])
        if speed < 3.0:
            continue
        ideal = speed * 0.1
        rollout_frames: dict[str, list[dict[str, Any]]] = {}
        lengths: dict[str, list[float]] = {}
        for label, model in models.items():
            collected: list[dict[str, Any]] = []
            seen: list[float] = []
            previous = 0.0
            for state in rollout(
                model,
                start,
                steps=args.steps,
                device=str(device),
                num_sampling_steps=10,
                replan_every=10,
                stop_beyond_map=False,
            ):
                collected.append(state.frame)
                seen.append(state.travelled_m - previous)
                previous = state.travelled_m
            rollout_frames[label] = collected
            lengths[label] = seen
        if any(len(v) < args.steps for v in lengths.values()):
            continue
        finals = {label: float(np.mean(v[-5:])) / ideal for label, v in lengths.items()}
        gap = finals[labels[1]] - finals[labels[0]]
        if finals[labels[0]] < 0.25 <= finals[labels[1]] and gap > best_gap:
            best_gap, chosen = gap, (index, start, rollout_frames, lengths, ideal)

    if chosen is None:
        print("closed loop: no scene found where exactly one model stalled")
        return
    index, start, rollout_frames, lengths, ideal = chosen
    _filmstrip(
        rollout_frames,
        lengths,
        labels,
        [0, args.steps // 2 - 1, args.steps - 1],
        ideal,
        plots / "example_closed_loop.png",
    )

    drift_frames = {
        label: {"start": start, "final": rollout_frames[label][-1]} for label in labels
    }
    drift_records = {
        (label, when): _records_for(
            models[label], drift_frames[label][when], normalizer, layout, device
        )
        for label in labels
        for when in ("start", "final")
    }
    drift_titles = []
    for label in labels:
        row = []
        for when in ("start", "final"):
            entries = drift_records[(label, when)]
            total = sum(r["attention"] for r in entries) or 1.0
            agents = (
                sum(r["attention"] for r in entries if r["block"] == "neighbors")
                / total
            )
            row.append(f"{label} · {when} of rollout · agent share {agents:.1%}")
        drift_titles.append(row)
    _grid_2x2(
        [
            [drift_frames[labels[0]]["start"], drift_frames[labels[0]]["final"]],
            [drift_frames[labels[1]]["start"], drift_frames[labels[1]]["final"]],
        ],
        [
            [drift_records[(labels[0], "start")], drift_records[(labels[0], "final")]],
            [drift_records[(labels[1], "start")], drift_records[(labels[1], "final")]],
        ],
        drift_titles,
        "Attention drift across a rollout: where each model ends up looking",
        plots / "example_drift.png",
        _extent(drift_frames[labels[0]]["start"], list(drift_records.values()), None),
    )
    print(f"closed loop: scene {index}, retention gap {best_gap:+.3f}")


if __name__ == "__main__":
    main()
