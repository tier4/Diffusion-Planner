"""Run new-DP native H5/ONNX with the old DP scenario metrics and result layout."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.scenario_based_open_loop.open_loop import METRICS

from planner_metrics.scene_data import extract_metric_scene_data

from .dataset import H5FrameIndex
from .model import NewDpOnnxRunner

DEFAULT_PARAMETERS = {
    "centerline": {"horizon_seconds": 8.0},
    "departure": {"horizon_seconds": 3.0, "minimum_displacement_m": 2.0},
    "traffic_light_go": {"horizon_seconds": 3.0, "minimum_displacement_m": 2.0},
    "simple_turn": {"horizon_seconds": 8.0},
    "object_avoidance": {},
    "pedestrian_yield": {"horizon_seconds": 3.0, "maximum_forward_progress_m": 0.5},
    "vehicle_yield": {"horizon_seconds": 3.0, "maximum_forward_progress_m": 0.5},
    "temporal_stop": {"horizon_seconds": 3.0, "maximum_forward_progress_m": 0.5},
    "obstacle_stop": {"tolerance_m": 0.5},
    "traffic_light_stop": {"tolerance_m": 0.5},
}


def metric_view(frame: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Build only the legacy-shaped fields required by the unchanged scorers.

    Every value comes directly from a native field. The packed neighbor tensor is
    never returned to the model; columns unused by metrics remain zero.
    """
    ego_past = frame["ego_agent_past"]
    current = np.zeros(10, dtype=np.float32)
    current[:5] = ego_past[-1, :5]
    current[9] = ego_past[-1, 5]

    neighbor_pose = frame["neighbor_agents_past"]
    if frame["agent_shape"].shape != (neighbor_pose.shape[0], 2):
        raise ValueError("agent_shape does not match native neighbor slots")
    if frame["agent_label"].shape != (neighbor_pose.shape[0], 3):
        raise ValueError("agent_label does not match native neighbor slots")
    packed = np.zeros((*neighbor_pose.shape[:-1], 11), dtype=np.float32)
    packed[..., :4] = neighbor_pose
    packed[..., 6:8] = frame["agent_shape"][:, None, :]
    packed[..., 8:11] = frame["agent_label"][:, None, :]
    raw = {
        "ego_current_state": current,
        "ego_agent_future": frame["ego_agent_future"],
        "route_lanes": frame["route_lanes"],
        "lanes": frame["lanes"],
        "neighbor_agents_future": frame["neighbor_agents_future"],
        "neighbor_agents_past": packed,
        "ego_shape": frame["ego_shape"],
    }
    return {key: torch.from_numpy(value) for key, value in raw.items()}


def _stack_metric_views(frames: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    views = [metric_view(frame) for frame in frames]
    return {key: torch.stack([view[key] for view in views]) for key in views[0]}


def _valid_points(values: np.ndarray) -> np.ndarray:
    """Return a geometry row without H5's all-zero padding points."""
    return values[np.any(values != 0, axis=-1)]


def _draw_geometry(ax: object, geometries: np.ndarray, **kwargs: object) -> None:
    """Draw every non-empty native H5 geometry independently."""
    for geometry in geometries:
        points = _valid_points(geometry)
        if len(points):
            ax.plot(points[:, 0], points[:, 1], **kwargs)
            kwargs["label"] = "_nolegend_"


def visualize_h5_prediction(
    frame: dict[str, np.ndarray], prediction: np.ndarray, save_path: Path, title: str
) -> None:
    """Save one native-H5 scene without connecting zero-padded geometry to the origin."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    for lanes, color, label in (
        (frame["lanes"], "#9ca3af", "Lanes"),
        (frame["route_lanes"], "#00a6d6", "Route"),
    ):
        for lane in lanes:
            points = _valid_points(lane)
            if not len(points):
                continue
            center = points[:, :2]
            ax.plot(center[:, 0], center[:, 1], color=color, alpha=0.75, linewidth=1.2,
                    label=label)
            ax.plot(
                center[:, 0] + points[:, 2], center[:, 1] + points[:, 3],
                color=color, alpha=0.35, linewidth=0.8,
            )
            ax.plot(
                center[:, 0] + points[:, 4], center[:, 1] + points[:, 5],
                color=color, alpha=0.35, linewidth=0.8,
            )
            label = "_nolegend_"

    for geometries, color, label in (
        (frame["road_borders"], "#dc2626", "Road border"),
        (frame["stop_lines"], "#f59e0b", "Stop line"),
    ):
        _draw_geometry(ax, geometries, color=color, linewidth=1.0, label=label)

    for area in frame["intersection_area"]:
        points = _valid_points(area)
        if len(points) >= 3:
            ax.fill(points[:, 0], points[:, 1], color="#6b7280", alpha=0.15)

    ego_past = _valid_points(frame["ego_agent_past"])
    ax.plot(ego_past[:, 0], ego_past[:, 1], "--", color="#fb923c", label="Ego history")
    ego_future = _valid_points(frame["ego_agent_future"])
    ax.plot(ego_future[:, 0], ego_future[:, 1], "--", color="#111827", label="Ground truth")
    for neighbor in frame["neighbor_agents_past"]:
        valid = np.square(neighbor[:, 2]) + np.square(neighbor[:, 3]) > 0.5
        if np.any(valid):
            ax.plot(neighbor[valid, 0], neighbor[valid, 1], color="#64748b", alpha=0.35, linewidth=0.8)

    ax.plot(prediction[:, 0], prediction[:, 1], color="#f97316", linewidth=2, label="New DP output")
    ax.scatter(0.0, 0.0, color="#dc2626", marker="^", label="Ego")
    ax.scatter(prediction[-1, 0], prediction[-1, 1], color="black", marker="x", label="Prediction end")
    goal = frame["goal_pose"]
    if np.linalg.norm(goal[:2]) <= 100.0:
        ax.scatter(goal[0], goal[1], color="#2563eb", marker="*", s=80, label="Goal")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-60.0, 60.0)
    ax.set_ylim(-60.0, 60.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _resolve_matrix(matrix_path: Path, dataset: H5FrameIndex) -> dict[str, list[tuple[dict, int]]]:
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("open-loop matrix must be an object")
    result = {}
    for metric_name, references in payload.items():
        if not isinstance(references, list):
            raise ValueError(f"{metric_name} must be a list")
        resolved = []
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != {"h5_path", "frame_index"}:
                raise ValueError(f"{metric_name} entries must contain only h5_path and frame_index")
            index = dataset.index_for_frame(
                reference["h5_path"], int(reference["frame_index"]), relative_to=matrix_path.parent
            )
            resolved.append((reference, index))
        result[metric_name] = resolved
    return result


def run(
    matrix_path: Path,
    index_path: Path,
    onnx_path: Path,
    output: Path,
    batch_size: int = 8,
    seed: int = 0,
    providers: list[str] | None = None,
    visualize: bool = True,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    runner = NewDpOnnxRunner(str(onnx_path), providers)
    summaries: dict[str, dict[str, float]] = {}
    seed_cursor = 0
    with H5FrameIndex(index_path) as dataset:
        # Resolve everything before inference: partial/misaligned matrices fail atomically.
        resolved = _resolve_matrix(matrix_path, dataset)
        unknown = set(resolved).difference(METRICS)
        if unknown:
            raise ValueError(f"Unsupported metrics: {sorted(unknown)}")
        for metric_name, samples in resolved.items():
            totals: dict[str, float] = defaultdict(float)
            details: list[dict] = []
            for offset in range(0, len(samples), batch_size):
                chunk = samples[offset : offset + batch_size]
                frames = [dataset.frame(index) for _, index in chunk]
                trajectories, _ = runner.predict(
                    frames, [seed + seed_cursor + offset + i for i in range(len(chunk))]
                )
                ego_prediction = torch.from_numpy(trajectories[:, 0])
                evaluation = METRICS[metric_name](
                    ego_prediction,
                    extract_metric_scene_data(_stack_metric_views(frames)),
                    DEFAULT_PARAMETERS[metric_name],
                )
                for key, values in evaluation.scores.items():
                    totals[key] += float(values.float().sum().item())
                for i, (reference, row_index) in enumerate(chunk):
                    indexed = dataset.rows[row_index]
                    row = {
                        "sample_index": offset + i,
                        "h5_path": str(reference.get("h5_path", indexed["h5_path"])),
                        "frame_index": int(indexed["frame_index"]),
                        "frame_time_ns": int(indexed["frame_time_ns"]),
                        "metrics": {
                            key: float(value[i].float().item())
                            for key, value in evaluation.scores.items()
                        },
                    }
                    for section, fields in evaluation.details.items():
                        row[section] = {key: value[i].item() for key, value in fields.items()}
                    if visualize:
                        png_path = (
                            output
                            / "visualization"
                            / metric_name
                            / f"{offset + i:06d}_{Path(indexed['h5_path']).stem}.png"
                        )
                        visualize_h5_prediction(
                            frames[i], trajectories[i, 0], png_path, Path(indexed["h5_path"]).stem
                        )
                        row["visualization_png"] = str(png_path)
                    details.append(row)
            detail_path = output / "details" / metric_name / "details.jsonl"
            detail_path.parent.mkdir(parents=True, exist_ok=True)
            detail_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details),
                encoding="utf-8",
            )
            summaries[metric_name] = (
                {key: total / len(samples) for key, total in totals.items()} if samples else {}
            )
            seed_cursor += len(samples)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", action="append", dest="providers")
    parser.add_argument("--no-visualization", action="store_false", dest="visualize")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.matrix,
                args.index,
                args.onnx,
                args.output,
                args.batch_size,
                args.seed,
                args.providers,
                args.visualize,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
