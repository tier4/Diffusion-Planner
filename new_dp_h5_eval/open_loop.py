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


def run(matrix_path: Path, index_path: Path, onnx_path: Path, output: Path,
        batch_size: int = 8, seed: int = 0, providers: list[str] | None = None) -> dict:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    unknown = set(matrix).difference(METRICS)
    if unknown:
        raise ValueError(f"Unsupported metrics: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    runner = NewDpOnnxRunner(str(onnx_path), providers)
    summaries: dict[str, dict[str, float]] = {}
    seed_cursor = 0
    with H5FrameIndex(index_path) as dataset:
        # Resolve everything before inference: partial/misaligned matrices fail atomically.
        resolved = {name: [(path, dataset.index_for_source(path)) for path in paths]
                    for name, paths in matrix.items()}
        for metric_name, samples in resolved.items():
            totals: dict[str, float] = defaultdict(float)
            details: list[dict] = []
            for offset in range(0, len(samples), batch_size):
                chunk = samples[offset:offset + batch_size]
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
                for i, (source, _) in enumerate(chunk):
                    row = {
                        "sample_index": offset + i,
                        "source_npz": str(source),
                        "metrics": {key: float(value[i].float().item())
                                    for key, value in evaluation.scores.items()},
                    }
                    for section, fields in evaluation.details.items():
                        row[section] = {key: value[i].item() for key, value in fields.items()}
                    details.append(row)
            detail_path = output / "details" / metric_name / "details.jsonl"
            detail_path.parent.mkdir(parents=True, exist_ok=True)
            detail_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                           for row in details), encoding="utf-8")
            summaries[metric_name] = ({key: total / len(samples) for key, total in totals.items()}
                                      if samples else {})
            seed_cursor += len(samples)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n",
                                          encoding="utf-8")
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
    args = parser.parse_args()
    print(json.dumps(run(args.matrix, args.index, args.onnx, args.output,
                         args.batch_size, args.seed, args.providers), indent=2))


if __name__ == "__main__":
    main()
