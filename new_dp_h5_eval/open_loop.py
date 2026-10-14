"""Run new-DP native H5/ONNX with the old DP scenario metrics and result layout."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.scenario_based_open_loop.open_loop import METRICS

from planner_metrics.scene_data import extract_metric_scene_data

from .dataset import H5FrameIndex
from .metric_compat import legacy_lanes
from .model import NewDpOnnxRunner
from .render import render_view, save_prediction_png
from .vehicle import (
    VehicleDimensions,
    add_vehicle_params_argument,
    sim_ego_shape,
    vehicle_params_from_args,
)

DRAW_QUEUE_PER_WORKER = 4

DEFAULT_PARAMETERS = {
    "arrival": {"position_tolerance_m": 2.0, "heading_tolerance_deg": 10.0},
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
    "lane_change": {
        "horizon_seconds": 8.0,
        "minimum_lateral_shift_m": 1.0,
        "chain_tolerance_m": 1.0,
    },
}


def metric_view(frame: dict[str, np.ndarray], ego_shape: np.ndarray) -> dict[str, torch.Tensor]:
    """Build only the legacy-shaped fields required by the unchanged scorers.

    Every value comes directly from a native field, except ``ego_shape``: the scorers read
    ``[wheelbase, length, width]``, not the frame's native ``[base_link_to_front, length,
    width]``, so the caller passes the scoring shape from ``vehicle.sim_ego_shape``. The
    packed neighbor tensor is never returned to the model; columns unused by metrics remain
    zero.
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
        "ego_agent_future": frame["ego_agent_future"][..., :4],
        "route_lanes": frame["route_lanes"],
        "lanes": legacy_lanes(frame),
        "neighbor_agents_future": frame["neighbor_agents_future"],
        "neighbor_agents_past": packed,
        "ego_shape": ego_shape,
    }
    return {key: torch.from_numpy(value) for key, value in raw.items()}


def _stack_metric_views(
    frames: list[dict[str, np.ndarray]], ego_shapes: list[np.ndarray]
) -> dict[str, torch.Tensor]:
    views = [metric_view(frame, shape) for frame, shape in zip(frames, ego_shapes)]
    return {key: torch.stack([view[key] for view in views]) for key in views[0]}


def _scoring_ego_shapes(
    dataset: H5FrameIndex,
    resolved: dict[str, list[tuple[dict, int]]],
    vehicle_params: dict[str, VehicleDimensions],
) -> dict[int, np.ndarray]:
    """Scoring ``[wheelbase, length, width]`` per index row (see ``vehicle.py``).

    Each shard is checked once, over all of its frames, before any inference runs.
    """
    by_shard: dict[Path, np.ndarray] = {}
    shapes: dict[int, np.ndarray] = {}
    for samples in resolved.values():
        for _, index in samples:
            path = dataset.shard_path(index)
            if path not in by_shard:
                by_shard[path] = sim_ego_shape(
                    dataset.ego_shapes(path),
                    dataset.project_id(path),
                    vehicle_params,
                    source=str(path),
                )
            shapes[index] = by_shard[path]
    return shapes


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


def _prefetched_chunks(
    dataset: H5FrameIndex, samples: list[tuple[dict, int]], batch_size: int, reader: Executor
) -> Iterator[tuple[int, list[tuple[dict, int]], list[dict[str, np.ndarray]]]]:
    """Each ``(offset, chunk, frames)`` of ``samples``, reading the next chunk's frames meanwhile."""

    def read(offset: int) -> list[dict[str, np.ndarray]]:
        return [dataset.frame(index) for _, index in samples[offset : offset + batch_size]]

    pending = reader.submit(read, 0)
    for offset in range(0, len(samples), batch_size):
        frames = pending.result()
        if offset + batch_size < len(samples):
            pending = reader.submit(read, offset + batch_size)
        yield offset, samples[offset : offset + batch_size], frames


def _draw_pool(stack: ExitStack, workers: int) -> ProcessPoolExecutor:
    """A drawing process pool that cancels its queued drawings when the run fails."""
    if workers < 1:
        raise ValueError(f"--draw-workers must be at least 1 to save images: {workers}")
    pool = stack.enter_context(ProcessPoolExecutor(workers, mp_context=get_context("spawn")))
    stack.callback(pool.shutdown, cancel_futures=True)
    return pool


def _wait_for_drawings(drawing: deque[Future], keep: int) -> None:
    """Collect the oldest drawings, raising their errors, until at most ``keep`` are queued."""
    while len(drawing) > keep:
        drawing.popleft().result()


def _detail_row(reference: dict, indexed: dict, sample_index: int, evaluation, i: int) -> dict:
    """The details.jsonl row of one sample of a scored chunk."""
    row = {
        "sample_index": sample_index,
        "h5_path": str(reference.get("h5_path", indexed["h5_path"])),
        "frame_index": int(indexed["frame_index"]),
        "frame_time_ns": int(indexed["frame_time_ns"]),
        "metrics": {
            key: float(value[i].float().item()) for key, value in evaluation.scores.items()
        },
    }
    for section, fields in evaluation.details.items():
        row[section] = {key: value[i].item() for key, value in fields.items()}
    return row


def run(
    matrix_path: Path,
    index_path: Path,
    onnx_path: Path,
    output: Path,
    batch_size: int = 8,
    seed: int = 0,
    providers: list[str] | None = None,
    visualize: bool = True,
    draw_workers: int = 8,
    *,
    vehicle_params: dict[str, VehicleDimensions],
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    runner = NewDpOnnxRunner(str(onnx_path), providers)
    summaries: dict[str, dict[str, float]] = {}
    seed_cursor = 0
    drawing: deque[Future] = deque()
    with ExitStack() as stack:
        dataset = stack.enter_context(H5FrameIndex(index_path))
        reader = stack.enter_context(ThreadPoolExecutor(1))
        draw = _draw_pool(stack, draw_workers) if visualize else None
        # Resolve everything before inference: partial/misaligned matrices fail atomically.
        resolved = _resolve_matrix(matrix_path, dataset)
        unknown = set(resolved).difference(METRICS)
        if unknown:
            raise ValueError(f"Unsupported metrics: {sorted(unknown)}")
        scoring_shapes = _scoring_ego_shapes(dataset, resolved, vehicle_params)
        for metric_name, samples in resolved.items():
            totals: dict[str, float] = defaultdict(float)
            details: list[dict] = []
            for offset, chunk, frames in _prefetched_chunks(dataset, samples, batch_size, reader):
                trajectories, _ = runner.predict(
                    frames, [seed + seed_cursor + offset + i for i in range(len(chunk))]
                )
                ego_prediction = torch.from_numpy(trajectories[:, 0])
                metric_shapes = [scoring_shapes[index] for _, index in chunk]
                evaluation = METRICS[metric_name](
                    ego_prediction,
                    extract_metric_scene_data(_stack_metric_views(frames, metric_shapes)),
                    DEFAULT_PARAMETERS[metric_name],
                )
                for key, values in evaluation.scores.items():
                    totals[key] += float(values.float().sum().item())
                for i, (reference, row_index) in enumerate(chunk):
                    indexed = dataset.rows[row_index]
                    row = _detail_row(reference, indexed, offset + i, evaluation, i)
                    if visualize:
                        stem = Path(indexed["h5_path"]).stem
                        png_path = (
                            output / "visualization" / metric_name / f"{offset + i:06d}_{stem}.png"
                        )
                        drawing.append(
                            draw.submit(
                                save_prediction_png,
                                render_view(frames[i], scoring_shapes[row_index]),
                                trajectories[i, 0],
                                png_path,
                                stem,
                            )
                        )
                        row["visualization_png"] = str(png_path)
                    details.append(row)
                _wait_for_drawings(drawing, DRAW_QUEUE_PER_WORKER * draw_workers)
            _wait_for_drawings(drawing, 0)
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
    parser.add_argument("--draw-workers", type=int, default=8)
    add_vehicle_params_argument(parser)
    args = parser.parse_args()
    vehicle_params = vehicle_params_from_args(parser, args)
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
                args.draw_workers,
                vehicle_params=vehicle_params,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
