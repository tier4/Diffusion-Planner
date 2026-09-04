"""Run ONNX and render scenes with the legacy Open Loop Matplotlib visualizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from visualize_input import visualize_inputs

from diffusion_planner.data import PlannerDataNormalizer, PlannerDataset
from diffusion_planner.models.onnx import PLANNER_INPUT_NAMES


def heading_to_cos_sin(array: np.ndarray) -> np.ndarray:
    """Convert a legacy [..., 3] heading array to [..., 4] pose array."""
    heading = array[..., 2]
    return np.concatenate(
        (array[..., :2], np.cos(heading)[..., None], np.sin(heading)[..., None]),
        axis=-1,
    ).astype(np.float32)


def run_prediction(
    session: ort.InferenceSession, frame: dict[str, np.ndarray], seed: int
) -> np.ndarray:
    """Run one current-format H5 frame through the sampler."""
    normalized = PlannerDataNormalizer()(frame)
    inputs = {
        key: np.asarray(normalized[key], dtype=np.float32)[None]
        for key in PLANNER_INPUT_NAMES
    }
    inputs["turn_indicators"] = np.asarray(
        normalized["turn_indicators"], dtype=np.float32
    )[None]
    inputs["initial_noise"] = torch.randn(
        (1, 321, 80, 4), generator=torch.Generator().manual_seed(seed)
    ).numpy()
    trajectory = session.run(None, inputs)[0][0]
    return PlannerDataNormalizer().denormalize_trajectory(trajectory)


def legacy_inputs(npz_path: Path) -> dict[str, np.ndarray]:
    """Load an original NPZ in the exact input convention of the old visualizer."""
    with np.load(npz_path, allow_pickle=False) as source:
        data = {
            key: np.asarray(source[key]) for key in source.files if key != "version"
        }
    for key in ("ego_agent_past", "goal_pose"):
        data[key] = heading_to_cos_sin(data[key])
    return {key: value[None] for key, value in data.items()}


def centerline_errors(prediction: np.ndarray, route_lanes: np.ndarray) -> np.ndarray:
    """Return nearest route-centerline distance for every predicted timestep."""
    starts = route_lanes[:, :-1, :2].reshape(-1, 2)
    ends = route_lanes[:, 1:, :2].reshape(-1, 2)
    valid = np.linalg.norm(ends - starts, axis=1) > 1e-6
    starts, ends = starts[valid], ends[valid]
    points = prediction[0, :, :2]
    vectors = ends - starts
    ratios = ((points[:, None] - starts[None]) * vectors[None]).sum(axis=-1)
    ratios /= np.maximum((vectors * vectors).sum(axis=-1)[None], 1e-12)
    ratios = np.clip(ratios, 0.0, 1.0)
    projected = starts[None] + ratios[..., None] * vectors[None]
    return np.linalg.norm(points[:, None] - projected, axis=-1).min(axis=1)


def calculate_metrics(
    prediction: np.ndarray, frame: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Calculate the same aggregate/detail fields as scenario Open Loop."""
    errors = centerline_errors(prediction, frame["route_lanes"])
    centerline = {
        "average_lateral_error_m": float(errors[:80].mean()),
        "final_lateral_error_m": float(errors[min(79, len(errors) - 1)]),
    }
    initial = frame["ego_agent_past"][-1, :2]
    displacement = np.linalg.norm(prediction[0, :30, :2] - initial[None], axis=1)
    maximum = float(displacement.max())
    departure = {
        "failure_rate_percent": float(maximum < 2.0) * 100.0,
        "horizon_seconds": 3.0,
        "minimum_displacement_m": 2.0,
        "max_displacement_m": maximum,
        "departed": maximum >= 2.0,
    }
    return {"centerline": centerline, "departure": departure}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    samples = [
        (metric, Path(path)) for metric, paths in matrix.items() for path in paths
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    visualization_dir = args.output / "visualization"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(
        args.onnx.as_posix(), providers=["CPUExecutionProvider"]
    )
    dataset = PlannerDataset(args.parquet)
    predictions = []
    details = {"centerline": [], "departure": []}
    try:
        for index, (metric, npz_path) in enumerate(samples):
            frame = {key: value.numpy() for key, value in dataset[index].items()}
            prediction = run_prediction(session, frame, args.seed + index)
            metric_values = calculate_metrics(prediction, frame)
            inputs = legacy_inputs(npz_path)
            fig, ax = plt.subplots(figsize=(8, 8))
            visualize_inputs(
                inputs,
                ax=ax,
                view_ranges=[60],
                show_neighbors=True,
                show_ego_future=False,
                route_color="#00A6D6",
                route_label="Route centerline",
            )
            ax.plot(
                prediction[0, :, 0],
                prediction[0, :, 1],
                color="orange",
                linewidth=2,
                label="ONNX prediction",
            )
            ax.scatter(
                prediction[0, -1, 0],
                prediction[0, -1, 1],
                color="black",
                marker="x",
                label="final point",
            )
            ax.set_title(f"{npz_path.stem} · ONNX prediction")
            ax.legend(loc="best")
            fig.tight_layout()
            visualization_path = visualization_dir / metric / f"frame_{index:04d}.png"
            visualization_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(visualization_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            predictions.append(prediction)
            for metric_name, values in metric_values.items():
                if metric_name != metric:
                    continue
                details[metric_name].append(
                    {
                        "sample_index": index,
                        "source_npz": npz_path.as_posix(),
                        "metrics": values,
                        "visualization_png": visualization_path.as_posix(),
                        "matrix_group": metric,
                    }
                )
    finally:
        dataset.close()
    np.savez_compressed(
        args.output / "predictions.npz", trajectory=np.stack(predictions)
    )
    summary = {
        metric_name: {"sample_count": float(len(metric_details))}
        for metric_name, metric_details in details.items()
    }
    for metric_name, metric_details in details.items():
        keys = metric_details[0]["metrics"]
        summary[metric_name].update(
            {
                key: float(np.mean([item["metrics"][key] for item in metric_details]))
                for key in keys
                if key
                in (
                    "average_lateral_error_m",
                    "final_lateral_error_m",
                    "failure_rate_percent",
                )
            }
        )
        details_path = args.output / "details" / metric_name / "details.jsonl"
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False) + "\n" for item in metric_details
            ),
            encoding="utf-8",
        )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(predictions)} legacy Open Loop visualizations to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
