"""Run configured scenario-specific open-loop metrics on NPZ samples."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_planner.utils.dataset import DiffusionPlannerData
from planner_metrics.centerline import evaluate_centerline_with_details
from planner_metrics.departure import evaluate_departure_with_details

METRICS = {
    "centerline": evaluate_centerline_with_details,
    "departure": evaluate_departure_with_details,
}


def _metric_parameters_from_args(args) -> dict[str, dict[str, object]]:
    """Build metric parameters from prefixed argument fields.

    For each registered metric, fields named
    ``override_<metric>_<parameter>`` are collected under the shorter
    ``<parameter>`` key. This keeps metric configuration in ``TrainConfig``
    without requiring a second metric-to-parameter mapping here.
    """
    values = vars(args)
    parameters: dict[str, dict[str, object]] = {}
    for metric_name in METRICS:
        prefix = f"override_{metric_name}_"
        parameters[metric_name] = {
            field_name[len(prefix) :]: value
            for field_name, value in values.items()
            if field_name.startswith(prefix)
        }
    return parameters


def _load_json_object(path: str, label: str) -> dict:
    """Read a required UTF-8 JSON object and report path-specific errors."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Override Open-loop {label} not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Override Open-loop {label} must contain a JSON object: {resolved}")
    return payload


def load_override_open_loop_settings(list_path: str) -> dict[str, list[str]]:
    """Load and validate the metric-to-NPZ mapping.

    The mapping must contain registered metric names whose values are lists of
    readable NPZ file paths. An empty path disables this validation set.
    """
    if not list_path:
        return {}

    raw_lists = _load_json_object(list_path, "list")

    lists: dict[str, list[str]] = {}
    for metric_name, paths in raw_lists.items():
        if metric_name not in METRICS:
            raise ValueError(
                f"Unsupported Override Open-loop metric {metric_name!r}; "
                f"supported: {sorted(METRICS)}"
            )
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ValueError(
                f"Override Open-loop list for {metric_name!r} must be a list of strings"
            )
        for path in paths:
            resolved = Path(path)
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"Override Open-loop NPZ for {metric_name!r} not found: {resolved}"
                )
            try:
                with np.load(resolved, allow_pickle=True):
                    pass
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Override Open-loop NPZ for {metric_name!r} cannot be read: {resolved}"
                ) from exc
        lists[metric_name] = paths

    return lists


@torch.no_grad()
def run_override_open_loop_validation(
    model,
    args,
    visualization_dir: str | Path | None = None,
    details_dir: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Run configured open-loop metrics once and return scalar summaries.

    This function intentionally runs on one process only. Callers own rank
    selection and W&B logging, while registered scorers own metric-specific
    score and detail computation. Optional per-sample JSONL details and scene
    visualizations are written under the supplied output directories.
    """
    metric_lists = load_override_open_loop_settings(args.override_open_loop_list)
    if not metric_lists:
        return {}

    # Keep config parsing usable in lightweight environments that do not install
    # the full validation stack (including progress-bar dependencies).
    from diffusion_planner.validate_model import _prepare_validation_inputs

    was_training = model.training
    model.eval()
    try:
        summaries: dict[str, dict[str, float]] = {}
        metric_parameters = _metric_parameters_from_args(args)
        visualization_root = Path(visualization_dir) if visualization_dir is not None else None
        details_root = Path(details_dir) if details_dir is not None else None
        if visualization_root is not None:
            visualization_root.mkdir(parents=True, exist_ok=True)
            from diffusion_planner.override_validation.visualize import (
                visualize_override_prediction,
            )
        for root in (details_root,):
            if root is not None:
                root.mkdir(parents=True, exist_ok=True)
        for metric_name, paths in metric_lists.items():
            loader = DataLoader(
                DiffusionPlannerData(paths),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
            totals: dict[str, float] = defaultdict(float)
            count = 0
            scorer = METRICS[metric_name]
            parameters = metric_parameters.get(metric_name, {})
            details: list[dict] = []
            for inputs in loader:
                raw_inputs = {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in inputs.items()
                }
                prepared = _prepare_validation_inputs(inputs, args, args.device)
                _, outputs = model(prepared.inputs)
                # Match validate_model's convention: predictions are physical
                # coordinates and metric inputs are denormalized.
                metric_inputs = args.observation_normalizer.inverse(prepared.inputs)
                batch_size = int(outputs["prediction"].shape[0])
                ego_prediction = outputs["prediction"][:, 0]
                batch_start = count
                count += batch_size
                evaluation = scorer(ego_prediction, metric_inputs, parameters)
                per_sample_scores = evaluation.scores
                for key, value in per_sample_scores.items():
                    if (
                        not torch.is_tensor(value)
                        or value.ndim != 1
                        or value.shape[0] != batch_size
                    ):
                        raise ValueError(
                            f"Override metric {metric_name!r} output {key!r} must be a "
                            f"per-sample 1-D tensor of length {batch_size}"
                        )
                    totals[key] += float(value.detach().float().sum().item())

                for batch_index in range(batch_size):
                    sample_index = batch_start + batch_index
                    detail = {
                        "sample_index": sample_index,
                        "source_npz": str(paths[sample_index]),
                        "metrics": {
                            key: float(value[batch_index].detach().float().item())
                            for key, value in per_sample_scores.items()
                        },
                    }
                    for section, fields in evaluation.details.items():
                        detail[section] = {
                            key: value[batch_index].item() for key, value in fields.items()
                        }
                    if visualization_root is not None:
                        sample_inputs = {
                            key: value[batch_index : batch_index + 1]
                            if torch.is_tensor(value)
                            else value
                            for key, value in raw_inputs.items()
                        }
                        visualize_override_prediction(
                            sample_inputs,
                            ego_prediction[batch_index],
                            visualization_root
                            / metric_name
                            / f"sample_{batch_start + batch_index:08d}.png",
                            f"{metric_name} sample {sample_index}",
                            show_neighbors=False,
                            view_range=60.0,
                        )
                        detail["visualization_png"] = str(
                            visualization_root / metric_name / f"sample_{sample_index:08d}.png"
                        )
                    details.append(detail)

            if details_root is not None:
                details_path = details_root / metric_name / "details.jsonl"
                details_path.parent.mkdir(parents=True, exist_ok=True)
                details_path.write_text(
                    "".join(json.dumps(detail, ensure_ascii=False) + "\n" for detail in details),
                    encoding="utf-8",
                )

            summaries[metric_name] = {"sample_count": float(count)}
            summaries[metric_name].update(
                {key: total / count for key, total in totals.items()} if count else {}
            )
    finally:
        model.train(was_training)
    return summaries
