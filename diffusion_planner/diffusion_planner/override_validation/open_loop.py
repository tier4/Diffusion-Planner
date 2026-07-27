"""Open-loop Override Validation runner used from the training loop."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_planner.override_validation.metrics import METRICS


class _NpzPathDataset(Dataset):
    """Small dataset adapter for path lists embedded in an Override list JSON."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        with np.load(self.paths[index], allow_pickle=True) as loaded:
            data = dict(loaded)
        data.pop("version", None)
        return data


def _load_json_object(path: str, label: str) -> dict:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Override Open-loop {label} not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Override Open-loop {label} must contain a JSON object: {resolved}")
    return payload


def load_override_open_loop_settings(
    list_path: str, config_path: str
) -> tuple[dict[str, list[str]], dict]:
    """Load and validate the metric-to-NPZ list plus its shared configuration."""
    if bool(list_path) != bool(config_path):
        raise ValueError(
            "override_open_loop_list and override_open_loop_config must be supplied together"
        )
    if not list_path:
        return {}, {}

    raw_lists = _load_json_object(list_path, "list")
    raw_config = _load_json_object(config_path, "config")
    metric_parameters = raw_config.get("metrics", {})
    if not isinstance(metric_parameters, dict):
        raise ValueError("Override Open-loop config field 'metrics' must be an object")
    interval = raw_config.get("interval_epochs", 1)
    if not isinstance(interval, int) or interval < 1:
        raise ValueError(
            "Override Open-loop config field 'interval_epochs' must be an integer >= 1"
        )

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
        if not isinstance(metric_parameters.get(metric_name, {}), dict):
            raise ValueError(f"Override Open-loop parameters for {metric_name!r} must be an object")
        lists[metric_name] = paths

    return lists, raw_config


@torch.no_grad()
def run_override_open_loop_validation(model, args) -> dict[str, dict[str, float]]:
    """Run configured Override Open-loop metrics once and return scalar summaries.

    This intentionally runs on one process only.  Callers own rank selection and
    W&B logging; the scorer registry owns metric-specific computation.
    """
    metric_lists, config = load_override_open_loop_settings(
        args.override_open_loop_list, args.override_open_loop_config
    )
    if not metric_lists:
        return {}

    # Keep config parsing usable in lightweight environments that do not install
    # the full validation stack (including progress-bar dependencies).
    from diffusion_planner.validate_model import _prepare_validation_inputs

    was_training = model.training
    model.eval()
    try:
        summaries: dict[str, dict[str, float]] = {}
        metric_parameters = config.get("metrics", {})
        for metric_name, paths in metric_lists.items():
            loader = DataLoader(
                _NpzPathDataset(paths),
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
            for inputs in loader:
                prepared = _prepare_validation_inputs(inputs, args, args.device)
                _, outputs = model(prepared.inputs)
                # Match validate_model's metric convention: model predictions are
                # physical coordinates and metrics receive denormalized inputs.
                metric_inputs = args.observation_normalizer.inverse(prepared.inputs)
                batch_size = int(outputs["prediction"].shape[0])
                count += batch_size
                for key, value in scorer(
                    outputs["prediction"][:, 0], metric_inputs, parameters
                ).items():
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

            summaries[metric_name] = {"sample_count": float(count)}
            summaries[metric_name].update(
                {key: total / count for key, total in totals.items()} if count else {}
            )
    finally:
        model.train(was_training)
    return summaries
