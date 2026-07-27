"""Departure-decision metric for Override Open-loop validation."""

import torch

_PREDICTION_TIMESTEP_SECONDS = 0.1


def _parameters(parameters: dict) -> tuple[float, float]:
    missing = {"horizon_seconds", "minimum_displacement_m"} - parameters.keys()
    if missing:
        raise ValueError(f"departure metric requires parameters: {', '.join(sorted(missing))}")

    horizon_seconds = float(parameters["horizon_seconds"])
    minimum_displacement_m = float(parameters["minimum_displacement_m"])
    if horizon_seconds <= 0:
        raise ValueError("departure horizon_seconds must be positive")
    if minimum_displacement_m < 0:
        raise ValueError("departure minimum_displacement_m must be non-negative")
    return horizon_seconds, minimum_displacement_m


def evaluate_departure(
    prediction: torch.Tensor, inputs: dict[str, torch.Tensor], parameters: dict
) -> dict[str, torch.Tensor]:
    """Return a per-sample departure-failure percentage (0 or 100).

    The ``departure`` NPZ list must contain only scenes where departure is
    expected. A sample succeeds when its largest displacement from the current
    ego position through ``horizon_seconds`` is at least
    ``minimum_displacement_m``.
    """
    if prediction.ndim != 3 or prediction.shape[-1] < 2:
        raise ValueError(f"prediction must have shape (B, T, D>=2), got {tuple(prediction.shape)}")
    if "ego_current_state" not in inputs:
        raise ValueError("departure metric requires ego_current_state in the NPZ")

    _, minimum_displacement_m = _parameters(parameters)
    max_displacement_m = departure_max_displacement(prediction, inputs, parameters)
    failure_rate_percent = (max_displacement_m < minimum_displacement_m).to(
        prediction.dtype
    ) * 100.0
    return {"failure_rate_percent": failure_rate_percent}


def departure_max_displacement(
    prediction: torch.Tensor, inputs: dict[str, torch.Tensor], parameters: dict
) -> torch.Tensor:
    """Return each sample's maximum displacement within the departure horizon."""
    horizon_seconds, _ = _parameters(parameters)
    _, available_steps, _ = prediction.shape
    horizon_steps = min(int(round(horizon_seconds / _PREDICTION_TIMESTEP_SECONDS)), available_steps)
    if horizon_steps < 1:
        raise ValueError("departure horizon selects zero prediction steps")

    initial_xy = inputs["ego_current_state"][:, :2].to(
        device=prediction.device, dtype=prediction.dtype
    )
    if initial_xy.shape[0] != prediction.shape[0]:
        raise ValueError(
            "ego_current_state batch does not match prediction: "
            f"current={tuple(initial_xy.shape)}, prediction={tuple(prediction.shape)}"
        )
    displacement_m = (prediction[:, :horizon_steps, :2] - initial_xy[:, None]).norm(dim=-1)
    return displacement_m.max(dim=1).values
