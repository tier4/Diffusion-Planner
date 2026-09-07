"""Shared horizon-resolution helper for scenario-based open-loop metrics.

Every metric that takes a configurable ``horizon_seconds`` parameter
(``centerline``, ``departure``/``traffic_light_go``, ``pedestrian_yield``/
``vehicle_yield``/``temporal_stop``, ``simple_turn``) clamps it to a valid
prediction-step count the same way; this module holds that one shared
implementation.
"""

from __future__ import annotations


def resolve_horizon_steps(
    horizon_seconds: float,
    total_steps: int,
    *,
    label: str,
    timestep_seconds: float = 0.1,
) -> int:
    """Clamp a configured horizon (seconds) to a valid prediction-step count."""
    if horizon_seconds <= 0:
        raise ValueError(f"{label} horizon_seconds must be positive")
    steps = min(int(round(horizon_seconds / timestep_seconds)), total_steps)
    if steps < 1:
        raise ValueError(f"{label} horizon selects zero prediction steps")
    return steps


__all__ = ["resolve_horizon_steps"]
