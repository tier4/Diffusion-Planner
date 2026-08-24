"""Validated delay and vehicle-plant options for closed-loop rollouts.

All defaults are inert so importing this module or threading its options through an
existing rollout cannot change the baseline trajectory.  Non-zero delay behavior is
opt-in and invalid physical combinations fail at the public entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


WORLD_DELAY_MODES = ("none", "lag_extrapolated", "lag_raw")
PLANT_PARAMETER_SETS = ("official", "measured", "custom")
DELAY_TRAINING_MAX_STEPS = 5


@dataclass(frozen=True)
class PlantParameters:
    """Dead time and first-order time constant for steer and acceleration."""

    steer_dead_time_s: float
    steer_time_constant_s: float
    accel_dead_time_s: float
    accel_time_constant_s: float

    def validate(self) -> "PlantParameters":
        for name, value in self.__dict__.items():
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        return self


NAMED_PLANT_PARAMETERS = {
    "official": PlantParameters(
        steer_dead_time_s=0.17,
        steer_time_constant_s=0.15,
        accel_dead_time_s=0.10,
        accel_time_constant_s=0.10,
    ),
    "measured": PlantParameters(
        steer_dead_time_s=0.06,
        steer_time_constant_s=0.03,
        accel_dead_time_s=0.19,
        accel_time_constant_s=0.25,
    ),
}


def resolve_plant_parameters(
    parameter_set: str,
    *,
    steer_dead_time_s: float | None = None,
    steer_time_constant_s: float | None = None,
    accel_dead_time_s: float | None = None,
    accel_time_constant_s: float | None = None,
) -> PlantParameters:
    """Resolve a named set or require all four values for ``custom``."""

    overrides = {
        "steer_dead_time_s": steer_dead_time_s,
        "steer_time_constant_s": steer_time_constant_s,
        "accel_dead_time_s": accel_dead_time_s,
        "accel_time_constant_s": accel_time_constant_s,
    }
    if parameter_set not in PLANT_PARAMETER_SETS:
        raise ValueError(
            f"plant_parameter_set must be one of {PLANT_PARAMETER_SETS}, got {parameter_set!r}"
        )
    if parameter_set == "custom":
        missing = [name for name, value in overrides.items() if value is None]
        if missing:
            raise ValueError(
                "custom plant parameters require all four values; missing " + ", ".join(missing)
            )
        return PlantParameters(**overrides).validate()
    supplied = [name for name, value in overrides.items() if value is not None]
    if supplied:
        raise ValueError(
            f"explicit plant values require plant_parameter_set='custom'; got {', '.join(supplied)}"
        )
    return replace(NAMED_PLANT_PARAMETERS[parameter_set]).validate()


def validate_delay_options(
    *,
    timeline_progress_mode: str,
    world_delay_mode: str,
    k_lag: int,
    delay_step: int,
    tracker_mode: str,
    neighbor_history_mode: str,
    replan_interval: int,
    controller_compensation: bool,
    future_len: int = 80,
) -> None:
    """Fail loudly for combinations whose physical meaning is undefined."""

    if timeline_progress_mode not in ("pose", "clock"):
        raise ValueError(
            f"timeline_progress_mode must be 'pose' or 'clock', got {timeline_progress_mode!r}"
        )
    if world_delay_mode not in WORLD_DELAY_MODES:
        raise ValueError(
            f"world_delay_mode must be one of {WORLD_DELAY_MODES}, got {world_delay_mode!r}"
        )
    if k_lag < 0:
        raise ValueError(f"k_lag must be >= 0, got {k_lag}")
    if world_delay_mode == "none" and k_lag != 0:
        raise ValueError("k_lag must be 0 when world_delay_mode='none'")
    if world_delay_mode != "none":
        if k_lag == 0:
            raise ValueError("a lag world_delay_mode requires k_lag > 0")
        if timeline_progress_mode != "clock":
            raise ValueError("world delay is defined only for timeline_progress_mode='clock'")
        if neighbor_history_mode != "recorded":
            raise ValueError("world delay requires neighbor_history_mode='recorded'")
    if delay_step < 0:
        raise ValueError(f"delay_step must be >= 0, got {delay_step}")
    if delay_step > min(DELAY_TRAINING_MAX_STEPS, future_len):
        raise ValueError(
            f"delay_step must be <= {min(DELAY_TRAINING_MAX_STEPS, future_len)}, got {delay_step}"
        )
    if replan_interval < 1:
        raise ValueError(f"replan_interval must be >= 1, got {replan_interval}")
    if tracker_mode not in ("perfect", "mpc", "mpc_batched", "delayed"):
        raise ValueError(f"unknown tracker_mode={tracker_mode!r}")
    if controller_compensation and tracker_mode != "delayed":
        raise ValueError("controller_compensation requires tracker_mode='delayed'")
