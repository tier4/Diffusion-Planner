from collections.abc import Iterable
from dataclasses import dataclass

from .config_cli import cli


@dataclass
class ScenarioOpenLoopConfig:
    """Scenario-based open-loop validation configuration."""

    # ---------------------------------------------------------
    # Scenario-based open-loop validation
    # ---------------------------------------------------------
    scenario_based_open_loop_list: str = cli(
        "JSON matrix of scenario-based open-loop settings. Empty = disabled.",
        default="",
        path=True,
    )
    scenario_based_open_loop_only: bool = cli(
        "run the scenario-based open-loop validation and nothing else",
        default=False,
    )

    # ---------------------------------------------------------
    # Scenario-based Open-loop
    # ---------------------------------------------------------
    scenario_arrival_position_tolerance_m: float = 2.0
    scenario_arrival_heading_tolerance_deg: float = 10.0
    scenario_centerline_horizon_seconds: float = 8.0
    scenario_simple_turn_horizon_seconds: float = 8.0
    scenario_departure_horizon_seconds: float = 3.0
    scenario_departure_minimum_displacement_m: float = 2.0
    scenario_traffic_light_go_horizon_seconds: float = 3.0
    scenario_traffic_light_go_minimum_displacement_m: float = 2.0
    scenario_pedestrian_yield_horizon_seconds: float = 3.0
    scenario_pedestrian_yield_maximum_forward_progress_m: float = 0.5
    scenario_vehicle_yield_horizon_seconds: float = 3.0
    scenario_vehicle_yield_maximum_forward_progress_m: float = 0.5
    scenario_temporal_stop_horizon_seconds: float = 3.0
    scenario_temporal_stop_maximum_forward_progress_m: float = 0.5
    scenario_obstacle_stop_tolerance_m: float = 0.5
    scenario_traffic_light_stop_tolerance_m: float = 0.5
    scenario_lane_change_horizon_seconds: float = 8.0
    scenario_lane_change_minimum_lateral_shift_m: float = 1.0
    scenario_lane_change_chain_tolerance_m: float = 1.0


def scenario_metric_parameters(args, labels: Iterable[str]) -> dict[str, dict[str, object]]:
    """Group ``scenario_<label>_<parameter>`` fields of ``args`` by label.

    For each label, fields named ``scenario_<label>_<parameter>`` are collected under
    the shorter ``<parameter>`` key. Shared by the open-loop runner and the post-hoc
    closed-loop scenario metrics so both read the same thresholds.
    """
    values = vars(args)
    parameters: dict[str, dict[str, object]] = {}
    for label in labels:
        prefix = f"scenario_{label}_"
        parameters[label] = {
            field_name[len(prefix) :]: value
            for field_name, value in values.items()
            if field_name.startswith(prefix)
        }
    return parameters
