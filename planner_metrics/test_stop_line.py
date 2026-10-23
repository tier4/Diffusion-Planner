import math

import torch

from planner_metrics.stop_line import evaluate_stop_line_with_details

_PARAMETERS = {"horizon_seconds": 8.0, "stopping_deceleration_mps2": 1.0}
# wheel_base, length, width: the ego front is 0.5 * (2.0 + 4.0) = 3.0 m ahead of base_link.
_EGO_SHAPE = torch.tensor([2.0, 4.0, 1.8])
_STOP_LINE_X = 10.0


def _straight_route_lanes() -> torch.Tensor:
    xs = torch.arange(-5.0, 80.0, 1.0)
    lane = torch.zeros(len(xs), 4)
    lane[:, 0] = xs
    return lane.unsqueeze(0)  # (S=1, P, D)


def _stop_lines(x: float = _STOP_LINE_X) -> torch.Tensor:
    lines = torch.zeros(3, 2, 2)  # unused slots stay zero
    lines[0] = torch.tensor([[x, -2.0], [x, 2.0]])
    return lines


def _brake_to(stop_x: float, brake_steps: int = 40, steps: int = 80) -> torch.Tensor:
    """Constant deceleration from the origin to ``stop_x``, then hold."""
    phase = torch.linspace(0.0, 1.0, brake_steps)
    x = torch.cat([stop_x * (1 - (1 - phase) ** 2), torch.full((steps - brake_steps,), stop_x)])
    ego = torch.zeros(1, steps, 4)
    ego[0, :, 0] = x
    ego[0, :, 2] = 1.0
    return ego


def _evaluate(ego: torch.Tensor, **data) -> dict:
    data = {"route_lanes": _straight_route_lanes(), "ego_shape": _EGO_SHAPE, **data}
    data.setdefault("stop_lines", _stop_lines())
    return evaluate_stop_line_with_details(ego, data, _PARAMETERS)


def test_stop_line_passes_when_front_stops_before_the_line():
    result = _evaluate(_brake_to(stop_x=6.0))

    details = result.details["stop_line"]
    assert result.scores["success_rate_percent"].tolist() == [100.0]
    assert details["predicted_sustained_stop"].tolist() == [1.0]
    assert details["crossed_stop_line"].tolist() == [0.0]
    assert math.isclose(details["min_front_to_stop_line_m"].item(), 1.0, abs_tol=1e-3)


def test_stop_line_fails_when_front_crosses_the_line_even_if_base_link_does_not():
    result = _evaluate(_brake_to(stop_x=8.0))

    assert result.scores["success_rate_percent"].tolist() == [0.0]
    assert result.details["stop_line"]["crossed_stop_line"].tolist() == [1.0]


def _constant_speed(speed: float, steps: int = 80) -> torch.Tensor:
    ego = torch.zeros(1, steps, 4)
    ego[0, :, 0] = speed * 0.1 * torch.arange(steps)
    ego[0, :, 2] = 1.0
    return ego


def test_stop_line_passes_a_prediction_still_creeping_that_can_stop_before_the_line():
    # 0.6 m/s at 8 s with the front 2.26 m short of the line: 0.18 m to stop at 1 m/s^2.
    result = _evaluate(_constant_speed(0.6))

    details = result.details["stop_line"]
    assert result.scores["success_rate_percent"].tolist() == [100.0]
    assert details["predicted_sustained_stop"].tolist() == [0.0]
    assert details["can_stop_before_line"].tolist() == [1.0]
    assert math.isclose(details["speed_at_horizon_mps"].item(), 0.6, abs_tol=1e-4)
    assert math.isclose(details["front_to_stop_line_at_horizon_m"].item(), 2.26, abs_tol=1e-3)


def test_stop_line_fails_a_prediction_going_on_unbraked_towards_a_far_line():
    # 6 m/s at 8 s with the front 8.6 m short of a line 58 m ahead: 18 m to stop at 1 m/s^2.
    result = _evaluate(_constant_speed(6.0), stop_lines=_stop_lines(58.0))

    details = result.details["stop_line"]
    assert result.scores["success_rate_percent"].tolist() == [0.0]
    assert details["crossed_stop_line"].tolist() == [0.0]
    assert details["can_stop_before_line"].tolist() == [0.0]
    assert math.isclose(details["stopping_distance_m"].item(), 18.0, abs_tol=1e-3)


def test_stop_line_fails_a_prediction_that_could_stop_but_already_crossed():
    result = _evaluate(_constant_speed(1.0))  # front reaches 10.9 m: past the line at 10 m

    details = result.details["stop_line"]
    assert result.scores["success_rate_percent"].tolist() == [0.0]
    assert details["crossed_stop_line"].tolist() == [1.0]


def test_stop_line_fails_when_no_stop_line_crosses_the_route():
    lines = torch.zeros(1, 2, 2)
    lines[0] = torch.tensor([[10.0, 5.0], [10.0, 8.0]])  # beside the route
    result = _evaluate(_brake_to(stop_x=6.0), stop_lines=lines)

    assert result.scores["success_rate_percent"].tolist() == [0.0]
    assert result.details["stop_line"]["stop_line_found"].tolist() == [0.0]


def test_stop_line_ignores_lines_behind_and_uses_the_nearest_one_ahead():
    lines = torch.zeros(3, 2, 2)
    lines[0] = torch.tensor([[-3.0, -2.0], [-3.0, 2.0]])
    lines[1] = torch.tensor([[25.0, -2.0], [25.0, 2.0]])
    lines[2] = torch.tensor([[_STOP_LINE_X, -2.0], [_STOP_LINE_X, 2.0]])
    result = _evaluate(_brake_to(stop_x=6.0), stop_lines=lines)

    assert math.isclose(result.details["stop_line"]["stop_line_distance_m"].item(), _STOP_LINE_X)


def test_stop_line_reads_flagged_npz_line_strings():
    line_strings = torch.zeros(2, 2, 4)
    line_strings[0, :, :2] = torch.tensor([[5.0, -2.0], [5.0, 2.0]])
    line_strings[0, :, 3] = 1.0  # road border at 5 m: not a stop line
    line_strings[1, :, :2] = torch.tensor([[_STOP_LINE_X, -2.0], [_STOP_LINE_X, 2.0]])
    line_strings[1, :, 2] = 1.0
    data = {
        "route_lanes": _straight_route_lanes(),
        "ego_shape": _EGO_SHAPE,
        "line_strings": line_strings,
    }
    result = evaluate_stop_line_with_details(_brake_to(stop_x=6.0), data, _PARAMETERS)

    assert result.scores["success_rate_percent"].tolist() == [100.0]
    assert math.isclose(result.details["stop_line"]["stop_line_distance_m"].item(), _STOP_LINE_X)


def test_stop_line_prefers_the_native_base_link_to_front():
    # Box front is 3.0 m ahead; the real front 2.0 m ahead leaves the 8.5 m stop 0.5 m short.
    ego = _brake_to(stop_x=7.5)
    assert _evaluate(ego).scores["success_rate_percent"].tolist() == [0.0]
    result = _evaluate(ego, ego_base_link_to_front=torch.tensor([2.0]))

    assert result.scores["success_rate_percent"].tolist() == [100.0]
    assert math.isclose(
        result.details["stop_line"]["min_front_to_stop_line_m"].item(), 0.5, abs_tol=1e-3
    )
