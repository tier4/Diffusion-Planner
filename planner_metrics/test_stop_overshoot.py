import torch

from planner_metrics.stop_overshoot import evaluate_stop_overshoot_with_details

_PARAMETERS = {"tolerance_m": 0.5}


def _straight_route_lanes() -> torch.Tensor:
    xs = torch.arange(-5.0, 25.0, 1.0)
    lane = torch.zeros(len(xs), 4)
    lane[:, 0] = xs
    lane[:, 2] = 1.0
    return lane.unsqueeze(0)  # (S=1, P, D)


def _ramp_then_hold(stop_x: float, ramp_steps: int = 30, hold_steps: int = 30) -> torch.Tensor:
    ramp = torch.linspace(0.0, stop_x, ramp_steps)
    hold = torch.full((hold_steps,), stop_x)
    xy = torch.zeros(ramp_steps + hold_steps, 4)
    xy[:, 0] = torch.cat([ramp, hold])
    xy[:, 2] = 1.0
    return xy.unsqueeze(0)  # (N=1, T, D)


def _keeps_moving(final_x: float, steps: int = 60) -> torch.Tensor:
    xy = torch.zeros(steps, 4)
    xy[:, 0] = torch.linspace(0.0, final_x, steps)
    xy[:, 2] = 1.0
    return xy.unsqueeze(0)


def test_stop_overshoot_passes_when_prediction_stops_at_gt_position():
    lanes = _straight_route_lanes()
    gt = _ramp_then_hold(stop_x=5.0)
    pred = _ramp_then_hold(stop_x=5.0)
    result = evaluate_stop_overshoot_with_details(
        pred, {"route_lanes": lanes, "ego_agent_future": gt}, _PARAMETERS
    )

    details = result.details["stop_overshoot"]
    assert result.scores["failure_rate_percent"].tolist() == [0.0]
    assert details["overshoot_m"].item() < 1e-3
    assert details["gt_sustained_stop"].tolist() == [1.0]
    assert details["predicted_sustained_stop"].tolist() == [1.0]


def test_stop_overshoot_fails_when_prediction_overshoots_gt_position():
    lanes = _straight_route_lanes()
    gt = _ramp_then_hold(stop_x=5.0)
    pred = _ramp_then_hold(stop_x=7.0)
    result = evaluate_stop_overshoot_with_details(
        pred, {"route_lanes": lanes, "ego_agent_future": gt}, _PARAMETERS
    )

    details = result.details["stop_overshoot"]
    assert result.scores["failure_rate_percent"].tolist() == [100.0]
    assert details["overshoot_m"].item() > 1.5


def test_stop_overshoot_falls_back_to_terminal_position_when_gt_never_stops():
    lanes = _straight_route_lanes()
    gt = _keeps_moving(final_x=10.0)
    pred = _keeps_moving(final_x=10.0)
    result = evaluate_stop_overshoot_with_details(
        pred, {"route_lanes": lanes, "ego_agent_future": gt}, _PARAMETERS
    )

    details = result.details["stop_overshoot"]
    assert details["gt_sustained_stop"].tolist() == [0.0]
    assert details["gt_stop_position_s_m"].item() > 9.0
