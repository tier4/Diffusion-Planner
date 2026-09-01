import torch

from planner_metrics.yield_progress import (
    compute_max_forward_progress_batch,
    evaluate_yield_progress_with_details,
)

_PARAMETERS = {"horizon_seconds": 3.0, "maximum_forward_progress_m": 0.5}


def _ego_trajs(max_x: float, steps: int = 40) -> torch.Tensor:
    ego = torch.zeros(1, steps, 4)
    ego[0, :, 0] = torch.linspace(0.0, max_x, steps)
    ego[0, :, 2] = 1.0
    return ego


def test_compute_max_forward_progress_respects_horizon():
    ego = _ego_trajs(max_x=10.0, steps=40)
    within_horizon = compute_max_forward_progress_batch(ego, horizon_steps=30)
    assert within_horizon.item() == ego[0, 29, 0].item()


def test_yield_progress_passes_when_ego_stays_within_tolerance():
    ego = _ego_trajs(max_x=0.4)
    result = evaluate_yield_progress_with_details(ego, {}, _PARAMETERS)

    assert result.scores["failure_rate_percent"].tolist() == [0.0]
    assert result.details["yield_progress"]["yielded"].tolist() == [True]


def test_yield_progress_fails_when_ego_advances_past_tolerance():
    ego = _ego_trajs(max_x=5.0)
    result = evaluate_yield_progress_with_details(ego, {}, _PARAMETERS)

    assert result.scores["failure_rate_percent"].tolist() == [100.0]
    assert result.details["yield_progress"]["yielded"].tolist() == [False]
