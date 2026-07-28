import torch
from planner_metrics.departure import evaluate_departure


def test_departure_failure_rate_is_one_hundred_only_for_non_departing_predictions():
    prediction = torch.tensor(
        [
            [[0.2, 0.0], [0.5, 0.0], [0.8, 0.0], [3.0, 0.0]],
            [[0.2, 0.0], [0.6, 0.0], [1.0, 0.0], [1.2, 0.0]],
        ]
    )

    values = evaluate_departure(
        prediction,
        {"ego_current_state": torch.zeros((2, 4))},
        {"horizon_seconds": 0.3, "minimum_displacement_m": 1.0},
    )

    # The first sample only moves 0.8m in the first 0.3s; the second reaches 1.0m.
    torch.testing.assert_allclose(values["failure_rate_percent"], torch.tensor([100.0, 0.0]))
