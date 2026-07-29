# Scenario-based Open-loop Validation

Evaluate predictions on metric-specific NPZ samples during training or standalone validation.

## Input JSON

The JSON maps each metric name to a list of NPZ files:

```json
{
  "centerline": ["/path/to/centerline_scene.npz"],
  "departure": ["/path/to/departure_scene.npz"]
}
```

Supported metrics are `centerline` and `departure`. The NPZ files must use the standard planner input format; centerline evaluation requires `route_lanes` or `lanes`, and departure evaluation requires `ego_current_state`.

Pass the file with:

```bash
--scenario_based_open_loop_list /path/to/open_loop_matrix.json
```

## Adding a Metric

1. Implement the scorer in `planner_metrics/<metric_name>.py`.
2. Return `MetricEvaluation` from the open-loop scorer, including `scores` and optional `details`.
3. Register it in `scenario_based_open_loop/open_loop.py`.
4. Add configuration fields and tests as needed.
