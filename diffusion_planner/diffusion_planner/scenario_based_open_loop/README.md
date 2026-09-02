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

Supported metrics: `centerline`, `departure`, `traffic_light_go`, `simple_turn`, `object_avoidance`, `pedestrian_yield`, `vehicle_yield`, `temporal_stop`, `obstacle_stop`, `traffic_light_stop`. The NPZ files must use the standard planner input format.

- `centerline`, `departure`, `traffic_light_go` require `route_lanes`/`lanes` or `ego_current_state` as documented in `planner_metrics/centerline.py` and `planner_metrics/departure.py`. `traffic_light_go` reuses the `departure` scorer (predicted forward progress must clear a minimum within a horizon) under its own list key/parameters, for scenes where the ego is released from a red light rather than departing from a stop.
- `simple_turn` requires `ego_agent_future`: same lateral/longitudinal error decomposition as `centerline`, but measured against the GT trajectory instead of the route-lane centerline (route-lane geometry is coarser than the recorded turn shape).
- `object_avoidance` requires `neighbor_agents_future`, `neighbor_agents_past`, and `ego_shape` in the OR-interval-start NPZ, with at least one valid neighbor of any type.
- `pedestrian_yield`, `vehicle_yield`, and `temporal_stop` require no extra keys beyond the predicted ego trajectory: the predicted ego must not advance more than a configured forward-progress tolerance within a horizon. `pedestrian_yield`/`vehicle_yield` scenes put a pedestrian/cyclist or vehicle in the ego's path; `temporal_stop` reuses the same scorer under its own list key/parameters for scenes defined by a time-based (rather than actor-based) stop requirement.
- `obstacle_stop` and `traffic_light_stop` require `route_lanes`/`lanes` and `ego_agent_future`: the predicted stop position along the route must not overshoot the GT stop position by more than a configured tolerance.

Scorers never see the NPZ files or the model's prepared/normalized batch directly.
`open_loop.py` extracts exactly the fields each scorer needs via
`planner_metrics.scene_data.extract_metric_scene_data`, so the field-name
contract listed above is the whole interface — any data source (not just an
NPZ loader) that supplies a mapping with these keys can be scored the same
way.

Pass the file with:

```bash
--scenario_based_open_loop_list /path/to/open_loop_matrix.json
```

## Adding a Metric

1. Implement the scorer in `planner_metrics/<metric_name>.py`.
2. Return `MetricEvaluation` from the open-loop scorer, including `scores` and optional `details`.
3. Register it in `scenario_based_open_loop/open_loop.py`.
4. Add configuration fields and tests as needed.
