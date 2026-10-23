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

Supported metrics: `arrival`, `centerline`, `departure`, `traffic_light_go`, `pull_out`, `simple_turn`, `object_avoidance`, `pedestrian_yield`, `vehicle_yield`, `temporal_stop`, `obstacle_stop`, `traffic_light_stop`, `lane_change`. The NPZ files must use the standard planner input format.

- `arrival` requires `ego_agent_future`: a sample succeeds when the final displacement error against the GT endpoint is within `scenario_arrival_position_tolerance_m` (default 2.0 m) and the final heading error is within `scenario_arrival_heading_tolerance_deg` (default 10 deg); the raw errors are reported per sample in the details. GT may use either `[x, y, heading]` or `[x, y, cos(yaw), sin(yaw)]`.
- `centerline`, `departure`, `traffic_light_go`, `pull_out` require `route_lanes`/`lanes` or `ego_current_state` as documented in `planner_metrics/centerline.py` and `planner_metrics/departure.py`. `departure` requires the prediction to move at least 4 m within 5 s by default. `traffic_light_go` (released from a red light) and `pull_out` (pulling out of a bus stop) reuse the `departure` scorer under their own list keys/parameters.
- `simple_turn` requires `ego_agent_future`: same lateral/longitudinal error decomposition as `centerline`, but measured against the GT trajectory instead of the route-lane centerline (route-lane geometry is coarser than the recorded turn shape).
- `object_avoidance` requires `neighbor_agents_future`, `neighbor_agents_past`, and `ego_shape` in the OR-interval-start NPZ, with at least one valid neighbor of any type.
- `pedestrian_yield` and `vehicle_yield` require no extra keys beyond the predicted ego trajectory: the predicted ego must not advance more than a configured forward-progress tolerance (default 0.5 m) within a horizon (default 5 s). The scenes put a pedestrian/cyclist or vehicle in the ego's path while the ego is stopped.
- `temporal_stop` and `traffic_light_stop` require `route_lanes`/`lanes`, stop lines (`stop_lines`, or NPZ `line_strings` with the stop-line flag) and the ego front (`ego_base_link_to_front`, or `ego_shape`): within the horizon (default 8 s) its front must not cross the nearest stop line ahead on the route, and by then it must either come to a sustained stop or still be able to stop before the line from its speed at a comfortable deceleration (`scenario_<label>_stopping_deceleration_mps2`, default 1.0 m/s^2). The scenes are anchored around where the ego starts braking for the line. Scenes with no stop line on the route fail and record `stop_line_found = 0`; see `planner_metrics/stop_line.py`.
- `obstacle_stop` requires `route_lanes`/`lanes` and `ego_agent_future`: the predicted stop position along the route must not overshoot the GT stop position by more than a configured tolerance.
- `lane_change` requires `lanes` (or `route_lanes`) with the boundary-offset columns (`D>=8`) and `ego_agent_future`: within the horizon the prediction must cross the source lane's own boundary in the direction the GT went, and end within half a lane width of the GT's final lateral position. Lateral position is measured against the source lane — the starting lanelet extended through its endpoint-chained successors — so the score is independent of longitudinal error, and lane width comes from the map rather than a constant. Scenes whose GT never leaves its own lane carry no lane change and count as failures; the per-sample `gt_lane_change_detected` detail records which did carry one, which separates a planner failure from a mis-curated list. Completion ratio, final lateral offset error and lane-change time are per-sample details. The NPZ should be captured at or before the change starts — past the midpoint the nearest lane is the one being entered, and the scene reads as having no lane change.

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
