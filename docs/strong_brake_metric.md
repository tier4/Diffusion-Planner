# Strong braking from the active predicted trajectory

Closed-loop evaluation measures deceleration inside the active model plan. At each
simulation tick, let `p0`, `p1`, and `p2` be the first three XY points of the raw
prediction suffix selected for execution. `p0` is the predicted position for this
tick's outgoing step, not the current ego position. At `dt = 0.1 s`:

```text
v01 = norm(p1 - p0) / dt
v12 = norm(p2 - p1) / dt
brake_metric_accel_mps2 = (v12 - v01) / dt
```

The calculation uses the raw local prediction, avoiding world-coordinate rounding.
It excludes the ego-to-plan seam, actual executed speed, and predictions from other
replans. There is no EMA, moving average, virtual tracker, or jerk limit. On cached
ticks the suffix advances using the same point index as the existing rollout.
This is a plan-quality score, not a measurement of physical vehicle acceleration.
The rollout continues to record realized acceleration separately as `accel_mps2`.

## Event definition

The default threshold is `-2.5 m/s²`, with equality included:

1. Two consecutive simulation ticks at or below the threshold confirm an event.
   The onset is recorded at the second tick; a single isolated crossing cannot start one.
2. During an event, any raw score at or below the threshold resets its clear timer,
   even if that tick does not pass the two-frame confirmation mask.
3. Five consecutive raw scores above the threshold end the event (0.5 seconds).

Confirmation is across simulation ticks, including replans, not across two
accelerations inside one predicted trajectory. `steps` counts ticks passing the
two-frame mask; `count` uses the state machine above. Changing event boundaries
does not change `steps`. Frequent threshold crossings can form one long event,
so event count alone does not describe its duration or severity.

Warmup, snap/reset ticks, insufficient prediction suffixes (fewer than three points),
and non-finite points are unscored. Missing scores reset event confirmation and
release state; no interpolation or padding is applied.

## Saved results

The `strong_brake` block declares `filter_type=active_plan_three_point`,
`future_source=active_plan_raw_local_suffix`, `plan_points=3`,
`acceleration_window_s=0.1`,
`event_count_type=confirmed_start_raw_clear`, and `event_clear_frames=5`.
It retains `thresh_mps2`, `strongest_mps2`, `steps`, and `count`.
The trace saves `brake_metric_accel_mps2` and the metric definition, so trajectory
coloring uses the same scored values. Historical traces without the new definition
retain their original realized-acceleration interpretation. Aggregation rejects
mixed metric or event-counting definitions.

The tracker, model outputs, and MPC refresh schedule are unchanged. The colleague's
MPC path still runs only on replan ticks; cached ticks execute the existing raw
world-plan target.
