# Greenstop Converter Integration Design

Date: 2026-07-06
Branch: `feat/greenstop`

## Context

This branch was created from the red-stop work. The C++ data converter already
ports the red-light-run filter into the production skip path through
`frame_filters.hpp`, `FrameSkipInputs`, `FrameFilterParams`, and
`decide_frame_skip()`.

Sakayori's reference greenstop logic lives in
`reference/at-team-tools/sakayori/npz_cleansing/filters/greenstop.py`. It flags a
frame when the ego remains stopped at a heading-aligned green signal and there is
no object ahead that would justify stopping.

The reference detector checks both neighbor agents and `static_objects` as
possible lead blockers. The current C++ converter writes `static_objects` as
zero-filled placeholder data, so this integration will use neighbor-agent
blockers only. Static-object parity is out of scope for this branch.

## Goals

- Run greenstop by default during conversion, like the redrun integration.
- Mark detected frames as skipped with a new appended `GreenStop` skip label.
- Preserve default converter behavior: skipped frames do not write `.npz` unless
  `--write_skipped_npz=1` is set.
- Match Sakayori's reference thresholds for the first implementation.
- Keep the detector pure and unit-testable, following the existing redrun shape.
- Validate with a small, low-disk sampling workflow on sakurab rather than a
  broad full-dataset conversion.

## Non-Goals

- Do not add real static-object extraction in this branch.
- Do not tune thresholds beyond the reference defaults.
- Do not replace the existing Python `npz_cleansing` scan workflow.
- Do not run large dataset-wide conversion or video rendering jobs while the
  server is under training load.

## Architecture

Add a C++ greenstop detector beside `detect_red_light_run()` in
`cpp_tools/src/autoware_diffusion_planner_tools/include/processing/frame_filters.hpp`.

The detector will be wired into:

- `FrameFilterParams`, for threshold passing.
- `ConverterOptions`, for CLI defaults and validation.
- `decide_frame_skip()`, for production skip ordering.
- `SkippingLabel`, by appending `GreenStop` after existing labels to keep prior
  serialized integer values stable.

Skip priority will be:

1. stale data
2. invalid covariance
3. redrun related skips
4. greenstop
5. existing sustained no-future-progress
6. collision
7. off-lane
8. accepted

This order prevents generic no-future-progress from swallowing the more specific
greenstop label.

## Detection Logic

For each frame, greenstop uses data already built by the converter:

- `ego_current`: current heading and velocity.
- `ego_future`: ground-truth future trajectory.
- `route_lanes`: route lane geometry and traffic-light one-hot state.
- `neighbor_past`: neighbor state at the last past step.

A frame is `GreenStop` iff all checks pass:

1. The future is a real stationary window:
   - at most 5 exact `(0, 0)` future rows, to reject padded/truncated windows
   - max distance from `ego_future[0]` is less than `stay_radius`
2. Ego current speed is less than `speed_max`.
3. A green route lane is heading-aligned with the ego current heading and its
   entry point is ahead within `green_ahead`.
4. No active neighbor at the current past step lies in the forward corridor:
   `0.5 < forward < lead_fwd` and `abs(lateral) < lead_lat`.

If a neighbor is ahead, the frame is kept because stopping may be legitimate.
Static blockers are not checked because current converter `static_objects` are
placeholders.

## Defaults And CLI Options

Defaults match Sakayori's reference scan values:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--green_stop_heading_tol_deg` | `45.0` | Max heading difference for matching ego's green lane |
| `--green_stop_stay_radius_m` | `2.0` | Max future spatial extent for "stays put" |
| `--green_stop_speed_max_mps` | `1.0` | Max current speed for stationary ego |
| `--green_stop_ahead_m` | `40.0` | Max forward distance to the green lane entry |
| `--green_stop_lead_fwd_m` | `30.0` | Forward extent of lead-neighbor corridor |
| `--green_stop_lead_lat_m` | `2.0` | Half-width of lead-neighbor corridor |

Validation rejects negative threshold values.

## Output Behavior

Production conversion:

- greenstop detected: `is_skipped=true`, `skipping_info.label=GreenStop`, `.npz`
  omitted
- greenstop not detected: normal downstream skip checks continue

Inspection conversion:

- with `--write_skipped_npz=1`, detected greenstop frames still write `.npz`
  and sidecar JSON for review
- packed sequence behavior remains gap-free because `pack_sequence` already
  forces `write_skipped_npz`

## Tests

Unit tests for `detect_green_stop()`:

- stopped at green with no neighbor ahead returns true
- moving ego returns false
- red, yellow, white, or no-light route lane returns false
- green lane perpendicular to ego returns false
- neighbor ahead in corridor returns false
- neighbor outside corridor returns true
- padded or truncated future window returns false

Decision tests for `decide_frame_skip()`:

- greenstop returns `SkippingLabel::GreenStop`
- stale data and invalid covariance keep higher priority
- redrun keeps higher priority when both redrun and greenstop conditions are
  synthetically present
- greenstop is checked before `NoFutureProgress`

Option tests:

- defaults match the reference values
- negative greenstop thresholds fail validation

## Sampling And Visual Validation

Server validation must be low footprint because sakurab may be running large
training jobs and the available data pool is about 100K NPZ files.

Sampling will not scan or convert the whole server dataset. It will use existing
NPZ data on sakurab, with outputs under one small directory such as:

```text
/mnt/nvme/chenglin/greenstop_review_2026-07-06/
```

Preflight:

- run `df -h /mnt/nvme`
- confirm available space before rendering
- cap generated clips to roughly 40-80 total for the first pass

Use prior greenstop analysis to choose cohorts:

- `erga / hiratsuka`, because it had the largest absolute count
- `x2_dev / Fujiyoshida_diffusion_planner`, because prior review found real
  greenstop hits and a few false-negative-suspect cases

Build two small NPZ path lists:

- skipped candidates: greenstop-detected frames, sampled across known location
  clusters where possible
- kept hard negatives: stationary or green-adjacent frames kept because a
  neighbor is ahead, plus random kept frames from the same cohorts

Review workflow:

- render videos with `../clip-review-tool`
- save path lists, videos, render logs, and review JSONL under the review dir
- inspect that skipped samples are bad stopped-on-green clips
- inspect that kept samples have a blocker, are not actually greenstop, or move
  enough in the future to fall outside the filter

## Risks

- Neighbor-only blockers can over-filter rare cases where a static obstacle is
  present but not represented as a neighbor. This is accepted for this branch and
  documented as the main parity gap with the Python reference.
- The traffic-light one-hot is a t=0 snapshot, so the same caveat as redrun
  applies: later signal changes are not modeled by this detector.
- Some bus-stop-like greenstop data may be detected correctly but may require
  product judgment about whether it is useful training data. This design follows
  the existing cleansing goal and skips it by default.

## Approval

Approved approach: integrate greenstop into the existing C++ converter skip path,
run it by default like redrun, match reference thresholds, use neighbor blockers
only for this branch, and validate with small sakurab samples through
`../clip-review-tool`.
