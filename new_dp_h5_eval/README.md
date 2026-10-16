# New DP native-H5 evaluation

Evaluate a new-DP sampler ONNX using the shared open-loop metrics and closed-loop
reproducer. The reader supports native H5 version 4 and later. Models receive native
H5 tensors; the evaluator adapts lane geometry and ground truth for the shared
scorers.

## Dataset and model

Use the H5 manifests published by
[validation-scenario-selector](https://github.com/tier4/validation-scenario-selector/tree/main/validation_dataset).
They reference generated data on your server. Run the commands below from this
repository root with a Python environment containing its evaluation dependencies.
For the v0.0.2 release on Sakura:

```bash
DATASET_ROOT=/mnt/storage_rdma/diffusion_planner/dataset/validation_dataset/generation_runs/v0.0.2_20261008
PYTHON=.venv/bin/python
MODEL=/path/to/ml_planner.onnx
OUTPUT=/path/to/evaluation_results
```

## Ego dimensions

Native H5 stores `ego_shape` as `[base_link_to_front, length, width]`, while the scorers
shared with the NPZ evaluation need `[wheelbase, length, width]`: the closed-loop MPC
tracker, road-border and object boxes and renderer, and the open-loop object-avoidance box.
Both runners therefore read each shard's `project_id` and take the wheelbase from
`--vehicle_params`, a per-project json with `ego_wheel_base`, `ego_front_overhang`,
`ego_length` and `ego_width`. It defaults to the meta repository's
`dataset/data_converter_param.json`, the table both dataset converters read; pass it
explicitly when this repository is checked out on its own. A shard whose `ego_shape` does not
match its project's entry is rejected. The model input keeps the native `ego_shape`.

## Steering and wheel base

Data for models that read steering inserts `steering_tire_angle` as column 5 of
`ego_agent_past` (`yaw_rate` moves to column 6) and appends `wheel_base` to `ego_shape`. A shard
may also store each frame as one row of `frames_packed`, laid out by the `packed_frame_layout`
attribute, instead of a `frames/` group; both layouts are read. A model's `ego_agent_past` /
`ego_shape` widths (7/4 or 6/3) must match the data's.

In closed loop the live ego history carries the steering the model reads: the recorded value
while warming up, the MPC tracker's front-wheel angle, and for a placed ego (perfect tracker,
cached plans), which has no steering state, the kinematic angle `atan(wheelbase * yaw_rate /
speed)`, held below 1 m/s.

## Open-loop

The manifest maps metric names to `h5_path` / `frame_index` references. The Parquet
index supplies the corresponding frame timestamps. Relative H5 paths are resolved
from the manifest or index directory.

```bash
PYTHONPATH=.:diffusion_planner "$PYTHON" -m new_dp_h5_eval.open_loop \
  "$DATASET_ROOT/scenario_based_open_loop/h5/manifest.json" \
  "$DATASET_ROOT/scenario_based_open_loop/h5/index.parquet" \
  "$MODEL" "$OUTPUT/open_loop" \
  --provider CUDAExecutionProvider --no-visualization
```

Results include `summary.json` and per-sample scores under `details/`. Omit
`--no-visualization` to also save prediction images; `--draw-workers` sets how many
processes draw them (default 8).

## Closed-loop

The manifest maps group names to H5 paths or objects with `h5_path` and optional
`frame_start` / `frame_stop` (exclusive). Different windows of one H5 file are
separate routes. Optional `anchors`, `segment_start_ns`, and `segment_end_ns` are
retained in segment results.

```bash
PYTHONPATH=.:diffusion_planner "$PYTHON" -m new_dp_h5_eval.run_all_groups_closed_loop \
  --closed_loop_h5_root \
    "$DATASET_ROOT/scenario_based_closed_loop/h5/manifest.json" \
    "$DATASET_ROOT/closed_loop_lap/h5/manifest.json" \
  --closed_loop_labels scenario_based_closed_loop closed_loop_lap \
  --closed_loop_object_modes objects objects \
  --model_path "$MODEL" \
  --provider CUDAExecutionProvider \
  --out_root "$OUTPUT/closed_loop"
```

Explicit labels distinguish manifests that share the name `manifest.json`.
Results are saved under a timestamped directory with per-route `segments.jsonl`,
group summaries, and an aggregate `groups.json`. The runner supports the same
rendering, distributed execution, pass conditions, and plan optimizer options as
the shared closed-loop evaluator; see `--help` for the available arguments.

Closed-loop data must declare a 0.1 s frame interval and have strictly increasing
timestamps. The reader retains timestamp jitter and gaps without inserting
frames; dataset generation determines which recordings are usable. Version 4
poses use `ego_x,ego_y,ego_yaw`; later versions use map-frame
`x,y,z,qx,qy,qz,qw`, from which the evaluator derives planar position and yaw.

## Recompute pass fields

When only `closed_loop_pass_conditions.yaml` changes, recompute pass fields from
saved results without rerunning inference:

```bash
PYTHONPATH=.:diffusion_planner "$PYTHON" new_dp_h5_eval/recompute_pass_conditions.py \
  "$OUTPUT/closed_loop/YYYYMMDD_HHMM" \
  --pass-conditions diffusion_planner/diffusion_planner/config/closed_loop_pass_conditions.yaml
```

This updates `segments.jsonl`, group `summary.json` files, and `groups.json`.
Use `--dry-run` to print pass counts without changing files.
