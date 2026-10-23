# New DP native-H5 evaluation

Evaluate a new-DP sampler ONNX using the shared open-loop metrics and closed-loop
reproducer. The reader supports native H5 versions 4, 5 and 6, including ML-Planner #88 packed shards (`frames_packed`
and `packed_frame_layout`) and older per-tensor shards. Validation dataset 1.0.0
uses the packed layout while keeping native H5 format version 5. Models receive native
H5 tensors; the evaluator adapts lane geometry and ground truth for the shared
scorers.

## Dataset and model

Use the H5 manifests published by
[validation-scenario-selector](https://github.com/tier4/validation-scenario-selector/tree/main/validation_dataset).
They reference generated data on your server. Run the commands below from this
repository root with a Python environment containing its evaluation dependencies.
For validation dataset v1.0.0 on Sakura:

```bash
DATASET_ROOT=/mnt/storage_rdma/diffusion_planner/dataset/validation_dataset/generation_runs/v1.0.0_20261019
PYTHON=.venv/bin/python
VEHICLE_PARAMS="$DATASET_ROOT/vehicle_parameters.json"
MODEL=/path/to/ml_planner.onnx
OUTPUT=/path/to/evaluation_results
```

The dataset supplies `vehicle_parameters.json`, including the Komatsu
`Mu1UupL2` project with the confirmed x2/J6 gen2 dimensions. Pass this file
explicitly when the local meta-repository parameter table predates that project.

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

## H5 format 6 and steering-input models

H5 format 6 inserts `steering_tire_angle` before `yaw_rate` in `ego_agent_past` /
`ego_agent_future` and appends `wheel_base` to `ego_shape`. The evaluator reads every frame
in the format-5 layout and keeps the two extra values aside (`new_dp_h5_eval/ego_layout.py`),
and the format-6 `wheel_base` is checked against `--vehicle_params` like the rest of
`ego_shape`.

The model's layout is read from its ONNX `ego_agent_past` / `ego_shape` widths. Models with
`(31, 6)` / `3` (every model trained after ML-Planner #98, which drops the two values before
the model) run on format 4, 5 and 6 data. Steering-input models with `(31, 7)` / `4` need the
recorded steering angle, so they run only on format 6 data; format 5 data is refused rather
than given an estimated angle.

In closed loop the steering history starts from the recording; each simulated step appends
the MPC tracker's steering, or, for a placed (perfect-tracking) ego, the kinematic-bicycle
angle `atan(wheel_base * yaw_rate / speed)`, held below 0.5 m/s. Format 6 closed loop runs on
the per-segment backend only: the batched backend's devkit simulator rebuilds the ego history
from the format-5 columns, so it refuses format 6 routes.

## Open-loop

The manifest maps metric names to `h5_path` / `frame_index` references. The Parquet
index supplies the corresponding frame timestamps. Relative H5 paths are resolved
from the manifest or index directory.

```bash
PYTHONPATH=.:diffusion_planner "$PYTHON" -m new_dp_h5_eval.open_loop \
  "$DATASET_ROOT/scenario_based_open_loop/h5/manifest.json" \
  "$DATASET_ROOT/scenario_based_open_loop/h5/index.parquet" \
  "$MODEL" "$OUTPUT/open_loop" \
  --provider CUDAExecutionProvider --vehicle_params "$VEHICLE_PARAMS" --no-visualization
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
  --vehicle_params "$VEHICLE_PARAMS" \
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
poses use `ego_x,ego_y,ego_yaw`; version 5 uses map-frame
`x,y,z,qx,qy,qz,qw`, from which the evaluator derives planar position and yaw, and version 6
keeps that pose metadata.

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
