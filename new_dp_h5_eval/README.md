# New DP native-H5 evaluation

Evaluate a new-DP sampler ONNX using the shared open-loop metrics and closed-loop
reproducer. The reader supports native H5 versions 4 and 5. Models receive native
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
`--no-visualization` to also save prediction images.

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
poses use `ego_x,ego_y,ego_yaw`; version 5 uses map-frame
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
