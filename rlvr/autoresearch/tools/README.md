# Autoresearch Tools

Standalone evaluation, diagnostic, and data preparation tools for GRPO autoresearch.
All tools use `python -m` and require the project root in PYTHONPATH.

## Evaluation Tools

### eval_driving_metrics.py
Evaluates driving quality: speed, lateral acceleration (curvature-based), path length, stopped scenes.
```bash
python -m rlvr.autoresearch.tools.eval_driving_metrics \
  --model_path <base_model.pth> --lora_path <lora_dir> \
  --scenes <driving_scenes.json> --tag <name>
```

### eval_lane_border_distance.py
Combined lane departure + road border distance metrics. Reports lane_dep count, rb_crossings, min/avg border distance.
```bash
python -m rlvr.autoresearch.tools.eval_lane_border_distance \
  --model_path <model.pth> --scenes <scenes.json> --tag <name>
```

### eval_reward_vs_gt.py
Per-scene reward breakdown comparing model output vs ground truth. Useful for diagnosing which reward components are driving training.
```bash
python -m rlvr.autoresearch.tools.eval_reward_vs_gt \
  --model_path <model.pth> --scenes <scenes.json> --tag <name>
```

### grpo_viz.py
Visualizes all K GRPO trajectories per scene with reward breakdown. Each scene gets one figure:
left panel shows trajectories colored by rank (green=best, red=worst) with road borders and lane
boundaries; right panel shows per-trajectory reward table with progress, smoothness, lane status, path length.
```bash
python -m rlvr.autoresearch.tools.grpo_viz \
  --model_path <model.pth> --scenes <scenes.json> --output_dir <dir> \
  --indices 0 4 8 --K 16 --enable_lane --survival \
  --w_progress 1.0 --lane_near_scale 50.0
```

## Diagnostic Tools

### diagnose_grpo_signal.py
Diagnoses per-scene GRPO reward signal. Generates K trajectories per scene (batched), scores them, and reports advantage distribution. Useful for understanding why GRPO isn't learning on specific scenes.
```bash
python -m rlvr.autoresearch.tools.diagnose_grpo_signal \
  --model_path <model.pth> --scenes <scenes.json> --n_scenes 10
```

### viz_guidance_actual.py
Visualizes actual DiT inference with and without guidance. Shows the effect of different guidance configs on trajectory generation.
```bash
python -m rlvr.autoresearch.tools.viz_guidance_actual \
  --model_path <model.pth> --scenes <scenes.json> --output_dir <dir>
```

### viz_lane_departure.py
Visualizes lane departure and road border distance checks. Supports three modes:
- `--mode lane`: lane departure polygon containment + outer boundary distance
- `--mode rb`: road border point-to-segment distance (auto-picks closest-to-border timestep)
- `--mode both`: side-by-side comparison
```bash
python -m rlvr.autoresearch.tools.viz_lane_departure \
  --scenes <scenes.json> --indices 0 5 10 --output_dir <dir> \
  --mode rb --zoom 12
```

### viz_lane_gate_rank_flip.py
Visualizes the `_classify_outer_boundaries` classifier per lane-boundary segment:
outer (road-edge, black) vs shared-between-lanes (blue) vs junction-gap (orange dashed,
reclassified as shared to suppress false road-edge detections). Useful for debugging LD-gate
decisions on new scenes.
```bash
python -m rlvr.autoresearch.tools.viz_lane_gate_rank_flip \
  --scenes <scenes.json> --n_scenes 10 --output_dir <dir>
```

### viz_intersection_polygons.py
Overlays `POLYGON_TYPE_INTERSECTION_AREA` polygons from the NPZ `polygons` field on lane
segments. Use when tuning the intersection-polygon filter in the signed-distance lane gate.
```bash
python -m rlvr.autoresearch.tools.viz_intersection_polygons \
  --scenes <scenes.json> --output_dir <dir>
```

### viz_kinematic_floored.py
Visualizes trajectories flagged as kinematically infeasible by `compute_kinematic_gate`
(absolute yaw rate + bicycle-model curvature). Colors perimeter points by which gate fired
and at which timestep.
```bash
python -m rlvr.autoresearch.tools.viz_kinematic_floored \
  --model_path <model.pth> --scenes <scenes.json> --output_dir <dir>
```

### viz_t1_crossing_debug.py
Per-timestep signed-distance debug for a single scene: plots ego perimeter points colored by
signed distance to the nearest outer boundary. Useful when the gate flags a scene but the
trajectory looks fine visually (or vice versa).
```bash
python -m rlvr.autoresearch.tools.viz_t1_crossing_debug \
  --model_path <model.pth> --scenes <scenes.json> \
  --reference_config <config.json> \
  --scene_substr <scene_name_substring> --traj_idx <0..K-1> \
  --target_t <timestep> --lane_cross_thresh <threshold> \
  --output <fig.png>
```

### calibrate_rb_vs_lane.py
Measures lane vs RB reward-component contribution on a set of generated trajectories, and
sweeps RB scale vs top-1-agreement with the reference config. Use when picking
`rb_near_scale` / `rb_wide_scale` / `rb_cont_scale` for a new dataset.
```bash
python -m rlvr.autoresearch.tools.calibrate_rb_vs_lane \
  --model_path <model.pth> --scenes <scenes.json> \
  --reference_config <config.json> --output <calibration.json>
```

### det_path_drift.py
Tracks how the deterministic (no-noise) path length drifts across epochs vs a frozen
LoRA-less baseline. Useful for diagnosing progress collapse during ranked-SFT training.
```bash
python -m rlvr.autoresearch.tools.det_path_drift \
  --run_dir <experiment_run_dir> --model_path <base.pth> \
  --scenes <scenes.json> --epochs 1 3 5 7 9
```

### compute_baseline_cache.py
Precomputes baseline model and GT path lengths per scene. Used by `run_experiment.py --baseline_cache` to report progress ratios (model_path/gt_path and model_path/baseline_path) during training.
```bash
python -m rlvr.autoresearch.tools.compute_baseline_cache \
  --model_path <base_model.pth> --scenes <scenes.json> --output <cache.json>
```

### rb_campaign_launcher.py
Auto-queues experiment batches (max 2 in parallel). Requires environment variables:
```bash
export RB_CAMPAIGN_EXP_DIR=/path/to/experiments
export RB_CAMPAIGN_MODEL=/path/to/base_model.pth
python -m rlvr.autoresearch.tools.rb_campaign_launcher
```

## Data Preparation

### cleanse_lane_scenes.py
Filters scene lists by t=0 lane/border clearance. Removes scenes where ego starts out of lane or too close to road border. Used to clean training/eval scene lists.
```bash
python -m rlvr.autoresearch.tools.cleanse_lane_scenes \
  --scenes <input.json> --output <cleaned.json> --min_clearance 0.2
```

## Tests

Tests for closed-loop components are in `rlvr/autoresearch/tests/`:

```bash
# GAE computation
python -m pytest rlvr/autoresearch/tests/test_gae.py -x -q

# State update coordinate transforms
python -m pytest rlvr/autoresearch/tests/test_state_update.py -x -q

# Integration test with real scene data
python -m rlvr.autoresearch.tests.test_real_scene
```
