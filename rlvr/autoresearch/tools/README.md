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
