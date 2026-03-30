# GRPO Autoresearch

Run autonomous GRPO training experiments to optimize the Diffusion Planner model.

## Current Best Weights (zero-init baseline, to be beaten)

Zero-init exploration policy + Block 0 LoRA + 300 curated Odaiba scenes.

| Epoch | val_reward | val_rb_cross | val_collision | prob_rb_cross | miraikan_rb | border_t20 |
|-------|-----------|-------------|---------------|--------------|-------------|------------|
| 2 | +64.77 | 1/100 | 2.0% | 0/50 | 0/141 | 0.33m |
| 4 | +65.18 | 1/100 | 2.0% | 1/50 | 0/141 | 0.37m |
| **7** | **+66.00** | 2/100 | 2.0% | 0/50 | 0/141 | **0.53m** |

These weights use the exploration trainer framework (lateral/longitudinal guidance via Beta distribution) but the policy doesn't actually learn — it stays at zero-init (α≈β≈1.693, η_std≈0.48). The structured diversity from the Beta distribution provides ~5 points over standard GRPO.

**Goal: make the exploration policy LEARN per-scene guidance that beats zero-init.**

## What Works and What Doesn't (as of March 29-30, 2026)

### Works:
- **REINFORCE (inner_epochs=1)** — only update rule that keeps the policy alive
- **max_conc=10 clamp** on Beta alpha/beta — prevents distribution collapse
- **fc2 bias=False** — forces scene-dependent output (no global bias shortcut)
- **exploration_step_per_group=True** — ESSENTIAL. Step optimizer after each scene, not once per epoch. Otherwise the policy learns a global average instead of per-scene guidance.
- **raw_scale=10** — amplifies gradient through softplus compression

### Doesn't work:
- **PPO (inner_epochs>1)** — always kills the policy, even with clamp. Tested 4 times.
- **KL penalty** — KL is not the cause of collapse. KL=0 and KL=0.01 give identical results.
- **Accumulated stepping** (step once per epoch) — produces global bias, not per-scene guidance. Confirmed: trained policy had η_lat std=0.013 across 10 scenes (essentially uniform).
- **High LR with accumulated stepping** (lr=5e-3) — learns faster but concentrates too fast, same problem.
- **Curvature-based lat accel reward** — neutral. Identical results to old reward. Kept for correctness.

### Being tested (March 30):
- **Per-group stepping + no-bias + higher LR** — p6o (lr=5e-4) running. Previous test (p6m, lr=5e-5) showed scene_var but tiny η (2cm offset). Need higher LR to produce meaningful 20-30cm offsets.

## Launch Command

```bash
EXP_DIR=/media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207/auto_research/odaiba_grpo_experiments

source .venv/bin/activate && python -m rlvr.autoresearch.run_experiment \
  --config $EXP_DIR/configs/<config>.json \
  --name <experiment_name> \
  --model_path /media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207/v4.0/best_model.pth \
  --prob_scenes $EXP_DIR/miraikan_left_turn_train_53.json \
  --normal_scenes /media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207/auto_research/odaiba_medium_scale_rl/grpo_scenes_300.json \
  --val_scenes $EXP_DIR/val_v4_100.json \
  --output_dir $EXP_DIR \
  --skip_baseline \
  2>&1 | tee $EXP_DIR/<experiment_name>.log
```

## GPU & Parallelism

- 1x NVIDIA RTX PRO 6000 (98 GB). Each experiment uses ~3-5 GB.
- **Max 2 experiments in parallel.** Always check `nvidia-smi` first.
- When running 2, each runs ~30-50% slower.

## Critical Rules

1. **`lora_target: "first"`** (block 0 only). Other blocks damage neighbor predictions.
2. **Rejection sampling is critical** (`rejection_keep: 8`). Without it, model explodes in 2 epochs.
3. **REINFORCE only** — never use inner_epochs>1 (PPO), it always collapses.
4. **Per-group stepping** (`exploration_step_per_group: true`) — essential for per-scene learning.
5. **No bias in GuidanceHead fc2** — forces scene-dependent output.
6. **Always report full metrics** for every epoch (see checklist below).
7. **Kill experiments with no future** — don't waste GPU time on dead experiments.

## How to Check Experiment Results

Eval metrics are in the **.log file** (not the TSV). TSV only has training metrics.

```bash
# Eval results:
grep 'Eval \[' $EXP_DIR/<experiment>.log

# Training metrics (eta, entropy, scene variance):
grep 'Epoch [0-9].*scene_var' $EXP_DIR/<experiment>.log
```

**ALWAYS report the FULL picture. Never just eta or just val_reward.**

## Evaluation Checklist

For every epoch, report ALL of these:
- val_reward, val_rb_cross, val_collision, val_path, val_stopped
- prob_reward, prob_rb_cross, prob_stopped
- η_lat, η_lon, η_std, entropy, scene_var_lat, scene_var_lon
- For promising results: `python rlvr/eval_teleport_metrics.py` (accurate speed/lat_accel)
- For promising results: `python -m rlvr.autoresearch.eval_border_distance` on miraikan prob scenes (min/avg/t20 distance)
- For promising results: `python -m rlvr.autoresearch.visualize_scenes` on miraikan + teleport

## How to Verify Per-Scene Guidance Works

```python
# Load trained policy and run 10 random scenes — η should VARY across scenes
python3 << 'EOF'
# See the inline script used in March 30 session
# Load model + policy checkpoint, run 10 scenes, print η per scene
# scene_var_lat should be >> 0.013, η range should span positive AND negative
EOF
```

## Visualization

Use the proper visualize_scenes script for trajectory comparison with road borders + ego footprints:
```bash
# Compare baseline vs LoRA on miraikan problem scenes
python -m rlvr.autoresearch.visualize_scenes \
  --model_path <base_model.pth> \
  --scenes <miraikan_scenes.json> \
  --lora_path <lora_epoch_NNN> \
  --output_dir ~/Pictures/<experiment>_viz \
  --n_scenes 12

# For guidance direction analysis (with policy eta overlay)
python rlvr/viz_policy_guidance.py \
  --model_path <base_model.pth> \
  --policy_path <lora_epoch_NNN/exploration_policy.pth> \
  --scenes <scenes.json> \
  --output <output.png>
```
Shows: blue=baseline, orange=LoRA, green=GT, red=road borders, ego footprints, rb_crossing flags.

## Border Distance Metric

Computes min/avg/t20 distance from ego perimeter to road border on miraikan problem scenes:
```bash
python -m rlvr.autoresearch.eval_border_distance \
  --model_path <base_model.pth> --scenes <miraikan_prob.json> --tag <name> \
  --output_dir <ssd_path>/border_distance

# With visualization of worst N scenes:
python -m rlvr.autoresearch.eval_border_distance \
  --merged_model_path <merged.pth> --args_json <args.json> \
  --scenes <miraikan_prob.json> --tag <name> \
  --visualize --worst_n 10 --output_dir <ssd_path>/border_distance
```
Reports: rb_crossings, min_dist (mean/min/p5), avg_dist (mean/min), border_t20 (mean/min/p5).
Visualizations show trajectory colored by distance + time-series plot of border distance.

## Lat Accel Metric

reward.py uses curvature-based lat accel (accurate). For standalone reporting:
```bash
python rlvr/eval_teleport_metrics.py --model_path <path> --scenes <teleport.json> [--lora_path <dir>] --tag <name>
```
Corrected: baseline max=2.96 m/s² (not 8.58), ep7 max=1.91 m/s² (training improves smoothness).

## Merge + Deploy Pipeline

```bash
python3 -m preference_optimization.merge_lora --model_path <exp>/latest.pth --lora_dir <exp>/lora_epoch_NNN --output <exp>/merged.pth
python3 ros_scripts/torch2onnx.py <dir_containing_merged.pth>
```

## Files & Locations

| What | Where |
|------|-------|
| Experiment configs | `.../odaiba_grpo_experiments/configs/` |
| Best zero-init checkpoints | `.../20260328-150832_p4e_explore_300_block0_8ep/` |
| Base model | `/media/.../v4.0/best_model.pth` |
| Training scenes (300) | `.../odaiba_medium_scale_rl/grpo_scenes_300.json` |
| Validation 100 | `.../odaiba_grpo_experiments/val_v4_100.json` |
| Prob scenes (50, mixed locations) | `.../odaiba_grpo_experiments/prob_from_train_50.json` |
| Miraikan left-turn train (53) | `.../odaiba_grpo_experiments/miraikan_left_turn_train_53.json` |
| Miraikan left-turn all (111) | `.../odaiba_grpo_experiments/miraikan_left_turn_111.json` |
| Miraikan all (141) | `.../odaiba_grpo_experiments/miraikan_train_70.json` + `miraikan_val_71.json` |
| Teleport scenes (51) | `.../auto_research/v4_anchor_teleport_51.json` |

All paths on SSD: `/media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207/auto_research/`

## Miraikan Scene Selection

The 141 miraikan scenes come from 2 bags:
- **98d3d86e** (111 scenes): the critical **left turn** at miraikan exit (+54 deg yaw change). This is where baseline crosses road borders.
- **c62ee2a9** (30 scenes): straight section near border (+11 deg yaw change). Not the critical area.

**For training prob scenes, use `miraikan_left_turn_train_53.json`** (left-turn only, train split).
**For evaluation, use `miraikan_left_turn_111.json`** (all left-turn scenes) or the full 141.

Do NOT use `prob_from_train_50.json` for miraikan-specific training — it's a mix of locations (only 6 of 50 are miraikan).
All scenes verified: ego starts on-road with >= 30cm margin at t=0.
