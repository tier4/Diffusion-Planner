# GRPO Autoresearch Tools

Tools for running and evaluating GRPO reinforcement fine-tuning experiments
on the Diffusion Planner model. Designed for automated experiment loops
(autoresearch) where an AI agent or human iterates on hyperparameters.

## Quick Start

```bash
source .venv/bin/activate

# Run a GRPO experiment
python -m rlvr.autoresearch.run_experiment \
  --config rlvr/configs/grpo_onpolicy.json \
  --name my_experiment \
  --model_path /path/to/best_model.pth \
  --prob_scenes /path/to/prob_scenes.json \
  --normal_scenes /path/to/normal_scenes.json \
  --val_scenes /path/to/validation_scenes.json \
  --output_dir /path/to/output/

# Check if LoRA training is working
python -m rlvr.autoresearch.check_lora_training \
  /path/to/experiment_dir \
  --base-model /path/to/best_model.pth \
  --scene /path/to/test_scene.npz

# Visualize scenes (baseline only)
python -m rlvr.autoresearch.visualize_scenes \
  --model_path /path/to/best_model.pth \
  --scenes /path/to/scenes.json \
  --output_dir /path/to/output_images/ \
  --n_scenes 12

# Visualize comparison: baseline vs LoRA
python -m rlvr.autoresearch.visualize_scenes \
  --model_path /path/to/best_model.pth \
  --scenes /path/to/scenes.json \
  --lora_path /path/to/lora_epoch_004 \
  --output_dir /path/to/output_images/ \
  --indices 87 89 91 93 95
```

## Tools

### `run_experiment.py` — Single GRPO Experiment Runner

Runs one complete GRPO training experiment: loads model, generates trajectory
groups, trains with advantage-weighted loss, evaluates per epoch.

**Required args:**
- `--config`: JSON config file (see `rlvr/configs/grpo_onpolicy.json` for template)
- `--name`: Experiment name (used in output directory)
- `--model_path`: Path to base model `.pth` checkpoint
- `--prob_scenes`: JSON list of problem scene NPZ paths
- `--normal_scenes`: JSON list of normal/diverse scene NPZ paths
- `--val_scenes`: JSON list of validation scene NPZ paths
- `--output_dir`: Where to save checkpoints, logs, configs

**Output structure:**
```
output_dir/YYYYMMDD-HHMMSS_name/
├── grpo_config.json          # Config used
├── train_scenes.json         # Training scene list
├── grpo_train_log.tsv        # Per-epoch training metrics
├── grpo_eval_log.tsv         # Per-epoch eval metrics
├── lora_epoch_001/           # LoRA checkpoint per epoch
├── lora_best/                # Best checkpoint (by det reward)
└── run_summary.json          # Machine-readable summary
```

**Key eval metrics reported:**
- `rb_cross`: scenes where ego perimeter crosses road border (within 10cm)
- `rb_near`: fraction of timesteps with ego edge within 25cm of border
- `reward`: total reward (road border + progress + safety + feasibility)
- `collision`: collision rate
- `path`: mean trajectory path length

### `check_lora_training.py` — LoRA Verification

Checks that LoRA training actually changed the model's behavior by:
1. Inspecting LoRA weight norms (should be non-zero after training)
2. Comparing deterministic trajectories with and without LoRA

**Important:** Always uses `load_lora_checkpoint()` (not `PeftModel.from_pretrained()`
directly) because the v4 DiT uses `nn.MultiheadAttention` which must be replaced
with `UnfusedMHA` before loading LoRA weights.

### `visualize_scenes.py` — Scene Visualization with Road Borders

Generates publication-quality trajectory visualizations with:
- Lane boundaries (gray), road borders (red), GT trajectory (green)
- Ego vehicle rectangle at t=0 and footprints along the trajectory
- Road border metrics (rb_cross, rb_near) per scene
- Comparison mode: overlaps baseline (blue) and LoRA (orange) on same plot

**Modes:**
- **Single model**: `--model_path` only → grid of scenes with deterministic trajectory
- **Comparison**: add `--lora_path` → baseline vs LoRA overlapped on each scene

**Args:**
- `--model_path`: Base model `.pth` (required)
- `--scenes`: JSON list of NPZ paths (required)
- `--lora_path`: LoRA checkpoint dir for comparison (optional)
- `--output_dir`: Save images here (required)
- `--indices`: Specific scene indices to show (optional)
- `--n_scenes`: Number of evenly-spaced scenes if `--indices` not given (default: 12)
- `--cols`: Grid columns (default: 3)

### `eval_border_distance.py` — Border Distance Evaluation

Computes per-scene minimum distance from ego perimeter to road border. Reports
aggregate stats (min, mean, p5) and optionally visualizes worst scenes.

```bash
# Evaluate with merged model
python -m rlvr.autoresearch.eval_border_distance \
  --merged_model_path /path/to/merged.pth \
  --args_json /path/to/args.json \
  --scenes /path/to/miraikan_scenes.json \
  --tag model_name \
  --output_dir /path/to/output/

# With visualization of worst 10 scenes
python -m rlvr.autoresearch.eval_border_distance \
  --model_path /path/to/best_model.pth \
  --scenes /path/to/scenes.json \
  --tag baseline \
  --visualize --worst_n 10 \
  --output_dir /path/to/output/
```

### `compare_models.py` — Multi-Model Comparison

Overlays baseline + multiple trained models on the same scenes. Shows road borders,
lanes, ego footprints, and per-model border distance annotations.

```bash
python -m rlvr.autoresearch.compare_models \
  --base_model /path/to/best_model.pth \
  --models name1:/path/to/merged1.pth name2:/path/to/merged2.pth \
  --args_jsons /path/to/args1.json /path/to/args2.json \
  --scenes /path/to/scenes.json \
  --output_dir /path/to/output/ \
  --indices 48 47 49 1 3 --cols 2
```

## Scene Lists

Training requires three JSON files, each a list of NPZ paths:

- **Problem scenes**: Scenes where the baseline model has issues (e.g., road border
  crossings at miraikan exit). Should be < 30% of total training scenes.
- **Normal scenes**: Diverse driving scenes for general performance preservation.
- **Validation scenes**: Fixed set for per-epoch evaluation (not used in training).
  Should include a mix of problem + normal + anchor scenes.

**Important:** Remove "poison scenes" where the ego perimeter at t=0 already
touches a road border — these always get minimum reward regardless of model
output, adding noise to the training signal.

## Config Reference

See `rlvr/configs/grpo_onpolicy.json` for the recommended config. Key parameters:

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `inner_epochs` | 1 | On-policy is most stable for v4 |
| `learning_rate` | 5e-4 | Lower (2e-4) too slow, higher diverges |
| `kl_coef` | 0.2 | Higher pulls back to bad base model |
| `w_progress` | 7.0 | Prevents overly cautious short paths |
| `near_edge_scale` | 5.0 | Road border proximity penalty strength |
| `wide_edge_scale` | 0.5 | Wider proximity penalty |
| `train_epochs` | 3-4 | On-policy converges in 3-4 epochs |
| `rejection_keep` | 8 | Keep top 8 of 16 trajectories |
| `n_prob_scenes` | 0-50 | Problem scenes in training set. **Always set explicitly.** |
| `n_normal_scenes` | 250-300 | Normal scenes. **Always set explicitly.** |
| `lora_target` | `"first"` | Block 0 only. Other blocks degrade neighbor predictions. |
| `advantage_mode` | `"normalized"` | `"normalized"` or `"vd_grpo"` (see below) |
| `advantage_fixed_scale` | 10.0 | Denominator for VD-GRPO mode |
| `kl_schedule` | `"constant"` | `"constant"`, `"linear"`, `"cosine"`, or `"step"` (see below) |
| `kl_coef_final` | 0.05 | Target KL coef at end of training (for non-constant schedules) |
| `kl_warmup_fraction` | 0.5 | Fraction of epochs to hold initial `kl_coef` (for `"step"` schedule) |

### Available Config Templates

| Config | Description |
|--------|-------------|
| `grpo_onpolicy.json` | Standard on-policy (M=1), normalized advantages |
| `grpo_onpolicy_vdgrpo.json` | On-policy with VD-GRPO advantage computation |
| `grpo_onpolicy_kl_cosine.json` | On-policy with cosine KL decay (0.3 → 0.05) |
| `grpo_onpolicy_vdgrpo_kl_cosine.json` | On-policy + VD-GRPO + cosine KL decay |
| `grpo_onpolicy_hybrid.json` | All features: VD-GRPO + survival reward + cosine KL decay |
| `grpo_multi_epoch.json` | Multi-epoch (M=4), PPO-clipped IS, normalized advantages |
| `grpo_multi_epoch_vdgrpo.json` | Multi-epoch with VD-GRPO advantage computation |
| `grpo_zi_300sc.json` | **Best result**: zero-init exploration, 300sc, block 0 LoRA |

### Advantage Mode: VD-GRPO (Variance-Decoupled GRPO)

Inspired by [Plan-R1](https://arxiv.org/abs/2505.17659). Standard GRPO normalizes
advantages per group to zero mean and unit variance. This compresses the signal
from rare high-risk groups: if 30 of 32 trajectories are safe and 2 crash, the
crash advantage gets normalized to ~-1.5 regardless of the actual reward gap.

VD-GRPO replaces per-group std normalization with a fixed denominator
(`advantage_fixed_scale`). Advantages are still centered (subtract group mean)
but divided by the fixed scale instead of per-group std. This preserves the
absolute magnitude of negative rewards for crashes across groups.

**When to use VD-GRPO:**
- When training on safety-critical scenes where crash signals are being diluted
- When most trajectories in a group are similar quality (low variance) but a
  few are catastrophically bad
- Pair with `rejection_keep` to filter out the worst trajectories while still
  preserving the crash signal magnitude for the remaining ones

**`advantage_fixed_scale` tuning:** Controls advantage magnitude. Larger values
produce smaller advantages (more conservative updates). Start with 10.0. If
training is too aggressive, increase to 20.0. If too slow, decrease to 5.0.

### KL Schedule: Decaying KL Coefficient

Inspired by Plan-R1's multi-stage objective alignment. Early in training, a
high KL coefficient keeps the model close to the SFT base (preserving driving
realism). As training progresses, decaying KL gives the model more freedom to
deviate from the base to fix safety issues (e.g., road border avoidance).

**Schedules available:**
- `"constant"` (default): `kl_coef` stays fixed throughout training.
- `"linear"`: linearly interpolates from `kl_coef` to `kl_coef_final`.
- `"cosine"`: cosine annealing — slow decay early, faster in the middle, slow
  at the end. Good default for most runs.
- `"step"`: holds `kl_coef` for `kl_warmup_fraction` of training, then drops
  to `kl_coef_final`. Sharp transition, useful when you want a clear "alignment
  phase" followed by a "freedom phase".

**Example (cosine, 4 epochs, 0.3 → 0.05):**
```
epoch 1: kl=0.300  (conservative, stay close to base)
epoch 2: kl=0.238
epoch 3: kl=0.113
epoch 4: kl=0.050  (more freedom to deviate for safety)
```

**Tuning guidance:**
- `kl_coef` (start): higher = more conservative early training. 0.2-0.3 for
  short runs (3-4 epochs), 0.3-0.5 for longer runs (10+ epochs).
- `kl_coef_final` (end): how much freedom at the end. 0.05 is a good default.
  Set to 0.0 for maximum freedom (no KL penalty at all in final epochs).
- For short runs (3-4 epochs), cosine and linear behave similarly.
- For longer runs (10+ epochs), cosine decays slowly at first which is safer.
- The `"step"` schedule is useful when you want explicit "phases" (e.g., hold
  KL high for 50% of training, then release).

### Reward Mode: Survival Reward (PlannerRFT)

Inspired by [PlannerRFT](https://arxiv.org/abs/2601.12901). Standard gate mode
applies a flat -50 floor penalty when any terminal event occurs (collision, road
border crossing), regardless of *when* it happens. This kills the gradient signal
on hard scenes where all trajectories fail.

Survival reward gives proportional credit: a trajectory that crashes at t=60/80
gets 75% of the quality score, while one crashing at t=10/80 gets only 12.5%.
This preserves ranking among failed trajectories and provides gradient signal
even on the hardest scenes.

**Config:** `reward_mode: "survival"` (default: `"gate"`)

**When to use survival mode:**
- Training on safety-critical scenes where most/all trajectories have failures
- When gate mode produces flat rewards (all trajectories hit the -50 floor)
- When you want to teach the model to "delay" failures even if it can't fully
  avoid them yet

**The hybrid config** (`grpo_onpolicy_hybrid.json`) combines all three features:
VD-GRPO advantages + survival reward + cosine KL decay. This is the recommended
starting point for difficult training scenarios.

## V4-Specific Notes

- **Delay key**: The v4 decoder requires `inputs["delay"]`. This is automatically
  injected by `load_npz_data()` as `torch.zeros(1, dtype=torch.long)`.
- **4D timestep**: The v4 DiT expects `t` as `[B, P, T+1, 1]`, not `[B]`.
  The GRPO loss wrapper handles this expansion automatically.
- **Road border data**: V4 NPZ files have `line_strings` with shape `(60, 20, 4)`
  where channel 3 is the road border flag. Used by the reward and guidance.
- **LoRA loading**: Always use `load_lora_checkpoint()` from
  `preference_optimization/lora_utils.py`, never `PeftModel.from_pretrained()`.
- **LoRA merge for ONNX**: After `merge_lora_and_unload()`, use
  `fuse_unfused_mha_state_dict()` to convert q/k/v back to `in_proj_weight`.

## Reward Design

The reward uses road border perimeter-based detection only (lane polygon check disabled):

- **Hard gate**: ego perimeter (80 sample points) within 10cm of road border → reward floor (-50)
- **Soft penalties**: proximity at 25cm (`near_edge_scale`) and 40cm (`wide_edge_scale`)
- **Progress**: path length toward goal, weighted by `w_progress`
- **Safety**: proximity to neighbor vehicles
- **Centerline**: distance to nearest route lane centerline
- **Feasibility**: acceleration check only (lane protrusion/margin disabled)
