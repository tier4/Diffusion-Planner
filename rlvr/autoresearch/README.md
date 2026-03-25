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
| `n_prob_scenes` | 50 | Problem scenes in training set |
| `n_normal_scenes` | 150 | Normal scenes (total ~200) |

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
