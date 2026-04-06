# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diffusion-Planner is a diffusion-based trajectory planner for autonomous driving (Tier4 fork). It uses an encoder-decoder architecture: observations are encoded via MLP-Mixer + fusion attention, then a diffusion transformer (DiT) decoder generates ego/neighbor trajectories through iterative denoising.

Training has four phases:
1. **SFT** (supervised fine-tuning) — base model training on driving data (`diffusion_planner/`)
2. **DPO** (direct preference optimization) — preference-based fine-tuning (`preference_optimization/`)
3. **GRPO** (group relative policy optimization) — RL with rule-based rewards (`rlvr/`)
4. **Ranked SFT** (GRPO-ranked self-distillation) — generate N trajectories, select best by reward, SG-filter, retrain with SFT loss (`rlvr/grpo_sft_trainer.py`). Inspired by self-distillation approaches (cf. [Zhang et al. 2025](https://arxiv.org/abs/2604.01193)).

An optional **exploration policy** (`exploration_policy/`) learns scene-conditional guidance parameters for GRPO sampling.

## Setup

```bash
cd diffusion_planner
python3 -m pip install pip==24.1
pip install -r requirements.txt
pip install -e .
```

## Common Commands

### SFT Training (8-GPU DDP)
```bash
cd diffusion_planner
python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
  --exp_name <name> --train_set_list <train.json> --valid_set_list <valid.json> \
  --diffusion_model_type "x_start" --save_dir <dir> --train_epochs 50
```

### DPO Training
```bash
python3 -m preference_optimization.train_dpo \
  --model_path <model.pth> --train_npz_list <train.json> --valid_npz_list <valid.json> \
  --preference_mode rule --use_lora --lora_rank 16 --lora_target first
```

### GRPO Training
```bash
python3 -m rlvr.train_grpo \
  --model_path <model.pth> --train_npz_list <train.json> --valid_npz_list <valid.json> \
  --config rlvr/configs/grpo_onpolicy.json --exp_name <name>
```

### Ranked SFT Training
```bash
# Set ranked_sft_mode in config JSON to "gt_neighbor" or "baseline_neighbor"
python3 -m rlvr.autoresearch.run_experiment \
  --config <config.json> --name <name> --model_path <model.pth> \
  --prob_scenes <scenes.json> --normal_scenes <scenes.json> --val_scenes <val.json> \
  --output_dir <dir> --skip_baseline
```

### Validation
```bash
python3 -m diffusion_planner.valid_predictor  # SFT validation
```

### Evaluation Scripts
```bash
python3 -m rlvr.autoresearch.eval_border_distance  # Border distance metrics
python3 -m rlvr.autoresearch.tools.eval_driving_metrics  # Full driving metrics
python3 -m rlvr.autoresearch.visualize_scenes  # Scene visualization
```

### Tests
```bash
python3 -m pytest exploration_policy/test_exploration_policy.py
python3 -m pytest exploration_policy/test_integration.py
python3 -m pytest rlvr/test_grpo_sampler.py
python3 -m pytest rlvr/test_reward.py
```

### Linting
```bash
ruff check .  # Import sorting only (configured in pyproject.toml)
```

## Architecture

### Data Format
- Training data: NPZ files containing ego/neighbor trajectories, lane geometry, route info, goal poses
- Dataset lists: JSON files mapping to NPZ paths, created via `diffusion_planner/util_scripts/create_train_set_path.py`
- Normalization: `normalization.json` computed via `diffusion_planner/util_scripts/normalize.py`

### Model Architecture (`diffusion_planner/model/`)
- **Encoder** (`module/encoder.py`): MLP-Mixer for agent mixing + fusion attention for contextual fusion
- **Decoder** (`module/decoder.py`): DiT-based diffusion decoder, outputs ego trajectory [B, 80, 4] (x, y, heading, velocity) + neighbor predictions
- **Guidance system** (`model/guidance/`): Pluggable guidance functions (road_border, collision, lane_keeping, centerline, route, speed, lateral, longitudinal) applied during diffusion sampling via `guidance_wrapper.py`

### GRPO Pipeline (`rlvr/`)
- **Config**: `GRPOConfig` dataclass in `grpo_config.py` — 100+ params, JSON-serializable. Preset configs in `rlvr/configs/`
- **Sampling**: `grpo_sampler.py` / `grpo_sampler_batched.py` generate N diverse trajectories per scene with varying noise scales and guidance
- **Reward**: `reward.py` scores trajectories on safety (collision, border, lane departure), progress, smoothness, feasibility, centerline following. Lane departure uses polygon containment + midpoint nudge to classify road edges (see algorithm overview in reward.py)
- **Loss**: `grpo_loss.py` computes advantage-weighted diffusion loss with K=8 (noise, timestep) averaging
- **Closed-loop**: `closed_loop/` contains rollout, state update, per-step reward, and GAE for multi-step evaluation
- **Ranked SFT**: `grpo_sft_trainer.py` generates N trajectories, selects best by reward, applies Savitzky-Golay filter, trains with standard SFT diffusion loss including neighbor outputs. Config: `ranked_sft_mode` = "gt_neighbor" or "baseline_neighbor". Supports **neighbor regularization** (`neighbor_reg_weight`): MSE(lora_neighbor, base_neighbor) prevents neighbor L2 drift through shared DiT weights. `neighbor_reg_only=true` drops the neighbor SFT loss for a cleaner signal. Two-stage training recommended: stage 1 on 50 problem scenes (lr=5e-4), stage 2 warm-start on 500 mixed scenes (lr=1e-5)
- **Trainers**: `grpo_trainer_batched.py` (standard), `closed_loop_trainer.py` (hybrid CL+OL), `grpo_exploration_trainer.py` (joint DiT + exploration policy), `grpo_sft_trainer.py` (ranked self-distillation)

### LoRA Fine-tuning
- Applied to DiT attention layers via `peft` library
- `lora_target` controls which blocks: "all", "first" (block 0), "last", "blocks01"
- Load with `load_lora_checkpoint()` — never use `PeftModel.from_pretrained` directly
- **Block ablation**: zeroing LoRA blocks post-training can improve L2. With neighbor reg, block 2 is critical for lane keeping (removing it → 22/50 LD). Block 0 and block 1 are safe to zero — **no_block1 gives best ego L2** while preserving both LD and neighbor L2

### Key Conventions
- Always use `python -m <module>` to run scripts, never `sys.path` hacks
- Experiment artifacts go to external SSD, not the repo
- Config reproducibility: every experiment saves its full config (args.json or grpo_config.json)
- Checkpoints saved as `epoch_NNN/best_model.pth` with accompanying `best_model_info.json`
