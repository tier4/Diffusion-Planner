# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a fork of [ZhengYinan-AIR/Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner) — an autonomous driving motion planner using diffusion/flow-matching models. The repo is organized into several independent sub-packages.

## Environment

The active Python environment is `.venv` at the repo root. Always activate before running any Python:

```bash
source .venv/bin/activate
```

## Setup

```bash
python3 -m venv .venv
source ./.venv/bin/activate
cd diffusion_planner
pip install -r requirements_nuplan-devkit_fixed.txt
pip install -r requirements.txt
pip install -e .
```

## Key Commands

**Training (multi-GPU DDP):**
```bash
cd diffusion_planner
./train_run.sh <exp_name>     # pretraining → SFT → ONNX conversion
```

Or run a single stage manually:
```bash
python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
  --exp_name <name> --train_set_list <path.json> --valid_set_list <path.json> \
  --diffusion_model_type x_start --save_dir <dir> --train_epochs 100
```

**Convert trained model to ONNX:**
```bash
python3 ros_scripts/torch2onnx.py <save_dir>
```

**Dataset creation:**
```bash
# 1. Convert rosbags to .npz
python3 ros_scripts/parse_rosbag_for_directory.py <target_dir_list> --save_root <save_root>
# 2. Create path_list.json
python3 diffusion_planner/util_scripts/create_train_set_path.py <root_dir_list>
```

**DPO preference training:**
```bash
cd preference_optimization
python train_dpo.py --model_path <model.pth> --train_npz_list <train.json> \
  --valid_npz_list <valid.json> --preference_mode [rule|gui] --train_epochs 10
```

**Run tests:**
```bash
# DPO standalone FDE test (no model needed)
python3 preference_optimization/test_fde_standalone.py

# RLVR reward function tests (no model needed)
python3 rlvr/test_reward.py

# GRPO sampler tests (requires model + sample)
python3 rlvr/test_grpo_sampler.py --model_path <path.pth> --npz_path <path.npz>
```

**GRPO training (automatic, rule-based rewards):**
```bash
python rlvr/train_grpo.py \
  --model_path <model.pth> --train_npz_list <train.json> \
  --valid_npz_list <valid.json> --config rlvr/configs/grpo_onpolicy.json
```

**GRPO training (supervised GUI):**
```bash
python rlvr/trajectory_ranker_gui.py \
  --model_path <path.pth> --npz_list <npz.json> \
  --config rlvr/configs/grpo_onpolicy.json --use_lora
```

**GRPO trajectory visualization only (no training):**
```bash
python rlvr/trajectory_ranker_gui.py \
  --model_path <path.pth> --npz_list <npz.json> --no-training
```

**GRPO reward distribution analysis:**
```bash
python rlvr/analyze_rewards.py \
  --model_path <path.pth> --npz_list <npz.json> \
  [--lora_path <lora_dir>] --n_scenes 50
```

**Lint (import sort only via ruff):**
```bash
ruff check --select I --fix .
```

## Code Architecture

### Package Structure

| Path | Purpose |
|---|---|
| `diffusion_planner/` | Core PyTorch ML package; installed as editable package |
| `preference_optimization/` | DPO (Direct Preference Optimization) pipeline |
| `rlvr/` | RLVR integration with TeraSim simulator (Phase 1: ghost replay) |
| `guidance_gui/` | Gradio tool for interactive guidance exploration |
| `diffusion_planner_ros/` | ROS 2 Humble node for Autoware integration |
| `ros_scripts/` | Data pipeline utilities, rosbag parsing, ROS helpers |
| `cpp_tools/` | C++ tooling |
| `lichtblick_extensions/` | Lichtblick UI extensions |

### Core ML Model (`diffusion_planner/diffusion_planner/`)

**Entry point:** `model/diffusion_planner.py` — `Diffusion_Planner(config)` wraps `Encoder` + `Decoder`.

**Encoder** (`model/module/encoder.py`): Processes scene tokens using `MixerBlock` attention. Token types (defined by one-hot class encoding in `dimensions.py`):
- ego (CLASS_TYPE_EGO=0), neighbors (1), static objects (2), lanes (3), route (4), polygons (5), line strings (6), goal pose (7), ego shape (8), turn indicators (9)

**Decoder** (`model/module/decoder.py`): DiT-based (`model/module/dit.py`). Supports two diffusion paradigms, selected by `--diffusion_model_type`:
- `x_start`: VPSDE-linear SDE + DPM-Solver sampling (`model/diffusion_utils/`)
- flow matching: ODE solvers (Euler/Heun/RK4) in `model/flow_matching_utils/`

**Key dimensions** (`diffusion_planner/dimensions.py`):
- `INPUT_T = 30`, `OUTPUT_T = 80` — history and prediction horizons
- `POSE_DIM = 4` — `[x, y, cos(yaw), sin(yaw)]` (all pose data uses this format, NOT raw yaw angle)
- `MAX_NUM_NEIGHBORS = 32`, `NUM_SEGMENTS_IN_LANE = 140`, `NUM_SEGMENTS_IN_ROUTE = 25`

**Normalizers** (`utils/normalizer.py`):
- `StateNormalizer`: normalizes ego/neighbor future trajectories for training loss
- `ObservationNormalizer`: normalizes observation inputs (lanes, route, etc.)
- Both loaded from `normalization.json`; stored in `Config`/`args` as `state_normalizer` and `observation_normalizer`

**Config** (`utils/config.py`): Loads `args.json` from a checkpoint directory into a `Config` object. This is the canonical way to load model args at inference; `guidance_scale` defaults to `0.5`.

### Data Format

Every dataset sample is a `.npz` file (optionally paired with a `.json` sidecar for world-frame pose). All data inside `.npz` is in the **ego base_link frame at t=0**.

Key `.npz` keys:
| Key | Shape | Notes |
|---|---|---|
| `ego_agent_past` | `(21, 4)` | History t=-2s..t=0. Format: `[x, y, cos, sin]`. Last row is `[0,0,1,0]` |
| `ego_agent_future` | `(80, 3)` | GT future `[x, y, yaw_rad]` |
| `ego_current_state` | `(10,)` | `[x, y, cos, sin, vx, vy, ax, ay, steer, yaw_rate]` |
| `neighbor_agents_past` | `(32, 21, 11)` | NPC history: `[x, y, cos, sin, vx, vy, w, l, is_veh, is_ped, is_bike]` |
| `lanes` / `route_lanes` | `(140/25, 20, 33)` | Lane segments; dim 33 = see `SEGMENT_POINT_DIM` in `dimensions.py` |
| `goal_pose` | `(4,)` | `[x, y, cos, sin]` in ego frame |

The `.json` sidecar (same stem as `.npz`) stores the world-frame ego pose at t=0 in **MGRS local Cartesian coordinates** (`x`, `y`, `z`, `qx`, `qy`, `qz`, `qw`).

**Important:** `ego_agent_future` uses raw yaw radians `[x, y, yaw_rad]`, but `train_epoch.py`'s `heading_to_cos_sin()` converts it to `[x, y, cos, sin]` before passing to the model. Be consistent about which format is expected.

The dataset is a JSON list of `.npz` paths, loaded by `DiffusionPlannerData` (`utils/dataset.py`) and `openjson` from `utils/train_utils.py`.

### Guidance System (`model/guidance/`)

- `registry.py` — `@register` decorator, `build()`, `list_available()` for guidance function discovery
- `base.py` — `BaseGuidance` abstract class; `_compute()` returns raw energy, `energy()` applies `_energy_scale * config.scale * raw`
- `config.py` — `GuidanceConfig` (per-function), `GuidanceSetConfig` (global_scale + list of function configs)
- `composer.py` — `GuidanceComposer`: drop-in replacement for legacy `GuidanceWrapper`; uses registry
- `collision.py`, `route_following.py`, `lane_keeping.py`, `centerline_following.py` — individual guidance energies (all return `[B]` energy)
- `anchor_following.py` — MTR-style prototype-guided trajectory anchoring
- Guidance functions receive `x: [B, P, T+1, 4]` in **physical ego-centric meters** (normalizer.inverse already applied by the wrapper)
- Active only during DPM-Solver sampling for `t ∈ (0.005, 0.1)`
- The global `guidance_scale` in Config multiplies the summed gradient correction

### Inference Pattern

To generate a trajectory from a loaded `.npz`, the key call pattern from `preference_optimization/utils.py`:

```python
P = 1 + model_args.predicted_neighbor_num
ego_current = data["ego_current_state"][:, :4]                        # [B, 4]
neighbors_current = data["neighbor_agents_past"][:, :P-1, -1, :4]    # [B, P-1, 4]
current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

xT = current_states[:, :, None, :].expand(-1, -1, OUTPUT_T + 1, -1).clone()
xT[:, :, 1:, :] = noise_scale * torch.randn(B, P, OUTPUT_T, 4, device=device)
data["sampled_trajectories"] = xT.reshape(B, P, -1)

_, decoder_output = model(data)
ego_trajectory = decoder_output["prediction"][:, 0]  # [B, OUTPUT_T, 4]
```

### DPO Pipeline (`preference_optimization/`)

| Module | Role |
|---|---|
| `train_dpo.py` | Entry point; CLI args |
| `trainer.py` | `DPOTrainer` class |
| `dpo_loss.py` | DPO loss computation |
| `model_utils.py` | `load_model(path, device)` → `(model, model_args)` |
| `utils.py` | `generate_trajectory_pair_with_retry()` — FDE-gated pair generation |
| `annotation_gui.py` | Gradio web UI for human annotation |
| `preference_collection.py` | Rule-based preference generation |

Model loading for DPO/inference always uses `model_utils.load_model` which returns `(model, Config)`.

### RLVR / GRPO Integration (`rlvr/`)

GRPO (Group Relative Policy Optimization) reinforcement fine-tuning pipeline. Supports two modes via JSON config:
- **On-policy** (`grpo_onpolicy.json`): M=1, single gradient step per rollout. Simplest, most stable.
- **Multi-epoch** (`grpo_multi_epoch.json`): M>1, reuses rollouts with clipped importance sampling.

Key components:

| Module | Role |
|---|---|
| `grpo_config.py` | `GRPOConfig` dataclass, JSON load/save |
| `grpo_loss.py` | Advantage-weighted diffusion loss + KL regularization |
| `grpo_trainer.py` | `GRPOTrainer`: generation, scoring, training loop, eval, best-model tracking |
| `grpo_sampler.py` | `generate_diverse_group()`: N trajectories with randomized noise/guidance |
| `reward.py` | Rule-based reward: `R = w_safety*S + w_progress*P + w_smooth*M + w_feasibility*F + w_centerline*C` |
| `train_grpo.py` | CLI entry point (`--mode rule` or `--mode gui`) |
| `trajectory_ranker_gui.py` | Gradio GUI: visualization + optional supervised GRPO training |
| `analyze_rewards.py` | Standalone reward distribution diagnostic tool |
| `configs/grpo_onpolicy.json` | On-policy config template (M=1) |
| `configs/grpo_multi_epoch.json` | Multi-epoch config template (M=4) |

**Training outputs** (in experiment directory):
- `grpo_train_log.tsv` — per-epoch training metrics
- `grpo_eval_log.tsv` — per-epoch evaluation (deterministic + stochastic, on fixed validation scenes)
- `run_summary.json` — machine-readable run state for agentic auto-research
- `lora_best/` — best checkpoint (by deterministic reward on validation)
- `lora_epoch_NNN/` — per-epoch checkpoints
- `eval_scenes.json` — fixed validation scene subset (seeded, reproducible across runs)

**GRPO loss**: `L = (1/G) * sum(A_i * loss_i) + kl_coef * KL(policy || ref)` where `loss_i` is the diffusion denoising MSE (approximates `-log pi`), `A_i` is the group-relative advantage, and `ref` is the SFT base model (via LoRA `disable_adapter()`).

**Reward function**: progress is scaled by `(1 - off_road_fraction)` to prevent off-road shortcuts from gaming the progress component.

**Evaluation**: each epoch evaluates on 50 fixed validation scenes (seeded for cross-run comparability). Reports both deterministic trajectory metrics (the deployment output) and stochastic group metrics.

Additional components (TeraSim ghost replay):
- `rlvr/npz_utils.py` — coordinate transforms (ego-centric ↔ world MGRS frame)
- `rlvr/terasim_bridge.py` — HTTP client to TeraSim REST API (Docker container)
- `rlvr/scripts/` — map conversion and ghost replay validator

TeraSim runs in Docker; the `.venv` only uses the `traci` Python client (TCP). **SUMO angle convention**: degrees, clockwise from north. **ROS/Autoware convention**: radians, counterclockwise from +X.
