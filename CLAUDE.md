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

**Run tests** (all tests are standalone scripts — no pytest/tox):
```bash
# DPO standalone FDE test (no model needed)
python3 preference_optimization/test_fde_standalone.py

# RLVR reward function tests (no model needed)
python3 rlvr/test_reward.py

# Exploration policy unit tests (no model needed, 12 tests)
python3 exploration_policy/test_exploration_policy.py

# GRPO sampler tests (requires model + sample)
python3 rlvr/test_grpo_sampler.py --model_path <path.pth> --npz_path <path.npz>

# Lateral/longitudinal guidance tests (standalone + model visualization)
python3 rlvr/test_lateral_longitudinal_guidance.py
python3 rlvr/test_lateral_longitudinal_guidance.py --model_path <path.pth> --npz_path <path.npz>
```

**GRPO autoresearch experiment:**
```bash
python -m rlvr.autoresearch.run_experiment \
  --config rlvr/configs/grpo_onpolicy.json \
  --name my_experiment \
  --model_path <model.pth> \
  --prob_scenes <prob.json> --normal_scenes <normal.json> \
  --val_scenes <val.json> --output_dir <output/>
```

**GRPO trajectory visualization (supervised GUI):**
```bash
python3 -m rlvr.trajectory_ranker_gui \
  --model_path <path.pth> --npz_list <npz.json> --use_lora
```

**GRPO trajectory visualization only (no training):**
```bash
python3 -m rlvr.trajectory_ranker_gui \
  --model_path <path.pth> --npz_list <npz.json> --no-training
```

**Check LoRA training effectiveness:**
```bash
python -m rlvr.autoresearch.check_lora_training \
  <experiment_dir> --base-model <model.pth> --scene <test.npz>
```

**Visualize scenes (baseline vs LoRA comparison):**
```bash
python -m rlvr.autoresearch.visualize_scenes \
  --model_path <model.pth> --scenes <scenes.json> \
  --lora_path <lora_dir> --output_dir <images/>
```

**GRPO reward distribution analysis:**
```bash
python rlvr/analyze_rewards.py \
  --model_path <path.pth> --npz_list <npz.json> \
  [--lora_path <lora_dir>] --n_scenes 50
```

**Guidance GUI (interactive guidance exploration):**
```bash
# Generate prototypes first (once per dataset):
python guidance_gui/scripts/generate_prototypes.py \
  --npz_list <train.json> --k 16 --output guidance_gui/prototypes_k16.npy
# Launch:
python guidance_gui/app.py \
  --model_path <model.pth> --npz_list <train.json> \
  --prototypes guidance_gui/prototypes_k16.npy
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
| `exploration_policy/` | Learned exploration policy for adaptive guidance (PlannerRFT-inspired) |
| `guidance_gui/` | Gradio tool for interactive guidance exploration |
| `scene_search/` | Gradio GUI for searching/curating NPZ scenes on a lanelet2 map |
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
- `lateral_guidance.py` — PlannerRFT Eq. 2: `Ψ_lat = (1/T) Σ (n⊥·(x-x_ref) - λ_lat·η_lat)²`. Params: `lambda_lat` (max offset metres), `eta_lat` (scale ∈ [-1,1], later learned by PPO)
- `longitudinal_guidance.py` — PlannerRFT Eq. 3: `Ψ_lon = (1/T) Σ (n∥·(v - λ_lon·η_lon·v_ref))²`. Velocity-based speed scaling. Params: `lambda_lon` (max speed deviation fraction), `eta_lon` (scale ∈ [-1,1])
- Both lateral/longitudinal require `inputs["reference_trajectory"]` `[B, T, 4]` (deterministic model output). The GRPO sampler injects this automatically when `enable_lateral`/`enable_longitudinal` are set. η values are currently sampled uniformly from [-1,1]; a future PPO exploration policy will output Beta-distributed η per scene.
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

Each mode also has a VD-GRPO variant (`grpo_onpolicy_vdgrpo.json`, `grpo_multi_epoch_vdgrpo.json`) that uses Variance-Decoupled advantage computation (Plan-R1): centers advantages but divides by a fixed scale instead of per-group std, preserving crash signal magnitude. Controlled by `advantage_mode: "vd_grpo"` and `advantage_fixed_scale: 10.0` in the config.

KL coefficient scheduling is available via `kl_schedule` (`"constant"`, `"linear"`, `"cosine"`, `"step"`), decaying from `kl_coef` to `kl_coef_final` over training. Survival reward mode (`reward_mode: "survival"`, from PlannerRFT) gives proportional credit based on when failures occur instead of a flat penalty. The hybrid config (`grpo_onpolicy_hybrid.json`) combines VD-GRPO + survival + cosine KL. See `rlvr/autoresearch/README.md` for full tuning guidance on all features and config templates.

Key components:

| Module | Role |
|---|---|
| `grpo_config.py` | `GRPOConfig` dataclass, JSON load/save |
| `grpo_loss.py` | Advantage-weighted diffusion loss + KL regularization |
| `grpo_trainer.py` | `GRPOTrainer`: generation, scoring, training loop, eval, best-model tracking |
| `grpo_sampler.py` | `generate_diverse_group()`: N trajectories with randomized noise/guidance. Supports PlannerRFT-style lateral/longitudinal exploration (disabled by default via `enable_lateral`/`enable_longitudinal` in `SamplerConfig`). |
| `test_lateral_longitudinal_guidance.py` | Unit + visualization tests for lateral/longitudinal guidance |
| `reward.py` | Rule-based reward: `R = w_safety*S + w_progress*P + w_smooth*M + w_feasibility*F + w_centerline*C` |
| `train_grpo.py` | CLI entry point (`--mode rule` or `--mode gui`) |
| `trajectory_ranker_gui.py` | Gradio GUI: visualization + optional supervised GRPO training |
| `analyze_rewards.py` | Standalone reward distribution diagnostic tool |
| `configs/grpo_onpolicy.json` | On-policy config template (M=1) |
| `configs/grpo_onpolicy_vdgrpo.json` | On-policy + VD-GRPO advantages |
| `configs/grpo_onpolicy_kl_cosine.json` | On-policy + cosine KL decay (0.3→0.05) |
| `configs/grpo_onpolicy_vdgrpo_kl_cosine.json` | On-policy + VD-GRPO + cosine KL decay |
| `configs/grpo_onpolicy_hybrid.json` | All features: VD-GRPO + survival + cosine KL |
| `configs/grpo_multi_epoch.json` | Multi-epoch config template (M=4) |
| `configs/grpo_multi_epoch_vdgrpo.json` | Multi-epoch + VD-GRPO advantages |

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

### Exploration Policy (`exploration_policy/`)

Learned exploration policy inspired by PlannerRFT (arxiv 2601.12901) that adaptively modulates trajectory sampling during GRPO training. Instead of sampling guidance scales uniformly at random, a small neural network learns scene-dependent lateral and longitudinal guidance parameters from Beta distributions.

**Key difference from PlannerRFT**: PlannerRFT uses a separate PPO loop with per-step closed-loop rewards from a simulator. Our approach uses a **joint single-loop design** where both the DiT planner and the exploration policy are trained with the same GRPO group-relative advantages in one pass. The policy outputs one Beta distribution per scene, K different eta values are sampled from it, and K deterministic trajectories (noise_scale=0) are generated — one per eta. This eliminates the noise confound (reward reflects guidance quality, not noise luck) and avoids the need for a separate simulator, replay buffer, or GAE computation.

**Architecture** (mirrors `diffusion_planner/model/` structure):
```
exploration_policy/
  __init__.py                  Package exports
  model.py                     ExplorationPolicy — top-level model (like diffusion_planner.py)
  utils.py                     get_frozen_encoder(), generate_reference_trajectory()
  loss.py                      REINFORCE + entropy bonus + KL penalty
  module/
    ref_mixer.py               RefTrajectoryMixer — MLP-Mixer compressing x_ref [B,T,4] → [B,H]
    ref_fusion.py              RefFusionAttention — cross-attention fusing ref with scene encoding
    heads.py                   GuidanceHead (Beta params) + ValueHead (Phase 2)
  test_exploration_policy.py   Unit tests (12 tests, no model needed)
  test_integration.py          Integration test with real model + scenes
```

**Network flow**:
```
[Frozen Encoder] → scene_encoding [B, N, 256]
[LoRA-disabled DiT] → x_ref [B, 80, 4]
        ↓
  RefTrajectoryMixer (MLP-Mixer, 2 layers)  → ref_token [B, 128]
        ↓
  RefFusionAttention (cross-attn: ref queries scene) → fused [B, 128]
        ↓                    ↓
  GuidanceHead           ValueHead
  (4 Beta params)        (V(s) scalar)
        ↓
  η ~ Beta(α, β) mapped to [-1, 1]
```

**Zero-initialization**: GuidanceHead fc2 weights are zero-initialized (no bias term — `bias=False`). This produces `softplus(0)+1 ≈ 1.693` for both Alpha and Beta, giving `η_mean = 0.0` (unbiased) and `η_std ≈ 0.48` (good exploration spread) at init. The bias is intentionally removed to force scene-dependent output: `output = W @ fused_input` must use the input, preventing the policy from learning a global bias that applies the same η to all scenes.

**Concentration clamping**: Alpha and Beta are clamped to `max_conc=10` to prevent distribution collapse. Without clamping, `raw_scale=10` can produce alpha/beta > 20 after one gradient step, making the Beta near-deterministic (entropy≈0). The clamp keeps worst-case at Beta(10,1) with std≈0.09.

**Joint training flow** (per scene in `GRPOExplorationTrainer`):
1. Frozen encoder → scene_encoding; LoRA-disabled DiT → x_ref
2. Policy(scene_encoding, x_ref) → Beta(α_lat, β_lat), Beta(α_lon, β_lon)
3. Sample K=16 η values from distributions → K trajectories (noise=0, lateral+longitudinal guidance)
4. Score all K → GRPO group-relative advantages
5. DiT loss: standard GRPO diffusion loss (same as `grpo_trainer.py`)
6. Policy loss: REINFORCE `L = -mean(A_k · log_prob_k) + c_e · (-entropy)`

**Per-group optimizer stepping** (`exploration_step_per_group: true`): The policy optimizer steps after each scene (group) instead of accumulating gradients across all scenes and stepping once per epoch. This is ESSENTIAL for per-scene learning. Without it, the gradient is averaged across 300 scenes and the policy learns only a global mean shift (confirmed: η_lat std=0.013 across scenes, essentially uniform). With per-group stepping, each scene provides its own gradient signal, forcing the network to use scene_encoding to produce different η for different scenes.

**Important**: Only REINFORCE (inner_epochs=1) works. PPO (inner_epochs>1) always collapses the distribution, even with clamping. KL penalty is not needed and not helpful (tested KL=0, 0.001, 0.01 — no difference).

**Key files in `rlvr/`**:

| File | Role |
|---|---|
| `grpo_exploration_trainer.py` | Joint GRPO+policy trainer (separate from `grpo_trainer.py`) |
| `grpo_config.py` | `exploration_*` fields (all default to off; backward-compatible) |
| `grpo_sampler.py` | `PolicyGroupMetadata` dataclass; `generate_diverse_group()` unchanged |
| `configs/grpo_onpolicy_exploration.json` | Config template with exploration policy enabled |

**Running tests**:
```bash
# Unit tests (no model needed):
python exploration_policy/test_exploration_policy.py

# Integration test (requires v4 model + scenes):
python exploration_policy/test_integration.py \
  --model_path <model.pth> --npz_list <scenes.json>
```

**Config** (recommended for policy learning):
```json
{
  "use_exploration_policy": true,
  "exploration_hidden_dim": 128,
  "exploration_lr": 5e-4,
  "exploration_entropy_coef": 0.05,
  "exploration_kl_coef": 0.0,
  "exploration_head_init": "zeros",
  "exploration_head_raw_scale": 10.0,
  "exploration_inner_epochs": 1,
  "exploration_step_per_group": true,
  "exploration_freeze_after_epoch": 0
}
```

**LoRA Block Targeting** (`lora_target` in config):
The DiT decoder has 3 blocks. Autoresearch found that targeting only **block 0** (first block) gives the best trade-off: ego L2 improves by ~5%, neighbor L2 only +6%, turn signals unaffected. Targeting all 3 blocks gives better ego improvement but +30-140% neighbor degradation. Block 0 handles general scene-level processing; later blocks specialize per-agent.

Options: `"all"` (default, all 3 blocks), `"first"` (block 0 only — recommended for exploration policy), `"last"` (block 2), `"blocks01"` (blocks 0+1).

**Critical**: The exploration policy trainer (`GRPOExplorationTrainer`) adds random noise alongside policy-guided trajectories. The original implementation used `noise_scale=0` which causes model collapse. Always keep `noise_scale_range` active + `rejection_keep > 0`.
