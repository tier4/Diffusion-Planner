# Exploration Policy

Learned exploration policy for adaptive guidance during GRPO reinforcement fine-tuning of diffusion-based planners. Inspired by [PlannerRFT](https://arxiv.org/abs/2601.12901) (Li et al., 2026).

## Overview

During GRPO training, the standard approach samples guidance parameters (lateral offset, longitudinal speed scaling) uniformly at random. This is scenario-agnostic — the same exploration distribution is used regardless of whether the ego vehicle is on a straight highway or navigating a tight intersection.

The exploration policy replaces this with a **learned, scene-conditional sampler**. A small neural network observes the scene context and reference trajectory, then outputs Beta distributions over the lateral and longitudinal guidance scales. Different samples from these distributions produce diverse trajectories that are scored and ranked by GRPO.

### Differences from PlannerRFT

This implementation adapts the core idea from PlannerRFT with several design changes:

| Aspect | PlannerRFT | This implementation |
|---|---|---|
| **Training loop** | Separate PPO (policy) + GRPO (planner) | Joint single-loop: both trained with GRPO advantages |
| **Trajectory noise** | DDIM noise (stochastic denoising) | noise_scale=0 (deterministic per η) |
| **Reward signal** | Per-step closed-loop rewards via simulator | Trajectory-level open-loop rewards |
| **Temporal structure** | MDP with GAE (γ=0.99, λ=0.95) | Contextual bandit (single η → single reward) |
| **Policy update** | PPO with replay buffer, clipped surrogate | REINFORCE with group-relative advantages |
| **KL scheduling** | Constant | Inverse: DiT KL decays, policy KL ramps up |

The **deterministic trajectory generation** (noise=0) is a deliberate choice: it eliminates the noise confound where a good guidance choice could receive a bad reward due to unlucky diffusion noise. Each trajectory's reward is a pure function of the chosen η values.

The **joint training loop** avoids the complexity of maintaining a separate replay buffer and PPO optimizer. Both networks receive the same GRPO group-relative advantage signal — the DiT learns which trajectory shapes are good, while the policy learns which guidance directions are productive for each scene.

**Inverse KL scheduling** addresses the cold-start problem: early in training the policy hasn't learned anything useful, so it needs freedom to explore (low KL). Meanwhile the DiT should stay close to the pretrained model (high KL). As training progresses, the policy has found good exploration directions and should be anchored (high KL), while the DiT can adapt more freely (low KL).

## Architecture

```
[Frozen Encoder] → scene_encoding [B, N, 256]
[LoRA-disabled DiT] → x_ref [B, 80, 4]
        ↓
  RefTrajectoryMixer (MLP-Mixer, 2 layers, hidden=128)
        ↓
  ref_token [B, 128]
        ↓
  RefFusionAttention (cross-attn: ref queries scene)
        ↓
  fused [B, 128]
     ↓              ↓
  GuidanceHead    ValueHead
  Beta(α,β) × 2   V(s) scalar
     ↓
  η_lat, η_lon ∈ [-1, 1]
```

- **RefTrajectoryMixer**: Compresses the 80-step reference trajectory into a single token using MLP-Mixer blocks (same architecture as `diffusion_planner.model.module.mixer.MixerBlock`).
- **RefFusionAttention**: Multi-head cross-attention where the reference token queries the frozen scene encoding, capturing the relationship between reference motion and surrounding environment.
- **GuidanceHead**: Outputs parameters for two Beta distributions (lateral and longitudinal). Zero-initialized output layer ensures η_mean=0.0 and η_std≈0.48 at initialization (PlannerRFT Section 4.5).
- **ValueHead**: Scalar state-value estimate V(s) for future PPO integration. Not used in current joint training.

Total parameters: ~227K (default config).

## Training

### Joint GRPO + Policy Training

Per scene:
1. Frozen encoder produces scene encoding
2. LoRA-disabled DiT produces reference trajectory x_ref
3. Exploration policy outputs Beta distributions conditioned on (scene, x_ref)
4. K=16 η values sampled from distributions
5. K deterministic trajectories generated (each with lateral + longitudinal guidance from its η)
6. All K trajectories scored → GRPO group-relative advantages
7. DiT updated with standard GRPO diffusion loss
8. Policy updated with REINFORCE: `L = -mean(A_k · log_prob_k) + c_e·(-H) + c_kl·KL(π||π_init)`

### Loss Components

| Component | Formula | Purpose |
|---|---|---|
| Policy gradient | `-mean(advantages · log_probs)` | Increase probability of high-reward η |
| Entropy bonus | `-c_e · (H_lat + H_lon)` | Prevent η distribution collapse |
| KL penalty | `c_kl · KL(π_current ‖ π_init)` | Anchor to zero-mean initial policy |

### Inverse KL Schedule

```
Epoch  | DiT KL (↓) | Policy KL (↑)
-------|------------|-------------
  1    |   0.30     |   0.01
  2    |   0.23     |   0.02
  3    |   0.17     |   0.04
  4    |   0.10     |   0.05
```

## File Structure

```
exploration_policy/
  __init__.py                  Package exports
  model.py                     ExplorationPolicy (top-level model)
  utils.py                     Frozen encoder access, reference trajectory generation
  loss.py                      REINFORCE + entropy + KL loss computation
  module/
    __init__.py
    ref_mixer.py               RefTrajectoryMixer (MLP-Mixer)
    ref_fusion.py              RefFusionAttention (cross-attention)
    heads.py                   GuidanceHead (Beta) + ValueHead (scalar)
  test_exploration_policy.py   Unit tests (12 tests, no model needed)
  test_integration.py          Integration test with real model + scenes
```

Related files in `rlvr/`:
```
rlvr/
  grpo_exploration_trainer.py              Joint trainer (separate from grpo_trainer.py)
  grpo_config.py                           exploration_* fields (backward-compatible)
  grpo_sampler.py                          PolicyGroupMetadata dataclass
  configs/grpo_onpolicy_exploration.json   Config template
```

## Usage

### Unit Tests (no model needed)
```bash
source .venv/bin/activate
PYTHONPATH=. python exploration_policy/test_exploration_policy.py
```

### Integration Test (requires model + scenes)
```bash
python exploration_policy/test_integration.py \
  --model_path <path/to/model.pth> \
  --npz_list <path/to/scenes.json>
```

### Config
```json
{
  "use_exploration_policy": true,
  "exploration_hidden_dim": 128,
  "exploration_n_mixer_layers": 2,
  "exploration_n_attn_heads": 4,
  "exploration_dropout": 0.1,
  "exploration_lr": 5e-5,
  "exploration_entropy_coef": 0.05,
  "exploration_kl_coef": 0.01,
  "exploration_kl_schedule": "linear",
  "exploration_kl_coef_final": 0.05
}
```

### Monitoring

Key metrics to watch during training:
- **η distribution width** (`exploration_eta_lat_std`): should stay above ~0.2. Below that, entropy bonus is too low.
- **η mean drift** (`exploration_eta_lat_mean`, `exploration_eta_lon_mean`): drift from 0 indicates the policy is learning directional preferences. Expected, but large drift (>0.5) may indicate collapse.
- **Policy entropy** (`exploration_entropy`): should decrease gradually, not crash.
- **Deterministic trajectory reward**: tracks DiT improvement independent of guidance.

## References

- PlannerRFT: Li et al., "Reinforcing Diffusion Planners through Closed-Loop and Sample-Efficient Fine-Tuning", arxiv 2601.12901, 2026.
- Lateral guidance: Eq. 2, `Ψ_lat = (1/T) Σ (n⊥·(x-x_ref) - λ_lat·η_lat)²`
- Longitudinal guidance: Eq. 3, `Ψ_lon = (1/T) Σ (n∥·(v - λ_lon·η_lon·v_ref))²`
