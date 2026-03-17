# RLVR -- Reinforcement Learning with Verifiable Rewards

Infrastructure for Group Relative Policy Optimization (GRPO) training of the Diffusion Planner.
Generates N diverse trajectories per scene, scores them with a rule-based reward function,
and computes group-relative normalized advantages.

## Overview

The GRPO pipeline differs from the existing DPO pipeline in two key ways:

1. **Group size**: DPO generates 2 trajectories per scene (pairwise preference). GRPO generates N (default 32 for training, 8 for GUI debugging).
2. **Automatic scoring**: DPO can use human annotation or simple heuristics. GRPO uses a multi-component reward function that evaluates safety, progress, smoothness, and feasibility -- all automatically.

## Architecture

```
rlvr/
  reward.py               Rule-based reward (batched over N trajectories)
  grpo_sampler.py          Diverse trajectory generation with random noise + guidance
  trajectory_ranker_gui.py Gradio GUI for visualizing rankings and tuning reward weights
  __init__.py              Package exports
  test_reward.py           Unit tests for reward (no model needed)
  test_grpo_sampler.py     Unit tests for sampler (needs model for full suite)
```

## Reward Function

`R = w_safety * S + w_progress * P + w_smooth * M + w_feasibility * F + w_centerline * C`

### Components

**Safety (S)**: Ego-NPC collision check using oriented bounding boxes and the Separating Axis Theorem.
Reuses `neighbor_clearance_penalty` from `diffusion_planner/loss.py`. Uses actual vehicle dimensions
(length, width from `ego_shape` and `neighbor_agents_past`) with a 0.5m safety margin. Returns
`collision_penalty` (default -10) on first collision, 0 otherwise.

**Progress (P)**: Euclidean distance reduction toward `goal_pose`. Falls back to total path length
if goal is unavailable (zeros).

**Smoothness (M)**: Negative mean absolute jerk computed via finite differences on positions.
Penalizes sharp direction changes and acceleration spikes.

**Feasibility (F)**: Three sub-components:
- *Lane boundary violations*: Reuses `lane_boundary_penalty` from `diffusion_planner/loss.py`.
  Builds the ego vehicle's 4 bounding box corners at each timestep and checks whether any corner
  protrudes beyond the actual left/right lane boundaries. Uses `route_lanes` (the ego's intended
  path) rather than all nearby `lanes` to avoid false negatives from adjacent/oncoming lanes.
  Includes a 0.3m margin before penalizing.
- *No-coverage penalty*: Timesteps where the ego center is >10m from any route lane center receive
  a large fixed penalty. Catches trajectories that leave the mapped route entirely.
- *Acceleration violations*: Fraction of timesteps where longitudinal acceleration exceeds
  `max_accel` (default 8 m/s^2).

**Centerline (C)**: Negative mean squared lateral deviation from the nearest route lane centerline.
Rewards trajectories that stay close to the center of the intended lane rather than weaving near
boundaries. Uses `route_lanes` to ensure the ego follows its assigned route. Computed as
`-mean(lateral_offset^2)` across all timesteps.

### Default Weights

| Weight | Default | Purpose |
|--------|---------|---------|
| `w_safety` | 5.0 | Collision avoidance dominates |
| `w_progress` | 1.0 | Goal-directed driving |
| `w_smooth` | 0.5 | Comfortable trajectories |
| `w_feasibility` | 5.0 | Stay on road, respect dynamics |
| `w_centerline` | 1.0 | Prefer lane center over lane edges |

All weights are tunable in the GUI without regenerating trajectories.

### Group Advantages

After scoring N trajectories, advantages are computed as:

```
advantage_i = (R_i - mean(R)) / (std(R) + epsilon)
```

This gives zero-mean, unit-variance advantages used as GRPO training signal.

## Diverse Trajectory Generation

`generate_diverse_group()` produces N trajectories per scene:

- **Trajectory 0**: Deterministic (noise_scale=0, no guidance) -- the model's MAP output.
- **Trajectories 1..N-1**: Each gets an independently randomized configuration:
  - Random noise scale from `[noise_min, noise_max]`
  - Each enabled guidance type is independently coin-flipped with `guidance_prob`
  - Anchor guidance picks a random prototype index per trajectory
  - Some trajectories end up with no guidance (all coin flips false) -- intentional diversity

### Guidance Types

Controlled by per-type checkboxes (GUI) or `SamplerConfig` booleans:

| Type | Default | Source |
|------|---------|--------|
| Centerline following | Enabled | `diffusion_planner/model/guidance/centerline_following.py` |
| Anchor following | Enabled | `diffusion_planner/model/guidance/anchor_following.py` |
| Collision | Disabled | `diffusion_planner/model/guidance/collision.py` |
| Route following | Disabled | `diffusion_planner/model/guidance/route_following.py` |
| Lane keeping | Disabled | `diffusion_planner/model/guidance/lane_keeping.py` |

Collision and lane keeping are disabled by default for sampling guidance because their `energy()`
path has known numerical issues. Their scoring/reward path is fine and is used indirectly through
the reward function (which calls `loss.py` functions directly).

## Trajectory Ranker GUI

Interactive visualization for debugging the reward function and validating trajectory quality.

### Launch

```bash
source .venv/bin/activate
python3 -m rlvr.trajectory_ranker_gui \
  --model_path /path/to/model.pth \
  --npz_list /path/to/path_list.json
```

Prototypes are auto-generated from the npz list on first run. Use `--regen-prototypes` to force
regeneration, or `--prototypes /path/to/file.npy` to use a specific file.

### GUI Controls

**Navigation**: Step through scenes (+-1, +-10, +-30), shuffle order, jump to index.

**Noise**: Min/max noise scale sliders control the sampling range for stochastic trajectories.

**Guidance**: Master enable checkbox + per-type checkboxes. Only checked types enter the random
pool. "Per-type inclusion probability" controls how likely each type is applied per trajectory.

**Reward weights**: Changing weights rescores existing trajectories instantly (no regeneration).

**Re-do button**: Regenerates all trajectories with fresh random noise and guidance configs.

**Display**: Zoom and time step sliders rerender without regeneration or rescoring.

### Visualization

- Trajectories colored red (worst) to green (best) by advantage rank
- Deterministic trajectory shown as blue dashed line with star marker, bold in reward table
- Ground truth shown as black dashed line
- Top-3 trajectories get diamond markers at the selected timestep
- Reward table shows all components sorted by total score
- Speed and curvature plots for top-3 trajectories

## Reused Code

The reward function reuses production code from the training/validation pipeline:

| Function | Source | Used For |
|----------|--------|----------|
| `lane_boundary_penalty` | `diffusion_planner/loss.py` | Feasibility: lane boundary check with vehicle footprint |
| `neighbor_clearance_penalty` | `diffusion_planner/loss.py` | Safety: SAT collision check with oriented bounding boxes |
| `_build_ego_bbox_corners` | Adapted from `compute_safety_penalty` in `loss.py` | Shared ego footprint construction |
| `generate_samples` | `guidance_gui/generate_samples.py` | Per-trajectory model inference |
| `GuidanceComposer` | `diffusion_planner/model/guidance/composer.py` | Guidance injection during sampling |

## Future: Imitation-Based Reward Components

`loss.py` also contains `loss_func(pred, gt)` which returns per-timestep losses that could serve
as an imitation-based reward component:

- `position_lat_loss` / `position_lon_loss`: Lateral and longitudinal error projected onto the GT
  heading direction. More meaningful than raw L2 -- separates lane-departure error from
  speed-mismatch error.
- `cosine_similarity_loss`: Heading alignment with GT (1 - cos_sim).
- `heading_l2_loss`: Raw heading vector error.

These would score "how close is this sampled trajectory to the ground truth log-replay?" Adding
them as a reward component (e.g., `w_imitation * -ADE_to_GT`) would anchor the GRPO reward
partially to imitation learning. This creates a tradeoff:

- **Pro**: Prevents reward hacking where trajectories score well on rule-based metrics but are
  unrealistic (e.g., driving perfectly on the centerline at an unreasonable speed profile).
- **Con**: Biases toward replicating logged behavior rather than discovering novel better
  trajectories. The GT trajectory is not always optimal (human drivers make mistakes).

The current reward is purely rule-based. Whether to mix in imitation signals depends on how much
the rule-based reward alone can shape the policy without reward hacking.
