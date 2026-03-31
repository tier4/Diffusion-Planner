# RLVR -- Reinforcement Learning with Verifiable Rewards

Infrastructure for Group Relative Policy Optimization (GRPO) training of the Diffusion Planner.
Generates N diverse trajectories per scene, scores them with a rule-based reward function,
and computes group-relative normalized advantages.

## Overview

The GRPO pipeline differs from the existing DPO pipeline in two key ways:

1. **Group size**: DPO generates 2 trajectories per scene (pairwise preference). GRPO generates N (default 32 for training, 8 for GUI debugging).
2. **Automatic scoring**: DPO can use human annotation or simple heuristics. GRPO uses a multi-component reward function that evaluates safety, progress, smoothness, feasibility, and centerline adherence -- all automatically.

## Architecture

```
rlvr/
  reward.py                Rule-based reward (road border + safety + progress + feasibility)
  grpo_loss.py             Advantage-weighted diffusion loss with PPO clipping + KL
  grpo_config.py           Dataclass config with JSON serialization
  grpo_trainer.py          Training loop with per-epoch eval, LoRA checkpointing
  grpo_sampler.py          Diverse trajectory generation with random noise + guidance
  trajectory_ranker_gui.py Gradio GUI for visualizing rankings and tuning reward weights
  configs/
    grpo_onpolicy.json     Recommended on-policy config (best for v4)
    grpo_multi_epoch.json  Multi-epoch PPO-clip config
  autoresearch/
    run_experiment.py      Single experiment runner (all paths via CLI)
    check_lora_training.py LoRA weight verification tool
    visualize_scenes.py    Scene visualization with road borders + ego footprints
    README.md              Full autoresearch documentation
  test_reward.py           Unit tests for reward (no model needed)
  test_grpo_sampler.py     Unit tests for sampler (needs model for full suite)
```

## Reward Function

`R = safety_product * quality_score + (1 - safety_product) * (-50)`

**Safety gates** (hard, binary — any trigger floors reward to -50):
- Collision gate: ego collides with neighbor vehicle
- Road border gate: ego perimeter (80 sample points) crosses road border (within 10cm)
- Red light gate: ego runs red light

**Quality score** (soft, weighted sum):
`quality = w_progress * progress + w_safety * safety + w_smooth * smoothness + w_centerline * centerline + ttc_bonus - rb_near_penalty - rb_wide_penalty`

Road border proximity penalties (`near_edge_scale`, `wide_edge_scale`) subtract from quality
even when the ego doesn't cross the border, penalizing trajectories that get close.

### Components

**Safety (S)**: Ego-NPC collision check using oriented bounding boxes and the Separating Axis
Theorem (SAT). Builds ego bbox corners from `ego_shape` (wheel_base, length, width) and NPC
bboxes from `neighbor_agents_past` dimensions. Uses `neighbor_agents_future` GT positions (with
yaw converted from radians to cos/sin). Returns `collision_penalty` (default -10) on first
bounding box overlap. Additionally, a soft **proximity penalty** is applied when the ego passes
within 1m of any NPC without colliding (scales with intrusion depth).

Known limitation: NPC futures are open-loop (from log replay). If the ego deviates significantly
(e.g., stops), NPCs don't react, causing ghost collisions that wouldn't happen in closed-loop
simulation. TeraSim integration will address this.

**Progress (P)**: Euclidean distance reduction toward `goal_pose`. Falls back to total path length
if goal is unavailable (zeros).

**Smoothness (M)**: Negative mean absolute jerk computed via finite differences on positions.
Penalizes sharp direction changes and acceleration spikes.

**Feasibility (F)**: Multi-level penalty system with four severity tiers:

| Severity | Condition | Penalty per step | Description |
|----------|-----------|------------------|-------------|
| Margin | In-lane, vehicle edge within 0.5m of boundary | 0-0.5 | Discourages riding lane edges |
| Off-route | On a road but not in any route lane | 5.0 | Wrong lane / missed turn |
| Off-road | Outside all lane boundaries | 10.0 + protrusion | Left the drivable surface |
| Off-map | >10m from any lane center | 20.0 + distance | Completely off the map |

Lane containment checks use ALL `lanes` data (not just route lanes) with both distance and
longitudinal proximity constraints to prevent false positives from distant lane segments.
Off-route detection uses `route_lanes` to identify wrong-lane driving where the ego is on a
road but not on the planned route.

Additionally penalizes acceleration violations (fraction of steps where `|accel| > max_accel`).

All violations are time-weighted: early violations (t=0s) are penalized ~3x more than late
violations (t=7.9s), and the result is a time-weighted mean.

**Centerline (C)**: Normalized lane-usage penalty from the nearest route lane centerline. Computes
`(|lateral_offset| + half_vehicle_width) / lane_half_width` to measure what fraction of the lane
the vehicle occupies, accounting for actual lane width and vehicle dimensions. Values range from
~0.5 (centered) to 1.0 (edge touching boundary). Capped at 1.0 (being beyond the boundary is
handled by feasibility).

When the trajectory leaves route lane coverage (was near route but drifted away), switches to a
distance-based route deviation penalty (capped at 5.0) so trajectories that abandon the route are
always penalized. Timesteps beyond where route data simply doesn't extend are not penalized.

Time-weighted mean with early deviations penalized more.

### Default Weights

| Weight | Default | Purpose |
|--------|---------|---------|
| `w_safety` | 5.0 | Collision avoidance dominates |
| `w_progress` | 2.0 | Goal-directed driving |
| `w_smooth` | 0.5 | Comfortable trajectories |
| `w_feasibility` | 5.0 | Stay on road, respect dynamics |
| `w_centerline` | 5.0 | Follow route centerline |

All weights are tunable in the GUI without regenerating trajectories. The reward table shows
weighted values (column * weight) so that columns add up to the total.

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
path has known numerical issues.

## Trajectory Ranker GUI

Interactive visualization for debugging the reward function and validating trajectory quality.

### Launch

```bash
source .venv/bin/activate
python3 -m rlvr.trajectory_ranker_gui \
  --model_path /path/to/model.pth \
  --npz_list /path/to/path_list.json
```

Generate `path_list.json` with:
```bash
python3 diffusion_planner/util_scripts/create_train_set_path.py /path/to/npz_directory
```

Prototypes are auto-generated from the npz list on first run. Use `--regen-prototypes` to force
regeneration, or `--prototypes /path/to/file.npy` to use a specific file.

### GUI Controls

**Navigation**: Step through scenes (+-1, +-10, +-30), shuffle order, jump to index.

**Re-do button**: Regenerates all trajectories with fresh random noise and guidance configs.

**Noise**: Min/max noise scale sliders control the sampling range for stochastic trajectories.

**Guidance**: Master enable checkbox + per-type checkboxes. Only checked types enter the random
pool. "Per-type inclusion probability" controls how likely each type is applied per trajectory.

**Reward weights**: Changing weights rescores existing trajectories instantly (no regeneration).
The table and trajectory colors update live.

**Save Scene**: Saves current trajectories, reward breakdowns, and a plot image to
`.datasets/trajectory-dump-YY-MM-DD-HH-MM-SS/`. Multiple scenes accumulate in the same session
directory. Saved data can be used for offline analysis and debugging.

**Display**: Zoom and time step sliders rerender without regeneration or rescoring.

### Visualization

- Trajectories colored red (worst) to green (best) by advantage rank
- Deterministic trajectory shown as blue dashed line with star marker, bold in reward table
- Ground truth shown as black dashed line
- Top-3 trajectories get diamond markers at the selected timestep
- Collision points shown as red X markers at the first collision timestep
- Reward table shows weighted component values (columns add up to total)
- Speed and curvature plots for top-3 trajectories

## Reused Code

| Function | Source | Used For |
|----------|--------|----------|
| `batch_signed_distance_rect` | `diffusion_planner/model/guidance/collision.py` | SAT signed distance for collision detection |
| `center_rect_to_points` | `diffusion_planner/model/guidance/collision.py` | Oriented bounding box corner computation |
| `generate_samples` | `guidance_gui/generate_samples.py` | Per-trajectory model inference |
| `GuidanceComposer` | `diffusion_planner/model/guidance/composer.py` | Guidance injection during sampling |
| `load_npz_data` | `preference_optimization/utils.py` | Scene data loading with heading conversion |
| `load_model` | `preference_optimization/model_utils.py` | Model checkpoint loading |

Note: `loss.py` functions (`lane_boundary_penalty`, `neighbor_clearance_penalty`) have known bugs
(P/T reshape issue, baseline penalty from loose nearest-lane matching) and are NOT used. The
reward module implements its own lane boundary and collision checks directly.

## Future: Imitation-Based Reward Components

`loss.py` contains `loss_func(pred, gt)` which returns per-timestep losses that could serve
as an imitation-based reward component:

- `position_lat_loss` / `position_lon_loss`: Lateral and longitudinal error projected onto the GT
  heading direction.
- `cosine_similarity_loss`: Heading alignment with GT (1 - cos_sim).
- `heading_l2_loss`: Raw heading vector error.

Adding these as a reward component would anchor the GRPO reward partially to imitation learning.
Tradeoff: prevents reward hacking but biases toward replicating logged behavior (which may not
be optimal). The current reward is purely rule-based.
