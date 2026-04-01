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
  reward.py                  Rule-based reward (road border + safety + progress + feasibility
                             + lane departure detection with K=3 nearest centerlines
                             + underprogress penalty, GT-normalized progress, survival mode
                             + lane_crossing_steps as terminal event in survival mode)
  grpo_loss.py               Advantage-weighted diffusion loss with K=8 (noise,t) averaging
                             + prefix mask matching SFT training distribution
                             + optional neighbor prediction regularization
                             + compute_batched_grpo_loss for N-trajectory batched loss
  grpo_config.py             Dataclass config with JSON serialization
                             + lane departure, underprogress, progress normalization config
                             + diffusion_k_steps, neighbor_loss_weight, lane_dep_trim_n
  grpo_trainer.py            Standard GRPO training loop (batched loss)
  grpo_trainer_batched.py    Fully batched GRPO trainer (all scenes in ~5 forward passes)
                             + lane_dep_trim_n: drop worst lane-departure scenes per epoch
                             + fixed gradient accumulation scaling for incomplete last chunks
  grpo_exploration_trainer.py  Joint GRPO + exploration policy trainer
                             + random_guidance_mode: skip explorer, sample η directly
  grpo_sampler.py            Diverse trajectory generation with random noise + guidance
  grpo_sampler_batched.py    Strong CL+SPD sampler (1 det + 8 CL5-10 guided + 7 random)
  configs/
    grpo_onpolicy.json       Recommended on-policy config (best for v4)
    grpo_zi_300sc.json       Zero-init exploration + Block 0 LoRA (best baseline)
  closed_loop/               Closed-loop explorer training (PlannerRFT-style)
    state_update.py          Scene re-centering after ego moves one step
    per_step_reward.py       Per-step collision, road border, progress reward
    gae.py                   Generalized Advantage Estimation
    rollout.py               Sequential rollout manager (B=1)
    batched_rollout.py       GPU-parallel rollout manager (B=N, all scenes per step)
    closed_loop_trainer.py   Hybrid trainer: CL rollout + open-loop GRPO for DiT
  autoresearch/
    run_experiment.py        Single experiment runner (batched eval, all paths via CLI)
    check_lora_training.py   LoRA weight verification tool
    visualize_scenes.py      Scene visualization with road borders + ego footprints
    eval_border_distance.py  Road border distance metrics
    README.md                Full autoresearch documentation
  autoresearch/tools/         Evaluation and diagnostic tools (see tools/README.md)
    cleanse_lane_scenes.py   Filter scenes by t=0 lane/border clearance
    diagnose_grpo_signal.py  Diagnose per-scene GRPO reward signal (batched)
    eval_lane_border_distance.py  Combined lane departure + border distance eval
    eval_reward_vs_gt.py     Per-scene reward breakdown vs ground truth
    eval_driving_metrics.py  Speed/lat_accel/path length/stopped metrics
    viz_guidance_actual.py   Visualize actual DiT inference with/without guidance
    grpo_viz.py              Visualize K trajectories per scene with reward ranking table
  autoresearch/tests/         Tests for closed-loop components
    test_gae.py              Unit tests for GAE computation
    test_state_update.py     Unit tests for coordinate transforms
    test_real_scene.py       Integration test with real NPZ scene
  test_reward.py             Unit tests for reward (no model needed)
  test_grpo_sampler.py       Unit tests for sampler (needs model for full suite)
```

## Training Modes

| Mode | Config | Trainer | Description |
|------|--------|---------|-------------|
| Standard GRPO | `use_exploration_policy: false` | `GRPOTrainer` → `train_epoch_batched` | Fully batched, ~5 forward passes per epoch |
| Random guidance | `use_exploration_policy: true, random_guidance_mode: "uniform"` | `GRPOExplorationTrainer` | Random η ∈ [-1,1] lateral/longitudinal guidance. **Best mode.** |
| Explorer (open-loop) | `use_exploration_policy: true, random_guidance_mode: "explorer"` | `GRPOExplorationTrainer` | Learned Beta guidance + GRPO |
| Explorer (closed-loop) | `use_closed_loop: true` | `ClosedLoopExplorationTrainer` | Per-step rollout + GAE + GRPO |

All modes support GPU-batched trajectory generation and evaluation.

### Random Guidance Mode

The `random_guidance_mode` config replaces the learned exploration policy with direct η sampling:

| Mode | η distribution | Notes |
|------|---------------|-------|
| `"uniform"` | U[-1, 1] | **Recommended.** Matches zero-init explorer, +2 pts over no guidance |
| `"gaussian"` | N(0, 0.3) clipped | Similar to uniform, slightly worse on rb_cross |
| `"narrow"` | U[-0.5, 0.5] | Too little diversity, -7 pts vs uniform |
| `"none"` | η = 0 always | No lateral/longitudinal guidance, pure noise diversity |
| `"explorer"` | Learned Beta | Default. Explorer never learns, matches uniform output |

When mode ≠ "explorer", the 227K-param exploration policy network is not loaded.

## Batching

All trajectory generation, GRPO loss computation, and evaluation support GPU batching:
- K trajectories per scene generated in one forward pass (B=K)
- N×K batched generation in closed-loop GRPO (B=N×K)
- Batched eval with configurable `batch_size` (default 32)
- `compute_batched_grpo_loss`: N diffusion losses in one forward pass

Configure via `closed_loop_batch_size` in `GRPOConfig` (default 8 for 24GB VRAM).

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

### Lane Departure Detection

`compute_lane_departure_penalty()` detects when the ego vehicle leaves its lane:

1. Find K=3 nearest centerlines from **different lane segments** (not just closest points)
2. For each of 80 ego perimeter sample points, check containment against all K lanes
3. Thresholds: crossing (clearance <10cm from lane edge, including outside), near (<25cm), wide (<40cm), continuous (<80cm)
4. Returns `(crossing_gate, near_frac, wide_frac, lane_crossing_steps, cont_penalty)`
   - `lane_crossing_steps`: first timestep of lane departure per trajectory (for survival mode)
   - `lane_wide_frac`: fraction of timesteps within 40cm of lane edge

Enabled via `enable_lane_departure: true` in config. Can be used as:
- **Soft penalty** via `lane_near_scale`, `lane_wide_scale`, `lane_cont_scale`
- **Hard gate** via `lane_gate_enabled: true` (lane crossing zeros the reward in gate mode)
- **Survival terminal event** when `reward_mode: "survival"` + `enable_lane_departure: true`,
  lane departure reduces `survival_frac` proportionally to when it occurs

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
