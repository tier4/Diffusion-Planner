# RLVR -- Reinforcement Learning with Verifiable Rewards

Infrastructure for Group Relative Policy Optimization (GRPO) training of the Diffusion Planner.
Generates N diverse trajectories per scene, scores them with a rule-based reward function,
and computes group-relative normalized advantages.

## Credits

The logprob GRPO implementation is based on **DiffusionDriveV2** by Li et al.:
- Paper: [DiffusionDriveV2: Multi-Modal Diffusion Policy Model for Closed-Loop Autonomous Driving](https://arxiv.org/abs/2512.07745)
- Code: [hustvl/DiffusionDriveV2](https://github.com/hustvl/DiffusionDriveV2)
- Key adaptations: VPSDE noise schedule (vs DDIM), ego-only log-prob (vs per-anchor),
  mean-normalized log-prob (vs sum), analytical mean-divergence KL (vs IL regularization)

## Overview

The GRPO pipeline differs from the existing DPO pipeline in two key ways:

1. **Group size**: DPO generates 2 trajectories per scene (pairwise preference). GRPO generates N (default 32 for training, 8 for GUI debugging).
2. **Automatic scoring**: DPO can use human annotation or simple heuristics. GRPO uses a multi-component reward function that evaluates safety, progress, smoothness, feasibility, and centerline adherence -- all automatically.

## PRiSM — Perturbation-Recovery iterative Self-Mining

Self-improvement loop built on top of Ranked SFT. The model trains on its own failure modes by deliberately constructing them:

1. **Source**: take a closed-loop sim NPZ pool produced by the current best model.
2. **Mine warm scenes**: use `scenario_generation/tools/classify_replay_steps.py` to band per-step replay NPZs by reward.py `cl_score` and select the `warm` band (ego near centerline).
3. **Perturb**: apply parallel offsets / yaw / velocity / jitter to the warm scenes via `rlvr/autoresearch/tools/disturb_and_replay.py`. The tool emits a `manifest.json` with per-NPZ perturbation metadata (`dx, dy, dtheta_deg, dv, lateral_offset_m, longitudinal_offset_m, source_scene, kind`).
4. **Score under the same model**: `rlvr/autoresearch/tools/viz_p4_recovery.py` runs K=N generation under the configured `generation_variant`, ranks the K trajectories with `compute_reward_batch` (real reward.py), and records per-scene `top1_cl`, `delta_cl = top1_cl - det_cl`, slot label, and rank-1 safety flags (`top1_rb_cross`, `top1_lane_cross`, `top1_kin_violated`).
5. **Filter** to scenes by SFT-target QUALITY using a percentile filter on `top1_cl` (parameter-light, distribution-adaptive). Keep top P% of perturbations by `top1_cl` (closer-to-0 = cleaner SFT target) via `rlvr/autoresearch/tools/percentile_filter_perturbed.py`. Combine with no-poison exclusions: drop scenes where rank-1 fires the kinematic gate, road-border crossing, lane crossing, or collision. Selects by SFT TARGET QUALITY rather than perturbation difficulty. Replaces the earlier σ-Δ filter (`delta_cl >= 1*sigma_train`), which mixed those two signals and produced unfittable training targets when the K=N Δ-distribution σ grew with model capacity.
6. **Train** ranked-SFT on the filtered set, warmstarted from the same model. Iterate.

Round-N protocol: re-evaluate the previous round's training scenes under the new winner (drop those that no longer pass the filter — they're "solved"), promote previously-rejected scenes that now pass (warm-start retry), and add fresh-mined random warm sources. Each round builds its own held-out perturbed val set (no Δ filter on val) and the round's winner is evaluated on every previous round's frozen val + the new round's val + the union to detect cross-distribution regressions.

### Visualization tools

- `rlvr/autoresearch/tools/viz_p4_recovery.py` — per-scene K=N + rank-1 viz on perturbed scenes. Renders the lanelet/road-border base, all K trajectories in faint grey, the deterministic prediction in blue, the rank-1 winner in red, with a yellow translucent ego footprint anchored at the perturbed pose `(dx, dy)` in the world frame. Used both as the per-round filter (writes `summary.json` and `improve_scenes.json`) and as a per-scene PNG generator.
- `rlvr/autoresearch/tools/viz_prism_compare.py` — multi-model overlay on the same perturbed scene. Up to three trajectories on one panel (baseline / warmstart / PRiSM), all in the lanelet frame so the perturbation magnitude and the per-model recovery are visually obvious. Optional summary-JSON-driven ranking by `Δ_PRiSM-vs-reference` for "show me the biggest gain scenes" workflows.
- `rlvr/autoresearch/tools/recovery_sim_ghost.py` — ghost-overlay 8-second closed-loop rollout. Runs the rollout under two models on the same perturbed scene and writes per-step PNGs with both egos overlaid + plans, plus an optional WebM via ffmpeg.

### Frame-transform note

`disturb_and_replay` shifts `ego_current_state` and `ego_agent_past` by `(dx, dy)` but leaves `lanes` / `route_lanes` in the original world frame. The model output is in an "ego-current-pose-relative" frame: `pred[0]` is always near `(1, 0)` for a 10 m/s ego, regardless of perturbation magnitude. To draw the trajectory in the world frame, add `(dx, dy)` to every step before plotting. `viz_prism_compare._to_lanelet_frame` is the canonical helper. Skip this and the perturbation will look invisible in your viz.

## Architecture

```
rlvr/
  reward.py                  Rule-based reward (road border + safety + progress + feasibility
                             + lane departure detection with K=12 nearest lanes
                             + underprogress penalty, GT-normalized progress, survival mode
                             + lane_crossing_steps as terminal event in survival mode
                             + SG-filtered lat_accel penalty, SG-trimmed jerk computation
                             + point-to-segment RB distance with pre-filtering
                             + rb_penalty_mode: "frac" or "survival" (first-violation time-decay)
                             + rear-end collision permanent exclusion
                             + rb_min_dist in RewardBreakdown for eval reporting)
  grpo_loss.py               MSE-based advantage-weighted diffusion loss with K=8 averaging
                             (DEPRECATED — use logprob loss instead for trajectory shape learning)
  grpo_logprob_loss.py       DDV2-style Gaussian log-probability GRPO loss (NEW)
                             + Two-stage: collect_logprob_rollout + compute_logprob_grpo_loss
                             + advantage-weighted gradient via truncated VPSDE denoising rollout
                             + Mean-divergence KL regularization (analytical, properly scaled)
                             + Adaptive IL regularization
  vpsde_logprob.py           VPSDE denoising step with Gaussian log-probability (NEW)
                             + create_timestep_schedule, compute_discount_weights
  grpo_config.py             Dataclass config with JSON serialization
                             + logprob config: grpo_loss_type, logprob_num_steps, logprob_t_start,
                               logprob_discount, logprob_min_std, il_loss_weight, il_adaptive
                             + advantage_mode: "ddv2" (inter-anchor truncated GRPO)
                             + per-epoch scheduling: schedules dict with linear/cosine/step/peak types
  grpo_sft_trainer.py        Ranked SFT trainer: generate N trajs, pick best by reward,
                             SG-filter, train with SFT diffusion loss (ego + neighbor GT)
                             + "gt_neighbor" and "baseline_neighbor" modes
                             + neighbor_reg_only: MSE(lora_neighbor, base_neighbor) regularization
                             + Post-hoc block ablation trick for L2 preservation
                             + Per-epoch scheduled reward weights and guidance
  grpo_trainer.py            Standard GRPO training loop (batched loss)
                             + advantage_logprob loss path when grpo_loss_type="advantage_logprob"
  grpo_trainer_batched.py    Fully batched GRPO trainer (all scenes in ~5 forward passes)
                             + logprob loss path with per-scene collect + train
  grpo_exploration_trainer.py  Joint GRPO + exploration policy trainer
  grpo_sampler.py            Diverse trajectory generation with random noise + guidance
  grpo_sampler_batched.py    Strong CL+SPD sampler (1 det + 8 CL5-10 guided + 7 random)
  configs/
    grpo_onpolicy.json       Recommended on-policy config (best for v4)
    grpo_zi_300sc.json       Zero-init exploration + Block 0 LoRA (best baseline)
    logprob_curve20_norm.json    Logprob GRPO on 20 exit-curve scenes
    logprob_balanced_40sc.json   Logprob GRPO balanced 50/50 prob/normal
  tests/
    test_vpsde_logprob.py    9 unit tests for log-prob math
    test_logprob_loss.py     7 unit tests for loss computation
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
    cleanse_lane_scenes.py   Filter scenes by t=0 lane/border clearance + GT-lane + GT rb-future margin
    diagnose_grpo_signal.py  Diagnose per-scene GRPO reward signal (batched)
    eval_lane_border_distance.py  Combined lane departure + border distance eval
    eval_reward_vs_gt.py     Per-scene reward breakdown vs ground truth
    eval_driving_metrics.py  Speed/lat_accel/path length/stopped metrics
    viz_guidance_actual.py   Visualize actual DiT inference with/without guidance
    viz_lane_departure.py    Lane departure + road border distance viz (--mode lane/rb/both)
    grpo_viz.py              Visualize K trajectories per scene with reward ranking table
    compute_baseline_cache.py  Precompute baseline/GT paths for progress ratio metrics
    rb_campaign_launcher.py  Auto-queue experiment batches (env: RB_CAMPAIGN_EXP_DIR, RB_CAMPAIGN_MODEL)
  autoresearch/tests/         Tests for closed-loop components
    test_gae.py              Unit tests for GAE computation
    test_state_update.py     Unit tests for coordinate transforms
    test_real_scene.py       Integration test with real NPZ scene
  test_reward.py             Unit tests for reward (no model needed)
  test_grpo_sampler.py       Unit tests for sampler (needs model for full suite)
  test_scheduling.py         Unit tests for per-epoch scheduling (14 tests)
```

## Training Modes

| Mode | Config | Trainer | Description |
|------|--------|---------|-------------|
| **Ranked SFT** | `ranked_sft_mode: "gt_neighbor"` | `train_epoch_ranked_sft` | **Best for lane keeping.** Generate N trajs, pick best by reward, SG-filter, SFT loss on ego+neighbor. |
| **Curated SFT** | `ranked_sft_mode: "curated"` | `train_epoch_ranked_sft` | Use pre-saved trajectories from Scene Branch Editor GUI as SFT target. Skips generation+ranking. |
| Advantage Logprob GRPO | `grpo_loss_type: "advantage_logprob"` | `train_epoch_batched` | DDV2-style Gaussian log-prob. Can learn curvature but degrades neighbor L2. |
| Advantage MSE GRPO | `grpo_loss_type: "advantage_mse"` | `train_epoch_batched` | Legacy. Cannot change trajectory shape, only length. |
| Random guidance | `random_guidance_mode: "uniform"` | `GRPOExplorationTrainer` | Random η guidance diversity. |
| Explorer (open-loop) | `random_guidance_mode: "explorer"` | `GRPOExplorationTrainer` | Learned Beta guidance + GRPO |
| Explorer (closed-loop) | `use_closed_loop: true` | `ClosedLoopExplorationTrainer` | Per-step rollout + GAE + GRPO |

All modes support GPU-batched trajectory generation and evaluation.

### Advantage Logprob GRPO (DDV2-style)

The advantage_logprob loss computes actual Gaussian log-probabilities during a truncated VPSDE denoising
rollout, enabling proper advantage-weighted gradient for trajectory shape learning. Based on
DiffusionDriveV2 (arXiv:2512.07745).

**Two-stage approach:**
1. **Collection** (no grad): Run model through multi-step denoising, store chain states + log-probs
2. **Optimization** (with grad): Re-run model on stored chain, compute advantage-weighted loss

**Config:**
```json
{
    "grpo_loss_type": "advantage_logprob",
    "logprob_num_steps": 5,
    "logprob_t_start": 0.01,
    "logprob_discount": 0.8,
    "logprob_min_std": 0.1,
    "il_loss_weight": 0.0,
    "kl_coef": 0.1,
    "advantage_mode": "normalized"
}
```

**Advantage modes:**
- `"normalized"` (recommended): standard GRPO, works with logprob
- `"ddv2"`: inter-anchor truncated GRPO (clip negative→0, safety→-1). Too harsh for our scenes.
- `"positive_only"`: only reinforce above-mean. Model tends to stop instead of curve.

**KL regularization:** Mean-divergence KL: `KL = mean((μ_policy - μ_ref)² / (2σ²))`.
Uses LoRA `disable_adapter_layers()` for reference model. Properly scaled (0.01→7 over epochs).

### Ranked SFT (GRPO-Ranked Self-Distillation)

The ranked SFT approach combines GRPO's trajectory generation and reward scoring with standard
SFT diffusion training. Instead of computing RL gradients, it treats the best-ranked trajectory
as a pseudo-GT and trains with MSE loss. This preserves neighbor prediction quality because
the SFT loss naturally covers all agents (ego + neighbors).

Inspired by self-distillation approaches in LLMs ([Zhang et al. 2026](https://arxiv.org/abs/2604.01193)),
adapted for diffusion planners with reward-based selection and Savitzky-Golay trajectory smoothing.

**Pipeline:**
1. Generate N=16 trajectories per scene (batched, with noise + guidance diversity)
2. Score all trajectories with the rule-based reward function
3. Select the highest-reward trajectory per scene
4. Apply Savitzky-Golay filter (window=11, order=3) to smooth the selected trajectory
5. Train LoRA with standard SFT diffusion loss: sample random t, noise, denoise, MSE vs pseudo-GT
6. **Post-hoc: zero LoRA block 0 weights** for better L2 preservation (the "no_block0 trick")

**Modes:**
- `"gt_neighbor"` (recommended): generate K trajs, pick best by reward, use GT neighbors
- `"baseline_neighbor"`: same as gt_neighbor but use base model neighbor predictions
- `"curated"`: skip generation+ranking, use `ego_agent_future` from NPZ directly as SFT target (for trajectories saved via Scene Branch Editor GUI)

**Config:**
```json
{
    "ranked_sft_mode": "gt_neighbor",
    "sg_filter_window": 11,
    "sg_filter_order": 3,
    "learning_rate": 1e-5,
    "train_epochs": 20,
    "seed_lora_path": "<warm-start checkpoint>"
}
```

*With neighbor regularization:*
- `neighbor_reg_only=true, neighbor_reg_weight=1.0` reduces neighbor L2 degradation significantly
- noise_scale_range [0.5, 2.0] recommended

**Block ablation trick:** After training with `lora_target="all"` (3 DiT blocks), zero out
one block's LoRA weights post-hoc as a quick way to probe how the blocks divide up the
learned behaviour — useful for debugging regressions introduced by a particular block.

Training all blocks then removing one post-hoc is generally more stable than training only
a subset of blocks: distributing gradients across all blocks produces smaller per-parameter
changes and avoids the destabilisation seen when a single block carries all the update.

**neighbor_reg_weight must be ≥ 1.0** at lr=5e-4. Lower values tend to diverge because the
ego loss has too much freedom while neighbor weights drift through shared DiT parameters.

**Generation variants** (`generation_variant` config field, default `"rsft_v2"`).
Composition is 1 deterministic + N guided cl_spd configs + M noise-only configs +
(15 - N - M) random-CL passes. Variants are defined in `rlvr/generation_variants.py`
as `GenerationVariant(cl_spd_configs=..., noise_configs=...)` entries in the `_VARIANTS`
registry. Use `rlvr.generation_variants.list_variants()` for the full list.

The default `rsft_v2` has **6 guided cl_spd slots** — `CL5_SPD5_det`,
`CL8_SPD8_str13_n0825` (stretch 1.3), `CL6_SPD6_str11_n0515` (stretch 1.1),
`CL5_SPD5_noisy`, `CL7_SPD7_str14_n0820` (stretch 1.4), `CL10_SPD10_noisy` — plus
**9 pure-noise slots** sweeping ranges 0.1→5.0 (`noise_n0103`, `n0306`, `n0510`,
`n0515`, `n0818`, `n1025`, `n1530`, `n2040`, `n3050`). No random-CL pool.
Use `rsft_v2_legacy` for the previous slot composition (2 fixed-noise + 7 random-CL)
or `default` for the pre-variant layout (8 cl_spd + 7 random).

### Rank Analytics

Per-epoch instrumentation that tracks **which generation slot wins rank #1** for each scene
and **which reward component drives the win**. Implemented in `rlvr/rank_analytics.py`.
Outputs `rank_analytics_epoch_NNN.json` per epoch and a final `rank_analytics_summary.json`
in the run dir. Visualize with `python -m rlvr.autoresearch.tools.viz_rank_analytics --run_dir <dir>`.

Use to identify redundant/dead generation slots that never produce winners — those can be
swapped for new guidance variants. Adds ~no overhead (the reward breakdowns are already
computed during scoring; analytics just retain and aggregate them).

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
- Collision gate: ego collides with neighbor vehicle (rear-end collisions permanently excluded: once an NPC overlaps ego from behind at any timestep, that NPC is excluded from all future collision checks)
- Road border gate (`rb_gate_enabled`): ego perimeter (80 sample points) crosses road border (within `rb_cross_thresh`, default 0.20m)
- Red light gate: ego runs red light

**Quality score** (soft, weighted sum):
`quality = w_progress * progress + w_safety * safety + w_smooth * smoothness + w_centerline * centerline + ttc_bonus - rb_penalty - lane_penalty`

Road border proximity penalties (`rb_near_scale`, `rb_wide_scale`, `rb_cont_scale`) subtract from quality
even when the ego doesn't cross the border, penalizing trajectories that get close. Configurable
distance thresholds: `rb_cross_thresh`/`rb_near_thresh`/`rb_wide_thresh`/`rb_cont_thresh` (defaults: 0.20/0.45/0.60/1.00m).
`rb_penalty_mode`: `"frac"` (fraction of timesteps in violation) or `"survival"` (first-violation time-decay — early violations penalized more than late ones).

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

**Smoothness (M)**: Negative mean absolute jerk computed via Savitzky-Golay conv1d kernel
(window=11, poly=3, deriv=3). Edge artifacts trimmed by pad//2 from each side. Raw finite
differences amplify noise ~1000x on 10Hz data; SG gives physically correct jerk values.

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

Additionally penalizes acceleration violations (fraction of steps where `|accel| > max_accel`)
and lateral acceleration violations using Savitzky-Golay filtered derivatives (curvature-based,
accurate — replaces the old noisy double-finite-diff method that inflated values ~5x).

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

`compute_lane_departure_penalty()` detects when the ego vehicle leaves its lane using a
**signed-distance gate** (polygon containment is no longer used):

1. Find K=12 nearest lanes by min centerline distance
2. Build lane polygons from left/right boundary offsets, classify each boundary segment as
   outer (road-edge) vs shared (between lanes) via a midpoint nudge classifier, and compute an
   outward normal for each outer segment
3. For each of 36 ego perimeter sample points at each timestep, compute the **signed**
   point-to-segment distance against outer segments. Convention: **positive = outside the
   lane (along the outward normal), negative = inside**. Uses only unclamped projections to
   avoid endpoint artifacts at junction gaps
4. Points that fall inside an intersection-area polygon (from the NPZ `polygons` field, type
   `POLYGON_TYPE_INTERSECTION_AREA`) are forced to signed = -100 (deep inside). This
   suppresses false crossings at intersection mouths where lane polygons don't tile cleanly
5. A trajectory **crosses** if the per-timestep maximum signed distance (the most-outside
   perimeter point) ever exceeds `-lane_cross_thresh`. Default 0.20m matches
   `rb_cross_thresh`. **Buffer-from-inside semantics**: a higher threshold = stricter gate
   (wider buffer); the default fires when any perimeter point comes within 20cm of the
   boundary or beyond it
6. Soft near/wide/cont penalties (when their `*_scale` weights are non-zero) use the
   per-timestep minimum **unsigned** point-to-segment distance through these thresholds:
   `lane_near_thresh` (0.25m), `lane_wide_thresh` (0.40m), `lane_cont_thresh` (0.80m).
   These fire only on timesteps where the trajectory is still inside the lane (not on
   crossing timesteps)
7. Returns `(crossing_gate, near_frac, wide_frac, lane_crossing_steps, cont_penalty)`

Enabled via `enable_lane_departure: true` in config. Can be used as:
- **Soft penalty** via `lane_near_scale`, `lane_wide_scale`, `lane_cont_scale`
- **Hard gate** via `lane_gate_enabled: true` (lane crossing zeros the reward in gate mode)
- **Survival terminal event** when `reward_mode: "survival"` + `enable_lane_departure: true`,
  lane departure reduces `survival_frac` proportionally to when it occurs

### Kinematic Feasibility Gate

`compute_kinematic_gate()` filters trajectories whose commanded motion violates the
vehicle's bicycle-model constraints:

- **Absolute yaw rate** must stay below `max_yaw_rate` (default 1.0 rad/s)
- **Bicycle-model curvature** must stay below `κ_max = kinematic_margin · tan(max_steer) / wheelbase`
  (defaults: `max_steer=0.64` rad, `kinematic_margin=2.5`). Yaw and speed are
  Savitzky-Golay smoothed before curvature estimation

Returns a per-trajectory **binary** gate (0 = any timestep violated, 1 = all clean).
Applied as a hard flooring multiplier on `totals` **after** survival/gate aggregation, so a
single kinematic violation floors the entire trajectory reward regardless of `reward_mode`.

### Baseline-anchored Underprogress

When `underprogress_reference = "baseline"` (vs the default `"det"`), the underprogress
penalty compares the trajectory's path length to a **frozen** LoRA-less deterministic path
(injected as `baseline_path_len` per scene by the trainer). Unlike the `"det"` reference,
which adapts as the model collapses to shorter paths, the baseline anchor keeps firing and
prevents progress collapse during training. Used by the J6 overnight-sweep champion recipe.

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
