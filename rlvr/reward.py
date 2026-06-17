"""Rule-based trajectory reward for GRPO training.

Computes R = w_safety * S + w_progress * P + w_smooth * M + w_feasibility * F + w_centerline * C
using log-replay data. Reuses ego bbox construction and lane/neighbor penalty
functions from diffusion_planner.loss for proper vehicle-footprint-aware checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Issue #130: the raw subscore / geometry / config code now lives in the
# neutral planner_metrics workspace library. This module keeps the reward
# *shaping* (RewardBreakdown, compute_reward_batch, GRPO advantages) and
# re-exports every moved symbol so `from rlvr.reward import ...` keeps working.
# ---------------------------------------------------------------------------
from planner_metrics.aggregate import compute_subscores_batch  # noqa: F401
from planner_metrics.config import RewardConfig  # noqa: F401  (re-export)
from planner_metrics.geometry import *  # noqa: F401,F403
from planner_metrics.subscores import *  # noqa: F401,F403


@dataclass
class RewardBreakdown:
    safety: float
    progress: float
    smoothness: float
    feasibility: float
    centerline: float
    red_light: float
    total: float
    collision_step: int | None
    off_road_fraction: float
    rb_crossing: bool = False
    rb_near_penalty: float = 0.0  # near-zone penalty (frac or survival-style depending on mode)
    rb_wide_penalty: float = 0.0  # wide-zone penalty (frac or survival-style depending on mode)
    rb_min_dist: float = 99.0  # min ego-perimeter-to-border distance (metres, skip t=0)
    lane_crossing: bool = False
    lane_near_frac: float = 0.0
    lane_wide_frac: float = 0.0
    # Static-collision (stopped-neighbor clearance) diagnostics. Zero/False
    # when static_collision_enabled=False.
    static_crossing: bool = False
    sc_near_penalty: float = 0.0
    sc_wide_penalty: float = 0.0
    sc_cont_penalty: float = 0.0
    sc_min_dist: float = 99.0  # min OBB clearance to any stopped neighbor (t>=1, ego moving)
    sc_n_stopped: int = 0  # how many stopped neighbors were found in the scene
    # Kinematic feasibility violation (yaw rate + bicycle-model curvature).
    # When True, the trajectory is INFEASIBLE and compute_reward_batch floors
    # ``total`` to the offroad floor. Convention matches the other gate
    # booleans on this dataclass (rb_crossing, lane_crossing, static_crossing):
    # True = violation occurred.
    kinematic_violated: bool = False


@torch.no_grad()
def compute_reward_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig = RewardConfig(),
) -> list[RewardBreakdown]:
    """Compute reward breakdowns for N trajectories in a single batched pass.

    Thin wrapper: the raw subscores (input marshalling + per-subscore
    computation) come from ``compute_subscores_batch``; reward shaping (weights,
    hard gates, survival/gate aggregation) is applied by ``_shape_reward``.
    Output is byte-identical to the pre-split implementation (pinned by the
    golden tests).

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        data: Observation dict from load_npz_data (with batch dim).
        config: RewardConfig with component weights.

    Returns:
        List of N RewardBreakdown instances.
    """
    return _shape_reward(compute_subscores_batch(ego_trajs, data, config), ego_trajs, data, config)


@torch.no_grad()
def _shape_reward(
    subs: dict,
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig = RewardConfig(),
) -> list[RewardBreakdown]:
    """Apply reward shaping (weights, hard gates, survival/gate aggregation) to the
    raw subscores from ``compute_subscores_batch`` and build the RewardBreakdown
    list. RLVR-specific; the raw subscore computation is neutral (lives in
    ``planner_metrics``)."""
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device

    safety_scores = subs["safety"]
    collision_steps = subs["collision_step"]
    red_light_scores = subs["red_light"]
    progress_scores = subs["progress"]
    smoothness_scores = subs["comfort"]
    centerline_scores = subs["centerline"]
    feasibility_scores = subs["feasibility"]
    off_road_fractions = subs["off_road_fraction"]
    ttc_scores = subs["ttc"]
    kinematic_gate = subs["kinematic_gate"]
    rb_crossing_gate = subs["rb_crossing_gate"]
    rb_near_pen = subs["rb_near_penalty"]
    rb_wide_pen = subs["rb_wide_penalty"]
    rb_cont_penalty = subs["rb_cont_penalty"]
    rb_crossing_steps = subs["rb_crossing_steps"]
    rb_min_dist_t = subs["rb_min_dist"]
    lane_crossing_gate = subs["lane_crossing_gate"]
    lane_near_frac = subs["lane_near_frac"]
    lane_wide_frac = subs["lane_wide_frac"]
    lane_cont_penalty = subs["lane_cont_penalty"]
    lane_crossing_steps = subs["lane_crossing_steps"]
    sc_crossing_gate = subs["sc_crossing_gate"]
    sc_near_pen = subs["sc_near_penalty"]
    sc_wide_pen = subs["sc_wide_penalty"]
    sc_cont_pen = subs["sc_cont_penalty"]
    sc_crossing_steps = subs["sc_crossing_steps"]
    sc_min_dist_scalar = subs["sc_min_dist"]
    sc_n_stopped_scene = int(subs["sc_n_stopped"][0].item()) if N > 0 else 0

    # NAVSIM PDMS-style multiplicative reward aggregation.
    # Safety gates: binary 0/1 multipliers. If any gate is 0, total is 0.
    # This prevents reward hacking (e.g. stopping to avoid offroad penalty)
    # because stopped trajectories get progress=0 → total=0, same as offroad.
    # Only trajectories that drive AND stay on-road get positive reward.

    # Gate 1: No collision (binary: 1 if no collision, 0 if collision)
    has_collision = torch.tensor(
        [1.0 if cs is not None else 0.0 for cs in collision_steps],
        device=device,
    )
    collision_gate = 1.0 - has_collision  # (N,)

    # Gate 2: Drivable area compliance — steep sigmoid
    # Near-binary but allows ranking of partially-offroad trajectories.
    # Hard binary (offroad>0 → 0) was tested in exp028 but gave worse
    # prob offroad (5% vs 0.8% with sigmoid in exp023) because it kills
    # the ranking signal for scenes where ALL trajectories have some offroad.
    # Binary gate: ANY offroad → gate = 0. No partial credit.
    # Polygon drivable_gate removed — road border crossing gate handles offroad detection
    drivable_gate = torch.ones(N, device=device)  # always passes (polygon check disabled)

    # Gate 3: Red light compliance
    has_red_light_violation = (red_light_scores < -0.5).float()
    red_light_gate = 1.0 - has_red_light_violation  # (N,)

    # Multiplicative safety product (hard gates only)
    # TTC is included in quality_score as a soft penalty instead of a gate,
    # because at intersections many good trajectories pass near NPCs.
    # Gate 4: Road border compliance (crossing = instant fail)
    # Road border perimeter check is the primary offroad detection (v4).
    # Lane polygon drivable_gate is kept as a soft penalty only, not a hard gate,
    # since lane polygons can disagree with road borders at intersection corners.
    safety_product = collision_gate * red_light_gate  # (N,)
    if config.rb_gate_enabled:
        safety_product = safety_product * rb_crossing_gate
    if config.lane_gate_enabled:
        safety_product = safety_product * lane_crossing_gate
    if config.static_collision_enabled and config.sc_gate_enabled:
        safety_product = safety_product * sc_crossing_gate

    # Weighted quality metrics (only matter when safety gates pass)
    # Progress is the primary positive signal. Smoothness/centerline are penalties.
    # Safety score includes proximity penalty to NPCs (closer = more negative).
    clamped_progress = progress_scores.clamp(min=0)

    # Progress-related penalties (overprogress, stopped, underprogress) are floors,
    # not progress rewards — they must apply even when w_progress=0. Accumulate
    # them into `progress_penalty` and subtract from quality_score directly,
    # bypassing the w_progress multiplier.
    progress_penalty = torch.zeros(N, device=device)

    # Normalize progress as percentage of GT path length, then apply
    # overprogress/underprogress/stopped penalties.
    # This ensures a 10m path on a 12m GT scene and a 10m path on a 22m GT scene
    # get different progress scores (83% vs 45%).
    if config.enable_overprogress and "ego_agent_future" in data:
        gt_future = data["ego_agent_future"]
        if gt_future.dim() == 3:
            gt_future = gt_future[0]  # (T_gt, 3)
        gt_xy = gt_future[:, :2]
        gt_valid = gt_xy.abs().sum(dim=-1) > 0.1
        # ALWAYS compute model path lengths — they are used for stopped and
        # underprogress penalties which must fire whether or not GT is
        # present (synthetic-data RSFT passes zero GT; without this the
        # penalties silently no-op and the model collapses path).
        model_path_lens = torch.diff(ego_trajs[:, :, :2], dim=1).norm(dim=-1).sum(dim=-1)  # (N,)
        baseline_path_len_scalar = None
        if "baseline_path_len" in data:
            bpl_t = torch.as_tensor(
                data["baseline_path_len"],
                device=device,
                dtype=torch.float32,
            ).reshape(())
            baseline_path_len_scalar = float(bpl_t.clamp(min=1e-3).item())

        if gt_valid.sum() >= 10:
            gt_path_len = torch.diff(gt_xy[gt_valid], dim=0).norm(dim=-1).sum()

            # Normalize progress to [0, 1] as fraction of GT, capped at margin.
            # 100% GT = 1.0 (max), >margin% GT = capped + penalized.
            progress_frac = (clamped_progress / gt_path_len.clamp(min=1e-3)).clamp(
                max=config.overprogress_margin
            )
            clamped_progress = progress_frac * config.progress_norm_scale

            # Compute path ratio for symmetric over/under progress penalties.
            # Both use the same ratio-based method: penalty * |deviation from threshold|.
            path_ratio = model_path_lens / gt_path_len.clamp(min=1e-3)

            # Overprogress: penalize path exceeding margin × GT (ratio-based).
            # NOTE: Changed from meter-based (pre-April 2026) to ratio-based.
            # Old: penalty * relu(path_meters - cap_meters). New: penalty * relu(ratio - margin).
            # Configs must use ratio-scale penalties (e.g. 100.0), not meter-scale (e.g. 0.3).
            # E.g., margin=1.0, penalty=100: at 1.5x GT → 100*(1.5-1.0)=50 penalty.
            overprogress = torch.relu(path_ratio - config.overprogress_margin)
            progress_penalty = progress_penalty + config.overprogress_penalty * overprogress

        # Stopped penalty: fires on any trajectory that barely moves,
        # whenever an anchor scene "should have moved" — GT is the
        # canonical anchor when present, else baseline_path_len
        # (underprogress_reference). Without either, we can't distinguish
        # a legitimate stop (red light) from reward-hacking collapse.
        anchor_len: float | None = None
        if gt_valid.sum() >= 10:
            anchor_len = float(torch.diff(gt_xy[gt_valid], dim=0).norm(dim=-1).sum().item())
        elif baseline_path_len_scalar is not None:
            anchor_len = baseline_path_len_scalar
        if config.stopped_penalty > 0 and anchor_len is not None and anchor_len > 5.0:
            is_stopped = (model_path_lens < 1.0).float()
            progress_penalty = progress_penalty + config.stopped_penalty * is_stopped

        # Underprogress: penalize trajectories shorter than the reference path.
        # When ``underprogress_reference="baseline"`` AND ``data["baseline_path_len"]``
        # is present, ALWAYS fire (even at N=1, and even when GT is absent —
        # the whole point of the baseline anchor is it doesn't depend on the
        # current rollout). The N>1 guard only makes sense for the legacy
        # "det" reference where traj[0] is the reference and ratio ≡ 1.0.
        _have_baseline_ref = (
            config.underprogress_reference == "baseline" and baseline_path_len_scalar is not None
        )
        if config.underprogress_penalty > 0 and (N > 1 or _have_baseline_ref):
            # Reference selection:
            #   "det"      — path of the deterministic traj (traj[0]). Adapts to
            #                current model, but can collapse to short when model
            #                starts producing short det trajs.
            #   "baseline" — baseline LoRA-less det path length, passed via
            #                `data["baseline_path_len"]` (a scalar tensor). Frozen
            #                anchor that doesn't collapse with training.
            if config.underprogress_reference == "baseline" and "baseline_path_len" in data:
                # Accept tensor / numpy scalar / Python float — callers may inject metadata
                # in any of these forms when wiring custom data dicts.
                ref_path_len = torch.as_tensor(
                    data["baseline_path_len"],
                    device=device,
                    dtype=torch.float32,
                )
                if ref_path_len.numel() != 1:
                    raise ValueError(
                        "data['baseline_path_len'] must be a scalar value, got shape "
                        f"{tuple(ref_path_len.shape)}"
                    )
                ref_path_len = ref_path_len.reshape(()).clamp(min=1e-3)
            else:
                ref_path_len = model_path_lens[0].clamp(min=1e-3)
            ratio = model_path_lens / ref_path_len
            underprogress = torch.relu(config.underprogress_threshold - ratio.clamp(max=1.0))
            progress_penalty = progress_penalty + config.underprogress_penalty * underprogress

    # TTC as quality bonus
    ttc_bonus = config.w_safety * (ttc_scores - 0.5) * 2

    # Road border proximity penalties (soft, applied even when on-road)
    # Thresholds configurable via config.rb_near_thresh / rb_wide_thresh
    rb_penalty = (
        config.rb_near_scale * rb_near_pen
        + config.rb_wide_scale * rb_wide_pen
        + config.rb_cont_scale * rb_cont_penalty
    )

    # Lane departure proximity penalties
    lane_penalty = (
        config.lane_near_scale * lane_near_frac
        + config.lane_wide_scale * lane_wide_frac
        + config.lane_cont_scale * lane_cont_penalty
    )

    # Static-collision proximity penalties (stopped neighbors).
    sc_penalty = (
        config.sc_near_scale * sc_near_pen
        + config.sc_wide_scale * sc_wide_pen
        + config.sc_cont_scale * sc_cont_pen
    )

    # Penalty magnitude preserves legacy behavior for configs with w_progress >= 1
    # (historical default range: w_progress ∈ {2.0, 7.0}), where the old code
    # effectively multiplied the penalty by w_progress via the clamped_progress
    # sum. For w_progress < 1 we floor at 1.0 so penalties still fire on
    # CL-only / reward-sculpted configs (w_progress=0 was the original bug).
    penalty_mult = max(float(config.w_progress), 1.0)
    quality_score = (
        config.w_progress * clamped_progress
        + config.w_safety * safety_scores
        + config.w_smooth * smoothness_scores
        + config.w_centerline * centerline_scores
        + ttc_bonus
        - rb_penalty
        - lane_penalty
        - sc_penalty
        - penalty_mult * progress_penalty
    )

    _OFFROAD_FLOOR = -50.0

    if config.reward_mode == "survival":
        # PlannerRFT-style survival reward: proportional credit based on how
        # long the trajectory survives before the first terminal event.
        # survival_frac = first_terminal_step / T. A crash at t=60/80 gets 75%
        # of quality_score. This prevents gradient death on hard scenes where
        # all trajectories fail — later crashes still rank higher.
        survival_frac = torch.ones(N, device=device)
        for i in range(N):
            first_terminal = T  # no failure → full survival
            if collision_steps[i] is not None:
                first_terminal = min(first_terminal, collision_steps[i])
            if config.rb_gate_enabled and rb_crossing_steps[i] is not None:
                first_terminal = min(first_terminal, rb_crossing_steps[i])
            if config.enable_lane_departure and lane_crossing_steps[i] is not None:
                first_terminal = min(first_terminal, lane_crossing_steps[i])
            if (
                config.static_collision_enabled
                and config.sc_gate_enabled
                and sc_crossing_steps[i] is not None
            ):
                first_terminal = min(first_terminal, sc_crossing_steps[i])
            survival_frac[i] = max(first_terminal, 1) / T  # at least 1/T to avoid 0

        # Blend: survived portion gets quality, failed portion gets floor.
        # Red light violations still use a hard gate on top of survival —
        # red light doesn't have a per-timestep failure point, so we apply
        # it as a binary multiplier like in gate mode.
        totals = survival_frac * quality_score + (1.0 - survival_frac) * _OFFROAD_FLOOR
        totals = totals * red_light_gate + (1.0 - red_light_gate) * _OFFROAD_FLOOR
    else:
        # Default "gate" mode: binary safety gates × quality.
        # Any terminal event → full floor penalty regardless of when it happens.
        totals = safety_product * quality_score + (1.0 - safety_product) * _OFFROAD_FLOOR

    # Kinematic feasibility hard gate: trajectories violating yaw-rate or
    # bicycle-model curvature bounds get floored. Applied after survival/gate
    # aggregation so it overrides any otherwise-positive reward.
    totals = totals * kinematic_gate + (1.0 - kinematic_gate) * _OFFROAD_FLOOR

    # Also compute additive total for backward compat in breakdown
    on_road_factor = 1.0 - off_road_fractions
    adjusted_progress = progress_scores * on_road_factor

    results: list[RewardBreakdown] = []
    for i in range(N):
        results.append(
            RewardBreakdown(
                safety=float(safety_scores[i]),
                progress=float(adjusted_progress[i]),
                smoothness=float(smoothness_scores[i]),
                feasibility=float(feasibility_scores[i]),
                centerline=float(centerline_scores[i]),
                red_light=float(red_light_scores[i]),
                total=float(totals[i]),
                collision_step=collision_steps[i],
                off_road_fraction=float(
                    off_road_fractions[i]
                ),  # always 0 (polygon disabled); use rb_crossing/rb_near_penalty instead
                rb_crossing=bool(rb_crossing_gate[i] < 0.5),
                rb_near_penalty=float(rb_near_pen[i]),
                rb_wide_penalty=float(rb_wide_pen[i]),
                rb_min_dist=float(rb_min_dist_t[i].item()),
                lane_crossing=bool(lane_crossing_gate[i] < 0.5),
                lane_near_frac=float(lane_near_frac[i]),
                lane_wide_frac=float(lane_wide_frac[i]),
                static_crossing=bool(sc_crossing_gate[i] < 0.5),
                sc_near_penalty=float(sc_near_pen[i]),
                sc_wide_penalty=float(sc_wide_pen[i]),
                sc_cont_penalty=float(sc_cont_pen[i]),
                sc_min_dist=float(sc_min_dist_scalar[i].item()),
                sc_n_stopped=sc_n_stopped_scene,
                kinematic_violated=bool(kinematic_gate[i] < 0.5),
            )
        )

    return results


def compute_reward(
    ego_traj: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig = RewardConfig(),
) -> RewardBreakdown:
    """Single-trajectory convenience wrapper around compute_reward_batch."""
    return compute_reward_batch(ego_traj.unsqueeze(0), data, config)[0]


def compute_group_advantages(
    rewards: list[RewardBreakdown],
    epsilon: float = 1e-8,
    mode: str = "normalized",
    fixed_scale: float = 10.0,
) -> np.ndarray:
    """Compute GRPO-style group-relative advantages.

    Args:
        rewards: List of RewardBreakdown for each trajectory in the group.
        epsilon: Small constant for numerical stability.
        mode: Advantage computation mode:
            "normalized": Standard GRPO (mean=0, std=1 per group).
            "vd_grpo": Variance-Decoupled GRPO (center only, fixed scale).
                Preserves absolute magnitude of negative rewards across groups.
            "raw": Centered advantages without std normalization. Uses
                fixed_scale as denominator. If all trajectories in a group
                are bad (e.g., all leave lane), all get negative advantages
                instead of half getting positive weight.
            "positive_only": Like "normalized" but clips negative advantages
                to zero. Only updates on trajectories that are better than
                the group mean.
        fixed_scale: Denominator for vd_grpo and raw modes.

    Returns:
        (G,) array of advantages.
    """
    totals = np.array([r.total for r in rewards])
    mean = totals.mean()

    if mode == "vd_grpo":
        if fixed_scale <= 0.0:
            raise ValueError(f"advantage_fixed_scale must be positive, got {fixed_scale}")
        return (totals - mean) / max(fixed_scale, epsilon)
    elif mode == "normalized":
        std = totals.std()
        if std < epsilon:
            return np.zeros(len(rewards))
        return (totals - mean) / (std + epsilon)
    elif mode == "raw":
        # Centered advantages without per-group std normalization.
        # If all K trajectories are bad, all get negative advantages.
        # This prevents normalized advantages from giving half of an
        # all-bad group positive weight.
        if fixed_scale <= 0.0:
            raise ValueError(f"advantage_fixed_scale must be positive, got {fixed_scale}")
        return (totals - mean) / max(fixed_scale, epsilon)
    elif mode == "absolute":
        # No centering, no normalization. Advantage = total / fixed_scale.
        # Positive reward → positive advantage, negative reward → negative advantage.
        # A group where all trajs score -30 gets ALL negative advantages.
        # Only trajs with positive absolute reward get reinforced.
        if fixed_scale <= 0.0:
            raise ValueError(f"advantage_fixed_scale must be positive, got {fixed_scale}")
        return totals / max(fixed_scale, epsilon)
    elif mode == "softmax":
        # Softmax-weighted advantages. Temperature = fixed_scale.
        # Rank 1 gets disproportionately strong signal (~0.9), others decay sharply.
        # Low temperature (5) = very sharp (rank 1 dominates).
        # High temperature (20) = softer (more spread across top trajs).
        # Centered so mean≈0 for stable GRPO training.
        temp = max(fixed_scale, epsilon)
        logits = totals / temp
        logits = logits - logits.max()  # numerical stability
        exp_logits = np.exp(logits)
        weights = exp_logits / exp_logits.sum()
        # Center and scale: mean=0, max≈1
        advantages = (weights - weights.mean()) / max(weights.max(), epsilon)
        return advantages
    elif mode == "positive_only":
        # Standard normalization but clip negatives to zero.
        # Only reinforces trajectories better than the group mean.
        std = totals.std()
        if std < epsilon:
            return np.zeros(len(rewards))
        advantages = (totals - mean) / (std + epsilon)
        return np.maximum(advantages, 0.0)
    elif mode == "ddv2":
        # DiffusionDriveV2 Inter-Anchor Truncated GRPO (arXiv:2512.07745, Eq. 10):
        # 1. Standard intra-group normalization
        # 2. Clip negative advantages to 0 (only reinforce improvements over group mean)
        # 3. Hard -1 penalty for safety violations (collision, off-road, lane departure)
        # Extension vs paper: paper only penalizes collisions; we also penalize
        # road-border crossings and lane departures. No inter-anchor distinction
        # since we don't use DDV2's multi-anchor GMM architecture.
        std = totals.std()
        if std < epsilon:
            advantages = np.zeros(len(rewards))
        else:
            advantages = (totals - mean) / (std + epsilon)
        # Clip negative to 0
        advantages = np.maximum(advantages, 0.0)
        # Hard -1 for safety violations
        for i, rb in enumerate(rewards):
            if rb.collision_step is not None or rb.rb_crossing or rb.lane_crossing:
                advantages[i] = -1.0
        return advantages
    else:
        raise ValueError(
            f"Unknown advantage mode: {mode!r}. "
            f"Expected 'normalized', 'vd_grpo', 'raw', 'absolute', 'softmax', "
            f"'positive_only', or 'ddv2'."
        )
