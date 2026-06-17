"""Configuration thresholds for the ``diffusion_planner.metrics`` subscores."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    w_safety: float = 5.0
    w_progress: float = 2.0
    w_smooth: float = 0.5
    w_feasibility: float = 5.0
    w_centerline: float = 5.0
    # Centerline usage mode:
    #   "baselink" (default): lane_usage = |baselink_lat| / side_hw —
    #       perfectly centered = 0; readings are directly interpretable as
    #       "rear axle is X fraction of the way to the lane edge".
    #   "body" (DEPRECATED): lane_usage = (|baselink_lat| + ego_half_w) /
    #       side_hw. Adds half-vehicle-width to the offset so a centered
    #       wide ego already reads non-zero. Easy to misread as lateral
    #       metres when it is not. Kept for backward compatibility with
    #       configs from before 2026-04-27 — emits a DeprecationWarning.
    centerline_usage_mode: str = "baselink"
    # Centerline time-weight floor. The per-step centerline penalty is averaged
    # with weights torch.linspace(1.0, centerline_time_weight_min, T). The default
    # 0.3 matches the historical behavior (late timesteps count 30% of early).
    # Set to 1.0 for a flat (uniform) time-average — recommended when late-curve
    # lane-following matters as much as early, and the training signal is being
    # compressed by the decay.
    centerline_time_weight_min: float = 0.3
    collision_penalty: float = -10.0
    red_light_penalty: float = -10.0
    max_accel: float = 8.0  # m/s^2
    dt: float = 0.1  # 10 Hz

    # Road border penalty scales and thresholds
    rb_near_scale: float = 3.0
    rb_wide_scale: float = 0.2
    rb_cont_scale: float = 0.0  # continuous penalty (0=disabled)
    rb_gate_enabled: bool = True  # if True, rb crossing is a hard safety gate
    rb_penalty_mode: str = (
        "frac"  # "frac" = fraction of timesteps, "survival" = first-violation time-decay
    )
    rb_cross_thresh: float = 0.20  # metres — ego perimeter within this = crossing
    rb_near_thresh: float = 0.45  # metres — near zone boundary (+20cm vs lane)
    rb_wide_thresh: float = 0.60  # metres — wide zone boundary (+20cm vs lane)
    rb_cont_thresh: float = 1.00  # metres — continuous penalty max distance (+20cm vs lane)

    # Lane departure penalty scales and thresholds
    enable_lane_departure: bool = False
    lane_gate_enabled: bool = False  # if True, lane crossing kills reward
    lane_near_scale: float = 3.0
    lane_wide_scale: float = 0.2
    lane_cont_scale: float = 0.0
    lane_cross_thresh: float = (
        0.20  # metres — signed distance threshold for crossing (matches rb_cross_thresh)
    )
    lane_near_thresh: float = 0.25  # metres — near zone boundary
    lane_wide_thresh: float = 0.40  # metres — wide zone boundary
    lane_cont_thresh: float = 0.80  # metres — continuous penalty max distance

    # Static-collision (stopped-neighbor clearance) penalty. Same staged
    # pattern as rb_*: gate + near/wide/cont zones. Off by default — enabling
    # changes training reward math, so set `static_collision_enabled=True` and
    # non-zero scales explicitly.
    static_collision_enabled: bool = False
    sc_gate_enabled: bool = (
        False  # hard terminator if any predicted step overlaps a stopped neighbor
    )
    sc_penalty_mode: str = "frac"  # "frac" or "survival" (matches rb_penalty_mode semantics)
    sc_near_scale: float = 0.0
    sc_wide_scale: float = 0.0
    sc_cont_scale: float = 0.0
    sc_cross_thresh: float = 0.2  # clearance below this (metres) = crossing. 0.2 m matches the "visually touching" threshold observed on the bigcurve resim — below that, SAT signed distance is slightly positive but the boxes are in practice a collision.
    sc_near_thresh: float = 0.4
    sc_wide_thresh: float = 0.7
    sc_cont_thresh: float = 1.0
    sc_neighbor_vel_thresh: float = 0.1  # m/s — |v0| below this counts as stationary
    sc_neighbor_disp_thresh: float = (
        0.5  # m — max displacement across GT future below this counts as stationary
    )
    sc_ego_min_speed: float = (
        1.0  # m/s — timesteps below this are not scored (matches collision suppression)
    )

    # Lateral acceleration penalty
    max_lat_accel: float = 2.0  # m/s^2
    lat_accel_scale: float = 3.0

    # Yaw-rate feasibility gate (absolute cap). Thresholds chosen so GT
    # trajectories pass (GT peaks ≈0.5 rad/s on tight human turns) and only
    # clearly unphysical predictions (e.g. pivot-in-place) fail.
    max_yaw_rate: float = 1.0  # rad/s  (2× GT peak)

    # Bicycle-model kinematic feasibility gate. κ_max = tan(max_steer)/wheelbase.
    # Wheelbase is read from ego_shape[0] per scene; max_steer is configured below.
    # Effective curvature bound: kinematic_margin × tan(max_steer) / wheelbase.
    # Margin absorbs SG finite-differencing noise and tight human driving.
    max_steer: float = 0.64  # rad — bicycle-model steering range
    kinematic_margin: float = 2.5  # multiplier over physical bicycle-model bound

    # Overprogress: cap progress at GT path × margin, penalize excess
    enable_overprogress: bool = False
    overprogress_margin: float = 1.1
    overprogress_penalty: float = 0.3
    stopped_penalty: float = 50.0  # applied in compute_reward_batch progress section

    # Underprogress: penalize trajectories that drive much less than a reference.
    underprogress_penalty: float = (
        0.0  # scale (0=disabled). Penalty = scale * max(0, threshold - ratio)
    )
    underprogress_threshold: float = 0.5  # fire when ratio < threshold
    # Reference for underprogress:
    #   "baseline" — (default) frozen LoRA-less baseline det path, passed via
    #                data["baseline_path_len"]. Anchors ratio regardless of
    #                training drift — recommended.
    #   "det"      — deterministic traj (traj[0]) path length. Adaptive but can
    #                collapse when model output itself collapses (path shrinks
    #                while threshold follows it → penalty never fires).
    underprogress_reference: str = "baseline"

    # Progress normalization scale: when enable_overprogress=True, progress is
    # normalized to [0, 1] as fraction of GT, then multiplied by this scale.
    # 100% GT progress → progress_norm_scale points. Default 20.
    progress_norm_scale: float = 20.0

    # Reward aggregation mode:
    # "gate" (default): binary safety gates × quality. Any terminal event → floor (-50).
    # "survival" (PlannerRFT): proportional credit based on how long the trajectory
    #   survives before the first terminal event. A crash at t=60/80 gets 75% of the
    #   quality score. Prevents gradient death on hard scenes where all trajectories fail.
    reward_mode: str = "gate"


__all__ = [
    "RewardConfig",
]
