"""Rule-based trajectory reward for GRPO training.

Computes R = w_safety * S + w_progress * P + w_smooth * M + w_feasibility * F + w_centerline * C
using log-replay data. Reuses ego bbox construction and lane/neighbor penalty
functions from diffusion_planner.loss for proper vehicle-footprint-aware checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from diffusion_planner.model.guidance.collision import (
    batch_signed_distance_rect,
    center_rect_to_points,
)


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


# ---------------------------------------------------------------------------
# Ego bbox construction (adapted from loss.compute_safety_penalty)
# ---------------------------------------------------------------------------


def _build_ego_bbox_corners(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
) -> torch.Tensor:
    """Build oriented bounding box corners for ego trajectories.

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        ego_shape: (3,) wheel_base, length, width.

    Returns:
        (N, T, 4, 2) corner points in global frame.
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    dtype = ego_trajs.dtype

    heading = ego_trajs[..., 2:4]  # (N, T, 2)
    heading_unit = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ego_xy = ego_trajs[..., :2]

    wheel_base = ego_shape[0]
    ego_length = ego_shape[1]
    ego_width = ego_shape[2]

    cog_to_rear = 0.5 * wheel_base
    ego_center_xy = ego_xy + heading_unit * cog_to_rear

    half_length = ego_length / 2.0
    half_width = ego_width / 2.0
    half_sizes = torch.tensor([half_length, half_width], device=device, dtype=dtype).expand(N, T, 2)

    corner_signs = torch.tensor(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    local_corners = corner_signs[None, None, :, :] * half_sizes[:, :, None, :]  # (N, T, 4, 2)

    rot = torch.stack(
        [
            heading_unit[..., 0],
            -heading_unit[..., 1],
            heading_unit[..., 1],
            heading_unit[..., 0],
        ],
        dim=-1,
    ).reshape(N, T, 2, 2)

    rotated_corners = torch.einsum("btij,btkj->btki", rot, local_corners)
    return ego_center_xy[:, :, None, :] + rotated_corners  # (N, T, 4, 2)


# ---------------------------------------------------------------------------
# Safety: batched neighbor collision using SAT from loss.py
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_safety_score_batch(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    neighbor_futures: torch.Tensor,
    neighbor_shapes: torch.Tensor,
    neighbor_valid: torch.Tensor,
    config: RewardConfig,
) -> tuple[torch.Tensor, list[int | None]]:
    """Batched ego-NPC collision check using oriented bounding boxes.

    Filters out rear-end collisions where an NPC hits the ego from behind,
    since the ego cannot control the behavior of following vehicles.
    A collision is only counted if the NPC center is ahead of or beside the
    ego (dot product of ego heading with ego→NPC vector >= 0).

    Known limitation: this filter may miss ego-at-fault collisions during
    lane changes where the ego merges into a vehicle that is slightly behind.
    The heading-based check treats "behind" as the full rear hemisphere.

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        ego_shape: (3,) wheel_base, length, width.
        neighbor_futures: (N_nb, T, 4) GT NPC future trajectories.
        neighbor_shapes: (N_nb, 2) width, length per NPC.
        neighbor_valid: (N_nb, T) bool mask of valid neighbor timesteps.
        config: RewardConfig.

    Returns:
        scores: (N,) tensor -- collision_penalty minus proximity penalty on collision,
            or just negative proximity penalty if no collision. Proximity penalty is
            the mean intrusion depth when ego passes within 1m of any NPC.
        collision_steps: list of length N -- timestep of first collision or None.
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    N_nb = neighbor_futures.shape[0]

    if N_nb == 0:
        return torch.zeros(N, device=device), [None] * N

    ego_corners = _build_ego_bbox_corners(ego_trajs, ego_shape)  # (N, T, 4, 2)

    # Build NPC bounding box corners: (N_nb, T, 4, 2)
    npc_pos = neighbor_futures[:, :, :2]
    npc_cos = neighbor_futures[:, :, 2]
    npc_sin = neighbor_futures[:, :, 3]
    npc_norm = (npc_cos**2 + npc_sin**2).sqrt().clamp_min(1e-6)
    npc_cos = npc_cos / npc_norm
    npc_sin = npc_sin / npc_norm

    npc_width = neighbor_shapes[:, 0].unsqueeze(1).expand(-1, T)  # (N_nb, T)
    npc_length = neighbor_shapes[:, 1].unsqueeze(1).expand(-1, T)  # (N_nb, T)

    npc_rect = torch.stack(
        [
            npc_pos[..., 0],
            npc_pos[..., 1],
            npc_cos,
            npc_sin,
            npc_length,
            npc_width,
        ],
        dim=-1,
    )  # (N_nb, T, 6)
    npc_corners = center_rect_to_points(npc_rect.reshape(-1, 6)).reshape(N_nb, T, 4, 2)

    # Cross product: ego (N, T) x NPC (N_nb, T) -> (N, N_nb, T)
    ego_exp = ego_corners.unsqueeze(1).expand(-1, N_nb, -1, -1, -1)  # (N, N_nb, T, 4, 2)
    npc_exp = npc_corners.unsqueeze(0).expand(N, -1, -1, -1, -1)  # (N, N_nb, T, 4, 2)
    nv_exp = neighbor_valid.unsqueeze(0).expand(N, -1, -1)  # (N, N_nb, T)

    # Flatten for batch_signed_distance_rect
    ego_flat = ego_exp.reshape(-1, 4, 2)
    npc_flat = npc_exp.reshape(-1, 4, 2)
    distances = batch_signed_distance_rect(ego_flat, npc_flat)  # (N * N_nb * T,)
    distances = distances.reshape(N, N_nb, T)

    # Mask out invalid NPC timesteps
    distances = distances.masked_fill(~nv_exp, 1e6)

    # Collision: negative signed distance = overlap
    collision_mask = distances < 0  # (N, N_nb, T)

    # Filter out rear-end collisions: only count if NPC is ahead of or beside
    # the ego (not approaching from behind). Check via dot product of ego
    # heading with the ego→NPC displacement vector.
    ego_xy = ego_trajs[:, :, :2]  # (N, T, 2)
    ego_heading = ego_trajs[:, :, 2:4]  # (N, T, 2) [cos, sin]
    npc_xy = neighbor_futures[:, :, :2]  # (N_nb, T, 2)

    # ego→NPC vector: (N, N_nb, T, 2)
    ego_to_npc = npc_xy.unsqueeze(0) - ego_xy.unsqueeze(1)
    # Dot product with ego heading: positive = NPC ahead/beside, negative = NPC behind
    dot = (ego_to_npc * ego_heading.unsqueeze(1)).sum(dim=-1)  # (N, N_nb, T)
    npc_is_behind = dot < 0  # (N, N_nb, T)

    # Suppress rear-end collisions: if NPC overlaps ego from behind at any
    # timestep, exclude that NPC from collision checks at ALL subsequent
    # timesteps. Without this, a rear-ending NPC that passes the ego gets
    # detected as a "side/front collision" once it crosses into the forward
    # hemisphere — a false positive the ego cannot control.
    rear_overlap = (distances < 0) & npc_is_behind  # (N, N_nb, T)
    # cummax along time: once True, stays True for all later timesteps
    ever_rear_ended = rear_overlap.cummax(dim=2).values  # (N, N_nb, T)
    collision_mask = collision_mask & ~npc_is_behind & ~ever_rear_ended

    # Suppress low-speed bbox overlaps: two stopped/slow vehicles queued
    # bumper-to-bumper at a red light or in traffic is not a collision.
    # Only count collisions when the ego is moving faster than 1 m/s.
    _COLLISION_MIN_SPEED = 1.0  # m/s
    ego_vel = torch.diff(ego_xy, dim=1) / config.dt  # (N, T-1, 2)
    ego_speed = ego_vel.norm(dim=-1)  # (N, T-1)
    # Pad last timestep
    ego_speed = torch.cat([ego_speed, ego_speed[:, -1:]], dim=1)  # (N, T)
    ego_moving = ego_speed > _COLLISION_MIN_SPEED  # (N, T)
    collision_mask = collision_mask & ego_moving.unsqueeze(1)  # broadcast over N_nb

    has_collision_at_t = collision_mask.any(dim=1)  # (N, T)
    has_collision = has_collision_at_t.any(dim=1)  # (N,)
    first_t = has_collision_at_t.float().argmax(dim=1)  # (N,)

    # Proximity penalty: soft penalty for being close to any agent without
    # colliding. Min signed distance across all neighbors per timestep.
    # Penalize when closer than _PROXIMITY_MARGIN metres.
    _PROXIMITY_MARGIN = 1.0  # metres
    min_dist_to_any_npc = distances.min(dim=1).values  # (N, T)
    proximity_intrusion = torch.relu(_PROXIMITY_MARGIN - min_dist_to_any_npc)  # (N, T)
    # Don't double-count collision timesteps
    proximity_intrusion = proximity_intrusion.masked_fill(has_collision_at_t, 0.0)
    proximity_penalty = proximity_intrusion.mean(dim=-1)  # (N,)

    scores = torch.where(
        has_collision,
        torch.tensor(config.collision_penalty, device=device) - proximity_penalty,
        -proximity_penalty,
    )

    collision_steps: list[int | None] = []
    for i in range(N):
        if has_collision[i]:
            collision_steps.append(int(first_t[i].item()))
        else:
            collision_steps.append(None)

    return scores, collision_steps


# ---------------------------------------------------------------------------
# Time-to-Collision (TTC): penalize trajectories on collision course
# ---------------------------------------------------------------------------

_TTC_HORIZON = 1.0  # seconds ahead to check
_TTC_DT = 0.1  # trajectory timestep


@torch.no_grad()
def compute_ttc_score_batch(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    neighbor_futures: torch.Tensor,
    neighbor_shapes: torch.Tensor,
    neighbor_valid: torch.Tensor,
) -> torch.Tensor:
    """Check if ego would collide with NPCs within TTC_HORIZON seconds.

    For each trajectory timestep, extrapolates ego and NPC positions forward
    by TTC_HORIZON using current velocity. If the extrapolated positions would
    collide (using simplified distance check), the timestep is marked unsafe.

    Returns:
        (N,) score: fraction of timesteps that are TTC-safe (1.0 = all safe).
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    N_nb = neighbor_futures.shape[0]

    if N_nb == 0 or T < 3:
        return torch.ones(N, device=device)

    # Ego velocity at each timestep
    ego_vel = (ego_trajs[:, 1:, :2] - ego_trajs[:, :-1, :2]) / _TTC_DT  # (N, T-1, 2)
    # Pad to match T
    ego_vel = torch.cat([ego_vel, ego_vel[:, -1:]], dim=1)  # (N, T, 2)

    # NPC velocity
    npc_vel = (
        neighbor_futures[:, 1:, :2] - neighbor_futures[:, :-1, :2]
    ) / _TTC_DT  # (N_nb, T-1, 2)
    npc_vel = torch.cat([npc_vel, npc_vel[:, -1:]], dim=1)  # (N_nb, T, 2)

    # Extrapolate positions TTC_HORIZON seconds ahead
    n_steps = int(_TTC_HORIZON / _TTC_DT)
    ego_future_pos = ego_trajs[:, :, :2] + ego_vel * _TTC_HORIZON  # (N, T, 2)
    npc_future_pos = neighbor_futures[:, :, :2] + npc_vel * _TTC_HORIZON  # (N_nb, T, 2)

    # Simple distance check between ego future and NPC future positions
    # Use center-to-center distance with safety margin (half ego length + half NPC length)
    ego_half_len = float(ego_shape[1]) / 2 + 0.5  # + 0.5m margin
    npc_half_lens = neighbor_shapes[:, 1] / 2 + 0.5  # (N_nb,)

    # Distance: (N, N_nb, T)
    diff = ego_future_pos.unsqueeze(1) - npc_future_pos.unsqueeze(0)  # (N, N_nb, T, 2)
    dist = diff.norm(dim=-1)  # (N, N_nb, T)

    # Collision threshold per NPC
    threshold = (ego_half_len + npc_half_lens).unsqueeze(0).unsqueeze(-1)  # (1, N_nb, 1)

    # Mask invalid NPCs
    ttc_collision = (dist < threshold) & neighbor_valid.unsqueeze(0)  # (N, N_nb, T)
    ttc_unsafe_at_t = ttc_collision.any(dim=1)  # (N, T) — unsafe if ANY NPC collision predicted

    # Score: fraction of safe timesteps
    ttc_score = 1.0 - ttc_unsafe_at_t.float().mean(dim=1)  # (N,)
    return ttc_score


# ---------------------------------------------------------------------------
# Feasibility: lane boundary check with vehicle half-width
# ---------------------------------------------------------------------------

_LN_X, _LN_Y = 0, 1
_LN_DX, _LN_DY = 2, 3
_LN_LBX, _LN_LBY = 4, 5
_LN_RBX, _LN_RBY = 6, 7
_LN_MAX_DIST = 30.0


def _point_in_polygon(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """Ray casting point-in-polygon test.

    Args:
        points: (M, 2) query points.
        polygon: (V, 2) polygon vertices (no need to close — last edge connects
            vertex V-1 back to vertex 0 automatically).

    Returns:
        (M,) bool tensor — True if the point is inside the polygon.
    """
    px, py = points[:, 0:1], points[:, 1:2]  # (M, 1)
    v1 = polygon  # (V, 2)
    v2 = torch.roll(polygon, -1, dims=0)  # (V, 2)

    y1, y2 = v1[:, 1], v2[:, 1]  # (V,)
    x1, x2 = v1[:, 0], v2[:, 0]

    # Does horizontal ray from (px, py) cross edge (v1, v2)?
    cond_y = (y1[None, :] > py) != (y2[None, :] > py)  # (M, V)
    dy = y2 - y1  # (V,) — can be negative, must NOT clamp
    safe_dy = torch.where(dy.abs() < 1e-10, torch.ones_like(dy), dy)
    ix = x1[None, :] + (py - y1[None, :]) * (x2[None, :] - x1[None, :]) / safe_dy[None, :]
    cond_x = px < ix
    return ((cond_y & cond_x).sum(dim=1) % 2) == 1  # (M,)


def _points_in_polygons_batched(
    points: torch.Tensor,
    polygons_v1: torch.Tensor,
    polygons_v2: torch.Tensor,
    poly_valid: torch.Tensor,
) -> torch.Tensor:
    """Batched ray casting: check M points against P polygons simultaneously.

    Args:
        points: (M, 2) query points.
        polygons_v1: (P, V, 2) start vertices of each polygon edge.
        polygons_v2: (P, V, 2) end vertices of each polygon edge.
        poly_valid: (P, V) bool — which edges are real (not padding).

    Returns:
        (M, P) bool — True if point m is inside polygon p.
    """
    M = points.shape[0]
    P, V, _ = polygons_v1.shape

    px = points[:, 0:1, None]  # (M, 1, 1)
    py = points[:, 1:2, None]  # (M, 1, 1)

    y1 = polygons_v1[:, :, 1]  # (P, V)
    y2 = polygons_v2[:, :, 1]
    x1 = polygons_v1[:, :, 0]
    x2 = polygons_v2[:, :, 0]

    # (M, P, V)
    cond_y = (y1[None] > py) != (y2[None] > py)
    dy = y2 - y1  # (P, V)
    safe_dy = torch.where(dy.abs() < 1e-10, torch.ones_like(dy), dy)
    ix = x1[None] + (py - y1[None]) * (x2[None] - x1[None]) / safe_dy[None]
    cond_x = px < ix

    # Mask out padding edges
    valid = poly_valid[None, :, :]  # (1, P, V)
    crossings = (cond_y & cond_x & valid).sum(dim=2)  # (M, P)
    return (crossings % 2) == 1


def _build_lane_polygons(
    lanes: torch.Tensor,
) -> list[torch.Tensor]:
    """Build closed polygons from lane segment boundaries.

    Each lane segment becomes a polygon: left boundary points forward,
    then right boundary points reversed.

    Args:
        lanes: (S, P, 33) lane tensor.

    Returns:
        List of (V, 2) polygon vertex tensors (only segments with ≥3 valid
        points are included).
    """
    polys: list[torch.Tensor] = []
    for seg_idx in range(lanes.shape[0]):
        pts = lanes[seg_idx, :, :2]
        lb = lanes[seg_idx, :, 4:6]
        rb = lanes[seg_idx, :, 6:8]
        valid = pts.abs().sum(dim=-1) > 0.1
        if valid.sum() < 3:
            continue
        left = (pts + lb)[valid]  # (K, 2)
        right = (pts + rb)[valid]  # (K, 2)
        poly = torch.cat([left, right.flip(0)], dim=0)  # (2K, 2)
        polys.append(poly)
    return polys


@torch.no_grad()
def _ego_on_road_polygon(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    lane_polys: list[torch.Tensor],
) -> torch.Tensor:
    """Check if the ego vehicle is on-road using polygon containment.

    For each timestep, builds the 4 ego bounding-box corners and checks
    whether every corner lies inside at least one lane polygon (ray casting).

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        ego_shape: (3,) wheel_base, length, width.
        lane_polys: list of (V, 2) polygon tensors from _build_lane_polygons.

    Returns:
        (N, T) bool tensor — True where the ego is fully on-road.
    """
    if not lane_polys:
        return torch.ones(
            ego_trajs.shape[0], ego_trajs.shape[1], dtype=torch.bool, device=ego_trajs.device
        )

    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    half_l = float(ego_shape[1]) / 2
    half_w = float(ego_shape[2]) / 2

    cos_h = ego_trajs[:, :, 2]  # (N, T)
    sin_h = ego_trajs[:, :, 3]
    cx = ego_trajs[:, :, 0]
    cy = ego_trajs[:, :, 1]

    # Sample points along the ego rectangle perimeter for higher resolution.
    # 4 corners + 20 points per side = 84 sample points total.
    _PTS_PER_SIDE = 20
    local_pts: list[tuple[float, float]] = []
    # Front edge (left to right)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        local_pts.append((half_l, half_w * (1 - 2 * t)))
    # Right edge (front to rear)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        local_pts.append((half_l * (1 - 2 * t), -half_w))
    # Rear edge (right to left)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        local_pts.append((-half_l, -half_w * (1 - 2 * t)))
    # Left edge (rear to front)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        local_pts.append((-half_l * (1 - 2 * t), half_w))

    local_pts_t = torch.tensor(local_pts, device=device, dtype=ego_trajs.dtype)  # (K, 2)
    K = local_pts_t.shape[0]

    # Rotate + translate all sample points: (N, T, K, 2)
    rx = (
        local_pts_t[:, 0][None, None, :] * cos_h[:, :, None]
        - local_pts_t[:, 1][None, None, :] * sin_h[:, :, None]
    )
    ry = (
        local_pts_t[:, 0][None, None, :] * sin_h[:, :, None]
        + local_pts_t[:, 1][None, None, :] * cos_h[:, :, None]
    )
    pts_x = cx[:, :, None] + rx  # (N, T, K)
    pts_y = cy[:, :, None] + ry

    all_pts = torch.stack([pts_x, pts_y], dim=-1).reshape(-1, 2)  # (N*T*K, 2)

    # Batch all polygons: pad to same vertex count and run one vectorized check
    traj_center = ego_trajs[:, :, :2].reshape(-1, 2)
    traj_min = traj_center.min(dim=0).values - 10
    traj_max = traj_center.max(dim=0).values + 10

    # Filter nearby polygons by bounding box
    nearby_polys = []
    for poly in lane_polys:
        pmin = poly.min(dim=0).values
        pmax = poly.max(dim=0).values
        if (
            pmax[0] < traj_min[0]
            or pmin[0] > traj_max[0]
            or pmax[1] < traj_min[1]
            or pmin[1] > traj_max[1]
        ):
            continue
        nearby_polys.append(poly)

    if nearby_polys:
        max_v = max(p.shape[0] for p in nearby_polys)
        P = len(nearby_polys)
        padded_v1 = torch.zeros(P, max_v, 2, device=device)
        padded_v2 = torch.zeros(P, max_v, 2, device=device)
        poly_valid = torch.zeros(P, max_v, dtype=torch.bool, device=device)
        for i, poly in enumerate(nearby_polys):
            V = poly.shape[0]
            padded_v1[i, :V] = poly
            padded_v2[i, :V] = torch.roll(poly, -1, dims=0)
            poly_valid[i, :V] = True

        # (M, P) — True if point is inside polygon
        inside_matrix = _points_in_polygons_batched(all_pts, padded_v1, padded_v2, poly_valid)
        inside_any = inside_matrix.any(dim=1)  # (M,)
    else:
        inside_any = torch.zeros(all_pts.shape[0], dtype=torch.bool, device=device)

    # At least 95% of perimeter points must be inside a lane polygon.
    # Requiring 100% is too strict — a few points can protrude 1-2cm past
    # a lane boundary at polygon seams without the ego being truly offroad.
    inside_any = inside_any.reshape(N, T, K)
    _ON_ROAD_THRESHOLD = 0.95
    inside_frac = inside_any.float().mean(dim=-1)  # (N, T)
    on_road = inside_frac >= _ON_ROAD_THRESHOLD  # (N, T)

    # Also compute fraction of points outside for a soft proximity penalty:
    # fraction_outside = 0 means fully on-road, >0 means partially protruding.
    fraction_outside = 1.0 - inside_any.float().mean(dim=-1)  # (N, T)

    # Edge proximity check: sample points on a rectangle EXPANDED by 25cm.
    # If an expanded point is OUTSIDE all lane polygons, the lane boundary
    # is closer than 25cm to the ego at that location.
    _EDGE_MARGIN = 0.25  # metres
    margin_pts: list[tuple[float, float]] = []
    outer_half_l = half_l + _EDGE_MARGIN
    outer_half_w = half_w + _EDGE_MARGIN
    # Front edge (expanded)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        margin_pts.append((outer_half_l, outer_half_w * (1 - 2 * t)))
    # Right edge (expanded)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        margin_pts.append((outer_half_l * (1 - 2 * t), -outer_half_w))
    # Rear edge (expanded)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        margin_pts.append((-outer_half_l, -outer_half_w * (1 - 2 * t)))
    # Left edge (expanded)
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        margin_pts.append((-outer_half_l * (1 - 2 * t), outer_half_w))

    margin_pts_t = torch.tensor(margin_pts, device=device, dtype=ego_trajs.dtype)

    mrx = (
        margin_pts_t[:, 0][None, None, :] * cos_h[:, :, None]
        - margin_pts_t[:, 1][None, None, :] * sin_h[:, :, None]
    )
    mry = (
        margin_pts_t[:, 0][None, None, :] * sin_h[:, :, None]
        + margin_pts_t[:, 1][None, None, :] * cos_h[:, :, None]
    )
    mpts_x = cx[:, :, None] + mrx
    mpts_y = cy[:, :, None] + mry
    all_margin_pts = torch.stack([mpts_x, mpts_y], dim=-1).reshape(-1, 2)

    if nearby_polys:
        margin_inside_matrix = _points_in_polygons_batched(
            all_margin_pts,
            padded_v1,
            padded_v2,
            poly_valid,
        )
        margin_outside = ~margin_inside_matrix.any(dim=1)
    else:
        margin_outside = torch.ones(all_margin_pts.shape[0], dtype=torch.bool, device=device)

    # Fraction of expanded points that are OUTSIDE = fraction of ego perimeter
    # where the lane boundary is closer than 25cm
    margin_outside = margin_outside.reshape(N, T, K)
    near_edge_penalty = margin_outside.float().mean(dim=-1)  # (N, T)
    # 0 = well inside (all expanded points inside lanes = boundary >25cm away)
    # 1 = entire perimeter near edge (all expanded points outside = boundary <25cm)

    # Second wider margin at 40cm for stronger penalty when ego is very close
    _WIDE_MARGIN = 0.40
    wide_pts: list[tuple[float, float]] = []
    wide_half_l = half_l + _WIDE_MARGIN
    wide_half_w = half_w + _WIDE_MARGIN
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        wide_pts.append((wide_half_l, wide_half_w * (1 - 2 * t)))
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        wide_pts.append((wide_half_l * (1 - 2 * t), -wide_half_w))
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        wide_pts.append((-wide_half_l, -wide_half_w * (1 - 2 * t)))
    for i in range(_PTS_PER_SIDE):
        t = i / (_PTS_PER_SIDE - 1)
        wide_pts.append((-wide_half_l * (1 - 2 * t), wide_half_w))

    wide_pts_t = torch.tensor(wide_pts, device=device, dtype=ego_trajs.dtype)
    wrx = (
        wide_pts_t[:, 0][None, None, :] * cos_h[:, :, None]
        - wide_pts_t[:, 1][None, None, :] * sin_h[:, :, None]
    )
    wry = (
        wide_pts_t[:, 0][None, None, :] * sin_h[:, :, None]
        + wide_pts_t[:, 1][None, None, :] * cos_h[:, :, None]
    )
    wpts_x = cx[:, :, None] + wrx
    wpts_y = cy[:, :, None] + wry
    all_wide_pts = torch.stack([wpts_x, wpts_y], dim=-1).reshape(-1, 2)

    if nearby_polys:
        wide_inside = _points_in_polygons_batched(
            all_wide_pts,
            padded_v1,
            padded_v2,
            poly_valid,
        )
        wide_outside = ~wide_inside.any(dim=1)
    else:
        wide_outside = torch.ones(all_wide_pts.shape[0], dtype=torch.bool, device=device)

    wide_outside = wide_outside.reshape(N, T, K)
    wide_edge_penalty = wide_outside.float().mean(dim=-1)  # (N, T)

    return on_road, fraction_outside, near_edge_penalty, wide_edge_penalty


def compute_feasibility_score_batch(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Acceleration feasibility penalty.

    Penalizes longitudinal and lateral acceleration violations via
    Savitzky-Golay filtered derivatives. Lane-boundary off-road detection
    is disabled — offroad is handled by compute_road_border_penalty instead.

    Args:
        ego_trajs: (N, T, 4).
        ego_shape: (3,) wheel_base, length, width.
        data: Observation dict.
        config: RewardConfig.

    Returns:
        scores: (N,) negative acceleration penalty.
        off_road_fractions: (N,) always zeros (lane-boundary check disabled).
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    scores = torch.zeros(N, device=device)
    off_road_fractions = torch.zeros(N, device=device)

    # --- Acceleration check ---
    pos = ego_trajs[:, :, :2]
    vel = torch.diff(pos, dim=1) / config.dt  # (N, T-1, 2)
    speed = vel.norm(dim=-1)  # (N, T-1)
    acc = torch.diff(speed, dim=1) / config.dt  # (N, T-2) longitudinal
    if acc.numel() > 0:
        accel_violations = (acc.abs() > config.max_accel).float().mean(dim=-1)
        scores = scores - accel_violations

    # --- Lateral acceleration check ---
    # Penalize lateral acceleration exceeding what a human driver produces.
    # GT trajectories peak at ~2.5 m/s²; the model reaches 3.5 m/s² on curves.
    # Uses Savitzky-Golay filtered derivatives via torch conv1d (GPU, no CPU round-trip).
    # lat_accel = |v × a| / |v| (cross product formula for curvature × speed²)
    _MAX_LAT_ACCEL = config.max_lat_accel
    _LAT_ACCEL_SCALE = config.lat_accel_scale
    if T >= 5:
        global _SG_VEL_KERNEL, _SG_ACCEL_KERNEL, _SG_LAT_CACHE_KEY
        _sg_window = min(11, T - (1 if T % 2 == 0 else 0))
        if _sg_window >= 5:
            _lat_cache_key = (device, config.dt, _sg_window)
            if _SG_VEL_KERNEL is None or _SG_LAT_CACHE_KEY != _lat_cache_key:
                _SG_VEL_KERNEL = _build_sg_diff_kernel(
                    window=_sg_window, poly=3, deriv=1, delta=config.dt
                ).to(device)
                _SG_ACCEL_KERNEL = _build_sg_diff_kernel(
                    window=_sg_window, poly=3, deriv=2, delta=config.dt
                ).to(device)
                _SG_LAT_CACHE_KEY = _lat_cache_key

            pad = _sg_window // 2
            # pos: [N, T, 2] -> [N, 2, T] for conv1d
            pos_2d = pos.detach().permute(0, 2, 1)  # [N, 2, T]
            pos_padded = torch.nn.functional.pad(pos_2d, (pad, pad), mode="replicate")

            # Velocity via SG deriv=1
            vel_sg = torch.nn.functional.conv1d(
                pos_padded, _SG_VEL_KERNEL.view(1, 1, -1).expand(2, 1, -1), groups=2
            )  # [N, 2, T]
            # Acceleration via SG deriv=2
            accel_sg = torch.nn.functional.conv1d(
                pos_padded, _SG_ACCEL_KERNEL.view(1, 1, -1).expand(2, 1, -1), groups=2
            )  # [N, 2, T]

            vx, vy = vel_sg[:, 0], vel_sg[:, 1]  # [N, T]
            ax_sg, ay_sg = accel_sg[:, 0], accel_sg[:, 1]  # [N, T]
            speed_sg = (vx**2 + vy**2).sqrt()  # [N, T]

            # lat_accel = |vx*ay - vy*ax| / max(|v|, 0.5)
            cross = (vx * ay_sg - vy * ax_sg).abs()
            lat_accel_sg = cross / speed_sg.clamp(min=0.5)
            # Zero out low-speed regions
            lat_accel_sg = torch.where(speed_sg > 0.5, lat_accel_sg, torch.zeros_like(lat_accel_sg))

            # Trim SG edge artifacts (pad from each side)
            if lat_accel_sg.shape[1] > 2 * pad + 1:
                lat_accel_trimmed = lat_accel_sg[:, pad:-pad]
            else:
                lat_accel_trimmed = lat_accel_sg
            lat_violations = torch.relu(lat_accel_trimmed - _MAX_LAT_ACCEL)
            scores = scores - _LAT_ACCEL_SCALE * lat_violations.mean(dim=-1)

    # (Yaw-rate + kinematic-curvature hard gate is applied separately via
    # compute_kinematic_gate, not as a soft penalty here.)

    # Lane-boundary off-road check DISABLED — compute_road_border_penalty handles
    # offroad detection. Feasibility score returns only the base score above.
    off_road_fractions = torch.zeros(N, device=device)
    return scores, off_road_fractions


_SG_SMOOTH_KERNEL = None
_SG_SMOOTH_CACHE_KEY = None


@torch.no_grad()
def compute_kinematic_gate(
    ego_trajs: torch.Tensor,
    config: RewardConfig,
    ego_shape: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hard feasibility gate: 1.0 if trajectory is kinematically feasible, 0.0 if not.

    Two checks on a SG-smoothed trajectory:
      1. |yaw_rate| ≤ max_yaw_rate (absolute rate cap)
      2. |yaw_rate| ≤ κ_max * speed where κ_max = kinematic_margin *
         tan(max_steer) / wheelbase — bicycle-model curvature bound.

    Either violated at ANY timestep → 0.0 gate.
    SG filtering removes diffusion noise so noise doesn't fire the gate.

    Args:
        ego_trajs: (N, T, 4) x, y, cos_h, sin_h.
        config: RewardConfig with max_yaw_rate, max_steer, kinematic_margin, dt.
        ego_shape: (3,) wheel_base, length, width. Required for bicycle-model
            curvature bound. If None, skip the curvature check (absolute yaw
            cap still applied).

    Returns:
        (N,) float tensor: 1.0 = feasible, 0.0 = infeasible.
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    if T < 5:
        return torch.ones(N, device=device)

    # SG smoothing kernel (poly=3, deriv=0): same SG family as existing lat-accel check.
    global _SG_SMOOTH_KERNEL, _SG_SMOOTH_CACHE_KEY
    _sg_window = min(11, T - (1 if T % 2 == 0 else 0))
    if _sg_window < 5:
        return torch.ones(N, device=device)
    key = (device, _sg_window)
    if _SG_SMOOTH_KERNEL is None or _SG_SMOOTH_CACHE_KEY != key:
        _SG_SMOOTH_KERNEL = _build_sg_diff_kernel(
            window=_sg_window, poly=3, deriv=0, delta=config.dt
        ).to(device)
        _SG_SMOOTH_CACHE_KEY = key
    pad = _sg_window // 2

    # Smooth (cos_h, sin_h) via SG → recover filtered heading without wrap issues
    cos_h = ego_trajs[..., 2]
    sin_h = ego_trajs[..., 3]
    nh = (cos_h**2 + sin_h**2).sqrt().clamp_min(1e-6)
    cos_h = cos_h / nh
    sin_h = sin_h / nh
    heading_2d = torch.stack([cos_h, sin_h], dim=1)  # (N, 2, T)
    heading_padded = torch.nn.functional.pad(heading_2d, (pad, pad), mode="replicate")
    heading_sg = torch.nn.functional.conv1d(
        heading_padded, _SG_SMOOTH_KERNEL.view(1, 1, -1).expand(2, 1, -1), groups=2
    )  # (N, 2, T)
    cos_sg = heading_sg[:, 0]
    sin_sg = heading_sg[:, 1]
    theta_sg = torch.atan2(sin_sg, cos_sg)  # (N, T)
    dtheta = theta_sg[:, 1:] - theta_sg[:, :-1]
    dtheta = torch.atan2(dtheta.sin(), dtheta.cos())  # wrap to (-π, π)
    yaw_rate = dtheta.abs() / config.dt  # (N, T-1)

    # SG-smoothed speed from positions
    pos = ego_trajs[..., :2].detach().permute(0, 2, 1)  # (N, 2, T)
    pos_padded = torch.nn.functional.pad(pos, (pad, pad), mode="replicate")
    global _SG_VEL_KERNEL, _SG_ACCEL_KERNEL, _SG_LAT_CACHE_KEY
    _lat_key = (device, config.dt, _sg_window)
    if _SG_VEL_KERNEL is None or _SG_LAT_CACHE_KEY != _lat_key:
        _SG_VEL_KERNEL = _build_sg_diff_kernel(
            window=_sg_window, poly=3, deriv=1, delta=config.dt
        ).to(device)
        _SG_ACCEL_KERNEL = _build_sg_diff_kernel(
            window=_sg_window, poly=3, deriv=2, delta=config.dt
        ).to(device)
        _SG_LAT_CACHE_KEY = _lat_key
    vel_sg = torch.nn.functional.conv1d(
        pos_padded, _SG_VEL_KERNEL.view(1, 1, -1).expand(2, 1, -1), groups=2
    )
    speed_sg = (vel_sg[:, 0] ** 2 + vel_sg[:, 1] ** 2).sqrt()  # (N, T)
    speed_align = speed_sg[:, :-1]  # align with yaw_rate (N, T-1)

    # Check 1: absolute yaw rate cap
    abs_violated = yaw_rate > config.max_yaw_rate

    # Check 2: bicycle-model curvature cap. κ_max = margin * tan(max_steer) / wheelbase.
    if ego_shape is not None:
        wheelbase = float(ego_shape[0])
        kappa_max = config.kinematic_margin * math.tan(config.max_steer) / max(wheelbase, 1e-3)
        curv_violated = yaw_rate > kappa_max * speed_align
    else:
        curv_violated = torch.zeros_like(abs_violated)

    violated_per_t = abs_violated | curv_violated
    any_violation = violated_per_t.any(dim=1)  # (N,) bool

    gate = (~any_violation).float()
    return gate


# ---------------------------------------------------------------------------
# Centerline: reward for staying close to route lane centerlines
# ---------------------------------------------------------------------------

_CL_X, _CL_Y = 0, 1
_CL_DX, _CL_DY = 2, 3
_CL_MAX_DIST = 30.0


def compute_centerline_score_batch(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    data: dict[str, torch.Tensor],
    usage_mode: str = "baselink",
    time_weight_min: float = 0.3,
) -> torch.Tensor:
    """Batched normalized lane-usage penalty from nearest route lane centerline.

    Uses route_lanes to compute what fraction of the lane width the vehicle
    occupies. A centered vehicle uses ~half_w/lane_hw; one at the boundary uses 1.0.

    Args:
        ego_trajs: (N, T, 4).
        data: Observation dict with "route_lanes" or "lanes" key.

    Returns:
        (N,) scores (negative, closer to 0 = closer to centerline).
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    half_w = float(ego_shape[2]) / 2

    if "route_lanes" in data:
        lanes = data["route_lanes"]
    elif "lanes" in data:
        lanes = data["lanes"]
    else:
        return torch.zeros(N, device=device)

    if lanes.dim() == 4:
        lanes = lanes[0]  # (S, P, 33)

    S_P = lanes.shape[0] * lanes.shape[1]
    lane_centers = lanes[..., _CL_X : _CL_Y + 1].reshape(S_P, 2)
    lane_dirs = lanes[..., _CL_DX : _CL_DY + 1].reshape(S_P, 2)
    lane_left = lanes[..., 4:6].reshape(S_P, 2)
    lane_right = lanes[..., 6:8].reshape(S_P, 2)

    lane_valid = lane_centers.norm(dim=-1) > 1e-3  # (S_P,)
    lane_dirs_n = lane_dirs / (lane_dirs.norm(dim=-1, keepdim=True) + 1e-6)
    lane_lat = torch.stack([-lane_dirs_n[..., 1], lane_dirs_n[..., 0]], dim=-1)  # (S_P, 2)

    left_hw = (lane_left * lane_lat).sum(dim=-1)  # (S_P,)
    right_hw = (lane_right * lane_lat).sum(dim=-1)  # (S_P,)

    ego_pos = ego_trajs[:, :, :2]  # (N, T, 2)

    # Distance to each lane center: (N, T, S_P)
    diff = ego_pos.unsqueeze(2) - lane_centers.unsqueeze(0).unsqueeze(0)
    dist = diff.norm(dim=-1)
    dist = dist.masked_fill(~lane_valid.view(1, 1, -1).expand(N, T, -1), 1e6)

    nearest = dist.argmin(dim=-1)  # (N, T)
    min_dist = dist.min(dim=-1).values  # (N, T)

    # Gather nearest lane properties
    flat_idx = nearest.reshape(-1)
    c = lane_centers[flat_idx].reshape(N, T, 2)
    lat = lane_lat[flat_idx].reshape(N, T, 2)

    # Lateral offset from centerline
    ego_lat = ((ego_pos - c) * lat).sum(dim=-1)  # (N, T)

    # Lane half-width on the side the ego is offset toward
    lhw_gathered = left_hw[flat_idx].reshape(N, T)
    rhw_gathered = right_hw[flat_idx].reshape(N, T)

    # Half-width on the ego's side: if ego_lat > 0, use left_hw; if < 0, use |right_hw|
    side_hw = torch.where(
        ego_lat >= 0,
        lhw_gathered.clamp(min=0.5),
        (-rhw_gathered).clamp(min=0.5),
    )  # (N, T)

    # Normalized lane usage: how close the vehicle is to the boundary.
    # "baselink" (default): 0 = baselink on centerline, 1 = baselink at half-lane.
    #   Pure baselink offset; ego width doesn't matter. Directly interpretable.
    # "body" (DEPRECATED): 0 = centered, 1 = ego edge touching boundary.
    #   Includes ego_half_w → even a centered wide vehicle has non-zero usage.
    #   Emits DeprecationWarning. Kept only for loading pre-2026-04-27 configs.
    if usage_mode == "baselink":
        lane_usage = ego_lat.abs() / side_hw  # (N, T)
    elif usage_mode == "body":
        import warnings

        warnings.warn(
            "centerline_usage_mode='body' is deprecated as of 2026-04-27. "
            "Use 'baselink' for new configs. Body adds half-vehicle-width to "
            "offset, which produces unitless values that are easy to misread "
            "as lateral metres.",
            DeprecationWarning,
            stacklevel=2,
        )
        lane_usage = (ego_lat.abs() + half_w) / side_hw  # (N, T)
    else:
        raise ValueError(
            f"Unknown centerline_usage_mode={usage_mode!r}. "
            f"Use 'baselink' (default) or 'body' (deprecated)."
        )

    # When near a route lane (<5m): penalize by lane usage (lateral position)
    # When far from route (>5m): penalize by distance to route (off-route deviation)
    # BUT: only apply route deviation if the trajectory was previously near the
    # route and then drifted away. If route lanes simply don't extend far enough,
    # don't penalize -- the trajectory may be following the road correctly beyond
    # where route data ends.
    _PROXIMITY = 5.0
    near_route = min_dist <= _PROXIMITY  # (N, T)

    # Detect "was near route, now drifted": cumulative max of near_route over time.
    # If the trajectory was ever near the route, subsequent far timesteps are
    # treated as route abandonment. If never near, it's a coverage gap.
    was_near = near_route.cummax(dim=-1).values  # (N, T) -- True from first near timestep onward
    left_route = was_near & ~near_route  # (N, T) -- was near but now far

    # Route deviation penalty: only for timesteps where trajectory left the route
    _ROUTE_DEVIATION_SCALE = 0.5
    route_deviation = (min_dist * _ROUTE_DEVIATION_SCALE).clamp(max=5.0)  # cap to avoid explosion

    # lane_usage is unbounded — squared directly. Trajectories past the lane
    # boundary accrue penalty proportional to (lane_usage)² with no clip.
    per_step_penalty = torch.where(
        near_route,
        lane_usage**2,
        torch.where(
            left_route,
            route_deviation**2,
            torch.zeros_like(lane_usage),  # no penalty if route never covered this area
        ),
    )  # (N, T)

    # Time-weighted mean: early deviations penalized more by default (min=0.3).
    # Raise time_weight_min to 1.0 for flat uniform averaging.
    time_weights = torch.linspace(1.0, time_weight_min, T, device=device).unsqueeze(0)
    penalty = per_step_penalty * time_weights
    return -(penalty.sum(dim=-1) / time_weights.sum())  # (N,)


# ---------------------------------------------------------------------------
# Progress: batched distance reduction toward goal
# ---------------------------------------------------------------------------


def compute_progress_score_batch(
    ego_trajs: torch.Tensor,
    goal_pose: torch.Tensor,
    data: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Batched progress toward goal.

    Uses the goal_pose if nearby (<100m). If goal is far (e.g., 2km final
    destination on 71k pool scenes), falls back to GT endpoint as local goal.

    Args:
        ego_trajs: (N, T, 4).
        goal_pose: (4,) x, y, cos, sin -- zeros if unavailable.
        data: observation dict (optional, used to extract GT endpoint fallback).

    Returns:
        (N,) scores.
    """
    goal_xy = goal_pose[:2]
    _MAX_GOAL_DIST = 100.0

    # Try goal_pose first
    if goal_xy.abs().sum() > 1e-6 and goal_xy.norm() <= _MAX_GOAL_DIST:
        dist_start = (ego_trajs[:, 0, :2] - goal_xy).norm(dim=-1)
        dist_end = (ego_trajs[:, -1, :2] - goal_xy).norm(dim=-1)
        return dist_start - dist_end

    # Fallback: use GT final position as local goal
    if data is not None and "ego_agent_future" in data:
        gt = data["ego_agent_future"]
        if gt.dim() == 3:
            gt = gt[0]
        gt_xy = gt[:, :2]
        gt_valid = gt_xy.abs().sum(dim=-1) > 0.1
        if gt_valid.sum() >= 10:
            gt_end = gt_xy[gt_valid][-1]  # last valid GT position
            dist_start = (ego_trajs[:, 0, :2] - gt_end).norm(dim=-1)
            dist_end = (ego_trajs[:, -1, :2] - gt_end).norm(dim=-1)
            return dist_start - dist_end

    # Last resort: path length
    diffs = torch.diff(ego_trajs[:, :, :2], dim=1)
    return torch.sqrt((diffs**2).sum(dim=-1)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Smoothness: batched jerk penalty
# ---------------------------------------------------------------------------


def _build_sg_diff_kernel(
    window: int = 11, poly: int = 3, deriv: int = 3, delta: float = 0.1
) -> torch.Tensor:
    """Build Savitzky-Golay differentiation kernel (precomputed, cached).

    Returns a 1D convolution kernel that computes the deriv-th derivative
    using a local polynomial fit over `window` points.
    Pure numpy implementation — no scipy dependency.
    """
    # SG coefficients via least-squares polynomial fitting
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    # Build Vandermonde matrix
    A = np.vander(x, N=poly + 1, increasing=True)  # [window, poly+1]
    # Pseudo-inverse gives the coefficient extraction matrix
    pinv = np.linalg.pinv(A)  # [poly+1, window]
    # The deriv-th row of pinv gives smoothing coefficients for the deriv-th derivative
    import math as _math

    coeffs = pinv[deriv] * _math.factorial(deriv) / (delta**deriv)
    # Reverse to match convolution convention (scipy savgol_coeffs convention)
    return torch.tensor(coeffs.copy(), dtype=torch.float32).flip(0)  # flip for conv1d


# Precompute SG jerk kernel; cache by (device, dt).
# Safe in DDP: each process has its own Python interpreter and module-level state.
_SG_JERK_KERNEL = None
_SG_JERK_CACHE_KEY = None

# Precompute SG velocity/acceleration kernels for lat_accel; cache by (device, dt, window).
# Same DDP safety note as above.
_SG_VEL_KERNEL = None
_SG_ACCEL_KERNEL = None
_SG_LAT_CACHE_KEY = None


def compute_smoothness_score_batch(
    ego_trajs: torch.Tensor,
    config: RewardConfig,
) -> torch.Tensor:
    """Batched negative mean absolute jerk using Savitzky-Golay convolution.

    Uses a precomputed SG kernel applied via torch conv1d for GPU speed.
    Raw finite differences amplify noise ~1000x on 10Hz data.
    SG filtering gives physically meaningful jerk values.

    Args:
        ego_trajs: (N, T, 4).
        config: RewardConfig for dt.

    Returns:
        (N,) scores (negative, closer to 0 = smoother).
    """
    global _SG_JERK_KERNEL, _SG_JERK_CACHE_KEY
    N, T, _ = ego_trajs.shape
    if T < 12:
        return torch.zeros(N, device=ego_trajs.device)

    # Build kernel once, cache by (device, dt)
    _cache_key = (ego_trajs.device, config.dt)
    if _SG_JERK_KERNEL is None or _SG_JERK_CACHE_KEY != _cache_key:
        _SG_JERK_KERNEL = _build_sg_diff_kernel(window=11, poly=3, deriv=3, delta=config.dt).to(
            ego_trajs.device
        )
        _SG_JERK_CACHE_KEY = _cache_key

    kernel = _SG_JERK_KERNEL  # [11]
    pad = kernel.shape[0] // 2

    # pos: [N, T, 2] -> [N, 2, T] for conv1d
    pos = ego_trajs[:, :, :2].detach().permute(0, 2, 1)  # [N, 2, T]

    # Pad and convolve: conv1d with kernel [1, 1, W] on [N, 2, T]
    pos_padded = torch.nn.functional.pad(pos, (pad, pad), mode="replicate")
    jerk = torch.nn.functional.conv1d(
        pos_padded,
        kernel.view(1, 1, -1).expand(2, 1, -1),
        groups=2,
    )  # [N, 2, T]

    jerk_mag = jerk.norm(dim=1)  # [N, T]
    # Trim SG edge artifacts (pad from each side)
    if jerk_mag.shape[1] > 2 * pad + 1:
        jerk_mag = jerk_mag[:, pad:-pad]
    return -jerk_mag.mean(dim=1)  # [N]


# ---------------------------------------------------------------------------
# Red light: penalize trajectories that enter red-light route lane segments
# ---------------------------------------------------------------------------

# Traffic light one-hot indices within the 33-dim lane point descriptor
_TL_GREEN = 8
_TL_YELLOW = 9
_TL_RED = 10
_TL_WHITE = 11
_TL_NONE = 12

# Proximity threshold: ego must be within this distance of a red-light
# lane point AND moving along the lane direction to count as a violation.
_RED_LIGHT_PROXIMITY = 3.0  # metres
_RED_LIGHT_HEADING_THRESH = 0.5  # cos(60°) — ego heading must roughly align with lane


def compute_red_light_score_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig,
) -> torch.Tensor:
    """Batched red-light violation penalty.

    Checks whether the ego trajectory enters route lane segments that have a
    red traffic light. A violation requires both spatial proximity (< 3m) and
    heading alignment (cos > 0.5) to avoid penalizing trajectories that pass
    near but don't enter the red-light lane.

    Only checks route_lanes (the ego's planned route), NOT lanes[].

    IMPORTANT: lanes[] contains red lights for cross-traffic at intersections.
    These are ALWAYS red regardless of the ego's signal phase (they represent
    the opposing traffic direction). Using lanes[] would cause false positives
    because the cross-traffic red lanes don't change when the ego has green.
    The ego's own traffic light state is on route_lanes, encoded as RED when
    applicable or WHITE when the converter couldn't resolve it.

    Known limitation: the C++ converter sometimes records the ego's traffic
    light as WHITE (unresolved) instead of RED, even when the ego is clearly
    stopped at a red light. In these cases the penalty won't fire. This is a
    data-level issue, not a reward logic issue.

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        data: Observation dict with "route_lanes".
        config: RewardConfig.

    Returns:
        (N,) scores — 0 if no violation, negative penalty if violated.
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    scores = torch.zeros(N, device=device)

    if "route_lanes" not in data:
        return scores

    rl = data["route_lanes"]
    if rl.dim() == 4:
        rl = rl[0]  # (S, P, 33)

    # Find route lane points with red light
    red_mask = rl[:, :, _TL_RED] > 0.5  # (S, P)
    if not red_mask.any():
        return scores

    # Extract red-light lane point positions and directions
    red_pts = rl[red_mask]  # (R, 33)
    red_xy = red_pts[:, :2]  # (R, 2)
    red_dir = red_pts[:, 2:4]  # (R, 2)

    # Filter out zero-padded points
    valid = red_xy.norm(dim=-1) > 0.1
    if not valid.any():
        return scores
    red_xy = red_xy[valid]  # (R', 2)
    red_dir = red_dir[valid]  # (R', 2)
    R = red_xy.shape[0]

    # Normalize lane directions
    red_dir_norm = red_dir / (red_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6))

    # Ego positions and headings
    ego_xy = ego_trajs[:, :, :2]  # (N, T, 2)
    ego_cos = ego_trajs[:, :, 2]  # (N, T)
    ego_sin = ego_trajs[:, :, 3]  # (N, T)
    ego_heading = torch.stack([ego_cos, ego_sin], dim=-1)  # (N, T, 2)

    # Distance from each ego position to each red-light point: (N, T, R')
    diff = ego_xy.unsqueeze(2) - red_xy.unsqueeze(0).unsqueeze(0)  # (N, T, R', 2)
    dist = diff.norm(dim=-1)  # (N, T, R')

    # Heading alignment: dot product of ego heading with lane direction
    # (N, T, 1, 2) . (1, 1, R', 2) -> (N, T, R')
    cos_align = (ego_heading.unsqueeze(2) * red_dir_norm.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

    # Violation: close enough AND heading aligned AND ego is moving (not stopped)
    # Compute ego speed to distinguish stopped vs moving
    ego_vel = torch.diff(ego_xy, dim=1) / config.dt  # (N, T-1, 2)
    ego_speed = ego_vel.norm(dim=-1)  # (N, T-1)
    # Pad to match T timesteps
    ego_speed = torch.cat([ego_speed, ego_speed[:, -1:]], dim=1)  # (N, T)
    is_moving = ego_speed > 0.5  # m/s threshold — ignore near-stationary

    is_close = dist < _RED_LIGHT_PROXIMITY  # (N, T, R')
    is_aligned = cos_align > _RED_LIGHT_HEADING_THRESH  # (N, T, R')

    # Violation at timestep: close to any red point AND aligned AND moving
    violation_per_point = is_close & is_aligned  # (N, T, R')
    violation_at_t = violation_per_point.any(dim=-1) & is_moving  # (N, T)

    # Number of violation timesteps
    n_violations = violation_at_t.float().sum(dim=-1)  # (N,)

    # Penalty: hard penalty for any violation + soft per-step penalty
    has_violation = n_violations > 0
    scores = torch.where(
        has_violation,
        torch.tensor(config.red_light_penalty, device=device) - n_violations * 0.5,
        scores,
    )

    return scores


# ---------------------------------------------------------------------------
# Road border penalty: ego perimeter vs road_border line_strings
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_road_border_penalty(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int | None], torch.Tensor, torch.Tensor]:
    """Compute per-trajectory road border penalties using ego perimeter sampling.

    Uses 80 points around the ego rectangle (20 per side) and checks min
    distance to road_border line_string *segments* (channel 3 in line_strings).
    Consecutive valid border points within each polyline form segments;
    point-to-segment distance is more accurate than point-to-point.

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        ego_shape: (3,) wheel_base, length, width.
        data: Observation dict with 'line_strings' key.
        config: RewardConfig with threshold overrides. None = defaults.

    Returns:
        Tuple of (crossing_gate, near_penalty, wide_penalty, first_crossing_steps, cont_penalty, per_timestep_min):
        - crossing_gate: (N,) 1.0 if no crossing, 0.0 if any timestep crosses border
        - near_penalty: (N,) near zone penalty (frac or survival-style depending on config)
        - wide_penalty: (N,) wide zone penalty (exclusive of near)
        - first_crossing_steps: list of N (int | None) — first timestep of crossing
        - cont_penalty: (N,) continuous proximity penalty (linear decay from cont_thresh)
        - per_timestep_min: (N, T) min ego-perimeter-to-border distance per timestep
    """
    if config is None:
        config = RewardConfig()

    N, T, _ = ego_trajs.shape
    device = ego_trajs.device

    no_crossing_steps: list[int | None] = [None] * N
    _safe_return = (
        torch.ones(N, device=device),
        torch.zeros(N, device=device),
        torch.zeros(N, device=device),
        no_crossing_steps,
        torch.zeros(N, device=device),
        torch.full((N, T), 99.0, device=device),
    )

    if "line_strings" not in data:
        return _safe_return

    ls = data["line_strings"]
    if ls.dim() == 4:
        ls = ls[0]  # remove batch dim -> (num_ls, pts, D)
    if ls.shape[-1] < 4:
        return _safe_return

    # Build road border segments from consecutive valid points within each polyline
    border_flag = ls[..., 3]  # (num_ls, pts)
    border_xy = ls[..., :2]  # (num_ls, pts, 2)
    is_border = border_flag > 0.5
    has_coords = border_xy.norm(dim=-1) > 1e-3
    valid = is_border & has_coords  # (num_ls, pts)

    # Consecutive valid pairs within each polyline form segments
    valid_pair = valid[:, :-1] & valid[:, 1:]  # (num_ls, pts-1)
    idx = torch.where(valid_pair.reshape(-1))[0]

    if idx.shape[0] == 0:
        return _safe_return

    seg_p1_all = border_xy[:, :-1].reshape(-1, 2)[idx]  # (E, 2)
    seg_p2_all = border_xy[:, 1:].reshape(-1, 2)[idx]  # (E, 2)

    # Build ego perimeter points (20 per side = 80 total)
    wb = ego_shape[0].item()
    length = ego_shape[1].item()
    width = ego_shape[2].item()
    ro = (length - wb) / 2
    _PTS_PER_SIDE = 20
    local_pts = []
    for j in range(_PTS_PER_SIDE):
        f = j / (_PTS_PER_SIDE - 1)
        local_pts.append((-ro + f * length, -width / 2))  # bottom
        local_pts.append((-ro + f * length, width / 2))  # top
        local_pts.append((-ro, -width / 2 + f * width))  # left
        local_pts.append((length - ro, -width / 2 + f * width))  # right
    local_pts = torch.tensor(local_pts, device=device, dtype=ego_trajs.dtype)  # (80, 2)
    K_pts = local_pts.shape[0]

    # For each trajectory and timestep, transform perimeter to world frame
    cos_h = ego_trajs[..., 2]  # (N, T)
    sin_h = ego_trajs[..., 3]
    h_norm = (cos_h**2 + sin_h**2).sqrt().clamp_min(1e-6)
    cos_h = cos_h / h_norm
    sin_h = sin_h / h_norm

    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=-1).reshape(N, T, 2, 2)
    rotated = torch.einsum("btij,kj->btki", rot, local_pts)
    world_pts = ego_trajs[..., :2].unsqueeze(2) + rotated  # (N, T, 80, 2)

    # Pre-filter: keep only segments near the trajectory bbox to avoid
    # computing distance to all ~400 segments. Use segment midpoints for
    # a fast cdist pre-filter, then exact point-to-segment on the reduced set.
    E = seg_p1_all.shape[0]
    _MAX_SEGS = 60  # max segments to keep after pre-filter
    if E > _MAX_SEGS:
        seg_mid = (seg_p1_all + seg_p2_all) * 0.5  # (E, 2)
        # Trajectory center = mean of all ego positions across all trajs
        traj_xy = ego_trajs[:, :, :2].reshape(-1, 2)  # (N*T, 2)
        traj_center = (traj_xy.min(0).values + traj_xy.max(0).values) * 0.5  # (2,)
        # Distance from each segment midpoint to trajectory center
        mid_dist = (seg_mid - traj_center).norm(dim=-1)  # (E,)
        # Also include segments within traj bbox + margin
        traj_max = traj_xy.max(0).values
        traj_min = traj_xy.min(0).values
        half_diag = (traj_max - traj_min).norm() / 2 + 5.0  # generous margin
        n_nearby = int((mid_dist < half_diag).sum().item())
        k = min(max(_MAX_SEGS, n_nearby), E)  # keep all nearby but never exceed E
        _, topk_idx = mid_dist.topk(k, largest=False)
        seg_p1 = seg_p1_all[topk_idx]
        seg_p2 = seg_p2_all[topk_idx]
    else:
        seg_p1 = seg_p1_all
        seg_p2 = seg_p2_all

    # Point-to-segment min distance (chunked for OOM safety)
    world_flat = world_pts.reshape(N * T * K_pts, 2)  # (N*T*80, 2)
    min_dists = _point_to_segments_min_dist(world_flat, seg_p1, seg_p2)
    min_dists = min_dists.reshape(N, T, K_pts)  # (N, T, 80)

    # Per-timestep: min distance across all perimeter points (true values,
    # including t=0). The gate and near/wide penalties below still exclude t=0
    # because the ego's starting pose is not model-controllable — but the
    # returned `per_timestep_min[:, 0]` carries the real distance for any
    # downstream diagnostic (cleanse, viz, eval scripts).
    per_timestep_min = min_dists.min(dim=2).values  # (N, T)

    # Thresholds from config
    cross_thresh = config.rb_cross_thresh
    near_thresh = config.rb_near_thresh
    wide_thresh = config.rb_wide_thresh
    cont_thresh = config.rb_cont_thresh

    # Crossing gate: any t>=1 timestep with min dist < cross_thresh = crossing.
    # t=0 is excluded from the gate (can't control starting position).
    is_crossing = per_timestep_min < cross_thresh  # (N, T), full tensor for diag
    has_crossing = is_crossing[:, 1:].any(dim=1)  # (N,), t=0 excluded
    crossing_gate = (~has_crossing).float()  # (N,) 1.0=safe, 0.0=crossing

    # First crossing timestep per trajectory (among t>=1).
    first_crossing_steps: list[int | None] = []
    for i in range(N):
        if has_crossing[i]:
            first_crossing_steps.append(
                int(is_crossing[i, 1:].nonzero(as_tuple=True)[0][0].item()) + 1
            )
        else:
            first_crossing_steps.append(None)

    # Exclusive categories: crossing > near > wide > safe. No double counting.
    is_not_crossing = ~is_crossing[:, 1:]  # (N, T-1)
    T_valid = T - 1  # timesteps 1..T-1

    if config.rb_penalty_mode == "survival":
        # First-violation time-decay over valid window per_timestep_min[:, 1:]
        # (timesteps 1..T-1): penalty = (T_valid - first_violation) / T_valid,
        # where T_valid = T-1 and first_violation is 0-indexed within that window.
        # Early violations are expensive, late violations are cheap.
        is_near = is_not_crossing & (per_timestep_min[:, 1:] < near_thresh)
        is_wide = (
            is_not_crossing
            & (per_timestep_min[:, 1:] >= near_thresh)
            & (per_timestep_min[:, 1:] < wide_thresh)
        )

        # Vectorized first-violation timestep computation across the batch.
        near_penalty = torch.zeros(N, device=device)
        near_has = is_near.any(dim=1)
        near_first_t = is_near.to(torch.int64).argmax(dim=1)
        near_penalty[near_has] = (T_valid - near_first_t[near_has].float()) / T_valid

        wide_penalty = torch.zeros(N, device=device)
        wide_has = is_wide.any(dim=1)
        wide_first_t = is_wide.to(torch.int64).argmax(dim=1)
        wide_penalty[wide_has] = (T_valid - wide_first_t[wide_has].float()) / T_valid

        # Continuous: worst (minimum) distance within range, scaled by first in-range time.
        cont_penalty = torch.zeros(N, device=device)
        if cont_thresh > 0:
            in_range = is_not_crossing & (per_timestep_min[:, 1:] < cont_thresh)
            cont_has = in_range.any(dim=1)
            cont_first_t = in_range.to(torch.int64).argmax(dim=1)
            worst_dist = (
                per_timestep_min[:, 1:].masked_fill(~in_range, float("inf")).min(dim=1).values
            )
            cont_penalty[cont_has] = (
                (1.0 - worst_dist[cont_has] / cont_thresh).clamp(0, 1)
                * (T_valid - cont_first_t[cont_has].float())
                / T_valid
            )

        return (
            crossing_gate,
            near_penalty,
            wide_penalty,
            first_crossing_steps,
            cont_penalty,
            per_timestep_min,
        )
    else:
        # Original "frac" mode: fraction of timesteps in violation
        near_frac = (is_not_crossing & (per_timestep_min[:, 1:] < near_thresh)).float().mean(dim=1)
        wide_frac = (
            (
                is_not_crossing
                & (per_timestep_min[:, 1:] >= near_thresh)
                & (per_timestep_min[:, 1:] < wide_thresh)
            )
            .float()
            .mean(dim=1)
        )
        if cont_thresh <= 0:
            cont_penalty = torch.zeros(N, device=device)
        else:
            cont_penalty = torch.where(
                is_not_crossing,
                (1.0 - per_timestep_min[:, 1:] / cont_thresh).clamp(min=0, max=1),
                torch.zeros_like(per_timestep_min[:, 1:]),
            ).mean(dim=1)

        return (
            crossing_gate,
            near_frac,
            wide_frac,
            first_crossing_steps,
            cont_penalty,
            per_timestep_min,
        )


# ---------------------------------------------------------------------------
# Static-collision penalty (stopped-neighbor OBB clearance)
# ---------------------------------------------------------------------------
#
# Mirrors the road-border staged-threshold pattern but against stopped
# neighbors instead of road-border segments. A neighbor is counted as
# "stopped" when both (a) |v0| < sc_neighbor_vel_thresh and (b) total
# displacement across the GT future < sc_neighbor_disp_thresh.
#
# Distance primitive: OBB-OBB SAT signed distance via
# batch_signed_distance_rect (reused from compute_safety_score_batch).
# Positive = clearance in metres, negative = penetration depth.
#
# Timesteps where ego is moving < sc_ego_min_speed are not scored (matches
# the bumper-to-bumper suppression in compute_safety_score_batch).
#
# Returns a closest-pair (ego_pt, npc_pt) per timestep for visualization,
# mirroring the road-border distance viz.
# ---------------------------------------------------------------------------


def _closest_points_between_rects(
    rect1: torch.Tensor,
    rect2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closest-point pair between two rectangles, vectorised, pure PyTorch.

    For each pair checks all 32 vertex-to-edge queries (16 rect1 corners →
    rect2 edges + 16 rect2 corners → rect1 edges) and returns the vertex +
    its foot on the nearest edge. Exact for non-overlapping rectangles,
    approximate (points on the nearest edges) for overlapping ones — fine
    for visualisation.

    Args:
        rect1: (B, 4, 2) corners in CCW or CW order.
        rect2: (B, 4, 2) corners.

    Returns:
        pt1: (B, 2) closest point on rect1.
        pt2: (B, 2) closest point on rect2.
    """
    B = rect1.shape[0]
    device = rect1.device
    dtype = rect1.dtype

    # Build two vertex-to-edge configurations:
    #   Config A (rect1 vertices → rect2 edges): 4 verts × 4 edges = 16 queries per pair.
    #   Config B (rect2 vertices → rect1 edges): 4 verts × 4 edges = 16 queries per pair.
    # Concatenated → 32 queries per pair, shaped (B, 32, 2) for q/sa/sb.
    r1_edges_a = rect1  # (B, 4, 2)
    r1_edges_b = torch.roll(rect1, -1, dims=1)  # (B, 4, 2)
    r2_edges_a = rect2
    r2_edges_b = torch.roll(rect2, -1, dims=1)

    # Config A flat: 4 verts × 4 edges = 16 (B, 16, 2/2/2)
    vA_q = rect1.unsqueeze(2).expand(-1, -1, 4, -1).reshape(B, 16, 2)  # query pt
    vA_sa = r2_edges_a.unsqueeze(1).expand(-1, 4, -1, -1).reshape(B, 16, 2)  # seg start
    vA_sb = r2_edges_b.unsqueeze(1).expand(-1, 4, -1, -1).reshape(B, 16, 2)
    # Config B flat
    vB_q = rect2.unsqueeze(2).expand(-1, -1, 4, -1).reshape(B, 16, 2)
    vB_sa = r1_edges_a.unsqueeze(1).expand(-1, 4, -1, -1).reshape(B, 16, 2)
    vB_sb = r1_edges_b.unsqueeze(1).expand(-1, 4, -1, -1).reshape(B, 16, 2)

    q = torch.cat([vA_q, vB_q], dim=1)  # (B, 32, 2)
    sa = torch.cat([vA_sa, vB_sa], dim=1)
    sb = torch.cat([vA_sb, vB_sb], dim=1)

    # Foot of perpendicular from q onto segment (sa, sb), clamped to [0, 1].
    seg = sb - sa  # (B, 32, 2)
    seg_len2 = (seg * seg).sum(dim=-1).clamp_min(1e-12)
    t = ((q - sa) * seg).sum(dim=-1) / seg_len2
    t = t.clamp(0.0, 1.0)
    foot = sa + t.unsqueeze(-1) * seg  # (B, 32, 2)
    d = (q - foot).norm(dim=-1)  # (B, 32)

    # Best index per pair; in first 16 q is on rect1, in last 16 q is on rect2.
    best = d.argmin(dim=-1)  # (B,)
    arange = torch.arange(B, device=device)
    q_best = q[arange, best]  # (B, 2)
    foot_best = foot[arange, best]  # (B, 2)

    # If best < 16, q is on rect1, foot is on rect2. Else swap.
    is_on_r1 = best < 16
    pt1 = torch.where(is_on_r1.unsqueeze(-1), q_best, foot_best)
    pt2 = torch.where(is_on_r1.unsqueeze(-1), foot_best, q_best)
    return pt1.to(dtype), pt2.to(dtype)


@torch.no_grad()
def compute_static_collision_penalty(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    neighbor_futures: torch.Tensor,
    neighbor_shapes: torch.Tensor,
    neighbor_valid: torch.Tensor,
    config: RewardConfig | None = None,
) -> dict:
    """Penalise ego trajectories that approach/overlap STOPPED neighbors.

    Stopped mask per neighbor:
      (a) |v0| < sc_neighbor_vel_thresh  (approx from first two valid future points)
      (b) max displacement from t=0 across the valid future < sc_neighbor_disp_thresh

    Gotcha: the v0 estimate requires BOTH t=0 and t=1 valid in
    ``neighbor_valid``. Neighbors with sparse GT futures (e.g. a tracker
    that dropped the agent after the first frame) therefore fail the
    mask and are silently ignored, even if the agent was visibly parked
    at t=0. This is by design — ``v0`` isn't meaningful without t=1 —
    but callers that work off real-data NPZs with spotty perception
    should be aware.

    Returns a dict with:
      crossing_gate:        (N,) 1.0 if no crossing, 0.0 if any predicted timestep
                            (t>=1, ego moving) has clearance < sc_cross_thresh
      near_penalty:         (N,) staged near-zone penalty (frac or survival)
      wide_penalty:         (N,) staged wide-zone penalty (frac or survival)
      cont_penalty:         (N,) continuous decay penalty
      first_crossing_steps: list[N] first timestep of crossing per trajectory
      per_timestep_min:     (N, T) min OBB clearance to any stopped neighbor at t
                            (unmasked — contains true values even for t=0 and for
                            ego-stopped steps; the gate/penalty excludes them)
      argmin_neighbor:      (N, T) int — index into the stopped-neighbor subset for the
                            min at each timestep (-1 if no stopped neighbors at all)
      stopped_mask:         (N_nb,) bool — which input neighbors were classified as stopped
      ego_closest_pt:       (N, T, 2) world-frame ego point at the min-clearance pair
      npc_closest_pt:       (N, T, 2) world-frame neighbor point at the min-clearance pair
    """
    if config is None:
        config = RewardConfig()

    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    dtype = ego_trajs.dtype
    N_nb = neighbor_futures.shape[0]

    def _safe_empty(_stopped_mask: torch.Tensor | None = None) -> dict:
        return {
            "crossing_gate": torch.ones(N, device=device),
            "near_penalty": torch.zeros(N, device=device),
            "wide_penalty": torch.zeros(N, device=device),
            "cont_penalty": torch.zeros(N, device=device),
            "first_crossing_steps": [None] * N,
            "per_timestep_min": torch.full((N, T), 99.0, device=device),
            "argmin_neighbor": torch.full((N, T), -1, dtype=torch.long, device=device),
            "stopped_mask": _stopped_mask
            if _stopped_mask is not None
            else torch.zeros(N_nb, dtype=torch.bool, device=device),
            "ego_closest_pt": torch.zeros(N, T, 2, device=device),
            "npc_closest_pt": torch.zeros(N, T, 2, device=device),
        }

    if N_nb == 0:
        return _safe_empty()

    # --- Stopped-neighbor mask ---
    nb_xy = neighbor_futures[:, :, :2]  # (N_nb, T, 2)
    # v0: norm of (xy[:,1] - xy[:,0]) / dt, only valid if both steps valid.
    both_valid_01 = neighbor_valid[:, 0] & neighbor_valid[:, 1]
    v0 = torch.zeros(N_nb, device=device)
    if both_valid_01.any():
        v0[both_valid_01] = (nb_xy[both_valid_01, 1] - nb_xy[both_valid_01, 0]).norm(
            dim=-1
        ) / config.dt
    # max displacement from t=0 over valid timesteps
    disp_all = (nb_xy - nb_xy[:, 0:1]).norm(dim=-1)  # (N_nb, T)
    disp_masked = disp_all.masked_fill(~neighbor_valid, 0.0)
    max_disp = disp_masked.max(dim=1).values  # (N_nb,)
    has_any_valid = neighbor_valid.any(dim=1)  # (N_nb,)

    stopped_mask = (
        has_any_valid
        & both_valid_01  # need v0 to be meaningful
        & (v0 < config.sc_neighbor_vel_thresh)
        & (max_disp < config.sc_neighbor_disp_thresh)
    )

    if not stopped_mask.any():
        return _safe_empty(_stopped_mask=stopped_mask)

    # --- Ego + stopped-neighbor corners ---
    nb_f_s = neighbor_futures[stopped_mask]  # (S, T, 4)
    nb_shapes_s = neighbor_shapes[stopped_mask]  # (S, 2) [width, length]
    nb_valid_s = neighbor_valid[stopped_mask]  # (S, T)
    S = nb_f_s.shape[0]

    ego_corners = _build_ego_bbox_corners(ego_trajs, ego_shape)  # (N, T, 4, 2)

    npc_cos = nb_f_s[:, :, 2]
    npc_sin = nb_f_s[:, :, 3]
    npc_norm = (npc_cos**2 + npc_sin**2).sqrt().clamp_min(1e-6)
    npc_cos = npc_cos / npc_norm
    npc_sin = npc_sin / npc_norm
    npc_width = nb_shapes_s[:, 0].unsqueeze(1).expand(-1, T)  # (S, T)
    npc_length = nb_shapes_s[:, 1].unsqueeze(1).expand(-1, T)

    npc_rect = torch.stack(
        [
            nb_f_s[..., 0],
            nb_f_s[..., 1],
            npc_cos,
            npc_sin,
            npc_length,
            npc_width,
        ],
        dim=-1,
    )  # (S, T, 6)
    npc_corners = center_rect_to_points(npc_rect.reshape(-1, 6)).reshape(S, T, 4, 2)

    # --- Signed distance over all ego × stopped-nb × T pairs ---
    # Two-step: SAT gives a reliable overlap sign (negative = penetration depth),
    # but for separated rectangles SAT only returns the MIN axis gap, which
    # under-reports true Euclidean clearance in the corner-to-corner case
    # (e.g. x-gap 0.2m but y-gap 3m → true dist ≈ 3m, not 0.2m). So:
    #   * overlapping pairs  → use SAT penetration depth (signed negative).
    #   * separated pairs    → use true Euclidean closest-pair distance.
    ego_exp = ego_corners.unsqueeze(1).expand(-1, S, -1, -1, -1)  # (N, S, T, 4, 2)
    npc_exp = npc_corners.unsqueeze(0).expand(N, -1, -1, -1, -1)  # (N, S, T, 4, 2)
    nv_exp = nb_valid_s.unsqueeze(0).expand(N, -1, -1)  # (N, S, T)

    ego_flat = ego_exp.reshape(-1, 4, 2)
    npc_flat = npc_exp.reshape(-1, 4, 2)

    sat_dist_flat = batch_signed_distance_rect(ego_flat, npc_flat)  # (N*S*T,)
    pt_e_all, pt_n_all = _closest_points_between_rects(ego_flat, npc_flat)
    euclid_dist_flat = (pt_e_all - pt_n_all).norm(dim=-1)

    # Overlap → use SAT penetration (negative). Separated → use true
    # Euclidean closest-pair distance. SAT alone under-reports clearance
    # in corner-to-corner separated cases (e.g. 0.2 m axis-gap with 3 m
    # true distance), which would shift zone classifications and change
    # absolute reward magnitudes — we need the Euclidean branch for both
    # the training signal AND viz.
    is_overlap = sat_dist_flat < 0
    signed_dist_flat = torch.where(is_overlap, sat_dist_flat, euclid_dist_flat)

    distances = signed_dist_flat.reshape(N, S, T)
    pt_e_nst = pt_e_all.reshape(N, S, T, 2)
    pt_n_nst = pt_n_all.reshape(N, S, T, 2)

    distances = distances.masked_fill(~nv_exp, 1e6)

    # Per-timestep min across stopped neighbors, remember which offender
    per_ts_min, argmin_s = distances.min(dim=1)  # (N, T), (N, T)

    # --- Gather closest-point pair for the per-timestep winning neighbor ---
    arange_t = torch.arange(T, device=device)
    arange_n = torch.arange(N, device=device).unsqueeze(-1).expand(-1, T)  # (N, T)
    t_idx = arange_t.unsqueeze(0).expand(N, -1)  # (N, T)
    ego_closest_pt = pt_e_nst[arange_n, argmin_s, t_idx]  # (N, T, 2)
    npc_closest_pt = pt_n_nst[arange_n, argmin_s, t_idx]  # (N, T, 2)

    # --- Ego-speed mask: only score timesteps where ego is moving ---
    ego_xy = ego_trajs[:, :, :2]
    ego_vel = torch.diff(ego_xy, dim=1) / config.dt  # (N, T-1, 2)
    ego_speed = ego_vel.norm(dim=-1)  # (N, T-1)
    ego_speed = torch.cat([ego_speed, ego_speed[:, -1:]], dim=1)  # (N, T)
    ego_moving = ego_speed > config.sc_ego_min_speed  # (N, T)

    # Thresholds
    cross_thresh = float(config.sc_cross_thresh)
    near_thresh = float(config.sc_near_thresh)
    wide_thresh = float(config.sc_wide_thresh)
    cont_thresh = float(config.sc_cont_thresh)

    # Crossing: clearance below cross_thresh at any timestep where either
    # (a) t=0 (ego starts overlapping — always a collision regardless of
    #     speed), or (b) t>=1 with ego moving.
    is_crossing_full = per_ts_min < cross_thresh  # (N, T), raw
    gate_steps = ego_moving.clone()
    gate_steps[:, 0] = True  # t=0 overlap is always a crossing
    is_crossing_gated = is_crossing_full & gate_steps  # (N, T)
    has_crossing = is_crossing_gated.any(dim=1)
    crossing_gate = (~has_crossing).float()

    first_crossing_steps: list[int | None] = []
    for i in range(N):
        if has_crossing[i]:
            idx = is_crossing_gated[i].nonzero(as_tuple=True)[0][0]
            first_crossing_steps.append(int(idx.item()))
        else:
            first_crossing_steps.append(None)

    # Staged categories (among t>=1, ego moving, not crossing).
    valid_steps = gate_steps & ~is_crossing_full  # (N, T) — non-crossing scoreable
    pm_1 = per_ts_min[:, 1:]
    valid_1 = valid_steps[:, 1:]
    T_valid = T - 1

    if config.sc_penalty_mode == "survival":
        # Survival-mode first-violation time-decay over t=1..T-1.
        is_near = valid_1 & (pm_1 < near_thresh)
        is_wide = valid_1 & (pm_1 >= near_thresh) & (pm_1 < wide_thresh)

        near_penalty = torch.zeros(N, device=device)
        near_has = is_near.any(dim=1)
        near_first_t = is_near.to(torch.int64).argmax(dim=1)
        near_penalty[near_has] = (T_valid - near_first_t[near_has].float()) / T_valid

        wide_penalty = torch.zeros(N, device=device)
        wide_has = is_wide.any(dim=1)
        wide_first_t = is_wide.to(torch.int64).argmax(dim=1)
        wide_penalty[wide_has] = (T_valid - wide_first_t[wide_has].float()) / T_valid

        cont_penalty = torch.zeros(N, device=device)
        if cont_thresh > 0:
            in_range = valid_1 & (pm_1 < cont_thresh)
            cont_has = in_range.any(dim=1)
            cont_first_t = in_range.to(torch.int64).argmax(dim=1)
            worst_dist = pm_1.masked_fill(~in_range, float("inf")).min(dim=1).values
            cont_penalty[cont_has] = (
                (1.0 - worst_dist[cont_has] / cont_thresh).clamp(0, 1)
                * (T_valid - cont_first_t[cont_has].float())
                / T_valid
            )
    else:
        # "frac" mode: fraction of scoreable timesteps in each band.
        denom = float(T_valid) if T_valid > 0 else 1.0
        near_penalty = (valid_1 & (pm_1 < near_thresh)).float().sum(dim=1) / denom
        wide_penalty = (valid_1 & (pm_1 >= near_thresh) & (pm_1 < wide_thresh)).float().sum(
            dim=1
        ) / denom
        if cont_thresh <= 0:
            cont_penalty = torch.zeros(N, device=device)
        else:
            cont_terms = torch.where(
                valid_1,
                (1.0 - pm_1 / cont_thresh).clamp(min=0, max=1),
                torch.zeros_like(pm_1),
            )
            cont_penalty = cont_terms.sum(dim=1) / denom

    # Global stopped-neighbor index map: map argmin_s (into stopped subset)
    # back to the original neighbor index (so callers can cross-reference the
    # raw neighbor_agents_future tensor).
    stopped_global_idx = stopped_mask.nonzero(as_tuple=True)[0]  # (S,)
    argmin_global = stopped_global_idx[argmin_s]  # (N, T)

    return {
        "crossing_gate": crossing_gate,
        "near_penalty": near_penalty,
        "wide_penalty": wide_penalty,
        "cont_penalty": cont_penalty,
        "first_crossing_steps": first_crossing_steps,
        "per_timestep_min": per_ts_min,
        "argmin_neighbor": argmin_global,
        "stopped_mask": stopped_mask,
        "ego_closest_pt": ego_closest_pt,
        "npc_closest_pt": npc_closest_pt,
    }


# ---------------------------------------------------------------------------
# Lane departure penalty
# ---------------------------------------------------------------------------
#
# Detects whether the ego vehicle leaves the drivable lane area using polygon
# containment and measures proximity to the road edge (outer lane boundary).
#
# Algorithm overview:
#
#   1. **Lane polygon construction** (_build_lane_polygons):
#      Each lane in the NPZ data has left/right boundary offsets relative to
#      its centerline. We construct closed polygons per lane:
#        boundary_point = centerline + offset   (SFT interpretation)
#      Each polygon has left edges (forward winding), right edges (reversed
#      winding), and two closing edges connecting the ends.
#
#   2. **K-nearest lane selection**:
#      To bound GPU memory, only the K=12 closest lanes (by min centerline-
#      point distance to trajectory bbox center) are used. Verified to produce
#      identical results to using all lanes on 100 validation scenes.
#
#   3. **Outer boundary classification** (_classify_outer_boundaries):
#      Not all lane boundary segments are road edges — segments shared between
#      adjacent lanes are interior. We classify via midpoint nudge:
#        a) Nudge segment midpoint outward (perpendicular to lane direction)
#        b) If nudged point falls inside any lane polygon → shared boundary
#        c) If outside but within 0.5m of a different lane's segment → junction
#           gap (still shared, common at intersections)
#        d) Otherwise → road edge (outer boundary)
#      Only outer segments are used for distance-based soft penalties.
#
#   4. **Ego perimeter sampling**:
#      36 points around the ego rectangle (10 per side, corners not duplicated),
#      rotated and translated to world coordinates at each timestep.
#
#   5. **Containment check** (_point_in_polygons):
#      GPU ray casting (+x direction) against all polygon edges. A point is
#      inside if it has an odd number of edge crossings for any polygon.
#      Pre-filters edges by Y-range and X-min to reduce the Q×E matrix.
#      Chunks query points when Q×E > 10M to prevent OOM.
#
#   6. **Distance to road edge**:
#      For perimeter points that ARE inside a lane, compute min distance to
#      the nearest outer boundary segment. Points outside a lane get distance=0
#      (they are already penalized by the crossing gate, not by proximity).
#
#   7. **Exclusive zone categories** (per timestep, skipping t=0):
#      - OUT:  any perimeter point outside all lane polygons → crossing_gate=0
#      - NEAR: all points inside, but min distance to road edge < 0.25m
#      - WIDE: all points inside, min distance between 0.25m and 0.40m
#      - SAFE: all points inside, min distance >= 0.40m
#      Each timestep belongs to exactly one category (no double counting).
#      near_frac and wide_frac report the fraction of in-lane timesteps in
#      each band. cont_penalty is a continuous linear decay from 0.80m.
#
# ---------------------------------------------------------------------------

_LANE_NEAR_THRESH = 0.25
_LANE_WIDE_THRESH = 0.40
_LANE_CONT_THRESH = 0.80
_LANE_PTS_PER_SIDE = 10  # 36 unique perimeter points (corners not duplicated)


def _build_lane_polygons(
    lanes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build lane polygon edges from lane tensor. Vectorized, no python loops.

    Each lane polygon = left boundary edges + right boundary edges (reversed winding)
    + two closing edges connecting the ends. Boundary = center + offset.

    Args:
        lanes: (S, P, D) lane tensor.

    Returns:
        edge_v1: (E, 2) polygon edge start vertices (all polygons concatenated)
        edge_v2: (E, 2) polygon edge end vertices
        edge_poly_id: (E,) int — which polygon each edge belongs to
        n_polys: total number of polygons
    """
    S, P, D = lanes.shape
    device = lanes.device

    center = lanes[..., :2]  # (S, P, 2)
    valid = center.norm(dim=-1) > 1e-3  # (S, P)
    left_pts = center + lanes[..., 4:6]
    right_pts = center + lanes[..., 6:8]

    n_valid = valid.sum(dim=1)
    has_poly = n_valid >= 2

    if not has_poly.any():
        z = torch.zeros(0, 2, device=device)
        return z, z, torch.zeros(0, dtype=torch.int32, device=device), 0

    poly_id_per_lane = torch.cumsum(has_poly.int(), dim=0) - 1  # (S,)
    n_polys = int(has_poly.sum().item())

    # Consecutive boundary edges (left forward, right reversed for winding)
    valid_pair = valid[:, :-1] & valid[:, 1:]  # (S, P-1)
    lane_ids_pair = torch.arange(S, device=device).unsqueeze(1).expand(S, P - 1)
    idx = torch.where(valid_pair.reshape(-1))[0]

    if len(idx) == 0:
        z = torch.zeros(0, 2, device=device)
        return z, z, torch.zeros(0, dtype=torch.int32, device=device), 0

    l_v1 = left_pts[:, :-1].reshape(-1, 2)[idx]
    l_v2 = left_pts[:, 1:].reshape(-1, 2)[idx]
    r_v1 = right_pts[:, 1:].reshape(-1, 2)[idx]  # reversed winding
    r_v2 = right_pts[:, :-1].reshape(-1, 2)[idx]
    edge_pid = poly_id_per_lane[lane_ids_pair.reshape(-1)[idx]]

    # Closing edges: connect left end→right end and right start→left start
    pl = torch.where(has_poly)[0]
    fv = valid.float().argmax(dim=1)[pl]
    lv = P - 1 - valid.flip(1).float().argmax(dim=1)[pl]
    c1_v1 = left_pts[pl, lv]
    c1_v2 = right_pts[pl, lv]
    c2_v1 = right_pts[pl, fv]
    c2_v2 = left_pts[pl, fv]
    c_pid = poly_id_per_lane[pl].int()

    all_v1 = torch.cat([l_v1, r_v1, c1_v1, c2_v1])
    all_v2 = torch.cat([l_v2, r_v2, c1_v2, c2_v2])
    all_pid = torch.cat([edge_pid, edge_pid, c_pid, c_pid]).int()

    return all_v1, all_v2, all_pid, n_polys


def _point_in_polygons(
    points: torch.Tensor,
    edge_v1: torch.Tensor,
    edge_v2: torch.Tensor,
    edge_poly_id: torch.Tensor,
    n_polys: int,
) -> torch.Tensor:
    """GPU-parallel point-in-polygon via ray casting. No python loops.

    Args:
        points: (Q, 2) query points.
        edge_v1, edge_v2: (E, 2) polygon edge endpoints.
        edge_poly_id: (E,) which polygon each edge belongs to.
        n_polys: total number of polygons.

    Returns:
        inside: (Q,) bool — True if inside ANY polygon.
    """
    Q = points.shape[0]
    E = edge_v1.shape[0]
    device = points.device

    if E == 0 or n_polys == 0:
        return torch.zeros(Q, dtype=torch.bool, device=device)

    px = points[:, 0]
    py = points[:, 1]
    v1x, v1y = edge_v1[:, 0], edge_v1[:, 1]
    v2x, v2y = edge_v2[:, 0], edge_v2[:, 1]

    # Prefilter: discard edges that can't be crossed by any query point's +x ray
    keep = (
        (torch.maximum(v1x, v2x) >= px.min())
        & (torch.maximum(v1y, v2y) >= py.min())
        & (torch.minimum(v1y, v2y) <= py.max())
    )

    if not keep.any():
        return torch.zeros(Q, dtype=torch.bool, device=device)

    v1x = v1x[keep]
    v1y = v1y[keep]
    v2x = v2x[keep]
    v2y = v2y[keep]
    edge_poly_id = edge_poly_id[keep]
    E = v1x.shape[0]

    # Chunk over query points when Q×E is large to avoid OOM
    _MAX_QE = 10_000_000  # ~200 MB accounting for multiple intermediates (bool, float, int64 index)
    chunk_size = max(1, _MAX_QE // E) if E > 0 else Q

    if chunk_size >= Q:
        return _pip_core(px, py, v1x, v1y, v2x, v2y, edge_poly_id, E, n_polys, device)

    results = []
    for start in range(0, Q, chunk_size):
        end = min(start + chunk_size, Q)
        results.append(
            _pip_core(
                px[start:end],
                py[start:end],
                v1x,
                v1y,
                v2x,
                v2y,
                edge_poly_id,
                E,
                n_polys,
                device,
            )
        )
    return torch.cat(results)


def _pip_core(
    px: torch.Tensor,
    py: torch.Tensor,
    v1x: torch.Tensor,
    v1y: torch.Tensor,
    v2x: torch.Tensor,
    v2y: torch.Tensor,
    edge_poly_id: torch.Tensor,
    E: int,
    n_polys: int,
    device: torch.device,
) -> torch.Tensor:
    """Core ray-casting kernel for a chunk of query points."""
    Q = px.shape[0]
    py_exp = py[:, None]
    above1 = v1y[None, :] > py_exp
    above2 = v2y[None, :] > py_exp
    straddles = above1 != above2

    dy = (v2y - v1y)[None, :]
    dy_safe = dy.clone()
    dy_safe[dy_safe.abs() < 1e-10] = 1.0
    t = (py_exp - v1y[None, :]) / dy_safe
    x_int = v1x[None, :] + t * (v2x - v1x)[None, :]

    crossing = straddles & (x_int > px[:, None])

    counts = torch.zeros(Q, n_polys, dtype=torch.int32, device=device)
    counts.scatter_add_(1, edge_poly_id[None, :].expand(Q, E).long(), crossing.int())

    inside_any = ((counts % 2) == 1).any(dim=1)
    return inside_any


def _point_to_segments_dist(
    points: torch.Tensor,
    seg_p1: torch.Tensor,
    seg_p2: torch.Tensor,
) -> torch.Tensor:
    """Distance from each point to each segment. Fully parallel on GPU.

    Args:
        points: (Q, 2)
        seg_p1, seg_p2: (E, 2)

    Returns:
        dist: (Q, E) distance matrix.
    """
    seg = seg_p2 - seg_p1
    seg_len2 = (seg**2).sum(-1).clamp(min=1e-10)
    diff = points[:, None, :] - seg_p1[None, :, :]
    t = ((diff * seg[None, :, :]).sum(-1) / seg_len2[None, :]).clamp(0, 1)
    closest = seg_p1[None, :, :] + t[:, :, None] * seg[None, :, :]
    return (points[:, None, :] - closest).norm(dim=-1)


def _point_to_segments_min_dist(
    points: torch.Tensor,
    seg_p1: torch.Tensor,
    seg_p2: torch.Tensor,
) -> torch.Tensor:
    """Min distance from each point to nearest segment. Chunks to avoid OOM.

    Like _point_to_segments_dist but only returns (Q,) min distances
    instead of the full (Q, E) matrix. Chunks over query points when
    Q×E > 10M elements.

    Args:
        points: (Q, 2)
        seg_p1, seg_p2: (E, 2)

    Returns:
        min_dist: (Q,) min distance per point.
    """
    Q = points.shape[0]
    E = seg_p1.shape[0]
    _MAX_QE = 10_000_000
    chunk_size = max(1, _MAX_QE // E) if E > 0 else Q

    if chunk_size >= Q:
        return _point_to_segments_dist(points, seg_p1, seg_p2).min(dim=1).values

    results = []
    for start in range(0, Q, chunk_size):
        end = min(start + chunk_size, Q)
        d = _point_to_segments_dist(points[start:end], seg_p1, seg_p2)
        results.append(d.min(dim=1).values)
    return torch.cat(results)


def _points_inside_intersection_areas(
    points: torch.Tensor,
    polygons_tensor: torch.Tensor,
) -> torch.Tensor:
    """Test whether each point lies inside ANY intersection_area polygon.

    Uses horizontal-ray casting, fully batched.

    Args:
        points: (Q, 2).
        polygons_tensor: (Np, P, 2+K) per-scene polygons (from NPZ `polygons`).
            Per-point validity is derived from ||xy|| > 1e-3. Polygons with
            fewer than 3 valid points are ignored.

    Returns:
        (Q,) bool — True if the point is inside at least one polygon.
    """
    Q = points.shape[0]
    device = points.device
    inside_any = torch.zeros(Q, dtype=torch.bool, device=device)
    if polygons_tensor.shape[-1] < 2:
        return inside_any
    pg_xy = polygons_tensor[..., :2]  # (Np, P, 2)
    pg_valid = pg_xy.norm(dim=-1) > 1e-3  # (Np, P)
    Np = pg_xy.shape[0]
    for p_idx in range(Np):
        mask = pg_valid[p_idx]
        if mask.sum() < 3:
            continue
        verts = pg_xy[p_idx][mask]  # (Pv, 2)
        v1 = verts
        v2 = torch.roll(verts, -1, dims=0)
        px = points[:, 0:1]
        py = points[:, 1:2]
        y1 = v1[:, 1][None, :]
        y2 = v2[:, 1][None, :]
        x1 = v1[:, 0][None, :]
        x2 = v2[:, 0][None, :]
        cond_y = (y1 > py) != (y2 > py)
        denom = y2 - y1
        safe_denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
        x_intersect = x1 + (py - y1) * (x2 - x1) / safe_denom
        cond = cond_y & (x_intersect > px)
        crossings = cond.sum(dim=-1)
        inside_p = (crossings % 2) == 1
        inside_any = inside_any | inside_p
    return inside_any


def _point_to_segments_signed_min_dist(
    points: torch.Tensor,
    seg_p1: torch.Tensor,
    seg_p2: torch.Tensor,
    seg_outward: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each point, find nearest segment and return unsigned + signed distance.

    Signed distance: positive if point is on the outward side of its nearest
    segment, negative if inside. Magnitude equals unsigned distance.

    Fully batched on GPU, chunked to stay under ~10M Q×E elements.

    Args:
        points: (Q, 2).
        seg_p1, seg_p2: (E, 2) segment endpoints.
        seg_outward: (E, 2) outward unit vector per segment (perpendicular
            to segment direction, pointing away from the lane interior).

    Returns:
        unsigned_dist: (Q,) min distance per point.
        signed_dist: (Q,) (query - closest_on_segment) · seg_outward[argmin_seg].
    """
    Q = points.shape[0]
    E = seg_p1.shape[0]
    if E == 0:
        return (
            torch.full((Q,), 100.0, device=points.device, dtype=points.dtype),
            torch.full((Q,), -100.0, device=points.device, dtype=points.dtype),
        )

    seg = seg_p2 - seg_p1  # (E, 2)
    seg_len2 = (seg**2).sum(-1).clamp(min=1e-10)  # (E,)

    _MAX_QE = 10_000_000
    chunk_size = max(1, _MAX_QE // E)

    unsigned_all = []
    signed_all = []

    for start in range(0, Q, chunk_size):
        end = min(start + chunk_size, Q)
        chunk = points[start:end]
        diff = chunk[:, None, :] - seg_p1[None, :, :]
        t_raw = (diff * seg[None, :, :]).sum(-1) / seg_len2[None, :]
        is_unclamped = (t_raw > 0.0) & (t_raw < 1.0)
        t = t_raw.clamp(0, 1)
        closest = seg_p1[None, :, :] + t[:, :, None] * seg[None, :, :]
        to_query = chunk[:, None, :] - closest
        dist = to_query.norm(dim=-1)

        # Find the actually-nearest segment per query (clamped or not).
        min_dist, min_idx = dist.min(dim=1)

        # If the nearest segment's projection is CLAMPED (foot lies at an
        # endpoint), the query is past the segment's endpoint — don't flag as
        # crossing. Falling back to some other distant unclamped segment would
        # produce a spurious outward-projection reading because "outward" is
        # only meaningful perpendicular to the segment.
        nearest_unclamped = is_unclamped.gather(1, min_idx[:, None]).squeeze(-1)

        gathered_to_query = to_query.gather(1, min_idx[:, None, None].expand(-1, 1, 2)).squeeze(1)
        outward_for_min = seg_outward[min_idx]
        signed_raw = (gathered_to_query * outward_for_min).sum(-1)
        signed = torch.where(nearest_unclamped, signed_raw, torch.full_like(signed_raw, -100.0))

        unsigned_all.append(min_dist)
        signed_all.append(signed)

    return torch.cat(unsigned_all), torch.cat(signed_all)


def _classify_outer_boundaries(
    seg_p1: torch.Tensor,
    seg_p2: torch.Tensor,
    seg_dir: torch.Tensor,
    seg_lane: torch.Tensor,
    edge_v1: torch.Tensor,
    edge_v2: torch.Tensor,
    edge_poly_id: torch.Tensor,
    n_polys: int,
    nudge: float = 0.05,
    gap_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify boundary segments as outer (road edge) via midpoint nudge + containment.

    For each segment, nudge its midpoint outward (perpendicular to lane direction).
    If the nudged point lands inside any lane polygon → shared boundary.
    If outside but close to a different lane's boundary → junction gap (shared).
    Otherwise → road edge (outer).

    Segments alternate left/right per lane: even=left boundary, odd=right boundary.

    Args:
        seg_p1, seg_p2: (M, 2) boundary segment endpoints.
        seg_dir: (M, 2) unit lane direction at each segment.
        seg_lane: (M,) lane index.
        edge_v1, edge_v2: polygon edge vertices for containment check.
        edge_poly_id: polygon IDs for edges.
        n_polys: total polygon count.
        nudge: outward nudge distance in meters.
        gap_threshold: max distance to different-lane segment to be a junction gap.

    Returns:
        is_outer: (M,) bool.
        outward: (M, 2) outward unit vector per segment (away from lane interior).
    """
    M = seg_p1.shape[0]
    device = seg_p1.device

    # Midpoint of each segment
    mid = (seg_p1 + seg_p2) / 2

    # Outward normal from lane direction: left_normal = (-dy, dx)
    left_normal = torch.stack([-seg_dir[:, 1], seg_dir[:, 0]], dim=-1)

    # Even indices = left boundary → outward = left normal
    # Odd indices = right boundary → outward = -left normal (right normal)
    is_left = torch.arange(M, device=device) % 2 == 0
    outward = torch.where(is_left[:, None], left_normal, -left_normal)
    outward = outward / outward.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    nudged = mid + nudge * outward

    # Check if nudged point is inside any polygon
    inside = _point_in_polygons(nudged, edge_v1, edge_v2, edge_poly_id, n_polys)

    # Inside → shared. Outside → candidate road edge.
    candidate_outer = ~inside

    # At intersections, nudged point may land in gap between polygons.
    # If close to a different lane's boundary segment → junction gap, not road edge.
    if candidate_outer.any():
        nudged_outer = nudged[candidate_outer]
        d = _point_to_segments_dist(nudged_outer, seg_p1, seg_p2)  # (n_cand, M)
        # Mask out same-lane segments
        outer_lane = seg_lane[candidate_outer]
        same_lane_mask = outer_lane[:, None] == seg_lane[None, :]
        d[same_lane_mask] = 999.0
        # Close to different-lane segment → junction gap
        min_d = d.min(dim=1).values
        is_junction_gap = min_d < gap_threshold
        outer_indices = torch.where(candidate_outer)[0]
        candidate_outer[outer_indices[is_junction_gap]] = False

    return candidate_outer, outward


_LANE_K_NEAREST = 12  # number of nearest lanes to consider (0 = all)


@torch.no_grad()
def compute_lane_departure_penalty(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    data: dict[str, torch.Tensor],
    k_nearest_lanes: int = _LANE_K_NEAREST,
    config: RewardConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int | None], torch.Tensor]:
    """Compute lane departure using polygon containment + distance to road edge. Pure torch.

    1. Polygon containment (GPU ray casting) for crossing gate.
    2. Distance to outer boundary segments for near/wide/cont soft penalties.
    Lane boundaries use SFT interpretation: boundary = center + offset.

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        ego_shape: (3,) wheel_base, length, width.
        data: Observation dict with 'lanes' key.
        k_nearest_lanes: Only consider K nearest lanes (by min centerline distance).
            0 = use all lanes. Default 12.
        config: RewardConfig with threshold overrides. None = defaults.

    Returns:
        Tuple of (crossing_gate, near_frac, wide_frac, lane_crossing_steps, cont_penalty):
        - crossing_gate: (N,) 1.0 if fully inside lane, 0.0 if any timestep exits
        - near_frac: (N,) fraction of evaluated timesteps (excluding t=0) where
          the traj is in-lane AND min distance to outer boundary < lane_near_thresh.
          Crossing (out-of-lane) timesteps contribute 0.
        - wide_frac: (N,) fraction of evaluated timesteps (excluding t=0) where
          the traj is in-lane AND distance ∈ [lane_near_thresh, lane_wide_thresh).
          Crossing timesteps contribute 0.
        - lane_crossing_steps: list of N (int | None) — first timestep of lane exit
        - cont_penalty: (N,) continuous proximity penalty (linear decay from lane_cont_thresh)
    """
    if config is None:
        config = RewardConfig()

    N, T, _ = ego_trajs.shape
    device = ego_trajs.device

    no_steps: list[int | None] = [None] * N
    safe = (
        torch.ones(N, device=device),
        torch.zeros(N, device=device),
        torch.zeros(N, device=device),
        no_steps,
        torch.zeros(N, device=device),
    )

    if "lanes" not in data:
        return safe
    lanes = data["lanes"]
    if lanes.dim() == 4:
        lanes = lanes[0]
    if lanes.shape[-1] < 8:
        return safe

    S, P, D = lanes.shape

    # Select K nearest lanes by min centerline-point distance to any trajectory point
    if k_nearest_lanes > 0 and S > k_nearest_lanes:
        center_all = lanes[..., :2]  # (S, P, 2)
        valid_all = center_all.norm(dim=-1) > 1e-3
        # Use trajectory bbox center + half-diagonal as reference
        # to catch lanes near any part of the trajectory
        traj_xy = ego_trajs[:, :, :2].reshape(-1, 2)  # (N*T, 2)
        traj_min = traj_xy.min(dim=0).values
        traj_max = traj_xy.max(dim=0).values
        traj_center = (traj_min + traj_max) / 2
        # Min distance from each centerline point to trajectory center
        dist_to_pts = (center_all - traj_center).norm(dim=-1)  # (S, P)
        dist_to_pts[~valid_all] = 1e6
        min_dist_per_lane = dist_to_pts.min(dim=1).values  # (S,)
        # Also include lanes within bbox + margin
        half_diag = (traj_max - traj_min).norm() / 2 + 5.0  # margin
        has_lane = valid_all.any(dim=1)
        min_dist_per_lane[~has_lane] = 1e6
        # Take max(K, lanes within bbox) to be safe
        # Note: .sum().item() syncs CPU↔GPU but runs once per scene (not per traj), negligible cost
        n_nearby = (min_dist_per_lane < half_diag).sum().item()
        k = max(k_nearest_lanes, min(n_nearby, S))
        _, topk_idx = min_dist_per_lane.topk(k, largest=False)
        lanes = lanes[topk_idx]
        S = k

    # Build polygon edges for containment
    edge_v1, edge_v2, edge_poly_id, n_polys = _build_lane_polygons(lanes)
    if n_polys == 0:
        return safe

    # --- Step 3: Build boundary segments and classify outer vs shared ---
    # Lane tensor layout: [center_x, center_y, dir_cos, dir_sin, lb_dx, lb_dy, rb_dx, rb_dy, ...]
    center = lanes[..., :2]
    direction = lanes[..., 2:4]
    lb_offset = lanes[..., 4:6]
    rb_offset = lanes[..., 6:8]
    valid = center.norm(dim=-1) > 1e-3

    # Boundary = center + offset (SFT interpretation, NOT lateral projection)
    left_pts = center + lb_offset  # (S, P, 2)
    right_pts = center + rb_offset  # (S, P, 2)

    # Some centerline points have zero direction; fill with lane average
    dirs = direction.clone()
    has_dir = dirs.norm(dim=-1) > 1e-6  # (S, P)
    dir_sum = (dirs * has_dir.unsqueeze(-1)).sum(dim=1)  # (S, 2)
    dir_avg = dir_sum / dir_sum.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dirs = torch.where(has_dir.unsqueeze(-1), dirs, dir_avg.unsqueeze(1).expand_as(dirs))

    # Build segments between consecutive valid centerline points
    valid_pair = valid[:, :-1] & valid[:, 1:]  # (S, P-1)
    mid_dirs = (dirs[:, :-1] + dirs[:, 1:]) / 2
    mid_dirs = mid_dirs / mid_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    lane_ids = torch.arange(S, device=device).unsqueeze(1).expand(S, P - 1)

    vp_flat = valid_pair.reshape(-1)
    idx = torch.where(vp_flat)[0]

    if len(idx) == 0:
        return safe

    # Interleave left/right segments: [left_seg_0, right_seg_0, left_seg_1, right_seg_1, ...]
    # This ordering is required by _classify_outer_boundaries (even=left, odd=right)
    M = len(idx)
    l_p1 = left_pts[:, :-1].reshape(-1, 2)[idx]
    l_p2 = left_pts[:, 1:].reshape(-1, 2)[idx]
    r_p1 = right_pts[:, :-1].reshape(-1, 2)[idx]
    r_p2 = right_pts[:, 1:].reshape(-1, 2)[idx]
    md_f = mid_dirs.reshape(-1, 2)[idx]
    lid_f = lane_ids.reshape(-1)[idx]

    seg_p1 = torch.stack([l_p1, r_p1], dim=1).reshape(2 * M, 2)
    seg_p2 = torch.stack([l_p2, r_p2], dim=1).reshape(2 * M, 2)
    seg_dir = torch.stack([md_f, md_f], dim=1).reshape(2 * M, 2)
    seg_lane = torch.stack([lid_f, lid_f], dim=1).reshape(2 * M)

    is_outer, outward_all = _classify_outer_boundaries(
        seg_p1,
        seg_p2,
        seg_dir,
        seg_lane,
        edge_v1,
        edge_v2,
        edge_poly_id,
        n_polys,
    )

    # Authoritative intersection filter: drop any "outer" segment that is
    # fully covered by a map-authored intersection_area polygon. Test 5 points
    # along the segment (t=0, 0.25, 0.5, 0.75, 1.0). If ALL are inside the same
    # polygon (or any polygon), the segment lies inside an intersection and is
    # NOT a road edge — drop it from the outer set entirely. These segments
    # must not contribute to either the crossing gate or the near/wide/cont
    # distance penalties, since the ego is legally allowed to traverse them.
    inter_polys = None
    if "polygons" in data:
        pg = data["polygons"]
        if pg.dim() == 4:
            pg = pg[0]
        if pg.shape[-1] >= 2:
            inter_polys = pg
    if is_outer.any() and inter_polys is not None:
        outer_indices = torch.where(is_outer)[0]
        outer_p1_cand = seg_p1[outer_indices]
        outer_p2_cand = seg_p2[outer_indices]
        Nout = outer_p1_cand.shape[0]
        sample_ts = torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 1.0], device=device, dtype=outer_p1_cand.dtype
        )
        # (Nout, 5, 2)
        samples = (
            outer_p1_cand[:, None, :]
            + sample_ts[None, :, None] * (outer_p2_cand - outer_p1_cand)[:, None, :]
        )
        inside_flat = _points_inside_intersection_areas(
            samples.reshape(-1, 2), inter_polys
        ).reshape(Nout, -1)
        # Segment drops only if ALL sampled points are inside SOME polygon
        fully_covered = inside_flat.all(dim=-1)
        if fully_covered.any():
            is_outer_new = is_outer.clone()
            is_outer_new[outer_indices[fully_covered]] = False
            is_outer = is_outer_new

    outer_p1 = seg_p1[is_outer]
    outer_p2 = seg_p2[is_outer]
    outer_outward = outward_all[is_outer]

    # --- Step 4: Sample ego perimeter points at each timestep ---
    wb = ego_shape[0].item()
    length = ego_shape[1].item()
    width = ego_shape[2].item()
    ro = (length - wb) / 2
    lp_list = []
    for j in range(_LANE_PTS_PER_SIDE):
        f = j / (_LANE_PTS_PER_SIDE - 1)
        lp_list.append((-ro + f * length, -width / 2))  # bottom edge
        lp_list.append((-ro + f * length, width / 2))  # top edge
        if 0 < f < 1:  # skip corners (already in top/bottom)
            lp_list.append((-ro, -width / 2 + f * width))  # left edge
            lp_list.append((length - ro, -width / 2 + f * width))  # right edge
    local_pts = torch.tensor(lp_list, device=device, dtype=ego_trajs.dtype)
    K_pts = local_pts.shape[0]

    cos_h = ego_trajs[..., 2]
    sin_h = ego_trajs[..., 3]
    h_norm = (cos_h**2 + sin_h**2).sqrt().clamp_min(1e-6)
    cos_h = cos_h / h_norm
    sin_h = sin_h / h_norm
    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=-1).reshape(N, T, 2, 2)
    rotated = torch.einsum("btij,kj->btki", rot, local_pts)
    world_pts = ego_trajs[..., :2].unsqueeze(2) + rotated

    Q = N * T * K_pts
    query = world_pts.reshape(Q, 2)

    # --- Step 5: Signed-distance gate via nearest outer-boundary segment ---
    # For each ego perimeter point, find nearest outer segment and compute signed
    # distance (positive = outside lane, negative = inside). Gate mirrors RB:
    # (signed_dist > -lane_cross_thresh) → crossed.
    #
    # Intersection-area override: an ego perimeter point that lies inside a
    # map-authored intersection_area polygon is legally inside an intersection
    # and CANNOT be crossed. Its signed distance is forced to -100 so it never
    # fires the gate. Perimeter points OUTSIDE the polygon are evaluated
    # normally — the polygon is NOT a safe zone for points outside it.
    # Unsigned near/wide/cont penalties still use all perimeter points.
    lane_cross_thresh = config.lane_cross_thresh
    if outer_p1.shape[0] > 0:
        unsigned_q, signed_q = _point_to_segments_signed_min_dist(
            query, outer_p1, outer_p2, outer_outward
        )
        if inter_polys is not None:
            peri_in_inter = _points_inside_intersection_areas(query, inter_polys)
            if peri_in_inter.any():
                signed_q = torch.where(
                    peri_in_inter,
                    torch.full_like(signed_q, -100.0),
                    signed_q,
                )
        unsigned_2d = unsigned_q.reshape(N, T, K_pts)
        signed_2d = signed_q.reshape(N, T, K_pts)
        per_ts_max_signed = signed_2d.max(dim=2).values  # (N, T)
        per_ts_min = unsigned_2d.min(dim=2).values  # (N, T)
    else:
        per_ts_max_signed = torch.full((N, T), -100.0, device=device)
        per_ts_min = torch.full((N, T), 100.0, device=device)

    # `per_ts_min[:, 0]` and `per_ts_max_signed[:, 0]` now carry the TRUE t=0
    # values so downstream diagnostics (cleanse, viz) see the real starting
    # distance. The gate and near/wide penalties below still exclude t=0 from
    # their aggregation because the starting pose is not model-controllable.

    # Crossing rule (buffer-from-inside semantics, matches `rb_cross_thresh`):
    # `per_ts_max_signed` is the max signed distance across perimeter points per
    # timestep — i.e., the most-outside (highest signed) point. With sign convention
    # +outside / -inside, the gate fires when this point is less than `lane_cross_thresh`
    # METRES INSIDE the boundary, OR already past it. Larger threshold = stricter gate
    # (wider safety buffer), not looser. e.g., default 0.20m → fires when any perimeter
    # point comes within 20cm of the lane edge or beyond.
    is_crossing_ts = per_ts_max_signed > -lane_cross_thresh  # (N, T), full for diag
    has_crossing = is_crossing_ts[:, 1:].any(dim=1)  # t=0 excluded from gate
    crossing_gate = (~has_crossing).float()
    first_idx = is_crossing_ts[:, 1:].float().argmax(dim=1)
    lane_crossing_steps: list[int | None] = [
        int(first_idx[i].item()) + 1 if has_crossing[i] else None for i in range(N)
    ]

    # --- Step 6: Exclusive zone categories (skip t=0) ---
    # OUT > NEAR > WIDE > SAFE. "In-lane" = signed distance below -cross_thresh
    # (strictly inside by more than crossing margin).
    is_out_ts = is_crossing_ts[:, 1:]  # (N, T-1)
    is_in_ts = ~is_out_ts

    lane_near_thresh = config.lane_near_thresh
    lane_wide_thresh = config.lane_wide_thresh
    lane_cont_thresh = config.lane_cont_thresh

    near_frac = (is_in_ts & (per_ts_min[:, 1:] < lane_near_thresh)).float().mean(dim=1)
    wide_frac = (
        (
            is_in_ts
            & (per_ts_min[:, 1:] >= lane_near_thresh)
            & (per_ts_min[:, 1:] < lane_wide_thresh)
        )
        .float()
        .mean(dim=1)
    )
    if lane_cont_thresh <= 0:
        cont_penalty = torch.zeros(N, device=device)
    else:
        cont_penalty = torch.where(
            is_in_ts,
            (1.0 - per_ts_min[:, 1:] / lane_cont_thresh).clamp(min=0, max=1),
            torch.zeros_like(per_ts_min[:, 1:]),
        ).mean(dim=1)

    return crossing_gate, near_frac, wide_frac, lane_crossing_steps, cont_penalty


# ---------------------------------------------------------------------------
# Top-level batched reward computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_reward_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig = RewardConfig(),
) -> list[RewardBreakdown]:
    """Compute reward breakdowns for N trajectories in a single batched pass.

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        data: Observation dict from load_npz_data (with batch dim).
        config: RewardConfig with component weights.

    Returns:
        List of N RewardBreakdown instances.
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device

    # --- Ego shape ---
    # No silent fallback: the wrong default footprint silently undersized RB /
    # lane / collision gates by ~3 m of length and 0.5 m of width on larger
    # platforms, letting trajectories that visibly crossed the border pass the
    # gate. The NPZ MUST carry the correct ego_shape (wheel_base, length,
    # width); callers (parse-from-bag, disturb_and_replay, etc.) are
    # responsible for writing it.
    if "ego_shape" not in data:
        raise ValueError(
            "compute_reward_batch: data is missing 'ego_shape' (wheel_base, "
            "length, width). Refusing to fall back to a hardcoded default — "
            "this previously caused silent footprint undersizing. Populate "
            "ego_shape upstream (parse-from-bag / disturb_and_replay / scene "
            "builder) and re-run."
        )
    es = data["ego_shape"]
    if es.dim() == 2:
        es = es[0]
    if es.numel() < 3:
        raise ValueError(
            f"compute_reward_batch: ego_shape has shape {tuple(es.shape)}; "
            "expected at least 3 elements (wheel_base, length, width)."
        )
    ego_shape = es[:3].to(device)

    # --- Neighbor data for collision ---
    neighbor_futures = torch.zeros(0, T, 4, device=device)
    neighbor_shapes = torch.zeros(0, 2, device=device)
    neighbor_valid = torch.zeros(0, T, dtype=torch.bool, device=device)

    if "neighbor_agents_future" in data:
        nf = data["neighbor_agents_future"]
        if nf.dim() == 4:
            nf = nf[0]
        if nf.shape[1] >= T and nf.shape[2] >= 4:
            nf_data = nf[:, :T, :4]  # (N_nb, T, 4) = x, y, cos, sin
        elif nf.shape[0] > 0 and nf.shape[1] >= T and nf.shape[2] == 3:
            raise ValueError(
                f"neighbor_agents_future has 3 columns (x, y, heading_rad) but "
                f"4 columns (x, y, cos, sin) are required. Re-generate the NPZ "
                f"with the updated tensor_converter / _backfill_neighbor_futures."
            )
        if nf.shape[1] >= T and nf.shape[2] >= 4:
            slot_valid = nf_data[:, :, :2].abs().sum(dim=(1, 2)) > 1e-6
            if slot_valid.any():
                neighbor_futures = nf_data[slot_valid]
                neighbor_valid = neighbor_futures[:, :, :2].abs().sum(dim=-1) > 1e-6

                if "neighbor_agents_past" in data:
                    nap = data["neighbor_agents_past"]
                    if nap.dim() == 4:
                        nap = nap[0]
                    ns = nap[slot_valid, -1, :]
                    if ns.shape[-1] >= 8:
                        neighbor_shapes = ns[:, [6, 7]]  # width, length
                    else:
                        neighbor_shapes = torch.full(
                            (neighbor_futures.shape[0], 2), 2.0, device=device
                        )
                else:
                    neighbor_shapes = torch.full((neighbor_futures.shape[0], 2), 2.0, device=device)

    zero_shapes = neighbor_shapes.abs().sum(dim=-1) < 1e-3
    if zero_shapes.any():
        neighbor_shapes[zero_shapes] = torch.tensor([2.0, 4.5], device=device)

    # --- Goal pose ---
    goal_pose = torch.zeros(4, device=device)
    if "goal_pose" in data:
        gp = data["goal_pose"]
        if gp.dim() == 2:
            gp = gp[0]
        if gp.numel() >= 4:
            goal_pose = gp[:4].to(device)

    # --- Batched score computation ---
    safety_scores, collision_steps = compute_safety_score_batch(
        ego_trajs, ego_shape, neighbor_futures, neighbor_shapes, neighbor_valid, config
    )
    progress_scores = compute_progress_score_batch(ego_trajs, goal_pose, data)
    smoothness_scores = compute_smoothness_score_batch(ego_trajs, config)
    feasibility_scores, off_road_fractions = compute_feasibility_score_batch(
        ego_trajs, ego_shape, data, config
    )
    centerline_scores = compute_centerline_score_batch(
        ego_trajs,
        ego_shape,
        data,
        usage_mode=config.centerline_usage_mode,
        time_weight_min=config.centerline_time_weight_min,
    )
    red_light_scores = compute_red_light_score_batch(ego_trajs, data, config)
    ttc_scores = compute_ttc_score_batch(
        ego_trajs, ego_shape, neighbor_futures, neighbor_shapes, neighbor_valid
    )

    # Road border penalty using ego perimeter sampling
    # Returns fracs or survival penalties depending on config.rb_penalty_mode
    (
        rb_crossing_gate,
        rb_near_pen,
        rb_wide_pen,
        rb_crossing_steps,
        rb_cont_penalty,
        rb_per_ts_min,
    ) = compute_road_border_penalty(
        ego_trajs,
        ego_shape,
        data,
        config=config,
    )

    # Lane departure penalty
    if config.enable_lane_departure:
        (
            lane_crossing_gate,
            lane_near_frac,
            lane_wide_frac,
            lane_crossing_steps,
            lane_cont_penalty,
        ) = compute_lane_departure_penalty(
            ego_trajs,
            ego_shape,
            data,
            config=config,
        )
    else:
        lane_crossing_gate = torch.ones(N, device=device)
        lane_near_frac = torch.zeros(N, device=device)
        lane_wide_frac = torch.zeros(N, device=device)
        lane_crossing_steps: list[int | None] = [None] * N
        lane_cont_penalty = torch.zeros(N, device=device)

    # Static-collision penalty (stopped-neighbor OBB clearance).
    # Default-off: when disabled, returns safe zeros + no gate effect.
    if config.static_collision_enabled:
        # Check the predicted trajectory as usual.
        sc_result = compute_static_collision_penalty(
            ego_trajs,
            ego_shape,
            neighbor_futures,
            neighbor_shapes,
            neighbor_valid,
            config,
        )
        sc_crossing_gate = sc_result["crossing_gate"]
        sc_near_pen = sc_result["near_penalty"]
        sc_wide_pen = sc_result["wide_penalty"]
        sc_cont_pen = sc_result["cont_penalty"]
        sc_crossing_steps = sc_result["first_crossing_steps"]
        sc_per_ts_min = sc_result["per_timestep_min"]
        sc_n_stopped_scene = int(sc_result["stopped_mask"].sum().item())

    else:
        sc_crossing_gate = torch.ones(N, device=device)
        sc_near_pen = torch.zeros(N, device=device)
        sc_wide_pen = torch.zeros(N, device=device)
        sc_cont_pen = torch.zeros(N, device=device)
        sc_crossing_steps: list[int | None] = [None] * N
        sc_per_ts_min = torch.full((N, T), 99.0, device=device)
        sc_n_stopped_scene = 0

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
    kinematic_gate = compute_kinematic_gate(ego_trajs, config, ego_shape)
    totals = totals * kinematic_gate + (1.0 - kinematic_gate) * _OFFROAD_FLOOR

    # Also compute additive total for backward compat in breakdown
    on_road_factor = 1.0 - off_road_fractions
    adjusted_progress = progress_scores * on_road_factor

    # Breakdown-friendly static-collision min-distance: min across t>=1 only.
    # (Full per-step values live in sc_per_ts_min; the scalar breakdown
    # field excludes t=0 since it's not model-controllable, same as rb_min_dist.)
    if T > 1:
        sc_min_dist_scalar = sc_per_ts_min[:, 1:].min(dim=1).values
    else:
        sc_min_dist_scalar = torch.full((N,), 99.0, device=device)

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
                rb_min_dist=float(rb_per_ts_min[i, 1:].min().item()),
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


# ---------------------------------------------------------------------------
# Group advantage computation
# ---------------------------------------------------------------------------


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
