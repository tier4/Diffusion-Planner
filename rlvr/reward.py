"""Rule-based trajectory reward for GRPO training.

Computes R = w_safety * S + w_progress * P + w_smooth * M + w_feasibility * F
using log-replay data. Reuses ego bbox construction and lane/neighbor penalty
functions from diffusion_planner.loss for proper vehicle-footprint-aware checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from diffusion_planner.loss import neighbor_clearance_penalty


@dataclass
class RewardConfig:
    w_safety: float = 5.0
    w_progress: float = 1.0
    w_smooth: float = 0.5
    w_feasibility: float = 5.0
    w_centerline: float = 1.0
    collision_penalty: float = -10.0
    max_accel: float = 8.0  # m/s^2
    dt: float = 0.1  # 10 Hz


@dataclass
class RewardBreakdown:
    safety: float
    progress: float
    smoothness: float
    feasibility: float
    centerline: float
    total: float
    collision_step: int | None
    off_road_fraction: float


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
    half_sizes = torch.tensor(
        [half_length, half_width], device=device, dtype=dtype
    ).expand(N, T, 2)

    corner_signs = torch.tensor(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]],
        device=device, dtype=dtype,
    )
    local_corners = corner_signs[None, None, :, :] * half_sizes[:, :, None, :]  # (N, T, 4, 2)

    rot = torch.stack([
        heading_unit[..., 0], -heading_unit[..., 1],
        heading_unit[..., 1], heading_unit[..., 0],
    ], dim=-1).reshape(N, T, 2, 2)

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

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        ego_shape: (3,) wheel_base, length, width.
        neighbor_futures: (N_nb, T, 4) GT NPC future trajectories.
        neighbor_shapes: (N_nb, 2) width, length per NPC.
        neighbor_valid: (N_nb, T) bool mask of valid neighbor timesteps.
        config: RewardConfig.

    Returns:
        scores: (N,) tensor -- 0.0 if no collision, collision_penalty otherwise.
        collision_steps: list of length N -- timestep of first collision or None.
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    N_nb = neighbor_futures.shape[0]

    if N_nb == 0:
        return torch.zeros(N, device=device), [None] * N

    ego_corners = _build_ego_bbox_corners(ego_trajs, ego_shape)  # (N, T, 4, 2)

    # Build neighbor data in the shape neighbor_clearance_penalty expects:
    # neighbors_future: [B, P, T, 4], neighbors_future_valid: [B, P, T]
    # We evaluate each of the N ego trajectories against the same set of neighbors.
    # Expand neighbors to batch dimension N.
    nf = neighbor_futures.unsqueeze(0).expand(N, -1, -1, -1)  # (N, N_nb, T, 4)
    nv = neighbor_valid.unsqueeze(0).expand(N, -1, -1)        # (N, N_nb, T)

    # neighbor_agents_past needed for shapes: fake it with shapes in the right slots
    # neighbor_clearance_penalty reads [:, :P, -1, 6] for width and [:, :P, -1, 7] for length
    fake_past = torch.zeros(N, N_nb, 1, 8, device=device)
    fake_past[:, :, 0, 6] = neighbor_shapes[:, 0].unsqueeze(0).expand(N, -1)  # width
    fake_past[:, :, 0, 7] = neighbor_shapes[:, 1].unsqueeze(0).expand(N, -1)  # length
    denorm_inputs = {"neighbor_agents_past": fake_past}

    # neighbor_clearance_penalty returns [B, T] penalty
    penalty = neighbor_clearance_penalty(ego_corners, nf, nv, denorm_inputs)  # (N, T)

    # Collision = any timestep with penalty > 0
    has_collision_at_t = penalty > 0  # (N, T)
    has_collision = has_collision_at_t.any(dim=1)  # (N,)
    first_t = has_collision_at_t.float().argmax(dim=1)  # (N,)

    scores = torch.where(
        has_collision,
        torch.tensor(config.collision_penalty, device=device),
        torch.tensor(0.0, device=device),
    )

    collision_steps: list[int | None] = []
    for i in range(N):
        if has_collision[i]:
            collision_steps.append(int(first_t[i].item()))
        else:
            collision_steps.append(None)

    return scores, collision_steps


# ---------------------------------------------------------------------------
# Feasibility: lane boundary check with vehicle half-width
# ---------------------------------------------------------------------------

_LN_X, _LN_Y = 0, 1
_LN_DX, _LN_DY = 2, 3
_LN_LBX, _LN_LBY = 4, 5
_LN_RBX, _LN_RBY = 6, 7
_LN_MAX_DIST = 30.0


def compute_feasibility_score_batch(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched lane boundary violation + acceleration penalty.

    Checks whether the ego vehicle (center +/- half_width) protrudes beyond
    the actual left/right lane boundaries of route_lanes. Boundary offsets in
    the lane tensor (indices 4-7) are offset vectors from centerline, not
    absolute positions.

    Args:
        ego_trajs: (N, T, 4).
        ego_shape: (3,) wheel_base, length, width.
        data: Observation dict.
        config: RewardConfig.

    Returns:
        scores: (N,) negative penalty.
        off_road_fractions: (N,) fraction of timesteps with boundary violation.
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    scores = torch.zeros(N, device=device)
    off_road_fractions = torch.zeros(N, device=device)

    # --- Acceleration check ---
    pos = ego_trajs[:, :, :2]
    vel = torch.diff(pos, dim=1) / config.dt
    speed = vel.norm(dim=-1)
    acc = torch.diff(speed, dim=1) / config.dt
    if acc.numel() > 0:
        accel_violations = (acc.abs() > config.max_accel).float().mean(dim=-1)
        scores = scores - accel_violations

    # --- Lane boundary check ---
    # Use ALL lanes for boundary checking (not just route_lanes).
    # Off-road = leaving all drivable surface. Off-route (taking a different
    # road) is penalized softly by the centerline reward, not here.
    if "lanes" not in data:
        return scores, off_road_fractions
    lanes = data["lanes"]

    if lanes.dim() == 4:
        lanes = lanes[0]  # (S, P, 33)

    S_P = lanes.shape[0] * lanes.shape[1]
    lane_centers = lanes[..., _LN_X:_LN_Y + 1].reshape(S_P, 2)
    lane_dirs = lanes[..., _LN_DX:_LN_DY + 1].reshape(S_P, 2)
    lane_left = lanes[..., _LN_LBX:_LN_LBY + 1].reshape(S_P, 2)   # offset vectors
    lane_right = lanes[..., _LN_RBX:_LN_RBY + 1].reshape(S_P, 2)  # offset vectors

    lane_dirs_n = lane_dirs / (lane_dirs.norm(dim=-1, keepdim=True) + 1e-6)
    lane_lat = torch.stack([-lane_dirs_n[..., 1], lane_dirs_n[..., 0]], dim=-1)  # (S_P, 2)

    # Valid: both boundary offsets and direction must be nonzero
    lane_valid = (
        (lane_left.norm(dim=-1) + lane_right.norm(dim=-1)) > 1e-3
    ) & (lane_dirs.norm(dim=-1) > 1e-6)  # (S_P,)

    # Boundary half-widths: project offset vectors onto lateral normal
    # These are signed: left_hw > 0 (left side), right_hw < 0 (right side)
    left_hw = (lane_left * lane_lat).sum(dim=-1)    # (S_P,)
    right_hw = (lane_right * lane_lat).sum(dim=-1)  # (S_P,)

    ego_pos = ego_trajs[:, :, :2]  # (N, T, 2)
    half_w = float(ego_shape[2]) / 2  # vehicle half-width

    # For each ego position, check ALL nearby lanes (not just the nearest center).
    # The ego is off-road only if it is outside the boundaries of EVERY lane.
    # Compute lateral offset and boundary violations for all (ego, lane) pairs.

    # Distance from ego to all lane centers: (N, T, S_P)
    diff = ego_pos.unsqueeze(2) - lane_centers.unsqueeze(0).unsqueeze(0)
    dist = diff.norm(dim=-1)
    dist = dist.masked_fill(~lane_valid.view(1, 1, -1).expand(N, T, -1), 1e6)
    min_dist = dist.min(dim=-1).values  # (N, T)

    # Only check lanes within a reasonable distance
    _CHECK_RADIUS = 8.0
    nearby_mask = dist < _CHECK_RADIUS  # (N, T, S_P)

    # Lateral offset for all (ego, lane) pairs: (N, T, S_P)
    ego_lat_all = (diff * lane_lat.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

    # Boundary violations per lane: (N, T, S_P)
    viol_left_all = torch.relu(ego_lat_all + half_w - left_hw.view(1, 1, -1))
    viol_right_all = torch.relu(right_hw.view(1, 1, -1) - ego_lat_all + half_w)
    protrusion_all = viol_left_all + viol_right_all  # (N, T, S_P)

    # A lane is "containing" the ego if protrusion == 0
    inside_lane = (protrusion_all == 0) & nearby_mask  # (N, T, S_P)

    # Ego is off-road at a timestep if NO nearby lane contains it
    in_any_lane = inside_lane.any(dim=-1)  # (N, T)

    # For off-road timesteps, compute the minimum protrusion across all nearby
    # lanes (how far outside the best candidate lane)
    protrusion_all = protrusion_all.masked_fill(~nearby_mask, 1e6)
    min_protrusion = protrusion_all.min(dim=-1).values  # (N, T)
    min_protrusion = min_protrusion.clamp(max=100.0)

    # Build per-step violations
    _BOUNDARY_STEP_PENALTY = 2.0
    violations = torch.where(
        ~in_any_lane,
        _BOUNDARY_STEP_PENALTY + min_protrusion,
        torch.zeros_like(min_protrusion),
    )

    # Completely far from any lane (>10m) -- harsh penalty
    _OFFROAD_DIST = 10.0
    _OFFROAD_STEP_PENALTY = 5.0
    far_from_any_lane = min_dist > _OFFROAD_DIST
    violations = torch.where(
        far_from_any_lane,
        _OFFROAD_STEP_PENALTY + min_dist,
        violations,
    )

    # Time weighting: going off road at t=2s is worse than at t=7s.
    # t=0 -> weight=1.0, t=T-1 -> weight=0.3
    time_weights = torch.linspace(1.0, 0.3, T, device=device).unsqueeze(0)
    weighted_violations = violations * time_weights

    off_road_fractions = (~in_any_lane).float().mean(dim=-1)  # (N,)
    scores = scores - weighted_violations.sum(dim=-1)

    return scores, off_road_fractions


# ---------------------------------------------------------------------------
# Centerline: reward for staying close to route lane centerlines
# ---------------------------------------------------------------------------

_CL_X, _CL_Y = 0, 1
_CL_DX, _CL_DY = 2, 3
_CL_MAX_DIST = 30.0


def compute_centerline_score_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Batched negative mean lateral deviation from nearest route lane centerline.

    Uses route_lanes (the ego's intended path) to compute lateral offset at
    each timestep. Trajectories closer to the centerline score higher (less negative).

    Args:
        ego_trajs: (N, T, 4).
        data: Observation dict with "route_lanes" or "lanes" key.

    Returns:
        (N,) scores (negative, closer to 0 = closer to centerline).
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device

    if "route_lanes" in data:
        lanes = data["route_lanes"]
    elif "lanes" in data:
        lanes = data["lanes"]
    else:
        return torch.zeros(N, device=device)

    if lanes.dim() == 4:
        lanes = lanes[0]  # (S, P, 33)

    S_P = lanes.shape[0] * lanes.shape[1]
    lane_centers = lanes[..., _CL_X:_CL_Y + 1].reshape(S_P, 2)
    lane_dirs = lanes[..., _CL_DX:_CL_DY + 1].reshape(S_P, 2)

    lane_valid = lane_centers.norm(dim=-1) > 1e-3  # (S_P,)
    lane_dirs_n = lane_dirs / (lane_dirs.norm(dim=-1, keepdim=True) + 1e-6)
    lane_lat = torch.stack([-lane_dirs_n[..., 1], lane_dirs_n[..., 0]], dim=-1)  # (S_P, 2)

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

    # Mask out timesteps too far from any lane
    no_lane = min_dist > _CL_MAX_DIST
    ego_lat = ego_lat.masked_fill(no_lane, 0.0)

    return -(ego_lat ** 2).mean(dim=-1)  # (N,)


# ---------------------------------------------------------------------------
# Progress: batched distance reduction toward goal
# ---------------------------------------------------------------------------

def compute_progress_score_batch(
    ego_trajs: torch.Tensor,
    goal_pose: torch.Tensor,
) -> torch.Tensor:
    """Batched progress toward goal.

    Args:
        ego_trajs: (N, T, 4).
        goal_pose: (4,) x, y, cos, sin -- zeros if unavailable.

    Returns:
        (N,) scores.
    """
    goal_xy = goal_pose[:2]
    if goal_xy.abs().sum() < 1e-6:
        diffs = torch.diff(ego_trajs[:, :, :2], dim=1)
        return torch.sqrt((diffs ** 2).sum(dim=-1)).sum(dim=-1)

    dist_start = (ego_trajs[:, 0, :2] - goal_xy).norm(dim=-1)
    dist_end = (ego_trajs[:, -1, :2] - goal_xy).norm(dim=-1)
    return dist_start - dist_end


# ---------------------------------------------------------------------------
# Smoothness: batched jerk penalty
# ---------------------------------------------------------------------------

def compute_smoothness_score_batch(
    ego_trajs: torch.Tensor,
    config: RewardConfig,
) -> torch.Tensor:
    """Batched negative mean absolute jerk.

    Args:
        ego_trajs: (N, T, 4).
        config: RewardConfig for dt.

    Returns:
        (N,) scores (negative, closer to 0 = smoother).
    """
    pos = ego_trajs[:, :, :2]
    vel = torch.diff(pos, dim=1) / config.dt
    acc = torch.diff(vel, dim=1) / config.dt
    jerk = torch.diff(acc, dim=1) / config.dt
    if jerk.numel() == 0:
        return torch.zeros(ego_trajs.shape[0], device=ego_trajs.device)
    return -(jerk.abs().sum(dim=-1)).mean(dim=-1)


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
    ego_shape = torch.tensor([2.79, 4.34, 1.70], device=device)
    if "ego_shape" in data:
        es = data["ego_shape"]
        if es.dim() == 2:
            es = es[0]
        if es.numel() >= 3:
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
            nf_data = nf[:, :T, :4]
            slot_valid = nf_data.abs().sum(dim=(1, 2)) > 1e-6
            if slot_valid.any():
                neighbor_futures = nf_data[slot_valid]
                # Per-timestep validity
                neighbor_valid = neighbor_futures.abs().sum(dim=-1) > 1e-6  # (N_nb, T)

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
                    neighbor_shapes = torch.full(
                        (neighbor_futures.shape[0], 2), 2.0, device=device
                    )

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
    progress_scores = compute_progress_score_batch(ego_trajs, goal_pose)
    smoothness_scores = compute_smoothness_score_batch(ego_trajs, config)
    feasibility_scores, off_road_fractions = compute_feasibility_score_batch(
        ego_trajs, ego_shape, data, config
    )
    centerline_scores = compute_centerline_score_batch(ego_trajs, data)

    totals = (
        config.w_safety * safety_scores
        + config.w_progress * progress_scores
        + config.w_smooth * smoothness_scores
        + config.w_feasibility * feasibility_scores
        + config.w_centerline * centerline_scores
    )

    results: list[RewardBreakdown] = []
    for i in range(N):
        results.append(RewardBreakdown(
            safety=float(safety_scores[i]),
            progress=float(progress_scores[i]),
            smoothness=float(smoothness_scores[i]),
            feasibility=float(feasibility_scores[i]),
            centerline=float(centerline_scores[i]),
            total=float(totals[i]),
            collision_step=collision_steps[i],
            off_road_fraction=float(off_road_fractions[i]),
        ))

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
) -> np.ndarray:
    """Compute GRPO-style group-relative advantages.

    Args:
        rewards: List of RewardBreakdown for each trajectory in the group.
        epsilon: Small constant for numerical stability.

    Returns:
        (G,) array of normalized advantages with ~zero mean and ~unit variance.
    """
    totals = np.array([r.total for r in rewards])
    mean = totals.mean()
    std = totals.std()
    if std < epsilon:
        return np.zeros(len(rewards))
    return (totals - mean) / (std + epsilon)
