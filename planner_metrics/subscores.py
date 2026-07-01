"""Batched EPDMS-style subscores: safety, TTC, drivable-area / road-border, lane-keeping,
progress, comfort/smoothness, feasibility, kinematic gate, red-light, static-collision,
lane-departure.

Raw physical quantities only — reward shaping (weights, gates, GRPO aggregation) lives
in ``rlvr.reward``. Must not import from ``rlvr``.
"""

from __future__ import annotations

import math

import torch

from planner_metrics.collision_geometry import (
    batch_signed_distance_rect,
    center_rect_to_points,
)
from planner_metrics.config import RewardConfig
from planner_metrics.geometry import *  # noqa: F401,F403  geometry primitives


@torch.no_grad()
def compute_ego_neighbor_signed_clearance(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    neighbor_futures: torch.Tensor,
    neighbor_shapes: torch.Tensor,
    neighbor_valid: torch.Tensor,
    *,
    return_closest_points: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Signed OBB clearance between ego trajectories and neighbor futures.

    Returns one signed distance per ``(trajectory, neighbor, timestep)``.
    Overlapping rectangles keep the SAT penetration depth (negative). Separated
    rectangles use the exact Euclidean closest-point clearance; SAT alone only
    gives the minimum separating-axis gap and under-reports diagonal/corner
    clearances.
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    N_nb = neighbor_futures.shape[0]

    if N_nb == 0:
        distances = torch.empty(N, 0, T, device=device, dtype=ego_trajs.dtype)
        if return_closest_points:
            pts = torch.empty(N, 0, T, 2, device=device, dtype=ego_trajs.dtype)
            return distances, pts, pts
        return distances

    ego_corners = _build_ego_bbox_corners(ego_trajs, ego_shape)  # (N, T, 4, 2)

    npc_pos = neighbor_futures[:, :, :2]
    npc_cos = neighbor_futures[:, :, 2]
    npc_sin = neighbor_futures[:, :, 3]
    npc_norm = (npc_cos**2 + npc_sin**2).sqrt().clamp_min(1e-6)
    npc_cos = npc_cos / npc_norm
    npc_sin = npc_sin / npc_norm

    npc_width = neighbor_shapes[:, 0].unsqueeze(1).expand(-1, T)
    npc_length = neighbor_shapes[:, 1].unsqueeze(1).expand(-1, T)
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

    ego_exp = ego_corners.unsqueeze(1).expand(-1, N_nb, -1, -1, -1)
    npc_exp = npc_corners.unsqueeze(0).expand(N, -1, -1, -1, -1)
    nv_exp = neighbor_valid.unsqueeze(0).expand(N, -1, -1)

    ego_flat = ego_exp.reshape(-1, 4, 2)
    npc_flat = npc_exp.reshape(-1, 4, 2)

    sat_dist_flat = batch_signed_distance_rect(ego_flat, npc_flat)
    pt_e_all, pt_n_all = _closest_points_between_rects(ego_flat, npc_flat)
    euclid_dist_flat = (pt_e_all - pt_n_all).norm(dim=-1)
    signed_dist_flat = torch.where(sat_dist_flat < 0, sat_dist_flat, euclid_dist_flat)

    distances = signed_dist_flat.reshape(N, N_nb, T).masked_fill(~nv_exp, 1e6)
    if not return_closest_points:
        return distances

    return (
        distances,
        pt_e_all.reshape(N, N_nb, T, 2),
        pt_n_all.reshape(N, N_nb, T, 2),
    )


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

    distances = compute_ego_neighbor_signed_clearance(
        ego_trajs,
        ego_shape,
        neighbor_futures,
        neighbor_shapes,
        neighbor_valid,
    )

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


_TTC_HORIZON = 1.0  # seconds ahead to check
_TTC_DT = 0.1  # trajectory timestep


@torch.no_grad()
def compute_ttc_score_batch(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    neighbor_futures: torch.Tensor,
    neighbor_shapes: torch.Tensor,
    neighbor_valid: torch.Tensor,
) -> dict[str, torch.Tensor | list[int | None]]:
    """Check if ego would collide with NPCs within TTC_HORIZON seconds.

    For each trajectory timestep, looks forward over the time-aligned ego/NPC
    future poses and uses the same OBB signed-clearance primitive as moving
    collision scoring. If any OBB overlap occurs within the horizon, that base
    timestep is marked unsafe.

    Returns:
        dict with:
          score: (N,) fraction of timesteps that are TTC-safe (1.0 = all safe).
          unsafe_at_t: (N, T) bool, base timestep has a collision within horizon.
          first_unsafe_steps: list[N] earliest unsafe base timestep.
          first_collision_steps: list[N] earliest actual OBB collision timestep.
          min_clearance: (N, T) min OBB clearance within the horizon at each base timestep.
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    N_nb = neighbor_futures.shape[0]

    def _safe_empty() -> dict[str, torch.Tensor | list[int | None]]:
        return {
            "score": torch.ones(N, device=device),
            "unsafe_at_t": torch.zeros(N, T, dtype=torch.bool, device=device),
            "first_unsafe_steps": [None] * N,
            "first_collision_steps": [None] * N,
            "min_clearance": torch.full((N, T), 99.0, device=device),
        }

    if N_nb == 0 or T == 0:
        return _safe_empty()

    distances = compute_ego_neighbor_signed_clearance(
        ego_trajs,
        ego_shape,
        neighbor_futures,
        neighbor_shapes,
        neighbor_valid,
    )  # (N, N_nb, T)
    collision_at_t = (distances < 0).any(dim=1)  # (N, T)
    min_dist_at_t = distances.min(dim=1).values  # (N, T)

    horizon_steps = max(0, int(round(_TTC_HORIZON / _TTC_DT)))
    ttc_unsafe_at_t = torch.zeros(N, T, dtype=torch.bool, device=device)
    ttc_min_clearance = torch.full((N, T), float("inf"), device=device)
    for offset in range(horizon_steps + 1):
        if offset >= T:
            break
        ttc_unsafe_at_t[:, : T - offset] |= collision_at_t[:, offset:]
        ttc_min_clearance[:, : T - offset] = torch.minimum(
            ttc_min_clearance[:, : T - offset],
            min_dist_at_t[:, offset:],
        )
    ttc_min_clearance = ttc_min_clearance.masked_fill(torch.isinf(ttc_min_clearance), 99.0)

    # Score: fraction of safe timesteps
    ttc_score = 1.0 - ttc_unsafe_at_t.float().mean(dim=1)  # (N,)

    has_unsafe = ttc_unsafe_at_t.any(dim=1)
    first_unsafe_t = ttc_unsafe_at_t.float().argmax(dim=1)
    has_collision = collision_at_t.any(dim=1)
    first_collision_t = collision_at_t.float().argmax(dim=1)

    first_unsafe_steps: list[int | None] = []
    first_collision_steps: list[int | None] = []
    for i in range(N):
        first_unsafe_steps.append(int(first_unsafe_t[i].item()) if has_unsafe[i] else None)
        first_collision_steps.append(int(first_collision_t[i].item()) if has_collision[i] else None)

    return {
        "score": ttc_score,
        "unsafe_at_t": ttc_unsafe_at_t,
        "first_unsafe_steps": first_unsafe_steps,
        "first_collision_steps": first_collision_steps,
        "min_clearance": ttc_min_clearance,
    }


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


@torch.no_grad()
def compute_road_border_penalty(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig | None = None,
    *,
    return_closest_points: bool = False,
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
        When ``return_closest_points=True``, appends:
        - ego_closest_pt: (N, T, 2) ego perimeter sample at the min distance
        - border_closest_pt: (N, T, 2) closest point on the winning border segment
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
    if return_closest_points:
        _safe_return = (
            *_safe_return,
            torch.zeros(N, T, 2, device=device),
            torch.zeros(N, T, 2, device=device),
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
    closest_return = ()
    if return_closest_points:
        # Reuse the same reduced road-border segment set as the distance metric
        # and return the closest pair for the winning ego perimeter sample.
        closest_border_flat = torch.empty_like(world_flat)
        seg = seg_p2 - seg_p1
        seg_len2 = (seg * seg).sum(dim=1).clamp_min(1e-9)
        chunk = 32768
        for start in range(0, world_flat.shape[0], chunk):
            pts = world_flat[start : start + chunk]
            rel = pts[:, None, :] - seg_p1[None, :, :]
            t = (rel * seg[None, :, :]).sum(dim=-1) / seg_len2[None, :]
            t = t.clamp(0.0, 1.0)
            closest = seg_p1[None, :, :] + t[:, :, None] * seg[None, :, :]
            d2 = ((pts[:, None, :] - closest) ** 2).sum(dim=-1)
            best_seg = d2.argmin(dim=1)
            closest_border_flat[start : start + chunk] = closest[
                torch.arange(pts.shape[0], device=device), best_seg
            ]
        border_closest_all = closest_border_flat.reshape(N, T, K_pts, 2)
        best_perim = min_dists.argmin(dim=2)
        n_idx = torch.arange(N, device=device).view(N, 1).expand(N, T)
        t_idx = torch.arange(T, device=device).view(1, T).expand(N, T)
        ego_closest_pt = world_pts[n_idx, t_idx, best_perim]
        border_closest_pt = border_closest_all[n_idx, t_idx, best_perim]
        closest_return = (ego_closest_pt, border_closest_pt)

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
            *closest_return,
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
            *closest_return,
        )


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

    # --- Ego × stopped-neighbor signed clearance ---
    nb_f_s = neighbor_futures[stopped_mask]  # (S, T, 4)
    nb_shapes_s = neighbor_shapes[stopped_mask]  # (S, 2) [width, length]
    nb_valid_s = neighbor_valid[stopped_mask]  # (S, T)
    S = nb_f_s.shape[0]
    distances, pt_e_nst, pt_n_nst = compute_ego_neighbor_signed_clearance(
        ego_trajs,
        ego_shape,
        nb_f_s,
        nb_shapes_s,
        nb_valid_s,
        return_closest_points=True,
    )

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


_LANE_NEAR_THRESH = 0.25
_LANE_WIDE_THRESH = 0.40
_LANE_CONT_THRESH = 0.80
_LANE_PTS_PER_SIDE = 10  # 36 unique perimeter points (corners not duplicated)


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


__all__ = [
    "compute_safety_score_batch",
    "compute_ego_neighbor_signed_clearance",
    "_TTC_HORIZON",
    "_TTC_DT",
    "compute_ttc_score_batch",
    "compute_feasibility_score_batch",
    "_SG_SMOOTH_KERNEL",
    "_SG_SMOOTH_CACHE_KEY",
    "compute_kinematic_gate",
    "_CL_X",
    "_CL_Y",
    "_CL_DX",
    "_CL_DY",
    "_CL_MAX_DIST",
    "compute_centerline_score_batch",
    "compute_progress_score_batch",
    "_SG_JERK_KERNEL",
    "_SG_JERK_CACHE_KEY",
    "_SG_VEL_KERNEL",
    "_SG_ACCEL_KERNEL",
    "_SG_LAT_CACHE_KEY",
    "compute_smoothness_score_batch",
    "_TL_GREEN",
    "_TL_YELLOW",
    "_TL_RED",
    "_TL_WHITE",
    "_TL_NONE",
    "_RED_LIGHT_PROXIMITY",
    "_RED_LIGHT_HEADING_THRESH",
    "compute_red_light_score_batch",
    "compute_road_border_penalty",
    "compute_static_collision_penalty",
    "_LANE_NEAR_THRESH",
    "_LANE_WIDE_THRESH",
    "_LANE_CONT_THRESH",
    "_LANE_PTS_PER_SIDE",
    "_LANE_K_NEAREST",
    "compute_lane_departure_penalty",
]
