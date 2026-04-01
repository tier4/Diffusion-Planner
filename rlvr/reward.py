"""Rule-based trajectory reward for GRPO training.

Computes R = w_safety * S + w_progress * P + w_smooth * M + w_feasibility * F + w_centerline * C
using log-replay data. Reuses ego bbox construction and lane/neighbor penalty
functions from diffusion_planner.loss for proper vehicle-footprint-aware checks.
"""

from __future__ import annotations

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
    collision_penalty: float = -10.0
    red_light_penalty: float = -10.0
    max_accel: float = 8.0  # m/s^2
    dt: float = 0.1  # 10 Hz

    # Near-edge / wide-edge / continuous penalty scales (road border)
    near_edge_scale: float = 3.0
    wide_edge_scale: float = 0.2
    cont_edge_scale: float = 0.0  # continuous penalty within 80cm (0=disabled)

    # Lane departure penalty scales
    enable_lane_departure: bool = False
    lane_gate_enabled: bool = False  # if True, lane crossing kills reward (too strict for most scenes)
    lane_near_scale: float = 3.0
    lane_wide_scale: float = 0.2
    lane_cont_scale: float = 0.0

    # Lateral acceleration penalty
    max_lat_accel: float = 2.0  # m/s^2
    lat_accel_scale: float = 3.0

    # Overprogress: cap progress at GT path × margin, penalize excess
    enable_overprogress: bool = False
    overprogress_margin: float = 1.1
    overprogress_penalty: float = 0.3
    stopped_penalty: float = 50.0

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
    rb_near_frac: float = 0.0
    lane_crossing: bool = False
    lane_near_frac: float = 0.0


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
    npc_norm = (npc_cos ** 2 + npc_sin ** 2).sqrt().clamp_min(1e-6)
    npc_cos = npc_cos / npc_norm
    npc_sin = npc_sin / npc_norm

    npc_width = neighbor_shapes[:, 0].unsqueeze(1).expand(-1, T)   # (N_nb, T)
    npc_length = neighbor_shapes[:, 1].unsqueeze(1).expand(-1, T)  # (N_nb, T)

    npc_rect = torch.stack([
        npc_pos[..., 0], npc_pos[..., 1],
        npc_cos, npc_sin,
        npc_length, npc_width,
    ], dim=-1)  # (N_nb, T, 6)
    npc_corners = center_rect_to_points(
        npc_rect.reshape(-1, 6)
    ).reshape(N_nb, T, 4, 2)

    # Cross product: ego (N, T) x NPC (N_nb, T) -> (N, N_nb, T)
    ego_exp = ego_corners.unsqueeze(1).expand(-1, N_nb, -1, -1, -1)  # (N, N_nb, T, 4, 2)
    npc_exp = npc_corners.unsqueeze(0).expand(N, -1, -1, -1, -1)     # (N, N_nb, T, 4, 2)
    nv_exp = neighbor_valid.unsqueeze(0).expand(N, -1, -1)            # (N, N_nb, T)

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
    ego_xy = ego_trajs[:, :, :2]                           # (N, T, 2)
    ego_heading = ego_trajs[:, :, 2:4]                     # (N, T, 2) [cos, sin]
    npc_xy = neighbor_futures[:, :, :2]                    # (N_nb, T, 2)

    # ego→NPC vector: (N, N_nb, T, 2)
    ego_to_npc = npc_xy.unsqueeze(0) - ego_xy.unsqueeze(1)
    # Dot product with ego heading: positive = NPC ahead/beside, negative = NPC behind
    dot = (ego_to_npc * ego_heading.unsqueeze(1)).sum(dim=-1)  # (N, N_nb, T)
    npc_is_behind = dot < 0  # (N, N_nb, T)

    # Suppress rear-end collisions
    collision_mask = collision_mask & ~npc_is_behind

    # Suppress low-speed bbox overlaps: two stopped/slow vehicles queued
    # bumper-to-bumper at a red light or in traffic is not a collision.
    # Only count collisions when the ego is moving faster than 1 m/s.
    _COLLISION_MIN_SPEED = 1.0  # m/s
    ego_vel = torch.diff(ego_xy, dim=1) / config.dt  # (N, T-1, 2)
    ego_speed = ego_vel.norm(dim=-1)                  # (N, T-1)
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
_TTC_DT = 0.1       # trajectory timestep

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
    npc_vel = (neighbor_futures[:, 1:, :2] - neighbor_futures[:, :-1, :2]) / _TTC_DT  # (N_nb, T-1, 2)
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
    v1 = polygon                               # (V, 2)
    v2 = torch.roll(polygon, -1, dims=0)       # (V, 2)

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
        left = (pts + lb)[valid]   # (K, 2)
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
        return torch.ones(ego_trajs.shape[0], ego_trajs.shape[1],
                          dtype=torch.bool, device=ego_trajs.device)

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
    rx = local_pts_t[:, 0][None, None, :] * cos_h[:, :, None] \
       - local_pts_t[:, 1][None, None, :] * sin_h[:, :, None]
    ry = local_pts_t[:, 0][None, None, :] * sin_h[:, :, None] \
       + local_pts_t[:, 1][None, None, :] * cos_h[:, :, None]
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
        if (pmax[0] < traj_min[0] or pmin[0] > traj_max[0] or
                pmax[1] < traj_min[1] or pmin[1] > traj_max[1]):
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

    mrx = margin_pts_t[:, 0][None, None, :] * cos_h[:, :, None] \
        - margin_pts_t[:, 1][None, None, :] * sin_h[:, :, None]
    mry = margin_pts_t[:, 0][None, None, :] * sin_h[:, :, None] \
        + margin_pts_t[:, 1][None, None, :] * cos_h[:, :, None]
    mpts_x = cx[:, :, None] + mrx
    mpts_y = cy[:, :, None] + mry
    all_margin_pts = torch.stack([mpts_x, mpts_y], dim=-1).reshape(-1, 2)

    if nearby_polys:
        margin_inside_matrix = _points_in_polygons_batched(
            all_margin_pts, padded_v1, padded_v2, poly_valid,
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
    wrx = wide_pts_t[:, 0][None, None, :] * cos_h[:, :, None] \
        - wide_pts_t[:, 1][None, None, :] * sin_h[:, :, None]
    wry = wide_pts_t[:, 0][None, None, :] * sin_h[:, :, None] \
        + wide_pts_t[:, 1][None, None, :] * cos_h[:, :, None]
    wpts_x = cx[:, :, None] + wrx
    wpts_y = cy[:, :, None] + wry
    all_wide_pts = torch.stack([wpts_x, wpts_y], dim=-1).reshape(-1, 2)

    if nearby_polys:
        wide_inside = _points_in_polygons_batched(
            all_wide_pts, padded_v1, padded_v2, poly_valid,
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
    vel = torch.diff(pos, dim=1) / config.dt  # (N, T-1, 2)
    speed = vel.norm(dim=-1)  # (N, T-1)
    acc = torch.diff(speed, dim=1) / config.dt  # (N, T-2) longitudinal
    if acc.numel() > 0:
        accel_violations = (acc.abs() > config.max_accel).float().mean(dim=-1)
        scores = scores - accel_violations

    # --- Lateral acceleration check ---
    # Penalize lateral acceleration exceeding what a human driver produces.
    # GT trajectories peak at ~2.5 m/s²; the model reaches 3.5 m/s² on curves.
    # NOTE: This uses double finite diff which inflates values ~5x vs curvature-based.
    # For accurate REPORTING use eval_teleport_metrics.py. But do NOT change this
    # penalty — it was used for all prior successful training runs (p4e, p6m etc).
    _MAX_LAT_ACCEL = config.max_lat_accel
    _LAT_ACCEL_SCALE = config.lat_accel_scale
    if vel.shape[1] >= 2:
        accel_vec = torch.diff(vel, dim=1) / config.dt  # (N, T-2, 2)
        heading = vel[:, :-1]  # (N, T-2, 2)
        heading_norm = heading / (heading.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        lat_dir = torch.stack([-heading_norm[..., 1], heading_norm[..., 0]], dim=-1)
        lat_accel = (accel_vec * lat_dir).sum(dim=-1)  # (N, T-2)
        if lat_accel.shape[1] > 2:
            lat_accel_trimmed = lat_accel[:, 2:]
            lat_violations = torch.relu(lat_accel_trimmed.abs() - _MAX_LAT_ACCEL)
            scores = scores - _LAT_ACCEL_SCALE * lat_violations.mean(dim=-1)

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

    # A lane point can only "contain" the ego if both:
    # 1. Total distance is within radius (not too far in any direction)
    # 2. Longitudinal distance is small (ego is alongside this lane segment,
    #    not far ahead/behind where the lateral projection is meaningless)
    _CHECK_RADIUS = 4.0
    _MAX_LONGITUDINAL = 3.5  # must accommodate lane point spacing (median ~1.5m, max ~5.5m)

    # Decompose distance into lateral and longitudinal components
    lane_dir_n = lane_dirs / (lane_dirs.norm(dim=-1, keepdim=True) + 1e-6)  # (S_P, 2)
    ego_lon_all = (diff * lane_dir_n.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (N, T, S_P)
    ego_lat_all = (diff * lane_lat.unsqueeze(0).unsqueeze(0)).sum(dim=-1)    # (N, T, S_P)

    # All polygon/lane-boundary protrusion, margin, and off-route penalties DISABLED.
    # Road border perimeter check (compute_road_border_penalty) handles offroad.
    # Feasibility score keeps only the lane-proximity base score computed above.
    off_road_fractions = torch.zeros(N, device=device)
    return scores, off_road_fractions


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
    lane_centers = lanes[..., _CL_X:_CL_Y + 1].reshape(S_P, 2)
    lane_dirs = lanes[..., _CL_DX:_CL_DY + 1].reshape(S_P, 2)
    lane_left = lanes[..., 4:6].reshape(S_P, 2)
    lane_right = lanes[..., 6:8].reshape(S_P, 2)

    lane_valid = lane_centers.norm(dim=-1) > 1e-3  # (S_P,)
    lane_dirs_n = lane_dirs / (lane_dirs.norm(dim=-1, keepdim=True) + 1e-6)
    lane_lat = torch.stack([-lane_dirs_n[..., 1], lane_dirs_n[..., 0]], dim=-1)  # (S_P, 2)

    left_hw = (lane_left * lane_lat).sum(dim=-1)    # (S_P,)
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

    # Normalized lane usage: how close the vehicle edge is to the boundary
    # 0 = centered, 1 = edge touching boundary, >1 = over boundary
    lane_usage = (ego_lat.abs() + half_w) / side_hw  # (N, T)

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

    # Cap lane_usage at 1.0 for centerline scoring -- being at the boundary
    # is the max lateral penalty. Being beyond it (usage>1) means off-route,
    # handled by route_deviation.
    capped_usage = lane_usage.clamp(max=1.0)

    per_step_penalty = torch.where(
        near_route,
        capped_usage ** 2,
        torch.where(
            left_route,
            route_deviation ** 2,
            torch.zeros_like(lane_usage),  # no penalty if route never covered this area
        ),
    )  # (N, T)

    # Time-weighted mean: early deviations penalized more
    time_weights = torch.linspace(1.0, 0.3, T, device=device).unsqueeze(0)
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
    return torch.sqrt((diffs ** 2).sum(dim=-1)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Smoothness: batched jerk penalty
# ---------------------------------------------------------------------------

def _build_sg_diff_kernel(window: int = 11, poly: int = 3, deriv: int = 3, delta: float = 0.1) -> torch.Tensor:
    """Build Savitzky-Golay differentiation kernel (precomputed, cached).

    Returns a 1D convolution kernel that computes the deriv-th derivative
    using a local polynomial fit over `window` points.
    """
    from scipy.signal import savgol_coeffs
    coeffs = savgol_coeffs(window, poly, deriv=deriv, delta=delta)
    return torch.tensor(coeffs, dtype=torch.float32).flip(0)  # flip for conv1d

# Precompute SG jerk kernel; cache by (device, dt)
_SG_JERK_KERNEL = None
_SG_JERK_CACHE_KEY = None

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
        _SG_JERK_KERNEL = _build_sg_diff_kernel(
            window=11, poly=3, deriv=3, delta=config.dt
        ).to(ego_trajs.device)
        _SG_JERK_CACHE_KEY = _cache_key

    kernel = _SG_JERK_KERNEL  # [11]
    pad = kernel.shape[0] // 2

    # pos: [N, T, 2] -> [N, 2, T] for conv1d
    pos = ego_trajs[:, :, :2].detach().permute(0, 2, 1)  # [N, 2, T]

    # Pad and convolve: conv1d with kernel [1, 1, W] on [N, 2, T]
    pos_padded = torch.nn.functional.pad(pos, (pad, pad), mode='replicate')
    jerk = torch.nn.functional.conv1d(
        pos_padded, kernel.view(1, 1, -1).expand(2, 1, -1),
        groups=2,
    )  # [N, 2, T]

    jerk_mag = jerk.norm(dim=1)  # [N, T]
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
    red_xy = red_xy[valid]      # (R', 2)
    red_dir = red_dir[valid]    # (R', 2)
    R = red_xy.shape[0]

    # Normalize lane directions
    red_dir_norm = red_dir / (red_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6))

    # Ego positions and headings
    ego_xy = ego_trajs[:, :, :2]        # (N, T, 2)
    ego_cos = ego_trajs[:, :, 2]        # (N, T)
    ego_sin = ego_trajs[:, :, 3]        # (N, T)
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

    is_close = dist < _RED_LIGHT_PROXIMITY        # (N, T, R')
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int | None], torch.Tensor]:
    """Compute per-trajectory road border penalties using ego perimeter sampling.

    Uses 80 points around the ego rectangle (20 per side) and checks min
    distance to road_border line_string points (channel 3 in line_strings).

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        ego_shape: (3,) wheel_base, length, width.
        data: Observation dict with 'line_strings' key.

    Returns:
        Tuple of (crossing_gate, near_penalty, wide_penalty, first_crossing_steps, cont_penalty):
        - crossing_gate: (N,) 1.0 if no crossing, 0.0 if any timestep crosses border
        - near_penalty: (N,) mean penalty for being within 25cm (0=safe, 1=touching)
        - wide_penalty: (N,) mean penalty for being within 40cm
        - first_crossing_steps: list of N (int | None) — first timestep of crossing
        - cont_penalty: (N,) continuous proximity penalty (linear decay from 0.8m)
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device

    no_crossing_steps: list[int | None] = [None] * N

    if "line_strings" not in data:
        return (torch.ones(N, device=device),
                torch.zeros(N, device=device),
                torch.zeros(N, device=device),
                no_crossing_steps,
                torch.zeros(N, device=device))

    ls = data["line_strings"]
    if ls.dim() == 4:
        ls = ls[0]  # remove batch dim -> (num_ls, pts, D)
    if ls.shape[-1] < 4:
        return (torch.ones(N, device=device),
                torch.zeros(N, device=device),
                torch.zeros(N, device=device),
                no_crossing_steps,
                torch.zeros(N, device=device))

    # Extract road border points
    border_flag = ls[..., 3]  # (num_ls, pts)
    border_xy = ls[..., :2]   # (num_ls, pts, 2)
    is_border = border_flag > 0.5
    has_coords = border_xy.norm(dim=-1) > 1e-3
    valid = is_border & has_coords
    border_pts = border_xy[valid]  # (K, 2)

    if border_pts.shape[0] == 0:
        return (torch.ones(N, device=device),
                torch.zeros(N, device=device),
                torch.zeros(N, device=device),
                no_crossing_steps,
                torch.zeros(N, device=device))

    # Build ego perimeter points (20 per side = 80 total)
    wb = ego_shape[0].item()
    length = ego_shape[1].item()
    width = ego_shape[2].item()
    ro = (length - wb) / 2
    _PTS_PER_SIDE = 20
    local_pts = []
    for j in range(_PTS_PER_SIDE):
        f = j / (_PTS_PER_SIDE - 1)
        local_pts.append((-ro + f * length, -width / 2))    # bottom
        local_pts.append((-ro + f * length,  width / 2))    # top
        local_pts.append((-ro, -width / 2 + f * width))     # left
        local_pts.append((length - ro, -width / 2 + f * width))  # right
    local_pts = torch.tensor(local_pts, device=device, dtype=ego_trajs.dtype)  # (80, 2)
    K_pts = local_pts.shape[0]

    # For each trajectory and timestep, transform perimeter to world frame
    cos_h = ego_trajs[..., 2]  # (N, T)
    sin_h = ego_trajs[..., 3]
    h_norm = (cos_h ** 2 + sin_h ** 2).sqrt().clamp_min(1e-6)
    cos_h = cos_h / h_norm
    sin_h = sin_h / h_norm

    # Rotation: (N, T, 2, 2)
    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=-1).reshape(N, T, 2, 2)
    # Rotated perimeter: (N, T, 80, 2)
    rotated = torch.einsum("btij,kj->btki", rot, local_pts)
    world_pts = ego_trajs[..., :2].unsqueeze(2) + rotated  # (N, T, 80, 2)

    # Distance from each perimeter point to nearest border point
    # world_pts: (N, T, 80, 2), border_pts: (K, 2)
    # Do this in chunks to avoid OOM for large K
    K_border = border_pts.shape[0]
    world_flat = world_pts.reshape(N * T * K_pts, 2)  # (N*T*80, 2)

    # Chunked min distance computation
    chunk_size = 5000
    min_dists = torch.full((N * T * K_pts,), 1e6, device=device)
    for start in range(0, K_border, chunk_size):
        end = min(start + chunk_size, K_border)
        bp_chunk = border_pts[start:end]  # (chunk, 2)
        d = torch.cdist(world_flat, bp_chunk)  # (N*T*80, chunk)
        chunk_min = d.min(dim=1).values  # (N*T*80,)
        min_dists = torch.minimum(min_dists, chunk_min)

    min_dists = min_dists.reshape(N, T, K_pts)  # (N, T, 80)

    # Per-timestep: min distance across all perimeter points
    per_timestep_min = min_dists.min(dim=2).values  # (N, T)

    # Skip t=0 (can't control starting position)
    per_timestep_min[:, 0] = 10.0

    # Crossing gate: any timestep with min dist < 0.10m = crossing
    _CROSS_THRESH = 0.10
    is_crossing = per_timestep_min < _CROSS_THRESH  # (N, T)
    has_crossing = is_crossing.any(dim=1)  # (N,)
    crossing_gate = (~has_crossing).float()  # (N,) 1.0=safe, 0.0=crossing

    # First crossing timestep per trajectory
    first_crossing_steps: list[int | None] = []
    for i in range(N):
        if has_crossing[i]:
            first_crossing_steps.append(int(is_crossing[i].nonzero(as_tuple=True)[0][0].item()))
        else:
            first_crossing_steps.append(None)

    # Near penalty: fraction of timesteps within 25cm
    _NEAR_THRESH = 0.25
    near_frac = (per_timestep_min[:, 1:] < _NEAR_THRESH).float().mean(dim=1)  # (N,)

    # Wide penalty: fraction of timesteps within 40cm
    _WIDE_THRESH = 0.40
    wide_frac = (per_timestep_min[:, 1:] < _WIDE_THRESH).float().mean(dim=1)  # (N,)

    # Continuous proximity penalty: smooth gradient from 0 to _CONT_THRESH
    # penalty = mean over timesteps of max(0, 1 - dist/_CONT_THRESH)
    # This creates a linear gradient pulling the trajectory away from the border
    _CONT_THRESH = 0.80
    cont_penalty = (1.0 - per_timestep_min[:, 1:] / _CONT_THRESH).clamp(min=0, max=1).mean(dim=1)  # (N,)

    return crossing_gate, near_frac, wide_frac, first_crossing_steps, cont_penalty


# ---------------------------------------------------------------------------
# Lane departure penalty
# ---------------------------------------------------------------------------

_LANE_CROSS_THRESH = 0.10
_LANE_NEAR_THRESH = 0.25
_LANE_WIDE_THRESH = 0.40
_LANE_CONT_THRESH = 0.80
_LANE_PTS_PER_SIDE = 20  # 80 total perimeter points


@torch.no_grad()
def compute_lane_departure_penalty(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-trajectory lane departure penalties using ego perimeter sampling.

    For each of 80 ego perimeter points at each timestep, finds the K=3 nearest
    lane centerline points from different lane segments and checks if the point
    is inside any of those lanes. Uses the full `lanes` tensor (140 segments),
    not just route_lanes, since ego can legitimately be in any lane.

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        ego_shape: (3,) wheel_base, length, width.
        data: Observation dict with 'lanes' key.

    Returns:
        Tuple of (crossing_gate, near_frac, wide_frac, cont_penalty):
        - crossing_gate: (N,) 1.0 if always in-lane, 0.0 if leaves lane
        - near_frac: (N,) fraction of timesteps within 25cm of lane edge
        - wide_frac: (N,) fraction of timesteps within 40cm of lane edge
        - cont_penalty: (N,) continuous proximity penalty (linear decay from 80cm)
    """
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device

    safe_return = (
        torch.ones(N, device=device),
        torch.zeros(N, device=device),
        torch.zeros(N, device=device),
        torch.zeros(N, device=device),
    )

    if "lanes" not in data:
        return safe_return

    lanes = data["lanes"]
    if lanes.dim() == 4:
        lanes = lanes[0]  # remove batch dim → (S, P, D)
    if lanes.shape[-1] < 8:
        return safe_return

    # Extract lane geometry
    S, P, D = lanes.shape
    center = lanes[..., :2].reshape(-1, 2)       # (S*P, 2)
    direction = lanes[..., 2:4].reshape(-1, 2)    # (S*P, 2)
    lb_offset = lanes[..., 4:6].reshape(-1, 2)    # (S*P, 2)
    rb_offset = lanes[..., 6:8].reshape(-1, 2)    # (S*P, 2)

    dir_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dir_unit = direction / dir_norm
    n_left = torch.stack([-dir_unit[..., 1], dir_unit[..., 0]], dim=-1)  # (S*P, 2)

    # Lane half-widths: project boundary offsets onto n_left
    width_left = (lb_offset * n_left).sum(dim=-1)    # (S*P,) positive = left
    width_right = (rb_offset * n_left).sum(dim=-1)   # (S*P,) negative = right

    # Valid mask: nonzero direction and nonzero center
    valid = (direction.norm(dim=-1) > 1e-6) & (center.norm(dim=-1) > 1e-3)
    num_valid = valid.sum().item()
    if num_valid == 0:
        return safe_return

    # Segment IDs for each centerline point (for K=3 from different segments)
    seg_ids = torch.arange(S, device=device).unsqueeze(1).expand(S, P).reshape(-1)  # (S*P,)

    # Build ego perimeter points (same as road border)
    wb = ego_shape[0].item()
    length = ego_shape[1].item()
    width = ego_shape[2].item()
    ro = (length - wb) / 2
    local_pts = []
    for j in range(_LANE_PTS_PER_SIDE):
        f = j / (_LANE_PTS_PER_SIDE - 1)
        local_pts.append((-ro + f * length, -width / 2))
        local_pts.append((-ro + f * length,  width / 2))
        local_pts.append((-ro, -width / 2 + f * width))
        local_pts.append((length - ro, -width / 2 + f * width))
    local_pts = torch.tensor(local_pts, device=device, dtype=ego_trajs.dtype)
    K_pts = local_pts.shape[0]  # 80

    # Transform perimeter to world frame
    cos_h = ego_trajs[..., 2]
    sin_h = ego_trajs[..., 3]
    h_norm = (cos_h ** 2 + sin_h ** 2).sqrt().clamp_min(1e-6)
    cos_h = cos_h / h_norm
    sin_h = sin_h / h_norm
    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=-1).reshape(N, T, 2, 2)
    rotated = torch.einsum("btij,kj->btki", rot, local_pts)
    world_pts = ego_trajs[..., :2].unsqueeze(2) + rotated  # (N, T, 80, 2)

    # Flatten query points
    Q = N * T * K_pts
    query = world_pts.reshape(Q, 2)  # (Q, 2)

    # Filter to valid centerline points only
    valid_center = center[valid]       # (V, 2)
    valid_n_left = n_left[valid]       # (V, 2)
    valid_wl = width_left[valid]       # (V,)
    valid_wr = width_right[valid]      # (V,)
    valid_seg = seg_ids[valid]         # (V,)
    V = valid_center.shape[0]

    # Compute distances from query points to valid centerline points
    # Process in chunks to avoid OOM
    best_clearance = torch.full((Q,), -1e6, device=device)
    chunk_size = 4000

    for q_start in range(0, Q, chunk_size):
        q_end = min(q_start + chunk_size, Q)
        q_chunk = query[q_start:q_end]  # (C, 2)
        C = q_chunk.shape[0]

        # Distance to all valid centerline points
        dist2 = ((q_chunk.unsqueeze(1) - valid_center.unsqueeze(0)) ** 2).sum(-1)  # (C, V)

        # For K=3 candidates from different segments
        chunk_clearance = torch.full((C,), -1e6, device=device)

        remaining_mask = torch.ones(C, V, dtype=torch.bool, device=device)
        for _k in range(3):
            # Mask out already-used segments
            masked_dist2 = dist2.clone()
            masked_dist2[~remaining_mask] = float('inf')

            # Find nearest
            min_d2, min_idx = masked_dist2.min(dim=1)  # (C,)
            has_valid = torch.isfinite(min_d2)

            if not has_valid.any():
                break

            # Get lane geometry at nearest point
            sel_center = valid_center[min_idx]     # (C, 2)
            sel_n_left = valid_n_left[min_idx]     # (C, 2)
            sel_wl = valid_wl[min_idx]             # (C,)
            sel_wr = valid_wr[min_idx]             # (C,)
            sel_seg = valid_seg[min_idx]           # (C,)

            # Lateral distance
            lat = ((q_chunk - sel_center) * sel_n_left).sum(dim=-1)  # (C,)
            dist_left = sel_wl - lat    # positive = inside on left side
            dist_right = lat - sel_wr   # positive = inside on right side

            # Clearance = min distance to either boundary (positive = inside lane)
            clearance = torch.minimum(dist_left, dist_right)  # (C,)
            clearance = torch.where(has_valid, clearance, torch.full_like(clearance, -1e6))

            # Update best clearance (max across K candidates = least violation)
            chunk_clearance = torch.maximum(chunk_clearance, clearance)

            # Mask out this segment for next iteration
            seg_mask = valid_seg.unsqueeze(0) == sel_seg.unsqueeze(1)  # (C, V)
            remaining_mask = remaining_mask & ~seg_mask

        best_clearance[q_start:q_end] = chunk_clearance

    # Reshape to (N, T, 80)
    best_clearance = best_clearance.reshape(N, T, K_pts)

    # Per-timestep: min clearance across all 80 perimeter points
    per_ts_min = best_clearance.min(dim=2).values  # (N, T)

    # Skip t=0
    per_ts_min[:, 0] = 10.0

    # Crossing gate: clearance is positive when inside lane, negative when outside.
    # Threshold is +0.10m (conservative): triggers when within 10cm of edge OR outside,
    # treating near-edge trajectories as lane departures for safety margin.
    is_crossing = per_ts_min < _LANE_CROSS_THRESH
    has_crossing = is_crossing.any(dim=1)
    crossing_gate = (~has_crossing).float()

    # Near penalty
    near_frac = (per_ts_min[:, 1:] < _LANE_NEAR_THRESH).float().mean(dim=1)

    # Wide penalty
    wide_frac = (per_ts_min[:, 1:] < _LANE_WIDE_THRESH).float().mean(dim=1)

    # Continuous proximity penalty
    cont_penalty = (1.0 - per_ts_min[:, 1:] / _LANE_CONT_THRESH).clamp(min=0, max=1).mean(dim=1)

    return crossing_gate, near_frac, wide_frac, cont_penalty


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
        if nf.shape[1] >= T and nf.shape[2] >= 3:
            # NPZ stores (x, y, yaw_rad) -- convert to (x, y, cos, sin)
            if nf.shape[2] == 3:
                nf_xy = nf[:, :T, :2]
                nf_yaw = nf[:, :T, 2:3]
                nf_cos_sin = torch.cat([torch.cos(nf_yaw), torch.sin(nf_yaw)], dim=-1)
                nf_data = torch.cat([nf_xy, nf_cos_sin], dim=-1)  # (N_nb, T, 4)
            else:
                nf_data = nf[:, :T, :4]
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
    progress_scores = compute_progress_score_batch(ego_trajs, goal_pose, data)
    smoothness_scores = compute_smoothness_score_batch(ego_trajs, config)
    feasibility_scores, off_road_fractions = compute_feasibility_score_batch(
        ego_trajs, ego_shape, data, config
    )
    centerline_scores = compute_centerline_score_batch(ego_trajs, ego_shape, data)
    red_light_scores = compute_red_light_score_batch(ego_trajs, data, config)
    ttc_scores = compute_ttc_score_batch(
        ego_trajs, ego_shape, neighbor_futures, neighbor_shapes, neighbor_valid
    )

    # Road border penalty using ego perimeter sampling
    rb_crossing_gate, rb_near_frac, rb_wide_frac, rb_crossing_steps, rb_cont_penalty = compute_road_border_penalty(
        ego_trajs, ego_shape, data,
    )

    # Lane departure penalty
    if config.enable_lane_departure:
        lane_crossing_gate, lane_near_frac, lane_wide_frac, lane_cont_penalty = compute_lane_departure_penalty(
            ego_trajs, ego_shape, data,
        )
    else:
        lane_crossing_gate = torch.ones(N, device=device)
        lane_near_frac = torch.zeros(N, device=device)
        lane_wide_frac = torch.zeros(N, device=device)
        lane_cont_penalty = torch.zeros(N, device=device)

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
    safety_product = collision_gate * red_light_gate * rb_crossing_gate  # (N,)
    if config.lane_gate_enabled:
        safety_product = safety_product * lane_crossing_gate

    # Weighted quality metrics (only matter when safety gates pass)
    # Progress is the primary positive signal. Smoothness/centerline are penalties.
    # Safety score includes proximity penalty to NPCs (closer = more negative).
    clamped_progress = progress_scores.clamp(min=0)

    # Overprogress penalty: cap reward at GT path length × 1.3.
    # Trajectories that go much further than the human drove get penalized,
    # preventing the model from learning to drive too fast through curves.
    # Overprogress: cap progress reward at GT path length × 1.1, then penalize
    # excess mildly. Disabled by default — causes stopping when penalty is too
    if config.enable_overprogress and "ego_agent_future" in data:
        gt_future = data["ego_agent_future"]
        if gt_future.dim() == 3:
            gt_future = gt_future[0]  # (T_gt, 3)
        gt_xy = gt_future[:, :2]
        gt_valid = gt_xy.abs().sum(dim=-1) > 0.1
        if gt_valid.sum() >= 10:
            gt_path_len = torch.diff(gt_xy[gt_valid], dim=0).norm(dim=-1).sum()
            cap = config.overprogress_margin * gt_path_len
            model_path_lens = torch.diff(ego_trajs[:, :, :2], dim=1).norm(dim=-1).sum(dim=-1)  # (N,)
            # Cap: progress can't exceed cap value. Excess gets penalized.
            capped = torch.minimum(clamped_progress, cap.expand(N))
            excess = torch.relu(model_path_lens - cap)
            clamped_progress = capped - config.overprogress_penalty * excess

            # Stopped penalty: if GT drives (>5m) but model barely moves (<1m),
            # apply extra negative progress to discourage stopping.
            if gt_path_len > 5.0:
                is_stopped = (model_path_lens < 1.0).float()
                clamped_progress = clamped_progress - config.stopped_penalty * is_stopped

    # TTC as quality bonus
    ttc_bonus = config.w_safety * (ttc_scores - 0.5) * 2

    # Road border proximity penalties (soft, applied even when on-road)
    # near (< 25cm): considerable penalty; wide (< 40cm): lighter penalty
    _RB_NEAR_SCALE = config.near_edge_scale  # reuse near_edge config
    _RB_WIDE_SCALE = config.wide_edge_scale
    rb_penalty = _RB_NEAR_SCALE * rb_near_frac + _RB_WIDE_SCALE * rb_wide_frac + config.cont_edge_scale * rb_cont_penalty

    # Lane departure proximity penalties
    lane_penalty = config.lane_near_scale * lane_near_frac + config.lane_wide_scale * lane_wide_frac + config.lane_cont_scale * lane_cont_penalty

    quality_score = (
        config.w_progress * clamped_progress
        + config.w_safety * safety_scores
        + config.w_smooth * smoothness_scores
        + config.w_centerline * centerline_scores
        + ttc_bonus
        - rb_penalty
        - lane_penalty
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
            if rb_crossing_steps[i] is not None:
                first_terminal = min(first_terminal, rb_crossing_steps[i])
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

    # Also compute additive total for backward compat in breakdown
    on_road_factor = (1.0 - off_road_fractions)
    adjusted_progress = progress_scores * on_road_factor

    results: list[RewardBreakdown] = []
    for i in range(N):
        results.append(RewardBreakdown(
            safety=float(safety_scores[i]),
            progress=float(adjusted_progress[i]),
            smoothness=float(smoothness_scores[i]),
            feasibility=float(feasibility_scores[i]),
            centerline=float(centerline_scores[i]),
            red_light=float(red_light_scores[i]),
            total=float(totals[i]),
            collision_step=collision_steps[i],
            off_road_fraction=float(off_road_fractions[i]),  # always 0 (polygon disabled); use rb_crossing/rb_near_frac instead
            rb_crossing=bool(rb_crossing_gate[i] < 0.5),
            rb_near_frac=float(rb_near_frac[i]),
            lane_crossing=bool(lane_crossing_gate[i] < 0.5),
            lane_near_frac=float(lane_near_frac[i]),
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
    elif mode == "positive_only":
        # Standard normalization but clip negatives to zero.
        # Only reinforces trajectories better than the group mean.
        std = totals.std()
        if std < epsilon:
            return np.zeros(len(rewards))
        advantages = (totals - mean) / (std + epsilon)
        return np.maximum(advantages, 0.0)
    else:
        raise ValueError(
            f"Unknown advantage mode: {mode!r}. "
            f"Expected 'normalized', 'vd_grpo', 'raw', or 'positive_only'."
        )
