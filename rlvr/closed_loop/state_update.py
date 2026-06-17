"""Scene state update for closed-loop rollout.

Re-centers all scene elements after ego executes one trajectory step.
Uses the same transform convention as StatePerturbation.centric_transform
in diffusion_planner/utils/data_augmentation.py.
"""

import math

import torch
from diffusion_planner.utils.data_augmentation import (
    heading_transform,
    vector_transform,
)


def build_transform_matrix(cos_h: float, sin_h: float, device: torch.device) -> torch.Tensor:
    """Build 2x2 rotation matrix from heading (cos, sin).

    Same convention as StatePerturbation.get_transform_matrix_batch:
        R = [[cos, sin], [-sin, cos]]
    This rotates vectors by -theta, making the heading direction become [1, 0].

    Returns: [1, 2, 2] rotation matrix.
    """
    return torch.tensor(
        [[[cos_h, sin_h], [-sin_h, cos_h]]],
        dtype=torch.float32,
        device=device,
    )


def update_scene_state(
    data: dict[str, torch.Tensor],
    trajectory: torch.Tensor,
    step_idx: int = 0,
    dt: float = 0.1,
) -> tuple[dict[str, torch.Tensor], tuple[float, float, float]]:
    """Re-center scene to ego's new pose after executing one trajectory step.

    Caller should update neighbor_agents_past BEFORE calling this if advancing
    neighbors (see advance_neighbor_past).

    Args:
        data: Scene dict in current ego frame. All tensors have batch dim [1, ...].
        trajectory: [T, 4] or [1, T, 4] DiT output (x, y, cos_h, sin_h)
              in current ego frame.
        step_idx: Which trajectory timestep to execute (default 0 = first 0.1s step).
        dt: Time step duration in seconds.

    Returns:
        new_data: Deep-cloned scene dict re-centered on new ego pose.
        ego_delta: (dx, dy, delta_heading) of the step in the OLD ego frame,
                   useful for tracking absolute pose.
    """
    if trajectory.dim() == 3:
        trajectory = trajectory.squeeze(0)

    device = trajectory.device

    # --- Extract new ego pose from trajectory ---
    dx = trajectory[step_idx, 0].item()
    dy = trajectory[step_idx, 1].item()
    cos_new = trajectory[step_idx, 2].item()
    sin_new = trajectory[step_idx, 3].item()
    delta_heading = math.atan2(sin_new, cos_new)

    # --- Deep clone all tensors ---
    new_data: dict[str, torch.Tensor] = {}
    for k, v in data.items():
        new_data[k] = v.clone() if isinstance(v, torch.Tensor) else v

    # --- Build transform (same as get_transform_matrix_batch) ---
    center_xy = torch.tensor([[[dx, dy]]], dtype=torch.float32, device=device)
    T = build_transform_matrix(cos_new, sin_new, device)

    # --- Transform ego_current_state [1, 10]: x,y,cos,sin,vx,vy,ax,ay,steering,yaw_rate ---
    old_ego_vx = new_data["ego_current_state"][0, 4].item()
    old_ego_vy = new_data["ego_current_state"][0, 5].item()

    # New velocity from finite difference (in old frame), then rotate to new frame
    new_vx_old_frame = dx / dt
    new_vy_old_frame = dy / dt
    # Rotate to new frame
    new_vx = cos_new * new_vx_old_frame + sin_new * new_vy_old_frame
    new_vy = -sin_new * new_vx_old_frame + cos_new * new_vy_old_frame

    # Acceleration from velocity change (in new frame)
    old_v_rotated_x = cos_new * old_ego_vx + sin_new * old_ego_vy
    old_v_rotated_y = -sin_new * old_ego_vx + cos_new * old_ego_vy
    new_ax = (new_vx - old_v_rotated_x) / dt
    new_ay = (new_vy - old_v_rotated_y) / dt

    new_data["ego_current_state"][0, 0] = 0.0  # x
    new_data["ego_current_state"][0, 1] = 0.0  # y
    new_data["ego_current_state"][0, 2] = 1.0  # cos(0)
    new_data["ego_current_state"][0, 3] = 0.0  # sin(0)
    new_data["ego_current_state"][0, 4] = new_vx
    new_data["ego_current_state"][0, 5] = new_vy
    new_data["ego_current_state"][0, 6] = new_ax
    new_data["ego_current_state"][0, 7] = new_ay
    # steering and yaw_rate: approximate from heading change
    new_data["ego_current_state"][0, 9] = delta_heading / dt  # yaw_rate

    # --- Transform ego_agent_past [1, 31, 4]: x, y, cos_h, sin_h ---
    # Append old current pose [0, 0, 1, 0] (ego was at origin), drop oldest
    if "ego_agent_past" in new_data:
        old_current = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=device
        ).unsqueeze(0)  # [1, 1, 4]
        past = new_data["ego_agent_past"]  # [1, 31, 4]
        # Roll: drop index 0, append old current at the end
        new_data["ego_agent_past"] = torch.cat([past[:, 1:, :], old_current], dim=1)
        # Transform positions
        new_data["ego_agent_past"][..., :2] = vector_transform(
            new_data["ego_agent_past"][..., :2], T, center_xy
        )
        # Transform heading (cos, sin)
        new_data["ego_agent_past"][..., 2:4] = vector_transform(
            new_data["ego_agent_past"][..., 2:4], T
        )

    # --- Transform neighbor_agents_past [1, N_nb, 31, 11] ---
    if "neighbor_agents_past" in new_data:
        nb = new_data["neighbor_agents_past"]
        mask = torch.sum(torch.ne(nb[..., :6], 0), dim=-1) == 0
        # xy
        nb[..., :2] = vector_transform(nb[..., :2], T, center_xy)
        # cos, sin heading
        nb[..., 2:4] = vector_transform(nb[..., 2:4], T)
        # vx, vy
        nb[..., 4:6] = vector_transform(nb[..., 4:6], T)
        nb[mask] = 0.0

    # --- Transform lanes [1, 140, 20, 33] ---
    if "lanes" in new_data:
        la = new_data["lanes"]
        mask = torch.sum(torch.ne(la[..., :8], 0), dim=-1) == 0
        la[..., :2] = vector_transform(la[..., :2], T, center_xy)  # center xy
        la[..., 2:4] = vector_transform(la[..., 2:4], T)  # direction
        la[..., 4:6] = vector_transform(la[..., 4:6], T)  # left boundary
        la[..., 6:8] = vector_transform(la[..., 6:8], T)  # right boundary
        la[mask] = 0.0

    # --- Transform route_lanes [1, 25, 20, 33] ---
    if "route_lanes" in new_data:
        rl = new_data["route_lanes"]
        mask = torch.sum(torch.ne(rl[..., :8], 0), dim=-1) == 0
        rl[..., :2] = vector_transform(rl[..., :2], T, center_xy)
        rl[..., 2:4] = vector_transform(rl[..., 2:4], T)
        rl[..., 4:6] = vector_transform(rl[..., 4:6], T)
        rl[..., 6:8] = vector_transform(rl[..., 6:8], T)
        rl[mask] = 0.0

    # --- Transform line_strings [1, 60, 20, 4] ---
    if "line_strings" in new_data:
        ls = new_data["line_strings"]
        mask = torch.sum(torch.ne(ls, 0), dim=-1) == 0
        ls[..., :2] = vector_transform(ls[..., :2], T, center_xy)
        ls[mask] = 0.0

    # --- Transform polygons [1, 10, 40, 3] ---
    if "polygons" in new_data:
        pg = new_data["polygons"]
        mask = torch.sum(torch.ne(pg, 0), dim=-1) == 0
        pg[..., :2] = vector_transform(pg[..., :2], T, center_xy)
        pg[mask] = 0.0

    # --- Transform static_objects [1, 5, 10] ---
    if "static_objects" in new_data:
        so = new_data["static_objects"]
        mask = torch.sum(torch.ne(so[..., :10], 0), dim=-1) == 0
        so[..., :2] = vector_transform(so[..., :2], T, center_xy)
        so[..., 2:4] = vector_transform(so[..., 2:4], T)
        so[mask] = 0.0

    # --- Transform goal_pose [1, 4]: x, y, cos_h, sin_h ---
    if "goal_pose" in new_data:
        gp = new_data["goal_pose"]
        if gp.dim() == 2:
            # [1, 4] — single goal
            gp[..., :2] = vector_transform(gp[..., :2].unsqueeze(1), T, center_xy).squeeze(1)
            gp[..., 2:4] = vector_transform(gp[..., 2:4].unsqueeze(1), T).squeeze(1)
        elif gp.dim() == 3:
            # [1, N_goals, 4] — multiple goals
            gp[..., :2] = vector_transform(gp[..., :2], T, center_xy)
            gp[..., 2:4] = vector_transform(gp[..., 2:4], T)

    return new_data, (dx, dy, delta_heading)


def advance_neighbor_past(
    data: dict[str, torch.Tensor],
    new_nb_positions: torch.Tensor,
    dt: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Advance neighbor_agents_past by one step with new positions.

    Call BEFORE update_scene_state so the new positions get transformed
    along with everything else.

    Args:
        data: Scene dict (will be modified in-place on the clone).
        new_nb_positions: [N_nb, 4] (x, y, cos_h, sin_h) in CURRENT ego frame.
            Use zeros for invalid/absent neighbors.
        dt: Time step for velocity computation.

    Returns:
        Modified data dict.
    """
    if "neighbor_agents_past" not in data:
        return data

    nb = data["neighbor_agents_past"]  # [1, N_nb, 31, 11]
    N_nb = nb.shape[1]
    device = nb.device

    # Get old current positions (last timestep) for velocity computation
    old_pos = nb[0, :, -1, :2]  # [N_nb, 2]
    new_pos = new_nb_positions[:, :2]  # [N_nb, 2]

    # Compute velocities from position difference
    new_vel = (new_pos - old_pos) / dt  # [N_nb, 2]

    # Build new timestep entry [N_nb, 11]
    new_entry = torch.zeros(N_nb, 11, dtype=torch.float32, device=device)
    new_entry[:, :4] = new_nb_positions[:, :4]  # x, y, cos_h, sin_h
    new_entry[:, 4:6] = new_vel  # vx, vy
    # Copy static attributes (width, length, type) from last known entry
    new_entry[:, 6:] = nb[0, :, -1, 6:]

    # Zero out entries where new position is all zeros (invalid neighbor)
    invalid = new_nb_positions[:, :2].abs().sum(dim=-1) == 0
    new_entry[invalid] = 0.0

    # Roll: drop oldest (index 0), append new entry at end
    data["neighbor_agents_past"] = torch.cat(
        [nb[:, :, 1:, :], new_entry.unsqueeze(0).unsqueeze(2)],
        dim=2,
    )

    return data


def transform_positions_to_ego_frame(
    positions: torch.Tensor,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    device: torch.device,
) -> torch.Tensor:
    """Transform positions from original frame to current ego-centric frame.

    Used by RolloutManager to transform GT neighbor futures into current ego frame.

    Args:
        positions: [N, 3] (x, y, heading_rad) in original frame.
        ego_x, ego_y, ego_heading: Ego's absolute pose in original frame.
        device: Torch device.

    Returns:
        [N, 4] (x, y, cos_h, sin_h) in ego-centric frame.
    """
    cos_h = math.cos(ego_heading)
    sin_h = math.sin(ego_heading)

    # Translate: subtract ego position
    rel_x = positions[:, 0] - ego_x
    rel_y = positions[:, 1] - ego_y

    # Rotate by -ego_heading
    x_ego = cos_h * rel_x + sin_h * rel_y
    y_ego = -sin_h * rel_x + cos_h * rel_y

    # Transform heading
    rel_heading = positions[:, 2] - ego_heading
    cos_rel = torch.cos(rel_heading)
    sin_rel = torch.sin(rel_heading)

    result = torch.zeros(positions.shape[0], 4, dtype=torch.float32, device=device)
    result[:, 0] = x_ego
    result[:, 1] = y_ego
    result[:, 2] = cos_rel
    result[:, 3] = sin_rel

    return result
