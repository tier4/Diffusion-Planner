"""Extract goal poses and route lanelets from agent GT future trajectories.

For agents that have ground truth futures but no goal/route (i.e., neighbors
loaded from NPZ), this module:
1. Sets goal_pose from the final GT position
2. Finds route_lanes by matching GT trajectory points to the closest lane segments
"""

from __future__ import annotations

import numpy as np

from scenario_generation.scene_context import SceneContext


def _match_trajectory_to_lanes(
    trajectory_xy: np.ndarray,
    lanes: np.ndarray,
    max_segments: int = 25,
    sample_interval: int = 5,
) -> list[int]:
    """Find ordered lane segment indices that a trajectory passes through.

    Args:
        trajectory_xy: (T, 2) trajectory positions.
        lanes: (N_lanes, 20, 33) lane segments.
        max_segments: Maximum route segments to return.
        sample_interval: Sample every N-th trajectory point for matching.

    Returns:
        Ordered list of lane indices (deduplicated, preserving traversal order).
    """
    lane_centers = lanes[:, :, :2]  # (N_lanes, 20, 2)
    N_lanes = lanes.shape[0]

    # Identify valid lanes (non-zero geometry)
    lane_norms = np.abs(lane_centers).sum(axis=(1, 2))
    valid_mask = lane_norms > 1.0
    if not valid_mask.any():
        return []

    valid_indices = np.where(valid_mask)[0]
    valid_centers = lane_centers[valid_mask]  # (N_valid, 20, 2)

    # Sample trajectory points
    sample_pts = trajectory_xy[::sample_interval]  # (N_samples, 2)

    matched: list[int] = []
    for pt in sample_pts:
        # Distance from this point to all 20 points of each valid lane
        diffs = valid_centers - pt[np.newaxis, np.newaxis, :]  # (N_valid, 20, 2)
        dists_per_point = np.linalg.norm(diffs, axis=-1)  # (N_valid, 20)
        min_dist_per_lane = dists_per_point.min(axis=1)  # (N_valid,)

        closest_valid_idx = int(np.argmin(min_dist_per_lane))
        closest_lane = int(valid_indices[closest_valid_idx])

        # Deduplicate consecutive matches
        if not matched or matched[-1] != closest_lane:
            matched.append(closest_lane)

        if len(matched) >= max_segments:
            break

    return matched


def assign_gt_goals_and_routes(
    scene: SceneContext,
    overwrite_existing: bool = False,
    min_gt_timesteps: int = 10,
) -> int:
    """Assign goal_pose and route_lanes to agents from their GT futures.

    For each agent that has a future_trajectory:
    - goal_pose is set to the last valid (non-zero) GT position
    - route_lanes are extracted by matching GT to the closest map lanes

    Agents whose GT future has fewer than min_gt_timesteps valid entries
    are removed from the scene (likely misdetections).

    Agents that already have goal_pose / route_lanes are skipped unless
    overwrite_existing is True.

    Args:
        scene: SceneContext to modify in-place.
        overwrite_existing: If True, overwrite existing goals/routes.
        min_gt_timesteps: Minimum valid GT timesteps to keep an agent.
            Non-ego agents below this threshold are removed.

    Returns:
        Number of agents that were updated.
    """
    lanes = scene.map_data.lanes
    lanes_sl = scene.map_data.lanes_speed_limit
    lanes_hsl = scene.map_data.lanes_has_speed_limit
    updated = 0
    remove_ids: list[str] = []

    for agent in scene.agents:
        if agent.future_trajectory is None:
            continue

        gt = agent.future_trajectory  # (T, 3) [x, y, heading_rad]

        # Trim trailing zeros (lost tracking / padding)
        valid = np.abs(gt[:, :2]).sum(axis=1) > 1e-3
        n_valid = int(valid.sum())

        # Remove short-lived non-ego agents (misdetections)
        if n_valid < min_gt_timesteps and agent.id != scene.ego_agent_id:
            remove_ids.append(agent.id)
            continue

        if not valid.any():
            continue
        last_valid_idx = int(np.where(valid)[0][-1])
        gt_trimmed = gt[:last_valid_idx + 1]

        needs_goal = agent.goal_pose is None or overwrite_existing
        needs_route = agent.route_lanes is None or overwrite_existing

        if not needs_goal and not needs_route:
            continue

        if needs_goal:
            agent.goal_pose = gt_trimmed[-1].copy().astype(np.float32)

        if needs_route:
            matched = _match_trajectory_to_lanes(gt_trimmed[:, :2], lanes)
            if matched:
                agent.route_lanes = lanes[matched].copy()
                agent.route_speed_limit = lanes_sl[matched].copy()
                agent.route_has_speed_limit = lanes_hsl[matched].copy()

        updated += 1

    if remove_ids:
        print(f"Removing {len(remove_ids)} short-lived agents (<{min_gt_timesteps} GT steps): {remove_ids}")
        scene.agents = [a for a in scene.agents if a.id not in remove_ids]

    return updated
