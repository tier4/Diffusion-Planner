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

    # Per-point validity mask (exclude zero-padded points at [0,0])
    lane_point_valid = np.abs(lane_centers).sum(axis=-1) > 1e-6  # (N_lanes, 20)
    valid_mask = lane_point_valid.any(axis=1)  # lane has at least one real point
    if not valid_mask.any():
        return []

    valid_indices = np.where(valid_mask)[0]
    valid_centers = lane_centers[valid_mask]  # (N_valid, 20, 2)
    valid_point_mask = lane_point_valid[valid_mask]  # (N_valid, 20)

    # Sample trajectory points
    sample_pts = trajectory_xy[::sample_interval]  # (N_samples, 2)

    matched: list[int] = []
    for pt in sample_pts:
        diffs = valid_centers - pt[np.newaxis, np.newaxis, :]  # (N_valid, 20, 2)
        dists_per_point = np.linalg.norm(diffs, axis=-1)  # (N_valid, 20)
        # Mask out zero-padded points so they don't participate in matching
        dists_per_point = np.where(valid_point_mask, dists_per_point, np.inf)
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
        gt_trimmed = gt[: last_valid_idx + 1]

        needs_goal = agent.goal_pose is None or overwrite_existing
        needs_route = agent.route_lanes is None or overwrite_existing

        if not needs_goal and not needs_route:
            continue

        changed = False
        if needs_goal:
            agent.goal_pose = gt_trimmed[-1].copy().astype(np.float32)
            changed = True

        if needs_route:
            matched = _match_trajectory_to_lanes(gt_trimmed[:, :2], lanes)
            if matched:
                agent.route_lanes = lanes[matched].copy()
                agent.route_speed_limit = lanes_sl[matched].copy()
                agent.route_has_speed_limit = lanes_hsl[matched].copy()
                changed = True

        if changed:
            updated += 1

    if remove_ids:
        import logging

        logging.getLogger(__name__).info(
            "Removing %d short-lived agents (<%d GT steps): %s",
            len(remove_ids),
            min_gt_timesteps,
            remove_ids,
        )
        scene.agents = [a for a in scene.agents if a.id not in remove_ids]

    return updated
