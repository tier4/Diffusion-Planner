import torch
import torch.nn.functional as F

from diffusion_planner.dimensions import INPUT_T, OUTPUT_T, TURN_INDICATOR_OUTPUT_KEEP
from diffusion_planner.model.guidance.collision import center_rect_to_points

_NEIGHBOR_EVAL_STEPS = [0, 20, 40, 60, 79]

# One-hot agent type occupies columns 8..10 = [vehicle, pedestrian, bicycle]
# (matches synthetic_neighbors._TYPE_BASE / neighbor_db).
_TYPE_BASE = 8
_TYPE_VEHICLE = 0
_TYPE_PEDESTRIAN = 1
_TYPE_BICYCLE = 2


# ---------------------------------------------------------------------------
# Velocity (delta) representation utilities  (HDP paper, Section IV-B)
# ---------------------------------------------------------------------------


def waypoints_to_velocity(waypoints: torch.Tensor) -> torch.Tensor:
    """Convert absolute waypoints to per-frame displacement (velocity).

    Only the position channels (first 2) are converted to finite differences;
    heading channels (cos, sin) are taken from the destination frame.

    Args:
        waypoints: [..., T+1, D] where D >= 2.  The first two channels are (x, y).

    Returns:
        velocity: [..., T, D].  velocity[..., i, :2] = waypoints[..., i+1, :2] - waypoints[..., i, :2].
    """
    assert waypoints.shape[-2] == OUTPUT_T + 1, "Expected waypoints shape [..., T+1, D]"
    vel_pos = torch.diff(waypoints[..., :2], dim=-2)  # [..., T, 2]
    if waypoints.shape[-1] > 2:
        return torch.cat([vel_pos, waypoints[..., 1:, 2:]], dim=-1)
    return vel_pos


def velocity_to_waypoints(velocity: torch.Tensor) -> torch.Tensor:
    """Integrate per-frame displacement back to absolute waypoints (cumulative sum).

    Args:
        velocity: [..., T, D].  First two channels are (dx, dy).

    Returns:
        waypoints: [..., T, D].
    """
    pos = torch.cumsum(velocity[..., :2], dim=-2)
    if velocity.shape[-1] > 2:
        return torch.cat([pos, velocity[..., 2:]], dim=-1)
    return pos


def _detached_integral(v: torch.Tensor, W: int) -> torch.Tensor:
    """Compute waypoints from velocity with gradient detach window.

    This implements Algorithm 1 from the HDP paper appendix.
    Gradients only flow through a sliding window of size *W* so that the
    position loss does not cause gradient accumulation over the full horizon.

    Args:
        v: [..., T, 2] predicted displacement (position channels only).
        W: gradient detach window size.

    Returns:
        waypoints: [..., T, 2] integrated positions.
    """
    # Fully-detached cumsum shifted by W
    wpt_sg = torch.cumsum(v.detach(), dim=-2)  # [..., T, 2]
    shift_sg = torch.roll(wpt_sg, shifts=W, dims=-2)
    shift_sg[..., :W, :] = 0.0

    # Live cumsum (gradients flow)
    wpt = torch.cumsum(v, dim=-2)  # [..., T, 2]
    shift = torch.roll(wpt, shifts=W, dims=-2)
    shift[..., :W, :] = 0.0

    return wpt + shift_sg - shift


def hybrid_loss(
    pred_v: torch.Tensor,
    gt_v: torch.Tensor,
    omega: float,
    W: int,
) -> torch.Tensor:
    """Compute hybrid velocity + waypoint loss per the HDP paper (Eq. 5).

    Args:
        pred_v: [..., T, D] predicted per-frame displacement (D >= 2).
        gt_v:   [..., T, D] ground-truth per-frame displacement.
        omega:  balancing weight for the waypoint loss term.
        W:      gradient detach window size for waypoint loss.

    Returns:
        loss: [..., T] per-timestep loss (velocity L2 + omega * waypoint L2).
    """
    # Velocity loss on all D channels
    l_v = torch.sum((pred_v - gt_v) ** 2, dim=-1)  # [..., T]

    if omega <= 0.0:
        return l_v

    # Waypoint loss only on position channels
    pred_pos = _detached_integral(pred_v[..., :2], W)  # [..., T, 2]
    gt_pos = torch.cumsum(gt_v[..., :2], dim=-2)  # [..., T, 2]
    l_wpt = torch.sum((pred_pos - gt_pos) ** 2, dim=-1)  # [..., T]

    return l_v + omega * l_wpt


def make_turn_indicator_gt(
    turn_indicators: torch.Tensor,  # # [B, INPUT_T + 1]
) -> torch.Tensor:
    turn_indicators_gt = turn_indicators.long()  # [B, INPUT_T + 1]
    turn_indicators_gt_keep = turn_indicators_gt[:, -1] == turn_indicators_gt[:, -2]  # [B,]
    turn_indicators_gt = turn_indicators_gt[:, -1] * ~turn_indicators_gt_keep  # change to 0 if keep
    turn_indicators_gt = turn_indicators_gt + turn_indicators_gt_keep * TURN_INDICATOR_OUTPUT_KEEP
    return turn_indicators_gt


def loss_func(
    trajectory_pred: torch.Tensor, trajectory_gt: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Calculate the loss between predicted and ground truth trajectories.

    Args:
        trajectory_pred (torch.Tensor): Predicted trajectory of shape [..., T, D].
        trajectory_gt (torch.Tensor): Ground truth trajectory of shape [..., T, D].
        where, D=4 (x, y, cos, sin).

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the loss values.
        where, each loss' shape is [..., T].
    """
    result_dict = {}

    ###################
    # Basic L2 Losses #
    ###################
    # simple L2 loss
    result_dict["simple_l2_loss"] = torch.mean((trajectory_pred - trajectory_gt) ** 2, dim=-1)

    # Position loss (x, y coordinates)
    position_pred = trajectory_pred[..., :2]  # [..., T, 2]
    position_gt = trajectory_gt[..., :2]  # [..., T, 2]

    # Calculate L2 distance for each time step
    position_diff = position_pred - position_gt  # [..., T, 2]
    position_error = torch.sum(position_diff**2, dim=-1)  # [..., T]
    result_dict["position_l2_loss"] = position_error

    # Heading loss (cos, sin components)
    cos_sin_pred = trajectory_pred[..., 2:]  # [..., T, 2]
    cos_sin_gt = trajectory_gt[..., 2:]  # [..., T, 2]

    # heading l2 loss
    heading_loss = torch.sum((cos_sin_pred - cos_sin_gt) ** 2, dim=-1)  # [..., T]
    result_dict["heading_l2_loss"] = heading_loss

    ######################
    # Specialized Losses #
    ######################
    # Lateral or longitudinal error (along vehicle direction)
    cos_gt = cos_sin_gt[..., 0]  # [..., T]
    sin_gt = cos_sin_gt[..., 1]  # [..., T]
    lon_diff = +position_diff[..., 0] * cos_gt + position_diff[..., 1] * sin_gt  # [..., T]
    lat_diff = -position_diff[..., 0] * sin_gt + position_diff[..., 1] * cos_gt  # [..., T]
    lat_error = torch.abs(lat_diff)  # [..., T]
    lon_error = torch.abs(lon_diff)  # [..., T]
    result_dict["position_lat_loss"] = lat_error
    result_dict["position_lon_loss"] = lon_error

    # Cosine similarity loss
    cosine_similarity = torch.sum(cos_sin_pred * cos_sin_gt, dim=-1)  # [..., T]
    result_dict["cosine_similarity_loss"] = 1.0 - cosine_similarity  # [..., T]

    return result_dict


def point_to_segment_distance(
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute distance from points to line segments.

    Args:
        p: [..., 2] query points.
        a: [..., 2] segment start points.
        b: [..., 2] segment end points.

    Returns:
        dist: [...] non-negative distances.
    """
    ab = b - a
    ap = p - a
    t = (ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-8)
    t = t.clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    return ((p - closest) ** 2).sum(-1).clamp_min(1e-8).sqrt()


def compute_ego_bbox_corners(
    ego_traj: torch.Tensor,
    ego_shape: torch.Tensor,
) -> torch.Tensor:
    """Compute ego bounding box corners from trajectory and vehicle shape.

    Args:
        ego_traj: [B, T, 4] ego trajectory (x, y, cos_heading, sin_heading).
        ego_shape: [B, 3] (wheelbase, length, width).

    Returns:
        corners: [B, T, 4, 2] four corners per timestep.
    """
    B, T, _ = ego_traj.shape
    device = ego_traj.device
    dtype = ego_traj.dtype

    heading = ego_traj[..., 2:]
    heading_unit = heading / torch.linalg.norm(heading, dim=-1, keepdim=True).clamp_min(1e-6)
    ego_xy = ego_traj[..., :2]

    cog_to_rear = 0.5 * ego_shape[:, 0:1].unsqueeze(-1)  # [B, 1, 1]
    ego_center_xy = ego_xy + heading_unit * cog_to_rear

    half_length = (ego_shape[:, 1] / 2.0).unsqueeze(-1).expand(-1, T)
    half_width = (ego_shape[:, 2] / 2.0).unsqueeze(-1).expand(-1, T)
    half_sizes = torch.stack([half_length, half_width], dim=-1)  # [B, T, 2]

    corner_signs = torch.tensor(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    local_corners = corner_signs[None, None, :, :] * half_sizes[:, :, None, :]
    rot = torch.stack(
        [
            heading_unit[..., 0],
            -heading_unit[..., 1],
            heading_unit[..., 1],
            heading_unit[..., 0],
        ],
        dim=-1,
    ).reshape(B, T, 2, 2)
    rotated_corners = torch.einsum("btij,btkj->btki", rot, local_corners)
    return ego_center_xy[:, :, None, :] + rotated_corners


def compute_ego_edge_points(
    ego_traj: torch.Tensor,
    ego_shape: torch.Tensor,
    n_interp: int,
) -> torch.Tensor:
    """Compute sample points along ego bounding box edges.

    Args:
        ego_traj: [B, T, 4] ego trajectory (x, y, cos_heading, sin_heading).
        ego_shape: [B, 3] (wheelbase, length, width).
        n_interp: number of intermediate points per edge.
            n_interp=0: 4 points (corners only).
            n_interp=1: 8 points (corners + midpoints).

    Returns:
        points: [B, T, 4*(n_interp+1), 2] sampled points.
    """
    corners = compute_ego_bbox_corners(ego_traj, ego_shape)  # [B, T, 4, 2]

    starts = corners  # [B, T, 4, 2]
    ends = torch.roll(corners, -1, dims=2)  # [B, T, 4, 2]

    n_pts = n_interp + 1
    t = torch.linspace(0.0, 1.0, n_pts + 1, device=corners.device)[:-1]
    t = t.reshape(1, 1, 1, n_pts, 1)

    starts = starts.unsqueeze(3)  # [B, T, 4, 1, 2]
    ends = ends.unsqueeze(3)  # [B, T, 4, 1, 2]

    points = starts + t * (ends - starts)  # [B, T, 4, n_pts, 2]
    B, T = points.shape[:2]
    return points.reshape(B, T, 4 * n_pts, 2)


def compute_road_border_penalty(
    ego_edge_points: torch.Tensor,
    line_strings: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Compute road border penalty for ego trajectory.

    Args:
        ego_edge_points: [B, T, K, 2] sample points on ego bbox edges.
        line_strings: [B, N, P, D] denormalized line strings.
        margin: distance threshold (meters).

    Returns:
        penalty: [B, T] non-negative penalty per timestep.
    """
    line_strings_xy = line_strings[..., :2]  # [B, N, P, 2]
    road_border_mask = (line_strings[..., 3] > 0.5).any(dim=-1)  # [B, N]

    B, T, K, _ = ego_edge_points.shape
    device = ego_edge_points.device

    # Pre-filter: keep only line strings that are road border in any batch
    any_rb = road_border_mask.any(dim=0)  # [N]
    if not any_rb.any():
        return torch.zeros(B, T, device=device)

    # Segment endpoints: [B, N, S, 2]
    seg_a = line_strings_xy[:, :, :-1, :]
    seg_b = line_strings_xy[:, :, 1:, :]
    S = seg_a.shape[2]

    # Segment validity: both endpoints non-zero and line string is road border
    seg_valid = (
        (seg_a.abs().sum(-1) > 1e-6) & (seg_b.abs().sum(-1) > 1e-6) & road_border_mask[:, :, None]
    )  # [B, N, S]

    # Pre-filter valid line strings to reduce memory
    valid_ls_indices = any_rb.nonzero(as_tuple=True)[0]  # [M]
    seg_a = seg_a[:, valid_ls_indices]  # [B, M, S, 2]
    seg_b = seg_b[:, valid_ls_indices]  # [B, M, S, 2]
    seg_valid = seg_valid[:, valid_ls_indices]  # [B, M, S]
    M = valid_ls_indices.shape[0]

    # Flatten segments: [B, M*S, 2]
    seg_a_flat = seg_a.reshape(B, M * S, 2)
    seg_b_flat = seg_b.reshape(B, M * S, 2)
    seg_valid_flat = seg_valid.reshape(B, M * S)

    # Compute distances: ego_edge_points [B, T, K, 2] vs segments [B, M*S, 2]
    p = ego_edge_points.reshape(B, T * K, 1, 2)
    a = seg_a_flat[:, None, :, :]  # [B, 1, M*S, 2]
    b = seg_b_flat[:, None, :, :]  # [B, 1, M*S, 2]

    dist = point_to_segment_distance(p, a, b)  # [B, T*K, M*S]

    # Mask invalid segments
    dist = torch.where(
        seg_valid_flat[:, None, :],
        dist,
        torch.full_like(dist, float("inf")),
    )

    # Min over all segments and all edge points per timestep
    min_dist_per_point = dist.min(dim=-1).values  # [B, T*K]
    min_dist_per_point = min_dist_per_point.reshape(B, T, K)
    min_dist = min_dist_per_point.min(dim=-1).values  # [B, T]

    return torch.where(
        torch.isfinite(min_dist),
        F.relu(margin - min_dist),
        torch.zeros_like(min_dist),
    )


def _box_outward_normals(corners: torch.Tensor) -> torch.Tensor:
    """Unit outward edge normals of a convex box. corners: [..., 4, 2] -> [..., 4, 2]."""
    edges = torch.roll(corners, -1, dims=-2) - corners
    normals = torch.stack([-edges[..., 1], edges[..., 0]], dim=-1)
    return normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp_min(1e-9)


def sat_penetration_depth(ego_corners: torch.Tensor, nbr_corners: torch.Tensor) -> torch.Tensor:
    """SAT overlap (penetration) depth between an ego box and neighbor boxes.

    Unlike the point-to-segment distance, this registers *any* polygon overlap (including
    shallow / interpenetrating ones the discrete edge-point sampling misses), returning the
    minimum-translation-distance magnitude.

    Args:
        ego_corners: [B, S, 4, 2] ego box corners per (batch, eval step).
        nbr_corners: [B, S, Pn, 4, 2] neighbor box corners.

    Returns:
        depth: [B, S, Pn] >= 0 (0 when the boxes are disjoint).
    """
    B, S, Pn, _, _ = nbr_corners.shape
    ego = ego_corners[:, :, None].expand(B, S, Pn, 4, 2)
    # candidate separating axes: edge normals of both boxes.
    axes = torch.cat([_box_outward_normals(ego), _box_outward_normals(nbr_corners)], dim=-2)
    pe = torch.einsum("bspax,bspcx->bspac", axes, ego)  # [B, S, Pn, 8, 4]
    pn = torch.einsum("bspax,bspcx->bspac", axes, nbr_corners)  # [B, S, Pn, 8, 4]
    overlap = torch.minimum(pe.amax(-1), pn.amax(-1)) - torch.maximum(pe.amin(-1), pn.amin(-1))
    # disjoint <=> some axis has non-positive overlap -> min over axes <= 0 -> relu = 0.
    return overlap.amin(-1).clamp_min(0.0)  # [B, S, Pn]


def compute_neighbor_collision_penalty(
    ego_edge_points: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbors_future_valid: torch.Tensor,
    neighbor_agents_past: torch.Tensor,
    margin_vehicle: float,
    margin_pedestrian: float,
    margin_bicycle: float,
) -> torch.Tensor:
    """Compute neighbor collision penalty for ego trajectory.

    The neighbor box is inflated per agent type and the penalty is the SAT penetration depth of
    the ego box into that inflated box. This is equivalent to a ``relu(margin - distance)``
    proximity hinge while a buffer remains, smoothly continuing into a true overlap-depth penalty
    once the boxes actually touch -- and, unlike a point-to-segment distance, it does not depend
    on any discrete edge-point sampling (so it never misses shallow overlaps or a small agent
    sitting inside the ego footprint).

    The inflation amount is set independently per agent type (one-hot type in past cols 8..10);
    all three must be specified explicitly by the caller.

    Args:
        ego_edge_points: [B, T, K, 2] sample points on ego bbox edges (only the 4 corners,
            i.e. every ``K // 4``-th point, are used here).
        neighbors_future: [B, Pn, T, 4] neighbor future trajectories in world frame.
        neighbors_future_valid: [B, Pn, T] validity mask for neighbor timesteps.
        neighbor_agents_past: [B, Pn_max, T_past, D] denormalized neighbor past states.
        margin_vehicle: per-side inflation (meters) for vehicles.
        margin_pedestrian: per-side inflation (meters) for pedestrians.
        margin_bicycle: per-side inflation (meters) for bicycles.

    Returns:
        penalty: [B, T] non-negative penalty per timestep.
    """
    B, T_full, K, _ = ego_edge_points.shape
    Pn = neighbors_future.shape[1]
    device = ego_edge_points.device

    steps = torch.tensor(
        [s for s in _NEIGHBOR_EVAL_STEPS if s < T_full], device=device, dtype=torch.long
    )
    S = steps.shape[0]
    if S == 0:
        return torch.zeros(B, T_full, device=device)

    # Drop neighbor slots that are invalid at every eval step across the whole batch.
    # The penalty is an amax over neighbors and invalid neighbors contribute zero, so
    # removing globally-invalid (padding) slots leaves the result unchanged while
    # shrinking the SAT tensors from the padded Pn (e.g. 320) down to the real count.
    neighbor_agents_past = neighbor_agents_past[:, :Pn]
    keep = neighbors_future_valid[:, :, steps].any(dim=2).any(dim=0)  # [Pn]
    if not bool(keep.any()):
        return torch.zeros(B, T_full, device=device)
    neighbors_future = neighbors_future[:, keep]
    neighbors_future_valid = neighbors_future_valid[:, keep]
    neighbor_agents_past = neighbor_agents_past[:, keep]
    Pn = neighbors_future.shape[1]

    # Ego box corners at eval timesteps (corners are every K // 4-th edge sample point).
    ego_corners = ego_edge_points[:, steps][:, :, :: K // 4, :]  # [B, S, 4, 2]

    # Per-neighbor inflation margin selected by the agent type one-hot (cols 8..10).
    neighbor_sizes = neighbor_agents_past[:, :Pn, -1, :]
    margin_by_type = torch.tensor(
        [margin_vehicle, margin_pedestrian, margin_bicycle],
        device=device,
        dtype=neighbor_sizes.dtype,
    )  # ordered [vehicle, pedestrian, bicycle]
    type_idx = neighbor_sizes[..., _TYPE_BASE : _TYPE_BASE + 3].argmax(dim=-1)  # [B, Pn]
    neighbor_margin = margin_by_type[type_idx]  # [B, Pn]

    # Neighbor sizes from last past timestep, inflated by the per-type margin on each side.
    neighbor_width = torch.clamp(neighbor_sizes[..., 6], min=1e-3) + 2.0 * neighbor_margin  # [B, Pn]
    neighbor_length = (
        torch.clamp(neighbor_sizes[..., 7], min=1e-3) + 2.0 * neighbor_margin
    )  # [B, Pn]

    # Neighbor pose at eval timesteps.
    neighbor_pos = neighbors_future[:, :, steps, :2]  # [B, Pn, S, 2]
    neighbor_cos = neighbors_future[:, :, steps, 2]  # [B, Pn, S]
    neighbor_sin = neighbors_future[:, :, steps, 3]  # [B, Pn, S]
    orientation_norm = torch.sqrt(neighbor_cos**2 + neighbor_sin**2).clamp_min(1e-6)
    neighbor_cos = neighbor_cos / orientation_norm
    neighbor_sin = neighbor_sin / orientation_norm

    # Build the inflated neighbor rect: [B, Pn, S, 6] -> corners [B, Pn, S, 4, 2].
    neighbor_rect = torch.stack(
        [
            neighbor_pos[..., 0],
            neighbor_pos[..., 1],
            neighbor_cos,
            neighbor_sin,
            neighbor_length.unsqueeze(-1).expand(-1, -1, S),
            neighbor_width.unsqueeze(-1).expand(-1, -1, S),
        ],
        dim=-1,
    )
    neighbor_corners = center_rect_to_points(neighbor_rect.reshape(-1, 6)).reshape(B, Pn, S, 4, 2)

    # SAT penetration of the ego box into each inflated neighbor box.
    nbr_corners = neighbor_corners.permute(0, 2, 1, 3, 4)  # [B, S, Pn, 4, 2]
    penetration = sat_penetration_depth(ego_corners, nbr_corners)  # [B, S, Pn]
    valid = neighbors_future_valid[:, :, steps].permute(0, 2, 1)  # [B, S, Pn]
    penetration = torch.where(valid, penetration, torch.zeros_like(penetration))
    penalty_s = penetration.amax(dim=-1)  # worst (deepest) overlap per (B, step)

    # Scatter to full T
    penalty = torch.zeros(B, T_full, device=device)
    penalty[:, steps] = penalty_s

    return penalty


def _gather_feature(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    expanded_index = index.unsqueeze(-1).expand(-1, -1, values.size(-1))
    return torch.gather(values, 1, expanded_index)


def _lane_point_clearance(
    points_xy: torch.Tensor,
    center: torch.Tensor,
    n_left: torch.Tensor,
    width_left: torch.Tensor,
    width_right: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate lateral distances from points to the nearest lane boundary.

    Args:
        points_xy: [B, M, 2] points to evaluate.
        center: [B, N, 2] lane center samples.
        n_left: [B, N, 2] left-direction normals.
        width_left: [B, N] distance from center to left boundary along n_left.
        width_right: [B, N] distance from center to right boundary along n_left.
        valid_mask: [B, N] mask of valid lane samples.

    Returns:
        dist_left: [B, M] signed distance to left boundary.
        dist_right: [B, M] signed distance to right boundary.
        valid_point_mask: [B, M] boolean mask indicating valid proximity.
    """
    diff = points_xy.unsqueeze(2) - center.unsqueeze(1)  # [B, M, N, 2]
    dist2 = (diff**2).sum(-1)
    inf = torch.full_like(dist2, float("inf"))
    dist2 = torch.where(valid_mask[:, None, :], dist2, inf)

    min_dist, nearest_idx = torch.min(dist2, dim=-1)
    valid_point_mask = torch.isfinite(min_dist)

    nearest_idx = nearest_idx.clamp_min(0)
    selected_center = _gather_feature(center, nearest_idx)
    selected_n_left = _gather_feature(n_left, nearest_idx)
    selected_width_left = _gather_feature(width_left.unsqueeze(-1), nearest_idx).squeeze(-1)
    selected_width_right = _gather_feature(width_right.unsqueeze(-1), nearest_idx).squeeze(-1)

    lat = ((points_xy - selected_center) * selected_n_left).sum(-1)
    dist_left = selected_width_left - lat
    dist_right = lat - selected_width_right

    return dist_left, dist_right, valid_point_mask


def compute_lane_boundary_penalty(
    ego_edge_points: torch.Tensor,
    route_lanes: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Compute lane boundary clearance penalty for ego edge points.

    Args:
        ego_edge_points: [B, T, K, 2] sample points on ego bbox edges.
        route_lanes: [B, num_segments, num_points, dim] denormalized lane tensor.
        margin: distance threshold (meters).

    Returns:
        penalty: [B, T] non-negative penalty per timestep.
    """
    B, T, K, _ = ego_edge_points.shape

    center = route_lanes[..., :2].reshape(B, -1, 2)
    direction = route_lanes[..., 2:4].reshape(B, -1, 2)
    left_offset = route_lanes[..., 4:6].reshape(B, -1, 2)
    right_offset = route_lanes[..., 6:8].reshape(B, -1, 2)

    direction_norm = torch.linalg.norm(direction, dim=-1, keepdim=True)
    valid_mask = direction_norm.squeeze(-1) > 1e-6
    direction_norm = direction_norm.clamp_min(1e-6)
    n_left = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1) / direction_norm

    width_left = (left_offset * n_left).sum(-1)
    width_right = (right_offset * n_left).sum(-1)

    points_flat = ego_edge_points.reshape(B, T * K, 2)

    dist_left, dist_right, valid_point_mask = _lane_point_clearance(
        points_flat,
        center,
        n_left,
        width_left,
        width_right,
        valid_mask,
    )

    dist_left = dist_left.view(B, T, K)
    dist_right = dist_right.view(B, T, K)
    valid_point_mask = valid_point_mask.view(B, T, K)

    distance_to_boundary = torch.minimum(dist_left, dist_right)
    distance_to_boundary = torch.where(
        valid_point_mask, distance_to_boundary, torch.full_like(distance_to_boundary, float("inf"))
    )

    min_distance = distance_to_boundary.min(dim=-1).values  # [B, T]
    valid_time_mask = valid_point_mask.any(dim=-1)  # [B, T]

    return torch.where(
        valid_time_mask,
        F.relu(margin - min_distance),
        torch.zeros_like(min_distance),
    )
