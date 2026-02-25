"""Centerline following guidance for the diffusion planner.

Pulls the ego trajectory toward the nearest route-lane centerline by
penalising lateral deviation continuously (quadratic cost), unlike
lane_keeping which only fires when the vehicle protrudes beyond the boundary.

Uses ``route_lanes`` (25 segments along the planned route) instead of the
full ``lanes`` set so the reference is always the intended lane rather than
the geometrically nearest lane which may belong to an adjacent or oncoming
lane.

The gradient grows linearly with lateral offset, so the correction is
stronger the further the vehicle is from the center.
"""

import torch
import torch.nn.functional as F

_X, _Y = 0, 1
_DX, _DY = 2, 3

# Ignore lane points farther than this from the ego position.
_MAX_LANE_DIST = 30.0

# Scale factor: gradient of ego_lat^2 w.r.t. normalised trajectory is
# ~2 * lateral_offset * position_std ≈ 2 * 1m * 20 = 40 at 1m deviation.
# _SCALE = 0.1 keeps the DPM-Solver correction at a reasonable magnitude;
# increase with the guidance_scale UI slider for stronger centerline pull.
_SCALE = 0.1


def centerline_following_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """Compute centerline-following guidance energy.

    Penalises squared lateral distance from the nearest lane centerline,
    continuously attracting the trajectory toward the center of the lane.

    Args:
        x: [B, P, T+1, 4] denormalized trajectory (x, y, cos_h, sin_h).
        t: [B] diffusion timestep in [0, 1].
        cond: unused.
        inputs: observation dict (already inverse-normalised by the wrapper).
            Required keys:
                ``route_lanes`` – [B, N_seg=25, N_pts=20, SEGMENT_POINT_DIM]

    Returns:
        [B] energy tensor (higher = closer to centerline).
    """
    B, P, T_plus1, _ = x.shape
    T = T_plus1 - 1

    mask = (t < 0.1) * (t > 0.005)
    mask = mask.view(B, 1, 1, 1)
    x = torch.where(mask, x, x.detach())

    ego_pos = x[:, 0, 1:, :2]  # [B, T, 2]

    # Use route_lanes (planned route only) so the reference is always the
    # intended lane, not the geometrically nearest lane in the full map.
    lanes = inputs["route_lanes"]  # [B, 25, 20, 33]
    N = lanes.shape[1] * lanes.shape[2]

    lane_centers = lanes[..., _X:_Y + 1].reshape(B, N, 2)  # [B, N, 2]
    lane_dirs    = lanes[..., _DX:_DY + 1].reshape(B, N, 2) # [B, N, 2]

    lane_dirs_n = lane_dirs / (lane_dirs.norm(dim=-1, keepdim=True) + 1e-6)
    lane_lat = torch.stack([-lane_dirs_n[..., 1], lane_dirs_n[..., 0]], dim=-1)  # [B, N, 2]

    # Mark invalid lane points.
    lane_valid = lane_centers.norm(dim=-1) > 1e-3  # [B, N]

    # Nearest valid lane centerline point for each ego timestep.
    dist = (ego_pos.unsqueeze(2) - lane_centers.unsqueeze(1)).norm(dim=-1)  # [B, T, N]
    dist = dist.masked_fill(~lane_valid.unsqueeze(1).expand(-1, T, -1), 1e6)
    nearest   = dist.argmin(dim=-1)   # [B, T]
    min_dist  = dist.min(dim=-1).values  # [B, T]

    def gather2(tensor):
        idx = nearest.unsqueeze(-1).expand(-1, -1, 2)
        return tensor.unsqueeze(1).expand(-1, T, -1, -1) \
                     .gather(2, idx.unsqueeze(2)).squeeze(2)

    c   = gather2(lane_centers)  # [B, T, 2]
    lat = gather2(lane_lat)      # [B, T, 2]

    # Signed lateral offset from centerline (positive = left of center).
    ego_lat = ((ego_pos - c) * lat).sum(dim=-1)  # [B, T]

    # Zero out timesteps where no valid lane is nearby.
    no_lane = min_dist > _MAX_LANE_DIST
    ego_lat = ego_lat.masked_fill(no_lane, 0.0)

    # Quadratic penalty: grows with lateral deviation.
    reward = -(ego_lat ** 2).sum(dim=-1)  # [B]

    return _SCALE * reward
