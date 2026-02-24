"""Lane keeping guidance for the diffusion planner.

Penalises ego trajectory points where the vehicle footprint protrudes beyond
the boundaries of the nearest lane segment.  The vehicle width is read from
``inputs["ego_shape"]`` so that the guidance respects the actual vehicle
dimensions rather than a hardcoded constant.

Lane segment data layout (SEGMENT_POINT_DIM = 33, indices from dimensions.py):
  0-1 : centerline X, Y
  2-3 : direction  dX, dY
  4-5 : left boundary  LB_X, LB_Y
  6-7 : right boundary RB_X, RB_Y
"""

import torch
import torch.nn.functional as F

# Indices into the lane feature vector (must match dimensions.py)
_X, _Y = 0, 1
_DX, _DY = 2, 3
_LBX, _LBY = 4, 5
_RBX, _RBY = 6, 7

# Only penalise violations that are above this threshold (metres).
# Setting to 0 means any protrusion is penalised; a small positive value
# provides a soft margin before the penalty kicks in.
_MARGIN = 0.0

# Ignore the nearest lane if it is farther than this distance from the
# ego position (handles sparse/empty lane data gracefully).
_MAX_LANE_DIST = 30.0

# Energy scale factor (tuned to be comparable with collision guidance ~300).
_SCALE = 100.0


def lane_keeping_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """Compute lane-keeping guidance energy.

    Finds the nearest lane segment point for each future ego position and
    computes how far the vehicle extends beyond the left and right lane
    boundaries.  Violations accumulate as a negative reward (lower = worse).

    Args:
        x: [B, P, T+1, 4] denormalized trajectory (x, y, cos_h, sin_h).
           guidance_wrapper.py applies state_normalizer.inverse() before
           calling this function, so positions are in base_link frame metres.
        t: [B] diffusion timestep in [0, 1].
        cond: unused, kept for interface compatibility.
        inputs: observation dict (already inverse-normalised by the wrapper).
            Required keys:
                ``lanes``      – [B, N_seg, N_pts, SEGMENT_POINT_DIM]
                ``ego_shape``  – [B, 3] (wheel_base, length, width) in metres

    Returns:
        [B] energy tensor.  Gradient flows through x via autograd when
        t ∈ (0.005, 0.1) (same active window as collision guidance).
    """
    B, P, T_plus1, _ = x.shape
    T = T_plus1 - 1

    # Allow gradients to flow only in the middle of the denoising schedule.
    mask = (t < 0.1) * (t > 0.005)
    mask = mask.view(B, 1, 1, 1)
    x = torch.where(mask, x, x.detach())

    # Future ego positions only (index 0 is the pinned current state).
    ego_pos = x[:, 0, 1:, :2]  # [B, T, 2]

    # ------------------------------------------------------------------ #
    # Lane geometry                                                        #
    # ------------------------------------------------------------------ #
    lanes = inputs["lanes"]  # [B, N_seg, N_pts, 33]
    N = lanes.shape[1] * lanes.shape[2]  # total lane points

    lane_centers = lanes[..., _X:_Y + 1].reshape(B, N, 2)   # [B, N, 2]
    lane_dirs    = lanes[..., _DX:_DY + 1].reshape(B, N, 2)  # [B, N, 2]
    lane_left    = lanes[..., _LBX:_LBY + 1].reshape(B, N, 2)  # [B, N, 2]
    lane_right   = lanes[..., _RBX:_RBY + 1].reshape(B, N, 2)  # [B, N, 2]

    # Lateral direction = unit vector perpendicular to lane, pointing left.
    lane_dirs_n = lane_dirs / (lane_dirs.norm(dim=-1, keepdim=True) + 1e-6)
    lane_lat = torch.stack([-lane_dirs_n[..., 1], lane_dirs_n[..., 0]], dim=-1)  # [B, N, 2]

    # Mark invalid lane points (all-zero after inverse normalisation).
    lane_valid = (lane_left.norm(dim=-1) + lane_right.norm(dim=-1)) > 1e-3  # [B, N]

    # ------------------------------------------------------------------ #
    # Find nearest valid lane point for each ego timestep                 #
    # ------------------------------------------------------------------ #
    # dist: [B, T, N]
    dist = (ego_pos.unsqueeze(2) - lane_centers.unsqueeze(1)).norm(dim=-1)
    dist = dist.masked_fill(~lane_valid.unsqueeze(1).expand(-1, T, -1), 1e6)
    nearest = dist.argmin(dim=-1)        # [B, T]
    min_dist = dist.min(dim=-1).values   # [B, T]

    # Helper: gather a [B, N, 2] tensor at per-timestep indices → [B, T, 2].
    def gather2(tensor):
        idx = nearest.unsqueeze(-1).expand(-1, -1, 2)           # [B, T, 2]
        return tensor.unsqueeze(1).expand(-1, T, -1, -1) \
                     .gather(2, idx.unsqueeze(2)).squeeze(2)    # [B, T, 2]

    c   = gather2(lane_centers)   # [B, T, 2]  nearest centerline point
    lat = gather2(lane_lat)       # [B, T, 2]  lateral unit vector (pointing left)
    lb  = gather2(lane_left)      # [B, T, 2]  left boundary point
    rb  = gather2(lane_right)     # [B, T, 2]  right boundary point

    # ------------------------------------------------------------------ #
    # Signed lateral offsets (positive = to the left of centreline)       #
    # ------------------------------------------------------------------ #
    # Ego centre lateral offset from nearest centreline point.
    ego_lat = ((ego_pos - c) * lat).sum(dim=-1)           # [B, T]

    # Lane half-widths measured along the lateral direction.
    left_hw  = ((lb - c) * lat).sum(dim=-1)               # [B, T]  > 0
    right_hw = ((rb - c) * lat).sum(dim=-1)               # [B, T]  < 0

    # Vehicle half-width from ego_shape (wheel_base=0, length=1, width=2).
    half_w = inputs["ego_shape"][:, 2:3] / 2              # [B, 1]

    # ------------------------------------------------------------------ #
    # Violations                                                           #
    # left  violation: ego left  edge (ego_lat + half_w) past left  boundary
    # right violation: ego right edge (ego_lat - half_w) past right boundary
    # ------------------------------------------------------------------ #
    viol_left  = F.relu(ego_lat + half_w - left_hw  + _MARGIN)   # [B, T]
    viol_right = F.relu(right_hw - ego_lat + half_w + _MARGIN)   # [B, T]

    # Zero out contributions from timesteps where no valid lane is nearby.
    no_lane = min_dist > _MAX_LANE_DIST                           # [B, T]
    viol_left  = viol_left.masked_fill(no_lane, 0.0)
    viol_right = viol_right.masked_fill(no_lane, 0.0)

    reward = -(viol_left + viol_right).sum(dim=-1)  # [B]

    return _SCALE * reward
