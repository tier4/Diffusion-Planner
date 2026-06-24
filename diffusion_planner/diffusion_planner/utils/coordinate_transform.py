"""Coordinate transform utilities for neighbor local-frame conversion.

These functions extract the inline transform logic from ``_build_gt_representation``
(encoder/GT side) and ``_denoised_to_trajectory`` (decoder/inference side) so that
they can be tested and reused independently.

All tensors use the 4-channel convention ``(x, y, cos_heading, sin_heading)``.
"""

import torch


def transform_to_local_frame(
    history_4d: torch.Tensor,
    future_4d: torch.Tensor,
    *,
    preserve_invalid: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform trajectory from ego-centric frame to neighbor-local frame.

    The local frame is centred on the last history position with the heading
    aligned to the x-axis (heading = 0 => cos=1, sin=0).

    Args:
        history_4d: [..., T_hist, 4]  (x, y, cos, sin) in ego frame.
        future_4d:  [..., T, 4]  (x, y, cos, sin) in ego frame.
        preserve_invalid: if True, timesteps that are all-zero in *future_4d*
            remain all-zero after transformation (padding convention).

    Returns:
        history_local: [..., T_hist, 4] in local frame.
        future_local:  [..., T, 4] in local frame.
    """
    # Reference pose: last history timestep
    ref_pos = history_4d[..., -1:, :2]  # [..., 1, 2]
    ref_cos = history_4d[..., -1:, 2:3]  # [..., 1, 1]
    ref_sin = history_4d[..., -1:, 3:4]  # [..., 1, 1]

    history_local = _inverse_transform(history_4d, ref_pos, ref_cos, ref_sin)

    # Detect invalid (all-zero) future timesteps BEFORE transforming
    if preserve_invalid:
        invalid_mask = future_4d.ne(0).sum(dim=-1, keepdim=True) == 0  # [..., T, 1]

    future_local = _inverse_transform(future_4d, ref_pos, ref_cos, ref_sin)

    if preserve_invalid:
        future_local = future_local.masked_fill(invalid_mask.expand_as(future_local), 0.0)

    return history_local, future_local


def transform_to_ego_frame(
    traj_local: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_cos: torch.Tensor,
    ref_sin: torch.Tensor,
) -> torch.Tensor:
    """Transform trajectory from neighbor-local frame back to ego-centric frame.

    This is the forward rotation (inverse of :func:`transform_to_local_frame`).

    Args:
        traj_local: [..., T, 4] trajectory in local frame.
        ref_pos: [..., 2] or [..., 1, 2] reference position.
        ref_cos: [..., 1] or [..., 1, 1] cos(heading) of reference.
        ref_sin: [..., 1] or [..., 1, 1] sin(heading) of reference.

    Returns:
        traj_ego: [..., T, 4] trajectory in ego-centric frame.
    """
    # Ensure broadcastable shapes: [..., 1, *]
    if ref_pos.dim() < traj_local.dim():
        ref_pos = ref_pos.unsqueeze(-2)
    if ref_cos.dim() < traj_local.dim():
        ref_cos = ref_cos.unsqueeze(-2)
    if ref_sin.dim() < traj_local.dim():
        ref_sin = ref_sin.unsqueeze(-2)

    local_x = traj_local[..., 0:1]
    local_y = traj_local[..., 1:2]
    # Forward rotation: R(theta) * [x; y]
    rot_x = local_x * ref_cos - local_y * ref_sin
    rot_y = local_x * ref_sin + local_y * ref_cos

    local_cos = traj_local[..., 2:3]
    local_sin = traj_local[..., 3:4]
    rot_cos = local_cos * ref_cos - local_sin * ref_sin
    rot_sin = local_cos * ref_sin + local_sin * ref_cos

    return torch.cat(
        [
            rot_x + ref_pos[..., 0:1],
            rot_y + ref_pos[..., 1:2],
            rot_cos,
            rot_sin,
        ],
        dim=-1,
    )


# ---- internal helpers -------------------------------------------------------


def _inverse_transform(
    traj_4d: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_cos: torch.Tensor,
    ref_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply inverse rotation + translation (ego→local).

    R^{-1}(theta) = R(-theta) = [[cos, sin], [-sin, cos]]
    """
    xy = traj_4d[..., :2] - ref_pos  # translate
    x = xy[..., 0:1] * ref_cos + xy[..., 1:2] * ref_sin
    y = -xy[..., 0:1] * ref_sin + xy[..., 1:2] * ref_cos

    cos_h = traj_4d[..., 2:3] * ref_cos + traj_4d[..., 3:4] * ref_sin
    sin_h = -traj_4d[..., 2:3] * ref_sin + traj_4d[..., 3:4] * ref_cos

    return torch.cat([x, y, cos_h, sin_h], dim=-1)
