"""Frenet frame trajectory perturbation for structured exploration.

Implements PlannerRFT's (arxiv 2601.12901) lateral/longitudinal
decomposition: given a reference trajectory, apply offsets in the Frenet
frame (arc-length + signed lateral deviation) and convert back to
Cartesian coordinates.

The perturbed trajectory is used as the denoising initial condition (xT)
in the diffusion sampler, NOT as a classifier guidance energy. This is
the correct PlannerRFT approach — a future learned PPO exploration
policy will output (lateral_offset, longitudinal_offset) per scene
instead of the current random sampling.

Coordinate convention:
    - Longitudinal (s): arc-length along the reference path.
      Positive Δs = ahead on the path (faster). Negative = behind (slower).
    - Lateral (d): signed perpendicular distance from the reference path.
      Positive Δd = left of path (following right-hand rule with heading).
      Negative = right of path.
"""

import numpy as np
import torch


def _compute_arc_lengths(ref_xy: torch.Tensor) -> torch.Tensor:
    """Compute cumulative arc-length along a 2D path.

    Args:
        ref_xy: [B, T, 2] reference positions.

    Returns:
        [B, T] cumulative arc-length. First element is 0.
    """
    diffs = ref_xy[:, 1:, :] - ref_xy[:, :-1, :]  # [B, T-1, 2]
    seg_lengths = diffs.norm(dim=-1)  # [B, T-1]
    zeros = torch.zeros(ref_xy.shape[0], 1, device=ref_xy.device)
    return torch.cat([zeros, seg_lengths.cumsum(dim=-1)], dim=-1)  # [B, T]


def _compute_heading(ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract or compute unit heading vectors from reference trajectory.

    Uses cos/sin channels if available (dim>=4), otherwise finite-differences.

    Args:
        ref: [B, T, D] reference trajectory. D>=4 means (x, y, cos, sin).

    Returns:
        (tangent, normal) each [B, T, 2].
        tangent = unit heading direction.
        normal  = left-perpendicular (-sin, cos).
    """
    if ref.shape[-1] >= 4:
        cos_h = ref[..., 2]
        sin_h = ref[..., 3]
    else:
        dx = ref[:, 1:, 0] - ref[:, :-1, 0]
        dy = ref[:, 1:, 1] - ref[:, :-1, 1]
        # Pad last element by repeating
        dx = torch.cat([dx, dx[:, -1:]], dim=-1)
        dy = torch.cat([dy, dy[:, -1:]], dim=-1)
        norm = (dx ** 2 + dy ** 2).sqrt().clamp_min(1e-6)
        cos_h = dx / norm
        sin_h = dy / norm

    # Normalize to handle denormalized cos/sin
    h_norm = (cos_h ** 2 + sin_h ** 2).sqrt().clamp_min(1e-6)
    cos_h = cos_h / h_norm
    sin_h = sin_h / h_norm

    tangent = torch.stack([cos_h, sin_h], dim=-1)  # [B, T, 2]
    normal = torch.stack([-sin_h, cos_h], dim=-1)   # [B, T, 2]
    return tangent, normal


def cartesian_to_frenet(
    ref: torch.Tensor,
    points_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project Cartesian points onto a reference path in Frenet coordinates.

    For each point, finds the nearest reference waypoint and computes
    (s, d) where s = arc-length at that waypoint, d = signed lateral
    offset (positive = left of path).

    Args:
        ref: [B, T_ref, D] reference trajectory (D >= 2, uses first 2 as xy).
        points_xy: [B, T_pts, 2] points to project.

    Returns:
        (s, d) each [B, T_pts].
    """
    ref_xy = ref[..., :2]  # [B, T_ref, 2]
    arc_lengths = _compute_arc_lengths(ref_xy)  # [B, T_ref]
    _, normal = _compute_heading(ref)  # [B, T_ref, 2]

    # Find nearest ref point for each query point
    # points_xy: [B, T_pts, 2], ref_xy: [B, T_ref, 2]
    dists = torch.cdist(points_xy, ref_xy)  # [B, T_pts, T_ref]
    nearest_idx = dists.argmin(dim=-1)  # [B, T_pts]

    B, T_pts = points_xy.shape[:2]

    # Gather nearest reference data
    idx_expanded = nearest_idx.unsqueeze(-1).expand(-1, -1, 2)
    nearest_xy = ref_xy.gather(1, idx_expanded)  # [B, T_pts, 2]
    nearest_normal = normal.gather(1, idx_expanded)  # [B, T_pts, 2]
    s = arc_lengths.gather(1, nearest_idx)  # [B, T_pts]

    # Signed lateral offset: dot(point - nearest, normal)
    delta = points_xy - nearest_xy  # [B, T_pts, 2]
    d = (delta * nearest_normal).sum(dim=-1)  # [B, T_pts]

    return s, d


def frenet_to_cartesian(
    ref: torch.Tensor,
    s: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Convert Frenet (s, d) coordinates back to Cartesian using a reference path.

    For each (s_i, d_i), interpolates the reference path at arc-length s_i
    and offsets laterally by d_i.

    Args:
        ref: [B, T_ref, D] reference trajectory (D >= 2).
        s: [B, T_pts] arc-length values.
        d: [B, T_pts] signed lateral offsets (positive = left).

    Returns:
        [B, T_pts, 2] Cartesian (x, y) positions.
    """
    ref_xy = ref[..., :2]
    arc_lengths = _compute_arc_lengths(ref_xy)  # [B, T_ref]
    _, normal = _compute_heading(ref)  # [B, T_ref, 2]

    B, T_ref = arc_lengths.shape
    T_pts = s.shape[1]

    # Clamp s to valid range
    s_max = arc_lengths[:, -1:]  # [B, 1]
    s_clamped = torch.clamp(s, min=torch.zeros_like(s), max=s_max.expand_as(s))

    # For each s value, find the segment it falls in and interpolate
    # arc_lengths: [B, T_ref], s_clamped: [B, T_pts]
    # Find index such that arc_lengths[idx] <= s < arc_lengths[idx+1]
    s_expanded = s_clamped.unsqueeze(-1)  # [B, T_pts, 1]
    al_expanded = arc_lengths.unsqueeze(1)  # [B, 1, T_ref]

    # Number of arc-length values <= s gives the segment index
    idx = (al_expanded <= s_expanded).sum(dim=-1) - 1  # [B, T_pts]
    idx = idx.clamp(0, T_ref - 2)
    idx_next = idx + 1

    # Gather segment endpoints
    def gather2d(tensor, indices):
        """Gather along dim=1 for [B, T, 2] tensor."""
        return tensor.gather(1, indices.unsqueeze(-1).expand(-1, -1, 2))

    def gather1d(tensor, indices):
        """Gather along dim=1 for [B, T] tensor."""
        return tensor.gather(1, indices)

    p0 = gather2d(ref_xy, idx)      # [B, T_pts, 2]
    p1 = gather2d(ref_xy, idx_next)  # [B, T_pts, 2]
    n0 = gather2d(normal, idx)       # [B, T_pts, 2]
    n1 = gather2d(normal, idx_next)  # [B, T_pts, 2]
    s0 = gather1d(arc_lengths, idx)  # [B, T_pts]
    s1 = gather1d(arc_lengths, idx_next)  # [B, T_pts]

    # Interpolation factor
    seg_len = (s1 - s0).clamp_min(1e-6)
    alpha = ((s_clamped - s0) / seg_len).clamp(0.0, 1.0)  # [B, T_pts]
    alpha = alpha.unsqueeze(-1)  # [B, T_pts, 1]

    # Interpolated centerline point and normal
    center = p0 + alpha * (p1 - p0)  # [B, T_pts, 2]
    n_interp = n0 + alpha * (n1 - n0)  # [B, T_pts, 2]
    n_interp = n_interp / n_interp.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # Apply lateral offset
    xy = center + d.unsqueeze(-1) * n_interp  # [B, T_pts, 2]
    return xy


def perturb_trajectory(
    ref: torch.Tensor,
    lateral_offset: float = 0.0,
    longitudinal_offset: float = 0.0,
) -> torch.Tensor:
    """Apply Frenet frame perturbation to a reference trajectory.

    This is the core PlannerRFT exploration primitive: given a reference
    (typically the deterministic model output), create a perturbed
    version by shifting in the Frenet frame.

    The heading at each point is interpolated from the reference heading
    at the new arc-length position, preserving path curvature.

    Args:
        ref: [B, T, 4] reference trajectory (x, y, cos_yaw, sin_yaw)
             in physical ego-centric metres.
        lateral_offset: Metres to offset perpendicular to path.
            Positive = left. Negative = right.
        longitudinal_offset: Metres to shift along path.
            Positive = ahead (faster). Negative = behind (slower).

    Returns:
        [B, T, 4] perturbed trajectory (x, y, cos_yaw, sin_yaw).
        Heading is interpolated from reference at the new arc-length.
    """
    ref_xy = ref[..., :2]  # [B, T, 2]
    arc_lengths = _compute_arc_lengths(ref_xy)  # [B, T]
    tangent, normal = _compute_heading(ref)  # each [B, T, 2]

    # Shift arc-lengths (longitudinal)
    s_new = arc_lengths + longitudinal_offset  # [B, T]

    # Convert back to Cartesian with lateral offset
    xy_new = frenet_to_cartesian(ref, s_new, torch.full_like(s_new, lateral_offset))

    # Interpolate heading at new arc-length positions
    B, T = s_new.shape
    T_ref = arc_lengths.shape[1]
    s_max = arc_lengths[:, -1:]
    s_clamped = torch.clamp(s_new, min=torch.zeros_like(s_new), max=s_max.expand_as(s_new))

    # Find segment indices for heading interpolation
    al_expanded = arc_lengths.unsqueeze(1)  # [B, 1, T_ref]
    s_expanded = s_clamped.unsqueeze(-1)    # [B, T, 1]
    idx = (al_expanded <= s_expanded).sum(dim=-1) - 1
    idx = idx.clamp(0, T_ref - 2)
    idx_next = idx + 1

    s0 = arc_lengths.gather(1, idx)
    s1 = arc_lengths.gather(1, idx_next)
    seg_len = (s1 - s0).clamp_min(1e-6)
    alpha = ((s_clamped - s0) / seg_len).clamp(0.0, 1.0)

    if ref.shape[-1] >= 4:
        cos_ref = ref[..., 2]  # [B, T_ref]
        sin_ref = ref[..., 3]
        cos0 = cos_ref.gather(1, idx)
        cos1 = cos_ref.gather(1, idx_next)
        sin0 = sin_ref.gather(1, idx)
        sin1 = sin_ref.gather(1, idx_next)
        cos_new = cos0 + alpha * (cos1 - cos0)
        sin_new = sin0 + alpha * (sin1 - sin0)
        # Renormalize
        h_norm = (cos_new ** 2 + sin_new ** 2).sqrt().clamp_min(1e-6)
        cos_new = cos_new / h_norm
        sin_new = sin_new / h_norm
    else:
        # Compute heading from new positions
        dx = xy_new[:, 1:, 0] - xy_new[:, :-1, 0]
        dy = xy_new[:, 1:, 1] - xy_new[:, :-1, 1]
        dx = torch.cat([dx, dx[:, -1:]], dim=-1)
        dy = torch.cat([dy, dy[:, -1:]], dim=-1)
        norm = (dx ** 2 + dy ** 2).sqrt().clamp_min(1e-6)
        cos_new = dx / norm
        sin_new = dy / norm

    return torch.stack([xy_new[..., 0], xy_new[..., 1], cos_new, sin_new], dim=-1)
