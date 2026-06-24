"""Replan-consistency metric — the open-loop proxy for SAGE's closed-loop
prefix-cascade failure (SAGE-JEPA plan §4).

SAGE's target pathology ("commit to an unrealistic prefix → the mismatch cascades
under replanning") only manifests in closed loop. Our eval is open-loop single-shot,
so the closest observable signal is: take the prediction made at frame ``t`` and the
prediction made one step-offset ``g`` later at frame ``t+g``; both describe overlapping
wall-clock time. Re-express the first prediction in the second frame's ego coordinates
and measure how far the two disagree on the overlap. A temporally stable planner barely
changes its plan frame-to-frame → small jump; an unstable one jumps.

Pure geometry: the caller supplies the inter-frame ego transform (where the ego at
``t+g`` sits in frame-``t`` coordinates). This keeps the metric independent of how
frames are paired / where the transform comes from (the driver derives it from the
ego GT motion). All trajectories are (N, T, 4) = x, y, cos(yaw), sin(yaw), each in its
own frame's ego coordinates.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator

import numpy as np
import torch

__all__ = [
    "compute_replan_consistency_batch",
    "temporal_consistency_loss",
    "parse_frame_key",
    "group_frames_by_scenario",
    "consecutive_frame_pairs",
    "ego_future_to_4col",
    "inter_frame_transform",
]


def temporal_consistency_loss(
    traj_a: torch.Tensor,
    traj_b: torch.Tensor,
    step_offset: int,
    rel_pos: torch.Tensor,
    rel_heading: torch.Tensor,
    w_heading: float = 1.0,
    stop_grad_a: bool = True,
    per_sample: bool = False,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable cross-frame consistency loss (the training counterpart of
    :func:`compute_replan_consistency_batch`). Aligns frame-t's plan into frame-(t+g)
    and penalises overlap disagreement, so adjacent plans agree (less flicker).

    ``stop_grad_a=True`` treats the earlier-frame plan as a fixed anchor (causal
    "anchor the current plan to recent history"); gradient flows only into ``traj_b``.
    Returns a scalar (mean position jump + ``w_heading`` * mean heading jump).
    """
    N, Ta, _ = traj_a.shape
    Tb = traj_b.shape[1]
    g = int(step_offset)
    L = min(Ta - g, Tb)
    if L <= 0:
        return traj_a.new_zeros(N) if per_sample else traj_a.new_zeros(())

    a = traj_a[:, g : g + L]
    if stop_grad_a:
        a = a.detach()
    b = traj_b[:, :L]

    theta = rel_heading.view(N, 1)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    p = rel_pos.view(N, 1, 2)

    d = a[..., :2] - p
    ax = cos_t * d[..., 0] + sin_t * d[..., 1]
    ay = -sin_t * d[..., 0] + cos_t * d[..., 1]
    a_xy = torch.stack([ax, ay], dim=-1)

    phi_a = torch.atan2(a[..., 3], a[..., 2]) - theta
    phi_b = torch.atan2(b[..., 3], b[..., 2])

    pos = (a_xy - b[..., :2]).norm(dim=-1).mean(dim=1)  # [N] per-sample position jump
    dphi = torch.atan2((phi_a - phi_b).sin(), (phi_a - phi_b).cos()).abs().mean(dim=1)  # [N]
    per = pos + w_heading * dphi  # [N] per-sample consistency loss
    if per_sample:
        return per
    if sample_weight is not None:
        # scene-aware gating: weighted mean (e.g. w=exp(-gt_dev/tau) down-weights scene-change
        # frames so the consistency loss does not force copying where GT genuinely deviates).
        w = sample_weight.reshape(N).to(per.dtype)
        return (per * w).sum() / (w.sum() + 1e-8)
    return per.mean()


def ego_future_to_4col(future: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert an ego trajectory ``(T, 3)`` [x, y, heading_rad] to ``(T, 4)`` [x, y,
    cos, sin]. A ``(T, 4)`` input is returned as-is (already cos/sin). Mirrors the
    planner's ``heading_to_cos_sin`` so the metric sees the same representation.
    """
    t = torch.as_tensor(future, dtype=torch.float32)
    if t.shape[-1] == 4:
        return t
    if t.shape[-1] != 3:
        raise ValueError(f"expected last dim 3 (x,y,heading) or 4 (x,y,cos,sin), got {t.shape}")
    xy = t[..., :2]
    h = t[..., 2]
    return torch.cat([xy, torch.cos(h)[..., None], torch.sin(h)[..., None]], dim=-1)


def inter_frame_transform(
    future_a: torch.Tensor, g: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inter-frame ego transform for a replan pair, from frame-t's GT future.

    The frame at offset ``g`` sits at the ego's true pose ``g`` steps ahead, i.e.
    ``future_a`` step ``g-1`` (0-indexed), expressed in frame-t coordinates. Returns
    ``(rel_pos[...,2], rel_heading[...])`` ready for :func:`compute_replan_consistency_batch`.
    Validated on real consecutive frames: GT-vs-GT jump is exactly 0 at the true g.
    """
    pose = future_a[..., g - 1, :]  # [..., 4]
    rel_pos = pose[..., :2]
    rel_heading = torch.atan2(pose[..., 3], pose[..., 2])
    return rel_pos, rel_heading

# Frame identity lives only in the path (the NPZ payload is ego-centric, no scenario id
# / timestamp). Two filename conventions seen:
#   1-field: {session}/{session}_{frame}.npz                 -> group = session dir
#   2-field: {session}/{session}_{scene}_{frame}.npz         -> group = session dir # scene
# In both, ordering is the LAST integer token (frame index). For the 2-field converter
# (basic_dataset) the frame index steps by a fixed cadence and the offset between two
# consecutive saved frames equals g trajectory steps (verified: ΔB=3 -> g=3, GT-vs-GT
# replan jump = 0 exactly at g=3).


def parse_frame_key(path: str) -> tuple[str, int]:
    """Return ``(group_key, frame_index)`` for a per-frame NPZ path.

    group_key isolates a single contiguous ego timeline: the session directory, plus
    the scene field when the filename carries one (so different scenes in the same
    session are never paired). frame_index is the last integer token in the basename.
    """
    base = os.path.basename(path)
    name = base[:-4] if base.endswith(".npz") else base
    # trailing run of purely-integer underscore tokens (handles 1- and 2-field names)
    int_toks: list[str] = []
    for tok in reversed(name.split("_")):
        if tok.isdigit():
            int_toks.append(tok)
        else:
            break
    int_toks.reverse()
    if not int_toks:
        raise ValueError(f"cannot parse a trailing frame index from {path!r}")
    frame = int(int_toks[-1])
    scene = int_toks[-2] if len(int_toks) >= 2 else None
    group = os.path.dirname(path) if scene is None else f"{os.path.dirname(path)}#{scene}"
    return group, frame


def group_frames_by_scenario(paths: list[str]) -> dict[str, list[tuple[int, str]]]:
    """Group per-frame NPZ paths by scenario timeline, each sorted by frame index.

    Returns ``{group_key: [(frame_index, path), ...sorted ascending]}`` where group_key
    is from :func:`parse_frame_key` (session dir, plus scene for the 2-field convention).
    """
    groups: dict[str, list[tuple[int, str]]] = {}
    for p in paths:
        group, frame = parse_frame_key(p)
        groups.setdefault(group, []).append((frame, p))
    for group in groups:
        groups[group].sort(key=lambda t: t[0])
    return groups


def consecutive_frame_pairs(
    paths: list[str],
    expected_gap: int | None = None,
) -> Iterator[tuple[int, str, int, str, int]]:
    """Yield modal-step-adjacent frame pairs within each scenario timeline.

    Saved frames step by a fixed cadence with gaps between sub-runs (filtered frames).
    Only pairs whose frame_gap equals the timeline's sampling step are yielded, so the
    gap-straddling pairs (which are NOT temporally adjacent) are skipped.

    Each item is ``(idx_a, path_a, idx_b, path_b, frame_gap)``; frame_gap == g (the
    trajectory-step offset between the two frames). ``expected_gap`` overrides the
    auto-detected per-timeline modal step. Scenarios are emitted in sorted order.
    """
    groups = group_frames_by_scenario(paths)
    for group in sorted(groups):
        frames = groups[group]
        if len(frames) < 2:
            continue
        gaps = [frames[i + 1][0] - frames[i][0] for i in range(len(frames) - 1)]
        step = expected_gap if expected_gap is not None else Counter(gaps).most_common(1)[0][0]
        for (idx_a, path_a), (idx_b, path_b) in zip(frames[:-1], frames[1:]):
            if idx_b - idx_a == step:
                yield (idx_a, path_a, idx_b, path_b, idx_b - idx_a)


@torch.no_grad()
def compute_replan_consistency_batch(
    traj_a: torch.Tensor,
    traj_b: torch.Tensor,
    step_offset: int,
    rel_pos: torch.Tensor,
    rel_heading: torch.Tensor,
) -> dict:
    """Overlap jump between consecutive-frame predictions.

    Args:
        traj_a: (N, Ta, 4) prediction at frame t, in frame-t ego coordinates.
        traj_b: (N, Tb, 4) prediction at frame t+g, in frame-(t+g) ego coordinates.
        step_offset: g — number of trajectory timesteps between the two frames
            (frame t+g corresponds to traj_a's step g).
        rel_pos: (N, 2) position of the frame-(t+g) ego origin expressed in frame-t
            coordinates (i.e. where the ego moved to over the g steps).
        rel_heading: (N,) heading angle (radians) of frame-(t+g) expressed in frame-t.

    Returns:
        dict with:
          position_jump: (N,) mean ||·|| position disagreement over the overlap (metres).
          heading_jump:  (N,) mean |Δheading| over the overlap (radians, wrapped).
          overlap_len:   int L = min(Ta - g, Tb); 0 if the frames don't overlap.
        Both jumps are non-negative; lower = more temporally stable. When
        overlap_len == 0 the jumps are zeros (caller should skip such pairs).
    """
    N, Ta, _ = traj_a.shape
    Tb = traj_b.shape[1]
    g = int(step_offset)
    device = traj_a.device

    L = min(Ta - g, Tb)
    if L <= 0:
        z = torch.zeros(N, device=device)
        return {"position_jump": z, "heading_jump": z.clone(), "overlap_len": 0}

    a = traj_a[:, g : g + L]  # (N, L, 4) — the part of tau_a overlapping tau_b
    b = traj_b[:, :L]  # (N, L, 4)

    theta = rel_heading.view(N, 1)  # (N, 1)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    p = rel_pos.view(N, 1, 2)  # (N, 1, 2)

    # Re-express tau_a positions in frame-(t+g): q' = R(-theta) (q - p)
    d = a[..., :2] - p  # (N, L, 2)
    ax = cos_t * d[..., 0] + sin_t * d[..., 1]  # (N, L)
    ay = -sin_t * d[..., 0] + cos_t * d[..., 1]  # (N, L)
    a_xy = torch.stack([ax, ay], dim=-1)  # (N, L, 2)

    # Headings of tau_a in frame-(t+g): phi_a - theta
    phi_a = torch.atan2(a[..., 3], a[..., 2]) - theta  # (N, L)
    phi_b = torch.atan2(b[..., 3], b[..., 2])  # (N, L)

    position_jump = (a_xy - b[..., :2]).norm(dim=-1).mean(dim=-1)  # (N,)
    dphi = phi_a - phi_b
    dphi = torch.atan2(dphi.sin(), dphi.cos()).abs()  # wrap to (-π, π], then |·|
    heading_jump = dphi.mean(dim=-1)  # (N,)

    return {
        "position_jump": position_jump,
        "heading_jump": heading_jump,
        "overlap_len": L,
    }
