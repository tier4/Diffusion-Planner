"""Diagnose lane-departure detections: recreate model trajectories on flagged scenes
and visualize per-step outer-boundary clearance + where ego first reaches
lane-departure status.

For each flagged scene, we:
  1. Run deterministic inference (same recipe as analyze_problem_areas).
  2. Replicate ``compute_lane_departure_penalty`` internals per timestep using
     the ego footprint perimeter, classified outer boundary segments, and
     signed clearance to those segments.
  3. Compute the per-step min-over-perimeter clearance (what the gate uses) and
     label each timestep against the configured thresholds
     (CROSS / NEAR / WIDE / SAFE).
  4. Identify the first CROSS timestep + the perimeter point responsible.
  5. Produce a per-scene PNG with an overview + zoomed-in view at the crossing
     step + a per-step clearance timeline annotated with the configured
     cross/near/wide thresholds.
  6. Produce a per-scene JSON with crossing step, worst clearance, crossed
     side, predicted vs GT path length, GT LD status (to catch ghost LDs).

Usage:
    python -m rlvr.autoresearch.tools.viz_ld_diagnostic \
        --model_path <base.pth> \
        --lora_path <lora_dir> \
        --scenes <ld_scenes.json> \
        --output_dir <dir> \
        [--config <grpo_config.json>] \
        [--limit N]

Pass ``--config`` to align the diagnostic thresholds / weights with the
training run; without it, RewardConfig defaults are used.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as MplPolygon

from preference_optimization.model_utils import load_model
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.closed_loop.batched_rollout import _batched_generate
from rlvr.reward import (
    _LANE_K_NEAREST,
    _LANE_PTS_PER_SIDE,
    RewardConfig,
    _build_lane_polygons,
    _classify_outer_boundaries,
    _point_to_outer_clearance,
    compute_reward_batch,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SceneDiag:
    scene_path: str
    ego_shape: tuple[float, float, float]
    pred_traj: np.ndarray            # (T, 4) x, y, cos, sin
    gt_traj: np.ndarray              # (T, 3) x, y, heading
    pred_world_pts: np.ndarray       # (T, K, 2) perimeter points
    pred_pt_clearance: np.ndarray    # (T, K) signed on-road clearance per perimeter pt
    pred_min_clearance: np.ndarray   # (T,) min over perimeter — what the gate uses
    pred_crossing_step: int | None
    pred_crossing_point: np.ndarray | None       # (2,) world coord of worst perimeter pt at crossing step
    pred_closest_boundary_pt: np.ndarray | None  # (2,) foot on the outer segment
    outer_p1: np.ndarray
    outer_p2: np.ndarray
    outer_normal: np.ndarray         # (E, 2) outward normals
    lane_polygons_xy: list[np.ndarray]
    lb_lines: list[np.ndarray]
    rb_lines: list[np.ndarray]
    border_polylines: list[np.ndarray]
    gt_crossing_step: int | None
    gt_min_clearance: np.ndarray     # (T,) GT clearance — used to flag ghost LDs
    pred_path_length: float
    gt_path_length: float
    lane_cross_thresh: float
    lane_near_thresh: float
    lane_wide_thresh: float
    reward_lane_crossing: bool       # as reported by compute_reward_batch


# ---------------------------------------------------------------------------
# Core LD replication
# ---------------------------------------------------------------------------

def _build_ego_perimeter_local(ego_shape: torch.Tensor, device: torch.device, dtype: torch.dtype):
    """Build local-frame perimeter points identical to compute_lane_departure_penalty."""
    wb = ego_shape[0].item()
    length = ego_shape[1].item()
    width = ego_shape[2].item()
    ro = (length - wb) / 2
    lp_list = []
    for j in range(_LANE_PTS_PER_SIDE):
        f = j / (_LANE_PTS_PER_SIDE - 1)
        lp_list.append((-ro + f * length, -width / 2))
        lp_list.append((-ro + f * length,  width / 2))
        if 0 < f < 1:
            lp_list.append((-ro, -width / 2 + f * width))
            lp_list.append((length - ro, -width / 2 + f * width))
    return torch.tensor(lp_list, device=device, dtype=dtype), ro, length, width


def _select_k_nearest_lanes(lanes: torch.Tensor, traj_xy: torch.Tensor,
                            k: int = _LANE_K_NEAREST) -> torch.Tensor:
    """Same K-nearest selection as compute_lane_departure_penalty."""
    S = lanes.shape[0]
    if k <= 0 or S <= k:
        return lanes
    center_all = lanes[..., :2]
    valid_all = center_all.norm(dim=-1) > 1e-3
    traj_min = traj_xy.min(dim=0).values
    traj_max = traj_xy.max(dim=0).values
    traj_center = (traj_min + traj_max) / 2
    dist_to_pts = (center_all - traj_center).norm(dim=-1)
    dist_to_pts[~valid_all] = 1e6
    min_dist_per_lane = dist_to_pts.min(dim=1).values
    half_diag = (traj_max - traj_min).norm() / 2 + 5.0
    has_lane = valid_all.any(dim=1)
    min_dist_per_lane[~has_lane] = 1e6
    n_nearby = (min_dist_per_lane < half_diag).sum().item()
    k_eff = max(k, min(n_nearby, S))
    _, topk_idx = min_dist_per_lane.topk(k_eff, largest=False)
    return lanes[topk_idx]


def _run_ld_check(traj: torch.Tensor, ego_shape: torch.Tensor,
                  data: dict[str, torch.Tensor],
                  config: RewardConfig | None = None) -> dict:
    """Mirror compute_lane_departure_penalty internals for visualization.

    Returns per-point clearance (signed on-road distance), outer segments +
    their outward normals, and lane-polygon vertex loops used for the
    background fill. The crossing step is the first t>0 where any perimeter
    point has clearance ≤ lane_cross_thresh.

    traj: (T, 4) — x, y, cos, sin (single-scene).
    """
    if config is None:
        config = RewardConfig()
    device = traj.device
    T = traj.shape[0]

    lanes = data["lanes"]
    if lanes.dim() == 4:
        lanes = lanes[0]

    # K nearest lane selection (matches reward.py)
    traj_xy = traj[:, :2]
    lanes = _select_k_nearest_lanes(lanes, traj_xy)

    edge_v1, edge_v2, edge_poly_id, n_polys = _build_lane_polygons(lanes)

    # Build outer boundary segments
    center = lanes[..., :2]
    direction = lanes[..., 2:4]
    lb_offset = lanes[..., 4:6]
    rb_offset = lanes[..., 6:8]
    valid = center.norm(dim=-1) > 1e-3

    left_pts = center + lb_offset
    right_pts = center + rb_offset

    dirs = direction.clone()
    has_dir = dirs.norm(dim=-1) > 1e-6
    dir_sum = (dirs * has_dir.unsqueeze(-1)).sum(dim=1)
    dir_avg = dir_sum / dir_sum.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dirs = torch.where(has_dir.unsqueeze(-1), dirs, dir_avg.unsqueeze(1).expand_as(dirs))

    valid_pair = valid[:, :-1] & valid[:, 1:]
    mid_dirs = (dirs[:, :-1] + dirs[:, 1:]) / 2
    mid_dirs = mid_dirs / mid_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    S, P, _ = lanes.shape
    lane_ids = torch.arange(S, device=device).unsqueeze(1).expand(S, P - 1)

    vp_flat = valid_pair.reshape(-1)
    idx = torch.where(vp_flat)[0]

    lb_lines = []
    rb_lines = []
    for s in range(S):
        v = valid[s]
        if v.sum() > 1:
            lb_lines.append(left_pts[s, v].cpu().numpy())
            rb_lines.append(right_pts[s, v].cpu().numpy())

    lane_polygons_xy = []
    for s in range(S):
        v = valid[s]
        if v.sum() < 2:
            continue
        lp = left_pts[s, v].cpu().numpy()
        rp = right_pts[s, v].cpu().numpy()
        poly = np.concatenate([lp, rp[::-1]], axis=0)
        lane_polygons_xy.append(poly)

    # Build perimeter once (used for empty-edge case too, so we can still emit world_pts)
    local_pts, _, _, _ = _build_ego_perimeter_local(ego_shape, device, traj.dtype)
    K = local_pts.shape[0]
    cos_h = traj[..., 2]; sin_h = traj[..., 3]
    h_norm = (cos_h ** 2 + sin_h ** 2).sqrt().clamp_min(1e-6)
    cos_h = cos_h / h_norm; sin_h = sin_h / h_norm
    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=-1).reshape(T, 2, 2)
    rotated = torch.einsum("tij,kj->tki", rot, local_pts)
    world_pts = traj[..., :2].unsqueeze(1) + rotated  # (T, K, 2)

    if len(idx) == 0 or n_polys == 0:
        return {
            "world_pts": world_pts,
            "pt_clearance": torch.full((T, K), 100.0, device=device, dtype=traj.dtype),
            "outer_p1": torch.zeros(0, 2, device=device),
            "outer_p2": torch.zeros(0, 2, device=device),
            "outer_normal": torch.zeros(0, 2, device=device),
            "lane_polygons_xy": lane_polygons_xy,
            "lb_lines": lb_lines,
            "rb_lines": rb_lines,
        }

    # Reproduce the reward.py segment construction with explicit outward normals.
    M = len(idx)
    l_p1 = left_pts[:, :-1].reshape(-1, 2)[idx]
    l_p2 = left_pts[:, 1:].reshape(-1, 2)[idx]
    r_p1 = right_pts[:, :-1].reshape(-1, 2)[idx]
    r_p2 = right_pts[:, 1:].reshape(-1, 2)[idx]
    md_f = mid_dirs.reshape(-1, 2)[idx]
    lid_f = lane_ids.reshape(-1)[idx]

    left_perp = torch.stack([-md_f[:, 1], md_f[:, 0]], dim=-1)
    right_perp = -left_perp

    seg_p1 = torch.cat([l_p1, r_p1], dim=0)
    seg_p2 = torch.cat([l_p2, r_p2], dim=0)
    seg_dir = torch.cat([md_f, md_f], dim=0)
    seg_lane = torch.cat([lid_f, lid_f], dim=0)
    seg_outward = torch.cat([left_perp, right_perp], dim=0)

    interleave = torch.stack([torch.arange(M, device=device),
                              torch.arange(M, 2 * M, device=device)], dim=1).reshape(-1)
    is_outer_inter = _classify_outer_boundaries(
        seg_p1[interleave], seg_p2[interleave], seg_dir[interleave], seg_lane[interleave],
        edge_v1, edge_v2, edge_poly_id, n_polys,
    )
    is_outer = torch.empty_like(is_outer_inter)
    is_outer[interleave] = is_outer_inter

    outer_p1 = seg_p1[is_outer]
    outer_p2 = seg_p2[is_outer]
    outer_normal = seg_outward[is_outer]

    Q = T * K
    query = world_pts.reshape(Q, 2)
    if outer_p1.shape[0] > 0:
        pt_clearance = _point_to_outer_clearance(query, outer_p1, outer_p2, outer_normal).reshape(T, K)
    else:
        pt_clearance = torch.full((T, K), 100.0, device=device, dtype=traj.dtype)

    return {
        "world_pts": world_pts,
        "pt_clearance": pt_clearance,
        "outer_p1": outer_p1,
        "outer_p2": outer_p2,
        "outer_normal": outer_normal,
        "lane_polygons_xy": lane_polygons_xy,
        "lb_lines": lb_lines,
        "rb_lines": rb_lines,
    }


def _build_border_polylines(data: dict[str, torch.Tensor]) -> list[np.ndarray]:
    ls = data.get("line_strings")
    if ls is None:
        return []
    if ls.dim() == 4:
        ls = ls[0]
    if ls.shape[-1] < 4:
        return []
    ls = ls.cpu().numpy()
    out = []
    for j in range(ls.shape[0]):
        pts = ls[j]
        v = (pts[:, 3] > 0.5) & (np.abs(pts[:, :2]).sum(axis=-1) > 0.01)
        if v.sum() > 1:
            out.append(pts[v, :2])
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def diagnose_scenes(model, model_args, scene_paths: list[str],
                    batch_size: int = 16,
                    eval_config: RewardConfig | None = None) -> list[SceneDiag]:
    results: list[SceneDiag] = []
    if eval_config is None:
        eval_config = RewardConfig(enable_lane_departure=True)
    normalizer = copy.deepcopy(model_args.observation_normalizer)

    for chunk_start in range(0, len(scene_paths), batch_size):
        chunk_paths = scene_paths[chunk_start:chunk_start + batch_size]
        chunk_data = []
        for sp in chunk_paths:
            try:
                chunk_data.append(load_npz_data(sp, DEVICE))
            except Exception as e:
                print(f"  Skip {Path(sp).name}: {e}")
                chunk_data.append(None)

        valid_indices = [i for i, d in enumerate(chunk_data) if d is not None]
        if not valid_indices:
            continue
        valid_data = [chunk_data[i] for i in valid_indices]

        batch = {}
        for k in valid_data[0]:
            vals = [d[k] for d in valid_data]
            if isinstance(vals[0], torch.Tensor):
                batch[k] = torch.cat(vals, dim=0)
            else:
                batch[k] = vals[0]

        norm_batch = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                      for k, v in batch.items()}
        norm_batch = normalizer(norm_batch)

        det_trajs = _batched_generate(
            model, model_args, norm_batch,
            noise_scale=0.0, composer=None, device=DEVICE,
        )

        for local_i, _ in enumerate(valid_indices):
            sp = chunk_paths[valid_indices[local_i]]
            data_i = valid_data[local_i]

            # Ego shape
            es = data_i.get("ego_shape")
            ego_shape = es[0] if es is not None and es.dim() > 1 else es
            if ego_shape is None:
                ego_shape = torch.tensor([2.75, 4.34, 1.70], device=DEVICE)

            pred = det_trajs[local_i]  # (T, 4) x, y, cos, sin
            # LD check on prediction
            diag_pred = _run_ld_check(pred, ego_shape, data_i)

            # LD check on GT too (ghost LD detection)
            with np.load(sp) as raw:
                gt_raw = raw["ego_agent_future"].copy()
            gt_t = torch.tensor(gt_raw, device=DEVICE, dtype=pred.dtype)
            gt_cos = torch.cos(gt_t[:, 2])
            gt_sin = torch.sin(gt_t[:, 2])
            gt_traj4 = torch.stack([gt_t[:, 0], gt_t[:, 1], gt_cos, gt_sin], dim=-1)
            diag_gt = _run_ld_check(gt_traj4, ego_shape, data_i)

            # Per-step min clearance across the ego perimeter. Skip t=0.
            lane_cross_thresh = eval_config.lane_cross_thresh
            pred_min_clearance = diag_pred["pt_clearance"].min(dim=1).values
            pred_min_clearance[0] = 10.0
            pred_crossing_mask = pred_min_clearance <= lane_cross_thresh
            pred_crossing_mask[0] = False
            if pred_crossing_mask.any():
                pred_cross_step = int(pred_crossing_mask.float().argmax().item())
                # Worst perimeter point at the crossing step = min clearance.
                cross_pt_idx = int(diag_pred["pt_clearance"][pred_cross_step].argmin().item())
            else:
                pred_cross_step = None
                cross_pt_idx = None

            gt_min_clearance = diag_gt["pt_clearance"].min(dim=1).values
            gt_min_clearance[0] = 10.0
            gt_crossing_mask = gt_min_clearance <= lane_cross_thresh
            gt_crossing_mask[0] = False
            gt_cross_step = int(gt_crossing_mask.float().argmax().item()) if gt_crossing_mask.any() else None

            outer_p1 = diag_pred["outer_p1"].cpu().numpy()
            outer_p2 = diag_pred["outer_p2"].cpu().numpy()
            outer_normal = diag_pred["outer_normal"].cpu().numpy()
            world_pts_np = diag_pred["world_pts"].cpu().numpy()

            # For the arrow: foot of the worst perimeter point on its closest outer segment.
            if pred_cross_step is not None and outer_p1.shape[0] > 0:
                cross_world = world_pts_np[pred_cross_step, cross_pt_idx]
                seg_vec = outer_p2 - outer_p1
                seg_len2 = np.clip((seg_vec ** 2).sum(-1), 1e-10, None)
                diff = cross_world[None] - outer_p1
                t_param = ((diff * seg_vec).sum(-1) / seg_len2).clip(0, 1)
                closest = outer_p1 + t_param[:, None] * seg_vec
                d_sq = ((closest - cross_world[None]) ** 2).sum(-1)
                j = int(d_sq.argmin())
                closest_boundary_pt = closest[j]
                cross_world_pt = cross_world
            else:
                cross_world_pt = None
                closest_boundary_pt = None

            pred_xy = pred[:, :2].cpu().numpy()
            path_pred = float(np.linalg.norm(np.diff(pred_xy, axis=0), axis=1).sum())
            gt_xy = gt_raw[:, :2]
            path_gt = float(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1).sum())

            # Sanity: rerun compute_reward_batch for "official" flag
            r = compute_reward_batch(pred.unsqueeze(0), data_i, eval_config)[0]

            results.append(SceneDiag(
                scene_path=sp,
                ego_shape=tuple(float(x) for x in ego_shape.cpu().numpy()),
                pred_traj=pred.cpu().numpy(),
                gt_traj=gt_raw,
                pred_world_pts=world_pts_np,
                pred_pt_clearance=diag_pred["pt_clearance"].cpu().numpy(),
                pred_min_clearance=pred_min_clearance.cpu().numpy(),
                pred_crossing_step=pred_cross_step,
                pred_crossing_point=cross_world_pt,
                pred_closest_boundary_pt=closest_boundary_pt,
                outer_p1=outer_p1,
                outer_p2=outer_p2,
                outer_normal=outer_normal,
                lane_polygons_xy=diag_pred["lane_polygons_xy"],
                lb_lines=diag_pred["lb_lines"],
                rb_lines=diag_pred["rb_lines"],
                border_polylines=_build_border_polylines(data_i),
                gt_crossing_step=gt_cross_step,
                gt_min_clearance=gt_min_clearance.cpu().numpy(),
                pred_path_length=path_pred,
                gt_path_length=path_gt,
                lane_cross_thresh=lane_cross_thresh,
                lane_near_thresh=eval_config.lane_near_thresh,
                lane_wide_thresh=eval_config.lane_wide_thresh,
                reward_lane_crossing=bool(r.lane_crossing),
            ))

        print(f"  Processed {chunk_start + len(chunk_paths)}/{len(scene_paths)} scenes")
    return results


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_lanes_and_borders(ax, d: SceneDiag):
    for poly in d.lane_polygons_xy:
        ax.fill(poly[:, 0], poly[:, 1], color="lightgreen", alpha=0.18, zorder=1)
    for ln in d.lb_lines:
        ax.plot(ln[:, 0], ln[:, 1], "b-", linewidth=0.6, alpha=0.35, zorder=4)
    for ln in d.rb_lines:
        ax.plot(ln[:, 0], ln[:, 1], "b-", linewidth=0.6, alpha=0.35, zorder=4)
    for bp in d.border_polylines:
        ax.plot(bp[:, 0], bp[:, 1], "r-", linewidth=2.5, alpha=0.7, zorder=5)


def _draw_outer_segments(ax, d: SceneDiag):
    for i in range(d.outer_p1.shape[0]):
        ax.plot(
            [d.outer_p1[i, 0], d.outer_p2[i, 0]],
            [d.outer_p1[i, 1], d.outer_p2[i, 1]],
            color="darkorange", linewidth=1.0, alpha=0.6, zorder=6,
        )


def _draw_trajectories(ax, d: SceneDiag):
    ax.plot(d.gt_traj[:, 0], d.gt_traj[:, 1], "g--", linewidth=2.0,
            label="GT", zorder=7)
    ax.plot(d.pred_traj[:, 0], d.pred_traj[:, 1], "-", color="royalblue",
            linewidth=2.0, label="pred", zorder=7)


def _ego_box_corners(xy, heading, length, width, wb):
    ro = (length - wb) / 2
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    hw = width / 2
    return np.array([
        [xy[0] + (length - ro) * cos_h - hw * sin_h, xy[1] + (length - ro) * sin_h + hw * cos_h],
        [xy[0] + (length - ro) * cos_h + hw * sin_h, xy[1] + (length - ro) * sin_h - hw * cos_h],
        [xy[0] - ro * cos_h + hw * sin_h,            xy[1] - ro * sin_h - hw * cos_h],
        [xy[0] - ro * cos_h - hw * sin_h,            xy[1] - ro * sin_h + hw * cos_h],
    ])


def _draw_ego_box(ax, d: SceneDiag, step: int, color="darkred", lw=2.0):
    pred = d.pred_traj
    cos_h, sin_h = pred[step, 2], pred[step, 3]
    heading = float(np.arctan2(sin_h, cos_h))
    xy = pred[step, :2]
    wb, length, width = d.ego_shape
    corners = _ego_box_corners(xy, heading, length, width, wb)
    ax.add_patch(MplPolygon(corners, closed=True, fill=False,
                            edgecolor=color, linewidth=lw, zorder=8))


def _draw_perimeter_points(ax, d: SceneDiag, step: int, size: int = 30):
    """Color each perimeter point by its configured on-road clearance band.

    Thresholds come from SceneDiag (populated from RewardConfig) so the
    diagnostic stays aligned with the live lane-departure gate:

    clearance <= lane_cross_thresh → red X (within the lane-cross gate)
    clearance <= lane_near_thresh  → orange circle (near outer edge)
    clearance >  lane_near_thresh  → green circle (clear)
    """
    wp = d.pred_world_pts[step]            # (K, 2)
    clr = d.pred_pt_clearance[step]        # (K,)
    cross_thresh = d.lane_cross_thresh
    near_thresh = d.lane_near_thresh
    for k in range(wp.shape[0]):
        c = float(clr[k])
        if c <= cross_thresh:
            col, mk, sz = "red", "X", size * 2
        elif c <= near_thresh:
            col, mk, sz = "orange", "o", size
        else:
            col, mk, sz = "green", "o", size
        ax.scatter(wp[k, 0], wp[k, 1], c=col, s=sz, marker=mk,
                   edgecolors="black", linewidths=0.4, zorder=9)


def _auto_bbox(d: SceneDiag):
    xs = np.concatenate([d.gt_traj[:, 0], d.pred_traj[:, 0]])
    ys = np.concatenate([d.gt_traj[:, 1], d.pred_traj[:, 1]])
    xc = (xs.min() + xs.max()) / 2
    yc = (ys.min() + ys.max()) / 2
    span = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 8
    return xc, yc, span


def draw_scene(d: SceneDiag, out_path: str):
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 1.2], height_ratios=[1, 0.5])
    ax_over = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_time = fig.add_subplot(gs[:, 2])
    ax_leg = fig.add_subplot(gs[1, :2])
    ax_leg.axis("off")

    # Overview
    _draw_lanes_and_borders(ax_over, d)
    _draw_outer_segments(ax_over, d)
    _draw_trajectories(ax_over, d)
    xc, yc, span = _auto_bbox(d)
    ax_over.set_xlim(xc - span, xc + span)
    ax_over.set_ylim(yc - span, yc + span)
    ax_over.set_aspect("equal")
    ax_over.grid(True, alpha=0.2)

    cross_step = d.pred_crossing_step
    if cross_step is not None:
        _draw_ego_box(ax_over, d, cross_step, color="darkred", lw=2.0)
        _draw_ego_box(ax_over, d, 0, color="dimgray", lw=1.2)
        _draw_ego_box(ax_over, d, 79, color="steelblue", lw=1.2)
        if d.pred_crossing_point is not None and d.pred_closest_boundary_pt is not None:
            p = d.pred_crossing_point
            q = d.pred_closest_boundary_pt
            ax_over.plot([p[0], q[0]], [p[1], q[1]], "k-", linewidth=2, zorder=10)
            ax_over.plot(p[0], p[1], "kX", markersize=10, zorder=10)
    ax_over.set_title(
        f"Overview | pred LD step={cross_step} | GT LD step={d.gt_crossing_step} | "
        f"path pred={d.pred_path_length:.1f}m gt={d.gt_path_length:.1f}m",
        fontsize=10,
    )

    # Zoomed ROI at crossing
    _draw_lanes_and_borders(ax_zoom, d)
    _draw_outer_segments(ax_zoom, d)
    _draw_trajectories(ax_zoom, d)
    if cross_step is not None:
        _draw_ego_box(ax_zoom, d, cross_step, color="darkred", lw=2.5)
        _draw_perimeter_points(ax_zoom, d, cross_step, size=40)
        if d.pred_crossing_point is not None and d.pred_closest_boundary_pt is not None:
            p = d.pred_crossing_point
            q = d.pred_closest_boundary_pt
            ax_zoom.plot([p[0], q[0]], [p[1], q[1]], "k-", linewidth=2.5, zorder=10)
            ax_zoom.plot(p[0], p[1], "kX", markersize=14, zorder=10)
            clearance = float(d.pred_min_clearance[cross_step])
            ax_zoom.annotate(
                f"step {cross_step}\n"
                f"min clearance: {clearance:+.3f}m\n"
                f"(≤ thresh {d.lane_cross_thresh:.2f}m)",
                xy=(p[0], p[1]), xytext=(p[0] + 1.2, p[1] + 1.2),
                fontsize=11, fontweight="bold",
                bbox=dict(facecolor="yellow", alpha=0.9),
                arrowprops=dict(arrowstyle="->", lw=2), zorder=11,
            )
        cx, cy = d.pred_traj[cross_step, :2]
        ax_zoom.set_xlim(cx - 5, cx + 5)
        ax_zoom.set_ylim(cy - 5, cy + 5)
    else:
        ax_zoom.set_xlim(xc - span, xc + span)
        ax_zoom.set_ylim(yc - span, yc + span)
    ax_zoom.set_aspect("equal")
    ax_zoom.grid(True, alpha=0.2)
    ax_zoom.set_title("Zoomed ROI at crossing (+/- 5m)", fontsize=10)

    # Per-step min on-road clearance (positive = inside road, negative = past edge)
    T = len(d.pred_min_clearance)
    ts = np.arange(T)
    ax_time.plot(ts, d.pred_min_clearance, "-o", color="royalblue", markersize=3,
                 label="min clearance (pred)")
    cross_mask = d.pred_min_clearance <= d.lane_cross_thresh
    cross_mask[0] = False
    if cross_mask.any():
        ax_time.scatter(ts[cross_mask], d.pred_min_clearance[cross_mask],
                        color="red", s=60, marker="X", zorder=5,
                        label="timestep crossing (≤ thresh)")
    # Also overlay GT clearance for reference
    ax_time.plot(ts, d.gt_min_clearance, "--", color="forestgreen", alpha=0.6,
                 label="min clearance (GT)")
    ax_time.axhline(0.0, color="black", linewidth=0.8, label="road edge (clearance=0)")
    ax_time.axhline(d.lane_cross_thresh, color="red", linestyle=":", linewidth=1.2,
                    label=f"lane_cross_thresh {d.lane_cross_thresh:.2f}m")
    ax_time.axhline(d.lane_near_thresh, color="orange", linestyle="--", linewidth=1,
                    label=f"lane_near_thresh {d.lane_near_thresh:.2f}m")
    ax_time.axhline(d.lane_wide_thresh, color="gold", linestyle="--", linewidth=1,
                    label=f"lane_wide_thresh {d.lane_wide_thresh:.2f}m")
    if cross_step is not None:
        ax_time.axvline(cross_step, color="red", linewidth=1, alpha=0.6,
                        label=f"first LD @ step {cross_step}")
    ax_time.set_xlabel("timestep (0..79, each 0.1s)")
    ax_time.set_ylabel("min clearance to road edge (m)")
    ax_time.set_title("Per-step min on-road clearance (pred vs GT)")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(fontsize=8, loc="upper right")
    # Clip for display: the 100.0 sentinel ("no outer segment nearby → unconstrained")
    # crushes the useful range. Clip to [-1, 2] for readability.
    ax_time.set_ylim(-1.0, 2.0)

    # Legend in bottom left
    legend_handles = [
        Patch(facecolor="lightgreen", alpha=0.3, label="lane polygons (on-road reference shading)"),
        plt.Line2D([0], [0], color="blue", lw=0.8, alpha=0.5, label="lane centerline boundaries"),
        plt.Line2D([0], [0], color="darkorange", lw=1.2, alpha=0.8, label="outer boundary segments (classified)"),
        plt.Line2D([0], [0], color="red", lw=2, label="road border polyline"),
        plt.Line2D([0], [0], color="g", ls="--", lw=2, label="GT ego trajectory"),
        plt.Line2D([0], [0], color="royalblue", lw=2, label="predicted ego trajectory"),
        plt.Line2D([0], [0], marker="s", lw=0, markerfacecolor="none",
                   markeredgecolor="darkred", markersize=12, label="ego box @ first LD step"),
        plt.Line2D([0], [0], marker="s", lw=0, markerfacecolor="none",
                   markeredgecolor="dimgray", markersize=12, label="ego box @ step 0 (start)"),
        plt.Line2D([0], [0], marker="o", lw=0, markerfacecolor="green",
                   markeredgecolor="black", markersize=8,
                   label="perimeter pt clearance > lane_near_thresh"),
        plt.Line2D([0], [0], marker="o", lw=0, markerfacecolor="orange",
                   markeredgecolor="black", markersize=8,
                   label="perimeter pt lane_cross_thresh < clearance <= lane_near_thresh"),
        plt.Line2D([0], [0], marker="X", lw=0, markerfacecolor="red",
                   markeredgecolor="black", markersize=10,
                   label="perimeter pt clearance <= lane_cross_thresh"),
    ]
    ax_leg.legend(handles=legend_handles, ncol=2, fontsize=10, loc="center")

    scene_name = Path(d.scene_path).stem
    bag = Path(d.scene_path).parent.name
    fig.suptitle(
        f"{bag}/{scene_name} | ego={d.ego_shape[1]:.2f}×{d.ego_shape[2]:.2f}m | "
        f"reward says LD={d.reward_lane_crossing}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def draw_summary(results: list[SceneDiag], out_path: str):
    """Aggregate diagnostics across all scenes."""
    cross_steps = [r.pred_crossing_step for r in results if r.pred_crossing_step is not None]
    worst_clearances = []
    for r in results:
        if r.pred_crossing_step is not None:
            worst_clearances.append(float(r.pred_min_clearance[r.pred_crossing_step]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(cross_steps, bins=40, range=(0, 80), color="steelblue", edgecolor="black")
    axes[0].set_xlabel("first LD step (0..79)")
    axes[0].set_ylabel("scenes")
    axes[0].set_title(f"first LD step distribution — {len(cross_steps)}/{len(results)} scenes flagged")
    axes[0].axvline(np.median(cross_steps) if cross_steps else 0, color="red",
                    linestyle="--", label=f"median={np.median(cross_steps):.0f}" if cross_steps else "")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(worst_clearances, bins=40, color="coral", edgecolor="black")
    axes[1].set_xlabel("min clearance AT first LD step (m)  — negative = past the edge")
    axes[1].set_ylabel("scenes")
    axes[1].set_title("how far past / close to the edge at crossing?")
    axes[1].axvline(0, color="black", linewidth=1, label="road edge")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # GT LD overlap — scenes where GT itself departs
    gt_also_ld = sum(1 for r in results if r.gt_crossing_step is not None)
    both = sum(1 for r in results if r.gt_crossing_step is not None and r.pred_crossing_step is not None)
    pred_only = sum(1 for r in results if r.gt_crossing_step is None and r.pred_crossing_step is not None)
    clean_gt_clean_pred = sum(1 for r in results
                              if r.gt_crossing_step is None and r.pred_crossing_step is None)
    labels = ["GT LD\n(ghost)", "pred-only\nLD", "both LD", "clean"]
    counts = [gt_also_ld - both, pred_only, both, clean_gt_clean_pred]
    axes[2].bar(labels, counts, color=["orange", "red", "purple", "gray"], edgecolor="black")
    for i, c in enumerate(counts):
        axes[2].text(i, c + 0.5, str(c), ha="center", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("scenes")
    axes[2].set_title(f"LD source split — {gt_also_ld} scenes have GT LD too ({gt_also_ld}/{len(results)})")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("LD detection diagnostic summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--lora_path", type=Path, default=None)
    p.add_argument("--scenes", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--config", type=Path, default=None,
                   help="GRPO training config JSON. When given, reward thresholds "
                        "and weights come from here so the diagnostic matches the "
                        "live run (enable_lane_departure is always forced on).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    scenes_dir = os.path.join(out_dir, "per_scene")
    os.makedirs(scenes_dir, exist_ok=True)

    with open(args.scenes) as f:
        scene_paths = json.load(f)
    if args.limit:
        scene_paths = scene_paths[:args.limit]
    print(f"Diagnosing {len(scene_paths)} scenes")

    model, model_args = load_model(args.model_path, device=DEVICE)
    if args.lora_path is not None:
        from preference_optimization.lora_utils import load_lora_checkpoint
        model = load_lora_checkpoint(model, args.lora_path)
    model.eval()

    if args.config is not None:
        eval_config = load_reward_config(args.config)
        eval_config.enable_lane_departure = True  # always on for this diagnostic
        source = f"from {args.config}"
    else:
        eval_config = RewardConfig(enable_lane_departure=True)
        source = "RewardConfig defaults"
    print(
        f"Reward thresholds ({source}): "
        f"lane_cross={eval_config.lane_cross_thresh:.2f}m, "
        f"lane_near={eval_config.lane_near_thresh:.2f}m, "
        f"lane_wide={eval_config.lane_wide_thresh:.2f}m"
    )

    results = diagnose_scenes(
        model, model_args, scene_paths,
        batch_size=args.batch_size, eval_config=eval_config,
    )

    # Per-scene PNGs + JSON
    summary_rows = []
    for i, d in enumerate(results):
        name = f"{i:03d}_{Path(d.scene_path).parent.name}_{Path(d.scene_path).stem}"
        png_path = os.path.join(scenes_dir, f"{name}.png")
        draw_scene(d, png_path)

        if d.pred_crossing_step is not None:
            worst_clearance = float(d.pred_min_clearance[d.pred_crossing_step])
        else:
            worst_clearance = float(d.pred_min_clearance.min())

        summary_rows.append({
            "scene": d.scene_path,
            "pred_crossing_step": d.pred_crossing_step,
            "gt_crossing_step": d.gt_crossing_step,
            "reward_lane_crossing": d.reward_lane_crossing,
            "clearance_at_cross": worst_clearance,
            "pred_min_clearance_overall": float(d.pred_min_clearance.min()),
            "pred_path_length": d.pred_path_length,
            "gt_path_length": d.gt_path_length,
            "ego_shape_lw": [d.ego_shape[1], d.ego_shape[2]],
        })
        if (i + 1) % 5 == 0:
            print(f"  rendered {i + 1}/{len(results)} PNGs")

    with open(os.path.join(out_dir, "ld_diagnostic.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    draw_summary(results, os.path.join(out_dir, "ld_diagnostic_summary.png"))

    n_flagged = sum(1 for r in results if r.pred_crossing_step is not None)
    n_gt_also = sum(1 for r in results if r.gt_crossing_step is not None)
    n_reward_said_ld = sum(1 for r in results if r.reward_lane_crossing)
    print()
    print(f"=== DIAGNOSTIC DONE ({len(results)} scenes) ===")
    print(f"  pred LD (our replication):        {n_flagged}/{len(results)}")
    print(f"  reward_lane_crossing (official):  {n_reward_said_ld}/{len(results)}")
    print(f"  GT LD (ghost LDs):                {n_gt_also}/{len(results)}")
    print(f"  outputs in: {out_dir}")


if __name__ == "__main__":
    main()
