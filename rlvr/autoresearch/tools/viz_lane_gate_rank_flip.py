"""Visualize lane boundary segment classification (outer / shared / junction gap).

For each input scene, show lane polygons + road borders + all lane boundary
segments colored by how the nudge classifier in _classify_outer_boundaries
labels them:

  - OUTER (road edge): black solid — the nudged midpoint fell outside every
    lane polygon AND was NOT close to another lane's boundary.
  - SHARED (between lanes): blue solid — the nudged midpoint landed inside
    some lane polygon. Means there's another lane on that side.
  - JUNCTION GAP: bright orange dashed — the nudged midpoint fell outside all
    polygons but was within 0.5m of a DIFFERENT lane's boundary. Typical at
    intersection mouths where lane polygons don't tile cleanly. These are
    reclassified as "shared" (excluded from outer) to prevent false road-edge
    detections.

Usage:
    python -m rlvr.autoresearch.tools.viz_lane_gate_rank_flip \
        --scenes /media/.../j6_train_mixed75.json \
        --n_scenes 10 \
        --output_dir /media/.../viz_junction_gaps
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from preference_optimization.utils import load_npz_data
from rlvr.reward import (
    _build_lane_polygons,
    _point_in_polygons,
    _point_to_segments_dist,
)


def classify_all_segments(lanes: torch.Tensor, nudge: float = 0.05, gap_threshold: float = 0.5):
    """Return (seg_p1, seg_p2, is_outer, is_junction_gap, is_shared) on a scene.

    Mirrors _classify_outer_boundaries but returns all three categories.
    """
    device = lanes.device
    edge_v1, edge_v2, edge_poly_id, n_polys = _build_lane_polygons(lanes)

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
    dir_count = has_dir.sum(dim=1, keepdim=True).clamp(min=1)
    dir_avg = dir_sum / dir_count
    dir_avg = dir_avg / dir_avg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dirs = torch.where(has_dir.unsqueeze(-1), dirs, dir_avg.unsqueeze(1).expand_as(dirs))

    valid_pair = valid[:, :-1] & valid[:, 1:]
    mid_dirs = (dirs[:, :-1] + dirs[:, 1:]) / 2
    mid_dirs = mid_dirs / mid_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    S, P = center.shape[:2]
    lane_ids = torch.arange(S, device=device).unsqueeze(1).expand(S, P - 1)

    vp_flat = valid_pair.reshape(-1)
    idx_keep = torch.where(vp_flat)[0]
    if len(idx_keep) == 0:
        empty = torch.zeros((0,), dtype=torch.bool, device=device)
        return (torch.zeros((0, 2), device=device), torch.zeros((0, 2), device=device),
                empty, empty, empty)

    l_p1 = left_pts[:, :-1].reshape(-1, 2)[idx_keep]
    l_p2 = left_pts[:, 1:].reshape(-1, 2)[idx_keep]
    r_p1 = right_pts[:, :-1].reshape(-1, 2)[idx_keep]
    r_p2 = right_pts[:, 1:].reshape(-1, 2)[idx_keep]
    md_f = mid_dirs.reshape(-1, 2)[idx_keep]
    lid_f = lane_ids.reshape(-1)[idx_keep]
    M = len(idx_keep)

    seg_p1 = torch.stack([l_p1, r_p1], dim=1).reshape(2 * M, 2)
    seg_p2 = torch.stack([l_p2, r_p2], dim=1).reshape(2 * M, 2)
    seg_dir = torch.stack([md_f, md_f], dim=1).reshape(2 * M, 2)
    seg_lane = torch.stack([lid_f, lid_f], dim=1).reshape(2 * M)

    Mseg = seg_p1.shape[0]
    mid = (seg_p1 + seg_p2) / 2
    left_normal = torch.stack([-seg_dir[:, 1], seg_dir[:, 0]], dim=-1)
    is_left = torch.arange(Mseg, device=device) % 2 == 0
    outward = torch.where(is_left[:, None], left_normal, -left_normal)
    outward = outward / outward.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    nudged = mid + nudge * outward

    inside = _point_in_polygons(nudged, edge_v1, edge_v2, edge_poly_id, n_polys)
    is_shared = inside.clone()  # lands inside some lane polygon → shared

    candidate_outer = ~inside  # outside all polygons
    is_junction_gap = torch.zeros_like(candidate_outer)

    if candidate_outer.any():
        nudged_outer = nudged[candidate_outer]
        d = _point_to_segments_dist(nudged_outer, seg_p1, seg_p2)
        outer_lane = seg_lane[candidate_outer]
        same_lane_mask = (outer_lane[:, None] == seg_lane[None, :])
        d[same_lane_mask] = 999.0
        min_d = d.min(dim=1).values
        cand_is_junction = min_d < gap_threshold
        # Map back into full-length is_junction_gap
        outer_indices = torch.where(candidate_outer)[0]
        is_junction_gap[outer_indices[cand_is_junction]] = True

    # Final outer = candidate_outer AND not junction gap
    is_outer = candidate_outer & ~is_junction_gap

    return seg_p1, seg_p2, is_outer, is_junction_gap, is_shared


def plot_scene(data, out_path, scene_name):
    device = data["lanes"].device
    lanes = data["lanes"]
    if lanes.dim() == 4:
        lanes = lanes[0]

    seg_p1, seg_p2, is_outer, is_junction, is_shared = classify_all_segments(lanes)

    fig, ax = plt.subplots(figsize=(13, 12))

    # Lane polygon fills
    lanes_np = lanes.cpu().numpy()
    for s in range(lanes_np.shape[0]):
        c = lanes_np[s, :, :2]
        v = np.linalg.norm(c, axis=-1) > 1e-3
        if v.sum() < 2:
            continue
        lx = lanes_np[s, v, 0] + lanes_np[s, v, 4]
        ly = lanes_np[s, v, 1] + lanes_np[s, v, 5]
        rx = lanes_np[s, v, 0] + lanes_np[s, v, 6]
        ry = lanes_np[s, v, 1] + lanes_np[s, v, 7]
        ax.fill(np.concatenate([lx, rx[::-1]]),
                np.concatenate([ly, ry[::-1]]),
                color="lightgray", alpha=0.25, zorder=1)
        # Centerline
        cl = lanes_np[s, v, :2]
        ax.plot(cl[:, 0], cl[:, 1], color="gray", linewidth=0.4, alpha=0.4, zorder=2)

    # Road borders (red)
    if "line_strings" in data:
        ls_d = data["line_strings"]
        if ls_d.dim() == 4:
            ls_d = ls_d[0]
        ls_np = ls_d.cpu().numpy()
        for j in range(ls_np.shape[0]):
            pts = ls_np[j]
            if ls_np.shape[-1] >= 4:
                v = (pts[:, 3] > 0.5) & (np.abs(pts[:, :2]).sum(axis=-1) > 0.01)
            else:
                v = np.abs(pts[:, :2]).sum(axis=-1) > 0.01
            if v.sum() > 1:
                ax.plot(pts[v, 0], pts[v, 1], color="red", linewidth=2.5,
                        alpha=0.75, zorder=4)

    # Segments colored by category
    p1 = seg_p1.cpu().numpy()
    p2 = seg_p2.cpu().numpy()
    outer_mask = is_outer.cpu().numpy()
    junction_mask = is_junction.cpu().numpy()
    shared_mask = is_shared.cpu().numpy()

    for i in range(p1.shape[0]):
        if outer_mask[i]:
            color, ls, lw, alpha = "black", "-", 2.5, 0.95
        elif junction_mask[i]:
            color, ls, lw, alpha = "darkorange", "-", 3.0, 0.95
        elif shared_mask[i]:
            color, ls, lw, alpha = "steelblue", "-", 0.8, 0.5
        else:
            continue
        ax.plot([p1[i, 0], p2[i, 0]], [p1[i, 1], p2[i, 1]],
                color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                zorder=5 if color == "darkorange" else 3)

    # GT trajectory for context
    gt = data.get("ego_agent_future")
    if gt is not None:
        if gt.dim() == 3:
            gt = gt[0]
        gt_np = gt.cpu().numpy()
        gt_v = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
        if gt_v.sum() > 1:
            ax.plot(gt_np[gt_v, 0], gt_np[gt_v, 1], color="gold",
                    linestyle="--", linewidth=2, alpha=0.9, zorder=6)

    # Ego starting position
    ego_cur = data.get("ego_current_state")
    ego_xy = None
    if ego_cur is not None:
        ec = ego_cur
        if ec.dim() == 2:
            ec = ec[0]
        ec_np = ec.cpu().numpy()
        ego_xy = (float(ec_np[0]), float(ec_np[1]))
        ax.plot(ego_xy[0], ego_xy[1], marker="*", markersize=18,
                markerfacecolor="darkred", markeredgecolor="black",
                linestyle="None", zorder=10)

    # Counts
    n_outer = int(outer_mask.sum())
    n_junction = int(junction_mask.sum())
    n_shared = int(shared_mask.sum())

    legend = [
        Patch(facecolor="black", edgecolor="black", label=f"OUTER road edge  ({n_outer})"),
        Patch(facecolor="darkorange", edgecolor="darkorange",
              label=f"JUNCTION GAP  ({n_junction})"),
        Patch(facecolor="steelblue", edgecolor="steelblue",
              label=f"SHARED (between lanes)  ({n_shared})"),
        Patch(facecolor="red", edgecolor="red", label="road border (line_strings)"),
        Patch(facecolor="gold", edgecolor="gold", label="GT trajectory"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=10)

    ax.set_aspect("equal")
    # Zoom: bbox of segments
    all_p = np.vstack([p1, p2])
    if ego_xy is not None:
        all_p = np.vstack([all_p, np.array(ego_xy)[None, :]])
    cx = (all_p[:, 0].min() + all_p[:, 0].max()) / 2
    cy = (all_p[:, 1].min() + all_p[:, 1].max()) / 2
    span = max(all_p[:, 0].max() - all_p[:, 0].min(),
               all_p[:, 1].max() - all_p[:, 1].min()) / 2 + 5
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Lane boundary classification — {scene_name}\n"
                 f"orange = junction-gap segments (would be mis-classified as outer without the nudge proximity check)",
                 fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_scenes", type=int, default=20)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    if args.indices:
        chosen = args.indices
    else:
        chosen = list(range(min(args.n_scenes, len(scene_paths))))

    print(f"Rendering {len(chosen)} scenes...")
    rendered = 0
    for i, scene_idx in enumerate(chosen):
        if scene_idx >= len(scene_paths):
            continue
        scene_path = scene_paths[scene_idx]
        try:
            data = load_npz_data(scene_path, device)
        except Exception as e:
            print(f"  [skip] {scene_path}: {e}")
            continue
        name = Path(scene_path).stem
        out_path = os.path.join(args.output_dir, f"{i:03d}_{name}.png")
        plot_scene(data, out_path, name)
        rendered += 1
        if rendered % 5 == 0:
            print(f"  {rendered}/{len(chosen)} done")

    print(f"\n{rendered} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
