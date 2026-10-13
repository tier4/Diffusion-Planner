"""Visualize intersection-area polygons from the NPZ `polygons` field.

Overlays: lane polygons (gray), road borders (red), classified outer segments
(black dashed), intersection-area polygons from NPZ (purple fill), junction-gap
segments flagged by the nudge proximity check (orange).

The hypothesis being tested: junction-gap segments (orange) should land inside
or on the edge of intersection-area polygons (purple). If so, we can use the
intersection_area polygons directly to mask off out-of-lane detections in
intersections.

Usage:
    python -m rlvr.autoresearch.tools.viz_intersection_polygons \
        --scenes /media/.../j6_train_mixed75.json \
        --n_scenes 15 \
        --output_dir /media/.../viz_intersection_polygons
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.viz_lane_gate_rank_flip import classify_all_segments
from scenario_generation.canvas import Figure


def plot_scene(data, out_path, scene_name):
    device = data["lanes"].device
    lanes = data["lanes"]
    if lanes.dim() == 4:
        lanes = lanes[0]

    seg_p1, seg_p2, is_outer, is_junction, is_shared = classify_all_segments(lanes)

    fig = Figure((13, 12), dpi=110)
    ax = fig.panel(equal=True)

    # Lane polygons filled light
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
        ax.polygons(
            [np.stack([np.concatenate([lx, rx[::-1]]), np.concatenate([ly, ry[::-1]])], 1)],
            fill="#d3d3d3",
            edge="#d3d3d3",
            alpha=0.25,
            z=1,
        )

    # Intersection-area polygons from NPZ
    n_intersect = 0
    if "polygons" in data:
        pg = data["polygons"]
        if pg.dim() == 4:
            pg = pg[0]
        pg_np = pg.cpu().numpy()
        # Shape (N_poly, N_pts, 2 + n_types). Type col index 2 = intersection_area.
        for i in range(pg_np.shape[0]):
            pts = pg_np[i]
            valid = np.abs(pts[:, :2]).sum(axis=-1) > 1e-3
            if valid.sum() < 3:
                continue
            # Check type flag — skip non-intersection polygons
            type_col = pts[:, 2] if pts.shape[-1] > 2 else np.ones(pts.shape[0])
            if not (type_col[valid].max() > 0.5):
                continue
            poly_pts = pts[valid, :2]
            ax.polygons([poly_pts], fill="#800080", edge="#800080", alpha=0.25, z=2)
            ax.lines([poly_pts], "#800080", width=1.8, alpha=0.7, z=3)
            n_intersect += 1

    # Road borders
    if "line_strings" in data:
        ls_d = data["line_strings"]
        if ls_d.dim() == 4:
            ls_d = ls_d[0]
        ls_np = ls_d.cpu().numpy()
        for j in range(ls_np.shape[0]):
            pts = ls_np[j]
            v = (
                (pts[:, 3] > 0.5)
                if ls_np.shape[-1] >= 4
                else (np.abs(pts[:, :2]).sum(axis=-1) > 0.01)
            )
            if v.sum() > 1:
                ax.lines([pts[v, :2]], "#ff0000", width=2.5, alpha=0.75, z=4)

    # Boundary segments classified
    p1 = seg_p1.cpu().numpy()
    p2 = seg_p2.cpu().numpy()
    outer_mask = is_outer.cpu().numpy()
    junction_mask = is_junction.cpu().numpy()

    segments = np.stack([p1, p2], axis=1)
    ax.lines(list(segments[outer_mask]), "#000000", width=2.2, alpha=0.9, z=5)
    ax.lines(list(segments[junction_mask & ~outer_mask]), "#ff8c00", width=3.0, alpha=0.95, z=6)

    # GT trajectory
    gt = data.get("ego_agent_future")
    if gt is not None:
        if gt.dim() == 3:
            gt = gt[0]
        gt_np = gt.cpu().numpy()
        gt_v = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
        if gt_v.sum() > 1:
            ax.lines([gt_np[gt_v, :2]], "#ffd700", width=2.2, alpha=0.9, dash="dash", z=7)

    # Ego start
    ego_cur = data.get("ego_current_state")
    if ego_cur is not None:
        ec = ego_cur
        if ec.dim() == 2:
            ec = ec[0]
        ec_np = ec.cpu().numpy()
        ax.markers(ec_np[:2], "*", size=18, color="#8b0000", edge="#000000", z=10)

    n_outer = int(outer_mask.sum())
    n_junction = int(junction_mask.sum())
    ax.polygons([], fill="#800080", label=f"intersection_area polygon  ({n_intersect})")
    ax.polygons([], fill="#000000", label=f"OUTER road edge  ({n_outer})")
    ax.polygons([], fill="#ff8c00", label=f"JUNCTION GAP segment  ({n_junction})")
    ax.polygons([], fill="#ff0000", label="road border (line_strings)")
    ax.polygons([], fill="#ffd700", label="GT trajectory")
    ax.legend("upper left")

    # Zoom
    all_x = np.concatenate([p1[:, 0], p2[:, 0]])
    all_y = np.concatenate([p1[:, 1], p2[:, 1]])
    if "polygons" in data and n_intersect > 0:
        pg_np_v = data["polygons"]
        if pg_np_v.dim() == 4:
            pg_np_v = pg_np_v[0]
        pgv = pg_np_v.cpu().numpy()
        mask = np.abs(pgv[..., :2]).sum(axis=-1) > 1e-3
        all_x = np.concatenate([all_x, pgv[..., 0][mask]])
        all_y = np.concatenate([all_y, pgv[..., 1][mask]])

    cx = (all_x.min() + all_x.max()) / 2
    cy = (all_y.min() + all_y.max()) / 2
    span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 + 5
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.grid(0.25)
    ax.title(f"Intersection-area polygons vs junction-gap segments — {scene_name}", 11)

    fig.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_scenes", type=int, default=15)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.scenes) as f:
        scene_paths = json.load(f)
    chosen = args.indices if args.indices else list(range(min(args.n_scenes, len(scene_paths))))

    print(f"Rendering {len(chosen)} scenes...")
    rendered = 0
    for i, idx in enumerate(chosen):
        if idx >= len(scene_paths):
            continue
        p = scene_paths[idx]
        try:
            d = load_npz_data(p, device)
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        name = Path(p).stem
        out_path = os.path.join(args.output_dir, f"{i:03d}_{name}.png")
        plot_scene(d, out_path, name)
        rendered += 1
    print(f"{rendered} images in {args.output_dir}")


if __name__ == "__main__":
    main()
