"""Visualize lane departure check using reward.py's polygon containment + segment distance.

Shows lane polygons, outer boundary segments (red), ego footprint with perimeter
points colored by distance to road edge, and a line from the worst point to the
nearest outer segment.

Usage:
    python -m rlvr.autoresearch.tools.viz_lane_departure \
        --scenes val_v4_100.json \
        --indices 10 44 46 2 22 \
        --output_dir ~/Pictures/lane_departure_viz \
        [--step auto]
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch
import matplotlib.cm as cmx

from preference_optimization.utils import load_npz_data
from rlvr.reward import (
    _build_lane_polygons,
    _classify_outer_boundaries,
    _point_in_polygons,
    _point_to_segments_dist,
    _LANE_PTS_PER_SIDE,
)


def check_scene(scene_path, step=None):
    """Run lane departure check on a scene at a given step using reward.py methods.

    Args:
        scene_path: path to NPZ file
        step: timestep to check (None = auto-pick curve apex)

    Returns dict with all data needed for visualization.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_npz_data(scene_path, device)
    raw = np.load(scene_path)
    gt_raw = raw["ego_agent_future"]

    es = data.get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es
    if ego_shape is None:
        ego_shape = torch.tensor([2.75, 4.34, 1.70], device=device)
    wb = ego_shape[0].item()
    length = ego_shape[1].item()
    width_val = ego_shape[2].item()
    ro = (length - wb) / 2
    half_w = width_val / 2

    # Auto-pick step at curve apex if not specified
    if step is None:
        dh = np.diff(gt_raw[:, 2])
        dh = np.arctan2(np.sin(dh), np.cos(dh))
        cum_yaw = np.cumsum(np.abs(dh))
        total_yaw = cum_yaw[-1]
        step = max(1, min(int(np.searchsorted(cum_yaw, total_yaw * 0.5)), 78))

    cos_h = float(np.cos(gt_raw[step, 2]))
    sin_h = float(np.sin(gt_raw[step, 2]))
    cx, cy = float(gt_raw[step, 0]), float(gt_raw[step, 1])

    # Build polygons and segments
    lanes = data["lanes"]
    if lanes.dim() == 4:
        lanes = lanes[0]
    S, P, D = lanes.shape

    edge_v1, edge_v2, edge_poly_id, n_polys = _build_lane_polygons(lanes)

    center = lanes[..., :2]
    direction = lanes[..., 2:4]
    lb_offset = lanes[..., 4:6]
    rb_offset = lanes[..., 6:8]
    valid = center.norm(dim=-1) > 1e-3

    sp1_l = []; sp2_l = []; sd_l = []; sl_l = []
    for s_idx in range(S):
        idx = torch.where(valid[s_idx])[0]
        if len(idx) < 2:
            continue
        lp = center[s_idx, idx] + lb_offset[s_idx, idx]
        rp = center[s_idx, idx] + rb_offset[s_idx, idx]
        dirs = direction[s_idx, idx].clone()
        for i in range(1, len(idx)):
            if dirs[i].norm() < 1e-6:
                dirs[i] = dirs[i - 1]
        if dirs[0].norm() < 1e-6 and len(dirs) > 1:
            dirs[0] = dirs[1]
        for i in range(len(idx) - 1):
            md = (dirs[i] + dirs[i + 1]) / 2
            dn = md.norm()
            if dn > 1e-6:
                md = md / dn
            sp1_l.append(lp[i]); sp2_l.append(lp[i + 1]); sd_l.append(md); sl_l.append(s_idx)
            sp1_l.append(rp[i]); sp2_l.append(rp[i + 1]); sd_l.append(md); sl_l.append(s_idx)

    has_segs = bool(sp1_l)
    if has_segs:
        seg_p1 = torch.stack(sp1_l)
        seg_p2 = torch.stack(sp2_l)
        seg_dir = torch.stack(sd_l)
        seg_lane = torch.tensor(sl_l, device=device, dtype=torch.int64)

        is_outer = _classify_outer_boundaries(
            seg_p1, seg_p2, seg_dir, seg_lane,
            edge_v1, edge_v2, edge_poly_id, n_polys,
        )
        outer_p1 = seg_p1[is_outer]
        outer_p2 = seg_p2[is_outer]
    else:
        outer_p1 = torch.zeros(0, 2, device=device)
        outer_p2 = torch.zeros(0, 2, device=device)
        is_outer = torch.zeros(0, dtype=torch.bool, device=device)
        seg_p1 = torch.zeros(0, 2, device=device)

    # Build ego perimeter at step (must match reward.py: corners not duplicated)
    lp_list = []
    for j in range(_LANE_PTS_PER_SIDE):
        f = j / (_LANE_PTS_PER_SIDE - 1)
        lp_list.append((-ro + f * length, -half_w))
        lp_list.append((-ro + f * length, half_w))
        if 0 < f < 1:  # skip corners (already in top/bottom)
            lp_list.append((-ro, -half_w + f * width_val))
            lp_list.append((length - ro, -half_w + f * width_val))
    local_pts = torch.tensor(lp_list, device=device, dtype=torch.float32)

    rot_m = torch.tensor([[cos_h, -sin_h], [sin_h, cos_h]], device=device, dtype=torch.float32)
    world_pts = (rot_m @ local_pts.T).T + torch.tensor([cx, cy], device=device, dtype=torch.float32)

    # Containment
    inside = _point_in_polygons(world_pts, edge_v1, edge_v2, edge_poly_id, n_polys)

    # Distance to outer segments + closest point for viz
    if outer_p1.shape[0] > 0:
        d_full = _point_to_segments_dist(world_pts, outer_p1, outer_p2)
        pt_dists = d_full.min(dim=1).values.cpu().numpy()
        worst = int(pt_dists.argmin())
        closest_seg_idx = int(d_full[worst].argmin().item())
        seg_vec = outer_p2[closest_seg_idx] - outer_p1[closest_seg_idx]
        seg_len2 = (seg_vec ** 2).sum().clamp(min=1e-10)
        t_param = ((world_pts[worst] - outer_p1[closest_seg_idx]) * seg_vec).sum() / seg_len2
        t_param = t_param.clamp(0, 1)
        closest_boundary_pt = (outer_p1[closest_seg_idx] + t_param * seg_vec).cpu().numpy()
    else:
        pt_dists = np.full(world_pts.shape[0], 100.0)
        worst = 0
        closest_boundary_pt = np.array([cx, cy])

    # Yaw
    dh = np.diff(gt_raw[:, 2])
    total_yaw = np.degrees(np.abs(np.sum(np.arctan2(np.sin(dh), np.cos(dh)))))

    return {
        "scene_path": scene_path,
        "step": step,
        "total_yaw": total_yaw,
        "gt_raw": gt_raw,
        "cx": cx, "cy": cy, "cos_h": cos_h, "sin_h": sin_h,
        "length": length, "ro": ro, "half_w": half_w,
        "world_pts": world_pts.cpu().numpy(),
        "pt_dists": pt_dists,
        "pt_in_any": inside.cpu().numpy(),
        "worst": worst,
        "wp": world_pts[worst].cpu().numpy(),
        "closest_boundary_pt": closest_boundary_pt,
        "outer_p1": outer_p1.cpu().numpy(),
        "outer_p2": outer_p2.cpu().numpy(),
        "n_outer": int(is_outer.sum()) if has_segs else 0,
        "n_total_segs": seg_p1.shape[0],
        "data": data,
    }


def draw_scene(result, output_path, zoom=None):
    """Draw the lane departure visualization.

    Args:
        result: dict from check_scene
        output_path: where to save image
        zoom: if set, zoom to ego +/- this many meters. None = full view.
    """
    r = result
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    # Thin lane boundary lines
    lanes_raw = r["data"]["lanes"][0].cpu().numpy()
    S_lanes = lanes_raw.shape[0]
    for s in range(S_lanes):
        c = lanes_raw[s, :, :2]
        v = np.linalg.norm(c, axis=-1) > 1e-3
        if v.sum() < 2:
            continue
        lx = lanes_raw[s, v, 0] + lanes_raw[s, v, 4]
        ly = lanes_raw[s, v, 1] + lanes_raw[s, v, 5]
        rx = lanes_raw[s, v, 0] + lanes_raw[s, v, 6]
        ry = lanes_raw[s, v, 1] + lanes_raw[s, v, 7]
        ax.fill(np.concatenate([lx, rx[::-1]]),
                np.concatenate([ly, ry[::-1]]),
                color='lightgreen', alpha=0.3, zorder=1)
        ax.plot(lx, ly, 'b-', linewidth=0.4, alpha=0.25, zorder=4)
        ax.plot(rx, ry, 'b-', linewidth=0.4, alpha=0.25, zorder=4)

    # Outer boundary segments (red)
    for i in range(r["outer_p1"].shape[0]):
        ax.plot([r["outer_p1"][i, 0], r["outer_p2"][i, 0]],
                [r["outer_p1"][i, 1], r["outer_p2"][i, 1]],
                'r-', linewidth=1.0, alpha=0.4, zorder=5)

    # Road borders
    if "line_strings" in r["data"]:
        ls_d = r["data"]["line_strings"][0].cpu().numpy()
        for j in range(ls_d.shape[0]):
            pts = ls_d[j]
            v = np.abs(pts).sum(axis=-1) > 0.1
            if v.sum() > 1:
                ax.plot(pts[v, 0], pts[v, 1], 'r-', linewidth=3, alpha=0.8, zorder=6)

    # GT trajectory
    ax.plot(r["gt_raw"][:, 0], r["gt_raw"][:, 1], 'g--', linewidth=2.5, zorder=7)

    # Ego box
    cx, cy = r["cx"], r["cy"]
    cos_h, sin_h = r["cos_h"], r["sin_h"]
    l, ro, hw = r["length"], r["ro"], r["half_w"]
    corners = np.array([
        [cx + (l - ro) * cos_h - hw * sin_h, cy + (l - ro) * sin_h + hw * cos_h],
        [cx + (l - ro) * cos_h + hw * sin_h, cy + (l - ro) * sin_h - hw * cos_h],
        [cx - ro * cos_h + hw * sin_h, cy - ro * sin_h - hw * cos_h],
        [cx - ro * cos_h - hw * sin_h, cy - ro * sin_h + hw * cos_h],
    ])
    ax.add_patch(MplPolygon(corners, closed=True, fill=False, edgecolor='darkred', linewidth=2.5, zorder=8))

    # Perimeter points colored by distance
    cmap_c = cmx.RdYlGn
    norm_v = plt.Normalize(vmin=0, vmax=2.0)
    for p in range(len(r["pt_dists"])):
        col = cmap_c(norm_v(r["pt_dists"][p]))
        sz = 25; mk = 'o'
        if not r["pt_in_any"][p]:
            col = 'red'; mk = 'X'; sz = 60
        ax.scatter(r["world_pts"][p, 0], r["world_pts"][p, 1], c=[col], s=sz, marker=mk,
                   zorder=9, edgecolors='black', linewidths=0.3)

    # Distance line from worst point to closest boundary
    wp = r["wp"]
    cbp = r["closest_boundary_pt"]
    ax.plot(wp[0], wp[1], 'kX', markersize=25, markeredgewidth=3, zorder=10)
    ax.plot([wp[0], cbp[0]], [wp[1], cbp[1]], 'k-', linewidth=3, zorder=10)
    ax.plot(cbp[0], cbp[1], 'ko', markersize=10, zorder=10)
    ax.annotate(
        f"Dist to road edge: {r['pt_dists'][r['worst']]:.3f}m\n"
        f"All in lane: {r['pt_in_any'].all()}\n"
        f"Outer segs: {r['n_outer']}/{r['n_total_segs']}",
        xy=(wp[0], wp[1]),
        xytext=(wp[0] + 0.5, wp[1] + 2),
        fontsize=12, fontweight='bold',
        arrowprops=dict(arrowstyle='->', lw=2),
        bbox=dict(facecolor='yellow', alpha=0.9), zorder=11,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_v)
    plt.colorbar(sm, ax=ax, shrink=0.4, label="Distance to road edge (m)")

    ax.legend(handles=[
        Patch(facecolor='lightgreen', alpha=0.3, label='Lane polygon'),
        plt.Line2D([0], [0], color='r', linewidth=1.5, alpha=0.6, label='Outer boundary seg'),
        plt.Line2D([0], [0], color='r', linewidth=3, label='Road border'),
        plt.Line2D([0], [0], color='g', ls='--', linewidth=2.5, label='GT trajectory'),
    ], fontsize=10, loc='upper left')

    ax.set_aspect('equal')
    if zoom is not None:
        ax.set_xlim(cx - zoom, cx + zoom)
        ax.set_ylim(cy - zoom, cy + zoom)
    else:
        gt_x, gt_y = r["gt_raw"][:, 0], r["gt_raw"][:, 1]
        xc = (gt_x.min() + gt_x.max()) / 2
        yc = (gt_y.min() + gt_y.max()) / 2
        span = max(gt_x.max() - gt_x.min(), gt_y.max() - gt_y.min()) / 2 + 8
        ax.set_xlim(xc - span, xc + span)
        ax.set_ylim(yc - span, yc + span)

    scene_name = os.path.basename(r["scene_path"]).replace('.npz', '')
    ax.set_title(
        f"{scene_name} — step {r['step']}, {r['total_yaw']:.0f}° curve\n"
        f"Red = outer boundary segments, Green = lane polygons",
        fontsize=13,
    )
    ax.grid(True, alpha=0.2)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize lane departure check")
    parser.add_argument("--scenes", type=str, required=True, help="Path to scene list JSON")
    parser.add_argument("--indices", type=int, nargs='+', default=None, help="Scene indices to visualize")
    parser.add_argument("--n_scenes", type=int, default=5, help="Number of scenes (if indices not given)")
    parser.add_argument("--output_dir", type=str, default="~/Pictures/lane_departure_viz")
    parser.add_argument("--step", type=int, default=None, help="Timestep (None = auto curve apex)")
    parser.add_argument("--zoom", type=float, default=None, help="Zoom to ego +/- meters")
    args = parser.parse_args()

    with open(args.scenes) as f:
        scenes = json.load(f)

    if args.indices:
        indices = args.indices
    else:
        indices = list(range(min(args.n_scenes, len(scenes))))

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    for idx in indices:
        sp = scenes[idx]
        print(f"Processing scene {idx} ({os.path.basename(sp)})...")
        try:
            result = check_scene(sp, step=args.step)
            fname = os.path.join(out_dir, f"scene{idx:03d}.png")
            draw_scene(result, fname, zoom=args.zoom)
            status = "IN" if result["pt_in_any"].all() else "OUT"
            print(f"  {status}, dist={result['pt_dists'][result['worst']]:.3f}m, "
                  f"outer={result['n_outer']}/{result['n_total_segs']}, "
                  f"yaw={result['total_yaw']:.0f}°")
            print(f"  Saved: {fname}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
