"""Debug-visualize WHY the NEW signed-distance gate fires at t=1 on a target traj.

Takes a target scene + target traj index (e.g., the OLD polygon winner).
Renders geometry (lanes, borders, intersection polygons, all classified
boundary segments color-coded) and overlays the ego perimeter points at t=1
colored by signed distance. Marks the nearest outer segment per point and
draws a line to the closest point on that segment, labeled with signed dist.

Usage:
    python -m rlvr.autoresearch.tools.viz_t1_crossing_debug \
        --model_path <base.pth> --reference_config <cfg.json> \
        --scenes <scenes.json> --scene_substr 14-22-19_0000000000005967 \
        --traj_idx 6 --lane_cross_thresh 0.0 \
        --output /tmp/t1_debug.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.calibrate_rb_vs_lane import generate_for_all_scenes
from rlvr.autoresearch.tools.viz_lane_gate_rank_flip import classify_all_segments
from rlvr.grpo_config import GRPOConfig
from rlvr.reward import (
    _LANE_PTS_PER_SIDE,
    _build_lane_polygons,
    _classify_outer_boundaries,
)


def build_perimeter_world(traj_t, ego_shape):
    """Return (K_pts, 2) world perimeter points at a single timestep."""
    wb = ego_shape[0].item()
    length = ego_shape[1].item()
    width = ego_shape[2].item()
    ro = (length - wb) / 2
    pts = []
    for j in range(_LANE_PTS_PER_SIDE):
        f = j / (_LANE_PTS_PER_SIDE - 1)
        pts.append((-ro + f * length, -width / 2))
        pts.append((-ro + f * length, width / 2))
        if 0 < f < 1:
            pts.append((-ro, -width / 2 + f * width))
            pts.append((length - ro, -width / 2 + f * width))
    local = torch.tensor(pts, device=traj_t.device, dtype=traj_t.dtype)
    cos_h = traj_t[2]
    sin_h = traj_t[3]
    n = (cos_h**2 + sin_h**2).sqrt().clamp_min(1e-6)
    cos_h = cos_h / n
    sin_h = sin_h / n
    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h]).reshape(2, 2)
    world = traj_t[:2] + local @ rot.T
    return world


def compute_outer_with_outward(data):
    lanes = data["lanes"]
    if lanes.dim() == 4:
        lanes = lanes[0]
    device = lanes.device
    edge_v1, edge_v2, edge_poly_id, n_polys = _build_lane_polygons(lanes)
    center = lanes[..., :2]
    direction = lanes[..., 2:4]
    lb_off = lanes[..., 4:6]
    rb_off = lanes[..., 6:8]
    valid = center.norm(dim=-1) > 1e-3
    left_pts = center + lb_off
    right_pts = center + rb_off
    dirs = direction.clone()
    has_dir = dirs.norm(dim=-1) > 1e-6
    dir_sum = (dirs * has_dir.unsqueeze(-1)).sum(dim=1)
    dir_count = has_dir.sum(dim=1, keepdim=True).clamp(min=1)
    dir_avg = dir_sum / dir_count
    dir_avg = dir_avg / dir_avg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dirs = torch.where(has_dir.unsqueeze(-1), dirs, dir_avg.unsqueeze(1).expand_as(dirs))
    vp = valid[:, :-1] & valid[:, 1:]
    md = (dirs[:, :-1] + dirs[:, 1:]) / 2
    md = md / md.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    S, P = center.shape[:2]
    lane_ids = torch.arange(S, device=device).unsqueeze(1).expand(S, P - 1)
    idx_keep = torch.where(vp.reshape(-1))[0]
    if len(idx_keep) == 0:
        return None
    l_p1 = left_pts[:, :-1].reshape(-1, 2)[idx_keep]
    l_p2 = left_pts[:, 1:].reshape(-1, 2)[idx_keep]
    r_p1 = right_pts[:, :-1].reshape(-1, 2)[idx_keep]
    r_p2 = right_pts[:, 1:].reshape(-1, 2)[idx_keep]
    md_f = md.reshape(-1, 2)[idx_keep]
    lid_f = lane_ids.reshape(-1)[idx_keep]
    M = len(idx_keep)
    s1 = torch.stack([l_p1, r_p1], dim=1).reshape(2 * M, 2)
    s2 = torch.stack([l_p2, r_p2], dim=1).reshape(2 * M, 2)
    sd = torch.stack([md_f, md_f], dim=1).reshape(2 * M, 2)
    sl = torch.stack([lid_f, lid_f], dim=1).reshape(2 * M)
    is_outer, outward_all = _classify_outer_boundaries(
        s1,
        s2,
        sd,
        sl,
        edge_v1,
        edge_v2,
        edge_poly_id,
        n_polys,
    )
    return {
        "all_p1": s1,
        "all_p2": s2,
        "outer_p1": s1[is_outer],
        "outer_p2": s2[is_outer],
        "outer_outward": outward_all[is_outer],
        "seg_lane": sl,
        "is_outer": is_outer,
    }


def plot(
    traj_t, data, ego_shape, info, out_path, target_t, scene_name, traj_label, lane_cross_thresh
):
    fig, ax = plt.subplots(figsize=(14, 12))

    # Lanes
    lanes = data["lanes"]
    if lanes.dim() == 4:
        lanes = lanes[0]
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
        ax.fill(
            np.concatenate([lx, rx[::-1]]),
            np.concatenate([ly, ry[::-1]]),
            color="lightgray",
            alpha=0.3,
            zorder=1,
        )

    # Intersection-area polygons
    if "polygons" in data:
        pg = data["polygons"]
        if pg.dim() == 4:
            pg = pg[0]
        pg_np = pg.cpu().numpy()
        for i in range(pg_np.shape[0]):
            pts = pg_np[i]
            valid = np.abs(pts[:, :2]).sum(axis=-1) > 1e-3
            if valid.sum() < 3:
                continue
            pp = pts[valid, :2]
            ax.fill(
                pp[:, 0],
                pp[:, 1],
                color="purple",
                alpha=0.2,
                zorder=2,
                label="intersection_area" if i == 0 else None,
            )
            ax.plot(pp[:, 0], pp[:, 1], color="purple", linewidth=1.5, alpha=0.6, zorder=3)

    # Classified segments
    seg_p1, seg_p2, is_outer, is_junction, is_shared = classify_all_segments(lanes)
    p1 = seg_p1.cpu().numpy()
    p2 = seg_p2.cpu().numpy()
    o_m = is_outer.cpu().numpy()
    j_m = is_junction.cpu().numpy()
    s_m = is_shared.cpu().numpy()
    for i in range(p1.shape[0]):
        if o_m[i]:
            ax.plot(
                [p1[i, 0], p2[i, 0]],
                [p1[i, 1], p2[i, 1]],
                color="black",
                linewidth=2.2,
                alpha=0.9,
                zorder=5,
            )
        elif j_m[i]:
            ax.plot(
                [p1[i, 0], p2[i, 0]],
                [p1[i, 1], p2[i, 1]],
                color="darkorange",
                linewidth=2.5,
                alpha=0.9,
                zorder=6,
            )
        elif s_m[i]:
            ax.plot(
                [p1[i, 0], p2[i, 0]],
                [p1[i, 1], p2[i, 1]],
                color="steelblue",
                linewidth=0.7,
                alpha=0.4,
                zorder=4,
            )

    # Road border line_strings
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
                ax.plot(pts[v, 0], pts[v, 1], color="red", linewidth=2.5, alpha=0.75, zorder=5)

    # Ego footprint at target_t
    t_state = traj_t[target_t]  # (4,) x,y,cos,sin
    x = float(t_state[0])
    y = float(t_state[1])
    cos_h = float(t_state[2])
    sin_h = float(t_state[3])
    n = (cos_h**2 + sin_h**2) ** 0.5
    cos_h /= max(n, 1e-6)
    sin_h /= max(n, 1e-6)
    wb = ego_shape[0].item()
    length = ego_shape[1].item()
    width = ego_shape[2].item()
    # center-shift idiom: box center is rear-axle + (wheelbase/2) forward (loss.py convention)
    ro = wb / 2
    cx = x + cos_h * ro
    cy = y + sin_h * ro
    hl, hw = length / 2, width / 2
    local_corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    world_corners = np.array([cx, cy]) + local_corners @ R.T
    ax.add_patch(
        MplPolygon(
            world_corners,
            closed=True,
            fill=False,
            edgecolor="darkgreen",
            linewidth=3,
            zorder=10,
            label=f"ego @ t={target_t}",
        )
    )

    # Full trajectory faint
    tt = traj_t.cpu().numpy()
    ax.plot(tt[:, 0], tt[:, 1], color="darkgreen", linewidth=1.2, alpha=0.5, zorder=6)

    # Perimeter points at target_t, colored by signed distance
    device = traj_t.device
    world_pts = build_perimeter_world(traj_t[target_t], ego_shape)  # (K, 2)
    outer_p1 = info["outer_p1"]
    outer_p2 = info["outer_p2"]
    outer_outward = info["outer_outward"]

    seg = outer_p2 - outer_p1
    seg_len2 = (seg**2).sum(-1).clamp(min=1e-10)
    diff = world_pts[:, None, :] - outer_p1[None, :, :]
    t_on = ((diff * seg[None, :, :]).sum(-1) / seg_len2[None, :]).clamp(0, 1)
    closest = outer_p1[None, :, :] + t_on[:, :, None] * seg[None, :, :]
    to_q = world_pts[:, None, :] - closest
    dist = to_q.norm(dim=-1)
    min_dist, min_idx = dist.min(dim=1)
    gathered = to_q.gather(1, min_idx[:, None, None].expand(-1, 1, 2)).squeeze(1)
    out_for_min = outer_outward[min_idx]
    signed = (gathered * out_for_min).sum(-1)

    wp = world_pts.cpu().numpy()
    sd = signed.cpu().numpy()
    ud = min_dist.cpu().numpy()
    closest_np = closest.gather(1, min_idx[:, None, None].expand(-1, 1, 2)).squeeze(1).cpu().numpy()

    # Scatter colored by signed distance: red if > -thresh (crossed), green otherwise
    for p in range(wp.shape[0]):
        col = "red" if sd[p] > -lane_cross_thresh else "green"
        ax.scatter(
            wp[p, 0],
            wp[p, 1],
            c=col,
            s=55,
            marker="X" if col == "red" else "o",
            zorder=11,
            edgecolors="black",
            linewidths=0.4,
        )

    # Draw line to closest outer segment for the most-"crossed" point
    worst = int(sd.argmax())
    ax.plot(
        [wp[worst, 0], closest_np[worst, 0]],
        [wp[worst, 1], closest_np[worst, 1]],
        color="red",
        linewidth=2.5,
        linestyle="-",
        zorder=12,
    )
    ax.annotate(
        f"worst signed dist = {sd[worst]:+.3f}m (unsigned {ud[worst]:.3f}m)",
        xy=(wp[worst, 0], wp[worst, 1]),
        xytext=(wp[worst, 0] + 1.5, wp[worst, 1] + 1.5),
        fontsize=10,
        weight="bold",
        color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        zorder=13,
    )

    # Also highlight the segment NEW method claims ego is "crossing"
    worst_seg_idx = int(min_idx[worst].item())
    ws_p1 = outer_p1[worst_seg_idx].cpu().numpy()
    ws_p2 = outer_p2[worst_seg_idx].cpu().numpy()
    ax.plot(
        [ws_p1[0], ws_p2[0]],
        [ws_p1[1], ws_p2[1]],
        color="red",
        linewidth=4,
        alpha=0.7,
        zorder=9,
        label="offending outer segment",
    )

    # GT
    gt = data.get("ego_agent_future")
    if gt is not None:
        if gt.dim() == 3:
            gt = gt[0]
        gt_np = gt.cpu().numpy()
        gt_v = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
        if gt_v.sum() > 1:
            ax.plot(
                gt_np[gt_v, 0],
                gt_np[gt_v, 1],
                color="gold",
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                zorder=7,
            )

    # Zoom to ego + nearest features
    span = 15
    ax.set_xlim(x - span, x + span)
    ax.set_ylim(y - span, y + span)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    n_cross = int((sd > -lane_cross_thresh).sum())
    title = (
        f"{scene_name} — traj={traj_label}, t={target_t}\n"
        f"lane_cross_thresh={lane_cross_thresh}  |  "
        f"{n_cross}/{wp.shape[0]} perimeter points flagged as 'crossed'\n"
        f"worst signed dist = {sd[worst]:+.4f}m  (crossed if > {-lane_cross_thresh:.2f})"
    )
    ax.set_title(title, fontsize=11)

    legend_items = [
        Patch(facecolor="black", label="classified OUTER edge"),
        Patch(facecolor="darkorange", label="JUNCTION GAP segment"),
        Patch(facecolor="steelblue", label="shared boundary"),
        Patch(facecolor="purple", alpha=0.3, label="intersection_area polygon"),
        Patch(facecolor="red", label="road border (line_strings)"),
        Patch(facecolor="darkgreen", label=f"ego footprint @ t={target_t}"),
        Patch(facecolor="red", edgecolor="black", label="perimeter pt flagged crossed"),
        Patch(facecolor="green", edgecolor="black", label="perimeter pt OK"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(
        f"  Perimeter signed dists: min={sd.min():+.3f}, max={sd.max():+.3f}, mean={sd.mean():+.3f}"
    )
    print(f"  Unsigned dists: min={ud.min():.3f}, max={ud.max():.3f}")
    print(f"  Flagged as crossed: {n_cross}/{wp.shape[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--reference_config", required=True)
    parser.add_argument("--scene_substr", required=True)
    parser.add_argument(
        "--traj_idx",
        type=int,
        required=True,
        help="Trajectory index within the 16 generated trajs (the OLD polygon winner).",
    )
    parser.add_argument("--target_t", type=int, default=1)
    parser.add_argument("--lane_cross_thresh", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    cfg = GRPOConfig.from_json(args.reference_config)
    cfg.enable_lane_departure = True

    model_dir = Path(args.model_path).parent
    args_json = model_dir / "args.json"
    if not args_json.exists():
        args_json = model_dir.parent / "args.json"
    model_args = Config(str(args_json))
    model = Diffusion_Planner(model_args)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt.get("model", ckpt).items()}
    model.load_state_dict(state)
    model.to(device).eval()

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    # Generate all trajs (same seed → reproduce the scene_idx=6 traj 6)
    trajs, all_data, valid_paths = generate_for_all_scenes(
        model, model_args, scene_paths, cfg, device
    )

    target_idx = None
    for i, p in enumerate(valid_paths):
        if args.scene_substr in p:
            target_idx = i
            break
    if target_idx is None:
        print(f"Scene not found containing '{args.scene_substr}'")
        sys.exit(1)
    print(f"Target scene: {Path(valid_paths[target_idx]).stem}")

    data = all_data[target_idx]
    es = data.get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es
    if ego_shape is None:
        ego_shape = torch.tensor([2.79, 4.34, 1.70], device=device)

    traj = trajs[target_idx, args.traj_idx]  # (T, 4)
    print(f"Traj idx={args.traj_idx}, target_t={args.target_t}")

    info = compute_outer_with_outward(data)
    if info is None:
        print("No outer segments found")
        sys.exit(1)

    scene_name = Path(valid_paths[target_idx]).stem
    traj_label = f"idx={args.traj_idx}"
    plot(
        traj,
        data,
        ego_shape,
        info,
        args.output,
        args.target_t,
        scene_name,
        traj_label,
        args.lane_cross_thresh,
    )


if __name__ == "__main__":
    main()
