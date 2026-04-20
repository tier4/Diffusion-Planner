"""Visualize scenes where OLD polygon vs NEW signed-distance lane gate picks different rank-1.

Generates rsft_v2 trajectories on scenes with the base model, scores with BOTH
methods (OLD from /tmp/reward_old.py, NEW from current rlvr.reward with
configurable lane_cross_thresh). For each scene where the top-1 winner differs,
plot lane polygons, road borders, outer segments, and both winner trajectories
with their first-crossing timestep marked.

Prereq:
    git show HEAD:rlvr/reward.py > /tmp/reward_old.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from rlvr.autoresearch.tools.calibrate_rb_vs_lane import (
    build_reward_config_from_grpo,
    generate_for_all_scenes,
)
from rlvr.grpo_config import GRPOConfig
import rlvr.reward as new_mod


def load_old_module(path):
    spec = importlib.util.spec_from_file_location("reward_old", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reward_old"] = mod
    spec.loader.exec_module(mod)
    return mod


def score_all(mod, trajs, all_data, grpo_cfg, lane_cross_thresh=None):
    base = build_reward_config_from_grpo(grpo_cfg)
    rcfg = mod.RewardConfig()
    for f in rcfg.__dataclass_fields__:
        if hasattr(base, f):
            setattr(rcfg, f, getattr(base, f))
    if lane_cross_thresh is not None and hasattr(rcfg, "lane_cross_thresh"):
        rcfg.lane_cross_thresh = lane_cross_thresh
    N, K = trajs.shape[0], trajs.shape[1]
    totals = np.zeros((N, K))
    bds = []
    for i in range(N):
        rs = mod.compute_reward_batch(trajs[i], all_data[i], rcfg)
        bds.append(rs)
        for k, r in enumerate(rs):
            totals[i, k] = r.total
    return totals, bds, rcfg


def ego_corners(x, y, cos_h, sin_h, length, width, wheelbase):
    n = np.hypot(cos_h, sin_h)
    if n < 1e-6:
        n = 1.0
    cos_h, sin_h = cos_h / n, sin_h / n
    ro = (length - wheelbase) / 2
    cx = x + cos_h * ro
    cy = y + sin_h * ro
    hl, hw = length / 2, width / 2
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return np.array([cx, cy]) + local @ R.T


def draw_trajectory(ax, traj, ego_shape, color, label, cross_step,
                    cross_info=None):
    """Draw trajectory + footprints. If cross_info supplied, highlight the
    perimeter point that triggered the crossing and draw a line to the
    offending outer segment.

    cross_info: dict with keys
        'worst_pt': (2,) array — the perimeter point that fired the gate
        'closest_pt': (2,) array — closest point on offending outer segment
        'seg_p1', 'seg_p2': (2,) each — the offending segment endpoints
        'signed_dist': float — signed distance value at that point
    """
    x = traj[:, 0].cpu().numpy()
    y = traj[:, 1].cpu().numpy()
    cos_h = traj[:, 2].cpu().numpy()
    sin_h = traj[:, 3].cpu().numpy()
    wb, length, width = ego_shape[0].item(), ego_shape[1].item(), ego_shape[2].item()

    T = traj.shape[0]
    step = max(1, T // 10)
    draw_ts = list(range(0, T, step))
    if (T - 1) not in draw_ts:
        draw_ts.append(T - 1)

    for t in draw_ts:
        c = ego_corners(x[t], y[t], cos_h[t], sin_h[t], length, width, wb)
        ax.add_patch(MplPolygon(c, closed=True, fill=False,
                                edgecolor=color, linewidth=0.9, alpha=0.55, zorder=7))
    ax.plot(x, y, color=color, linewidth=1.4, alpha=0.85, zorder=6, label=label)

    if cross_step is not None and 0 <= cross_step < T:
        c = ego_corners(x[cross_step], y[cross_step], cos_h[cross_step], sin_h[cross_step],
                        length, width, wb)
        ax.add_patch(MplPolygon(c, closed=True, fill=True,
                                facecolor=color, alpha=0.35, edgecolor=color,
                                linewidth=2.5, zorder=8))
        cx_, cy_ = c[:, 0].mean(), c[:, 1].mean()
        ax.annotate(f"first cross t={cross_step}",
                    xy=(cx_, cy_), xytext=(cx_, cy_ + 2.0),
                    color=color, fontsize=9, weight="bold", ha="center",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2), zorder=10)

        if cross_info is not None:
            wp = cross_info["worst_pt"]
            cpt = cross_info["closest_pt"]
            sp1 = cross_info["seg_p1"]
            sp2 = cross_info["seg_p2"]
            sd = cross_info["signed_dist"]
            # Highlight the offending outer segment
            ax.plot([sp1[0], sp2[0]], [sp1[1], sp2[1]],
                    color=color, linewidth=5, alpha=0.8, zorder=11)
            # Big marker on the crossing perimeter point
            ax.scatter([wp[0]], [wp[1]], c=color, s=230, marker="X",
                       edgecolors="black", linewidths=1.5, zorder=14,
                       label=None)
            # Connecting line from crossing point to closest point on segment
            ax.plot([wp[0], cpt[0]], [wp[1], cpt[1]],
                    color="red", linewidth=2.5, linestyle="-", zorder=13)
            # Signed distance label
            ax.annotate(f"signed dist = {sd:+.3f}m",
                        xy=(wp[0], wp[1]),
                        xytext=(wp[0] + 2.0, wp[1] - 2.5),
                        color="red", fontsize=10, weight="bold",
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                        zorder=15)


def compute_cross_info(traj, cross_step, ego_shape, outer_p1, outer_p2,
                       outer_outward, inter_polys, lane_cross_thresh):
    """Return info about the specific perimeter point that fired the gate at cross_step.

    Returns None if cross_step is None or the point can't be identified.
    """
    if cross_step is None or outer_p1 is None or outer_p1.shape[0] == 0:
        return None
    from rlvr.reward import (
        _LANE_PTS_PER_SIDE, _points_inside_intersection_areas,
    )
    import torch
    device = outer_p1.device
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
    local = torch.tensor(pts, device=device, dtype=traj.dtype)

    state = traj[cross_step]
    cos_h = state[2]; sin_h = state[3]
    n = (cos_h ** 2 + sin_h ** 2).sqrt().clamp_min(1e-6)
    cos_h = cos_h / n; sin_h = sin_h / n
    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h]).reshape(2, 2)
    world_pts = state[:2] + local @ rot.T  # (K, 2)

    # Signed distance: match reward.py logic (unclamped-only filter)
    seg = outer_p2 - outer_p1
    seg_len2 = (seg ** 2).sum(-1).clamp(min=1e-10)
    diff = world_pts[:, None, :] - outer_p1[None, :, :]
    t_raw = (diff * seg[None, :, :]).sum(-1) / seg_len2[None, :]
    is_unclamped = (t_raw > 0.0) & (t_raw < 1.0)
    t_c = t_raw.clamp(0, 1)
    closest = outer_p1[None, :, :] + t_c[:, :, None] * seg[None, :, :]
    to_q = world_pts[:, None, :] - closest
    dist = to_q.norm(dim=-1)
    INF = torch.tensor(float("inf"), device=device, dtype=dist.dtype)
    dist_u = torch.where(is_unclamped, dist, INF)
    min_dist_u, min_idx_u = dist_u.min(dim=1)
    has_u = torch.isfinite(min_dist_u)
    gathered = to_q.gather(1, min_idx_u[:, None, None].expand(-1, 1, 2)).squeeze(1)
    out_for_min = outer_outward[min_idx_u]
    signed_raw = (gathered * out_for_min).sum(-1)
    signed = torch.where(has_u, signed_raw, torch.full_like(signed_raw, -100.0))

    # Mask: perimeter points inside intersection polygon → signed = -100
    if inter_polys is not None:
        peri_in_inter = _points_inside_intersection_areas(world_pts, inter_polys)
        signed = torch.where(peri_in_inter, torch.full_like(signed, -100.0), signed)

    # Find the worst (most positive signed) perimeter point — the one firing the gate
    worst = int(signed.argmax().item())
    worst_signed = float(signed[worst])
    if worst_signed <= -lane_cross_thresh:
        return None  # no point actually crosses

    worst_seg_idx = int(min_idx_u[worst].item())
    closest_pt = closest[worst, worst_seg_idx].cpu().numpy()
    return {
        "worst_pt": world_pts[worst].cpu().numpy(),
        "closest_pt": closest_pt,
        "seg_p1": outer_p1[worst_seg_idx].cpu().numpy(),
        "seg_p2": outer_p2[worst_seg_idx].cpu().numpy(),
        "signed_dist": worst_signed,
    }


def plot_flip(scene_path, data, ego_shape,
              old_w_idx, old_w_traj, old_cross_old, old_cross_new,
              new_w_idx, new_w_traj, new_cross_old, new_cross_new,
              outer_p1, outer_p2, outer_outward, inter_polys,
              lane_cross_thresh, totals_diff, out_path):
    fig, ax = plt.subplots(figsize=(13, 12))

    # Lane polygons
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
        ax.fill(np.concatenate([lx, rx[::-1]]),
                np.concatenate([ly, ry[::-1]]),
                color="lightgreen", alpha=0.12, zorder=1)
        ax.plot(lx, ly, color="steelblue", linewidth=0.5, alpha=0.35, zorder=3)
        ax.plot(rx, ry, color="steelblue", linewidth=0.5, alpha=0.35, zorder=3)

    # Intersection-area polygons (from NPZ `polygons` field)
    n_inter = 0
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
            ax.fill(pp[:, 0], pp[:, 1], color="purple", alpha=0.22, zorder=2,
                    label="intersection_area polygon" if n_inter == 0 else None)
            ax.plot(pp[:, 0], pp[:, 1], color="purple", linewidth=1.8,
                    alpha=0.7, zorder=3)
            n_inter += 1

    # Road borders
    if "line_strings" in data:
        ls_d = data["line_strings"]
        if ls_d.dim() == 4:
            ls_d = ls_d[0]
        ls_np = ls_d.cpu().numpy()
        for j in range(ls_np.shape[0]):
            pts = ls_np[j]
            v = ((pts[:, 3] > 0.5) if ls_np.shape[-1] >= 4
                 else (np.abs(pts[:, :2]).sum(axis=-1) > 0.01))
            if v.sum() > 1:
                ax.plot(pts[v, 0], pts[v, 1], color="red", linewidth=2.5,
                        alpha=0.75, zorder=4,
                        label="road border" if j == 0 else None)

    # Outer segments classified by nudge (new method's boundary surface)
    if outer_p1 is not None and outer_p1.shape[0] > 0:
        p1 = outer_p1.cpu().numpy()
        p2 = outer_p2.cpu().numpy()
        for i in range(p1.shape[0]):
            ax.plot([p1[i, 0], p2[i, 0]], [p1[i, 1], p2[i, 1]],
                    color="black", linestyle="--", linewidth=1.3, alpha=0.7,
                    zorder=4,
                    label="classified outer edge" if i == 0 else None)

    # Ego start
    ego_cur = data.get("ego_current_state")
    ego_xy = (0.0, 0.0)
    if ego_cur is not None:
        ec = ego_cur
        if ec.dim() == 2:
            ec = ec[0]
        ec_np = ec.cpu().numpy()
        ego_xy = (float(ec_np[0]), float(ec_np[1]))
        ec_corners = ego_corners(ec_np[0], ec_np[1], ec_np[2], ec_np[3],
                                 ego_shape[1].item(), ego_shape[2].item(), ego_shape[0].item())
        ax.add_patch(MplPolygon(ec_corners, closed=True, fill=False,
                                edgecolor="darkred", linewidth=2.5, zorder=9,
                                label="ego start"))

    # GT
    gt = data.get("ego_agent_future")
    if gt is not None:
        if gt.dim() == 3:
            gt = gt[0]
        gt_np = gt.cpu().numpy()
        gt_v = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
        if gt_v.sum() > 1:
            ax.plot(gt_np[gt_v, 0], gt_np[gt_v, 1], color="gold", linestyle="--",
                    linewidth=2.0, alpha=0.9, zorder=6, label="GT")

    # Compute cross_info only for the NEW method result (OLD polygon method
    # doesn't have a single "worst point" — it uses polygon containment, so the
    # crossing comes from any perimeter point outside the union). For OLD, we
    # just mark the crossing timestep without a point highlight.
    new_cross_info = compute_cross_info(new_w_traj, new_cross_new, ego_shape,
                                        outer_p1, outer_p2, outer_outward,
                                        inter_polys, lane_cross_thresh)
    old_cross_info_new_method = compute_cross_info(old_w_traj, old_cross_new, ego_shape,
                                                    outer_p1, outer_p2, outer_outward,
                                                    inter_polys, lane_cross_thresh)

    draw_trajectory(ax, old_w_traj, ego_shape, color="blue",
                    label=f"OLD winner idx={old_w_idx}",
                    cross_step=old_cross_old,
                    cross_info=old_cross_info_new_method)
    draw_trajectory(ax, new_w_traj, ego_shape, color="magenta",
                    label=f"NEW winner idx={new_w_idx}",
                    cross_step=new_cross_new,
                    cross_info=new_cross_info)

    # Zoom to both trajectories
    all_x = np.concatenate([old_w_traj[:, 0].cpu().numpy(),
                            new_w_traj[:, 0].cpu().numpy(),
                            [ego_xy[0]]])
    all_y = np.concatenate([old_w_traj[:, 1].cpu().numpy(),
                            new_w_traj[:, 1].cpu().numpy(),
                            [ego_xy[1]]])
    cx = (all_x.min() + all_x.max()) / 2
    cy = (all_y.min() + all_y.max()) / 2
    span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 + 8
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    title = (f"{Path(scene_path).stem}\n"
             f"OLD winner idx={old_w_idx}: cross(OLD)={old_cross_old}  cross(NEW)={old_cross_new}\n"
             f"NEW winner idx={new_w_idx}: cross(OLD)={new_cross_old}  cross(NEW)={new_cross_new}\n"
             f"total diff NEW-OLD = {totals_diff:+.2f}")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--reference_config", required=True)
    parser.add_argument("--old_reward_path", default="/tmp/reward_old.py")
    parser.add_argument("--lane_cross_thresh", type=float, default=0.0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if args.n_scenes:
        scene_paths = scene_paths[:args.n_scenes]

    print(f"Generating trajectories for {len(scene_paths)} scenes...")
    trajs, all_data, valid_paths = generate_for_all_scenes(model, model_args, scene_paths, cfg, device)

    old_mod = load_old_module(args.old_reward_path)
    print("Scoring OLD (polygon)...")
    old_totals, old_bds, _ = score_all(old_mod, trajs, all_data, cfg)
    print(f"Scoring NEW (signed-dist, thresh={args.lane_cross_thresh})...")
    new_totals, new_bds, _ = score_all(new_mod, trajs, all_data, cfg,
                                       lane_cross_thresh=args.lane_cross_thresh)

    old_winners = old_totals.argmax(axis=1)
    new_winners = new_totals.argmax(axis=1)
    flipped = np.where(old_winners != new_winners)[0]
    print(f"{len(flipped)}/{len(valid_paths)} scenes flipped. Rendering...")

    es = all_data[0].get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es
    if ego_shape is None:
        ego_shape = torch.tensor([2.79, 4.34, 1.70], device=device)

    from rlvr.reward import _build_lane_polygons, _classify_outer_boundaries

    def compute_outer_for(data):
        try:
            lanes = data["lanes"]
            if lanes.dim() == 4:
                lanes = lanes[0]
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
            lane_ids = torch.arange(S, device=lanes.device).unsqueeze(1).expand(S, P - 1)
            idx_keep = torch.where(vp.reshape(-1))[0]
            if len(idx_keep) == 0:
                return None, None
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
                s1, s2, sd, sl, edge_v1, edge_v2, edge_poly_id, n_polys,
            )
            # Apply the same intersection-polygon filter reward.py uses.
            inter_p = data.get("polygons")
            if inter_p is not None and inter_p.dim() == 4:
                inter_p = inter_p[0]
            if is_outer.any() and inter_p is not None and inter_p.shape[-1] >= 2:
                from rlvr.reward import _points_inside_intersection_areas
                outer_indices = torch.where(is_outer)[0]
                p1c = s1[outer_indices]
                p2c = s2[outer_indices]
                sample_ts = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0],
                                          device=s1.device, dtype=s1.dtype)
                samples = (p1c[:, None, :] +
                           sample_ts[None, :, None] * (p2c - p1c)[:, None, :])
                inside_flat = _points_inside_intersection_areas(
                    samples.reshape(-1, 2), inter_p
                ).reshape(p1c.shape[0], -1)
                fully_covered = inside_flat.all(dim=-1)
                if fully_covered.any():
                    new_outer = is_outer.clone()
                    new_outer[outer_indices[fully_covered]] = False
                    is_outer = new_outer
            return s1[is_outer], s2[is_outer], outward_all[is_outer], inter_p
        except Exception as e:
            print(f"  [outer ext failed] {e}")
            return None, None, None, None

    from rlvr.reward import RewardConfig as NewCfg
    rcfg_new = NewCfg(enable_lane_departure=True)
    rcfg_new.lane_cross_thresh = args.lane_cross_thresh
    rcfg_old = old_mod.RewardConfig(enable_lane_departure=True)

    for i, scene_idx in enumerate(flipped):
        scene_path = valid_paths[scene_idx]
        data = all_data[scene_idx]
        old_w = int(old_winners[scene_idx])
        new_w = int(new_winners[scene_idx])
        old_traj = trajs[scene_idx, old_w]
        new_traj = trajs[scene_idx, new_w]

        _, _, _, old_cross_old_list, _ = old_mod.compute_lane_departure_penalty(
            old_traj.unsqueeze(0), ego_shape, data, config=rcfg_old,
        )
        _, _, _, new_cross_old_list, _ = old_mod.compute_lane_departure_penalty(
            new_traj.unsqueeze(0), ego_shape, data, config=rcfg_old,
        )
        _, _, _, old_cross_new_list, _ = new_mod.compute_lane_departure_penalty(
            old_traj.unsqueeze(0), ego_shape, data, config=rcfg_new,
        )
        _, _, _, new_cross_new_list, _ = new_mod.compute_lane_departure_penalty(
            new_traj.unsqueeze(0), ego_shape, data, config=rcfg_new,
        )

        outer_p1, outer_p2, outer_outward, inter_p = compute_outer_for(data)

        totals_diff = float(new_totals[scene_idx, new_w] - old_totals[scene_idx, old_w])

        out_name = f"flip_{i:03d}_{Path(scene_path).stem}.png"
        out_path = os.path.join(args.output_dir, out_name)
        plot_flip(scene_path, data, ego_shape,
                  old_w, old_traj, old_cross_old_list[0], old_cross_new_list[0],
                  new_w, new_traj, new_cross_old_list[0], new_cross_new_list[0],
                  outer_p1, outer_p2, outer_outward, inter_p,
                  args.lane_cross_thresh, totals_diff, out_path)

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(flipped)} plots saved")

    print(f"\nDone. {len(flipped)} images in {args.output_dir}")


if __name__ == "__main__":
    main()
