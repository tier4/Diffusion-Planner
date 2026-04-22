"""Visualize trajectories flagged as kinematically infeasible by compute_kinematic_gate.

Renders one image per floored trajectory showing the scene geometry and the
traj with footprints. Title reports the max yaw rate and max curvature (both
SG-smoothed) and which threshold fired.
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
from matplotlib.patches import Polygon as MplPolygon

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from rlvr.autoresearch.tools.calibrate_rb_vs_lane import generate_for_all_scenes
from rlvr.grpo_config import GRPOConfig
from rlvr.reward import (
    RewardConfig,
    _build_sg_diff_kernel,
    compute_kinematic_gate,
)


def sg_yaw_and_speed(traj, dt=0.1, window=11):
    device = traj.device
    pad = window // 2
    K_smooth = _build_sg_diff_kernel(window=window, poly=3, deriv=0, delta=dt).to(device)
    K_vel = _build_sg_diff_kernel(window=window, poly=3, deriv=1, delta=dt).to(device)
    cos_h = traj[..., 2]; sin_h = traj[..., 3]
    n = (cos_h**2 + sin_h**2).sqrt().clamp_min(1e-6)
    cos_h = cos_h / n; sin_h = sin_h / n
    hd = torch.stack([cos_h, sin_h], dim=0).unsqueeze(0)
    hd_pad = torch.nn.functional.pad(hd, (pad, pad), mode="replicate")
    hd_sg = torch.nn.functional.conv1d(hd_pad, K_smooth.view(1,1,-1).expand(2,1,-1), groups=2)
    cos_sg, sin_sg = hd_sg[0, 0], hd_sg[0, 1]
    theta = torch.atan2(sin_sg, cos_sg)
    dth = theta[1:] - theta[:-1]
    dth = torch.atan2(dth.sin(), dth.cos())
    yaw_rate = dth.abs() / dt
    pos = traj[..., :2].unsqueeze(0).permute(0, 2, 1)
    pos_pad = torch.nn.functional.pad(pos, (pad, pad), mode="replicate")
    vel_sg = torch.nn.functional.conv1d(pos_pad, K_vel.view(1,1,-1).expand(2,1,-1), groups=2)
    speed = (vel_sg[0, 0]**2 + vel_sg[0, 1]**2).sqrt()
    return yaw_rate.cpu().numpy(), speed.cpu().numpy()


def plot_floored(traj, ego_shape, data, yaw_rate, speed, rcfg, out_path, title_prefix):
    fig, ax = plt.subplots(figsize=(12, 11))

    # Lanes
    lanes = data["lanes"]
    if lanes.dim() == 4: lanes = lanes[0]
    lanes_np = lanes.cpu().numpy()
    for s in range(lanes_np.shape[0]):
        c = lanes_np[s, :, :2]
        v = np.linalg.norm(c, axis=-1) > 1e-3
        if v.sum() < 2: continue
        lx = lanes_np[s, v, 0] + lanes_np[s, v, 4]
        ly = lanes_np[s, v, 1] + lanes_np[s, v, 5]
        rx = lanes_np[s, v, 0] + lanes_np[s, v, 6]
        ry = lanes_np[s, v, 1] + lanes_np[s, v, 7]
        ax.fill(np.concatenate([lx, rx[::-1]]),
                np.concatenate([ly, ry[::-1]]),
                color="lightgreen", alpha=0.12, zorder=1)
        ax.plot(lx, ly, color="steelblue", linewidth=0.5, alpha=0.35, zorder=3)
        ax.plot(rx, ry, color="steelblue", linewidth=0.5, alpha=0.35, zorder=3)

    # Intersection polygons
    if "polygons" in data:
        pg = data["polygons"]
        if pg.dim() == 4: pg = pg[0]
        pg_np = pg.cpu().numpy()
        for i in range(pg_np.shape[0]):
            pts = pg_np[i]
            valid = np.abs(pts[:, :2]).sum(axis=-1) > 1e-3
            if valid.sum() < 3: continue
            pp = pts[valid, :2]
            ax.fill(pp[:, 0], pp[:, 1], color="purple", alpha=0.2, zorder=2)

    # Road borders
    if "line_strings" in data:
        ls_d = data["line_strings"]
        if ls_d.dim() == 4: ls_d = ls_d[0]
        ls_np = ls_d.cpu().numpy()
        for j in range(ls_np.shape[0]):
            pts = ls_np[j]
            v = ((pts[:, 3] > 0.5) if ls_np.shape[-1] >= 4
                 else (np.abs(pts[:, :2]).sum(axis=-1) > 0.01))
            if v.sum() > 1:
                ax.plot(pts[v, 0], pts[v, 1], color="red", linewidth=2.5, alpha=0.75, zorder=4)

    # Trajectory footprints
    t_np = traj.cpu().numpy()
    T = traj.shape[0]
    wb = ego_shape[0].item(); length = ego_shape[1].item(); width = ego_shape[2].item()
    ro = (length - wb) / 2
    step = max(1, T // 12)
    draw_ts = list(range(0, T, step))
    if (T - 1) not in draw_ts: draw_ts.append(T - 1)
    for t in draw_ts:
        cos_h = t_np[t, 2]; sin_h = t_np[t, 3]
        n = np.hypot(cos_h, sin_h)
        if n < 1e-6: n = 1.0
        cos_h /= n; sin_h /= n
        cx = t_np[t, 0] + cos_h * ro
        cy = t_np[t, 1] + sin_h * ro
        hl, hw = length/2, width/2
        local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        corners = np.array([cx, cy]) + local @ R.T
        color = "red" if t == draw_ts[0] else ("darkred" if t == (T-1) else "black")
        ax.add_patch(MplPolygon(corners, closed=True, fill=False,
                                edgecolor=color, linewidth=1.3, zorder=7))
    ax.plot(t_np[:, 0], t_np[:, 1], color="black", linewidth=1.3, alpha=0.8, zorder=6)

    # Mark timesteps where gate fires
    abs_vio = yaw_rate > rcfg.max_yaw_rate
    speed_align = speed[:-1]
    wb_g = getattr(plot_floored, "_wb", 4.76)
    kappa_cap = rcfg.kinematic_margin * float(np.tan(rcfg.max_steer)) / max(wb_g, 1e-3)
    curv_vio = yaw_rate > kappa_cap * speed_align
    any_vio = abs_vio | curv_vio
    for t in np.where(any_vio)[0]:
        ax.scatter(t_np[t, 0], t_np[t, 1], c="orange", s=120, marker="X",
                   edgecolors="black", linewidths=1.2, zorder=11)

    # GT
    gt = data.get("ego_agent_future")
    if gt is not None:
        if gt.dim() == 3: gt = gt[0]
        gt_np = gt.cpu().numpy()
        gt_v = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
        if gt_v.sum() > 1:
            ax.plot(gt_np[gt_v, 0], gt_np[gt_v, 1], color="gold", linestyle="--",
                    linewidth=2, alpha=0.9, zorder=6)

    # Zoom
    cx_t = (t_np[:, 0].min() + t_np[:, 0].max()) / 2
    cy_t = (t_np[:, 1].min() + t_np[:, 1].max()) / 2
    span = max(t_np[:, 0].max() - t_np[:, 0].min(), t_np[:, 1].max() - t_np[:, 1].min()) / 2 + 6
    ax.set_xlim(cx_t - span, cx_t + span)
    ax.set_ylim(cy_t - span, cy_t + span)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    max_curv_speed_col = np.where(speed_align > 0.1, yaw_rate / speed_align.clip(min=0.1), 0.0)
    # Recompute κ_max from config (ego_shape is passed into the caller; use wheelbase from the caller's frame)
    wb_g = getattr(plot_floored, "_wb", 4.76)
    kappa_cap = rcfg.kinematic_margin * float(np.tan(rcfg.max_steer)) / max(wb_g, 1e-3)
    title = (f"{title_prefix}\n"
             f"max SG yaw_rate = {yaw_rate.max():.3f} rad/s (cap {rcfg.max_yaw_rate})\n"
             f"max SG curvature = {max_curv_speed_col.max():.3f} /m (cap {kappa_cap:.3f} @ wb={wb_g:.2f})\n"
             f"abs-yaw violations: {int(abs_vio.sum())}, curv violations: {int(curv_vio.sum())}")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--reference_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    cfg = GRPOConfig.from_json(args.reference_config); cfg.enable_lane_departure = True
    model_args = Config(str(Path(args.model_path).parent / "args.json"))
    model = Diffusion_Planner(model_args)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt.get("model", ckpt).items()}
    model.load_state_dict(state); model.to(device).eval()

    with open(args.scenes) as f:
        scene_paths = json.load(f)
    trajs, all_data, valid = generate_for_all_scenes(model, model_args, scene_paths, cfg, device)

    rcfg = RewardConfig(enable_overprogress=True)
    es = all_data[0].get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es
    plot_floored._wb = float(ego_shape[0]) if ego_shape is not None else 4.76

    count = 0
    for s_i in range(len(valid)):
        gate = compute_kinematic_gate(trajs[s_i], rcfg, ego_shape)
        for k in range(trajs.shape[1]):
            if float(gate[k]) < 0.5:
                yr, sp = sg_yaw_and_speed(trajs[s_i, k])
                name = Path(valid[s_i]).stem
                out_path = os.path.join(args.output_dir,
                                        f"floored_{count:03d}_{name}_traj{k}.png")
                plot_floored(trajs[s_i, k], ego_shape, all_data[s_i], yr, sp, rcfg,
                             out_path, f"{name} — traj idx={k}")
                count += 1
    print(f"Rendered {count} floored trajectories to {args.output_dir}")


if __name__ == "__main__":
    main()
