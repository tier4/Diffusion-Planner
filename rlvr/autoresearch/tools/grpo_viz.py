#!/usr/bin/env python3
"""Visualize all K GRPO trajectories per scene with reward breakdown.

One figure per scene showing:
- Left: bird's eye view with all K trajectories colored by rank (green=best, red=worst)
  with road borders, lane boundaries, GT, and ego footprints
- Right: reward breakdown table for each trajectory

Usage:
    python -m rlvr.autoresearch.tools.grpo_viz \
        --model_path /path/to/best_model.pth \
        --scenes /path/to/scenes.json \
        --output_dir /path/to/output \
        --indices 0 1 2 3 4 \
        --K 16 --enable_lane --survival \
        --w_progress 3.0 --lane_near_scale 30.0 --lane_wide_scale 10.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import load_npz_data
from rlvr.grpo_sampler import SamplerConfig
from rlvr.grpo_sampler_batched import generate_diverse_group_batched
from rlvr.reward import RewardConfig, compute_lane_departure_penalty, compute_reward_batch
from scenario_generation.canvas import Figure, oriented_box
from scenario_generation.colormaps import map_values

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _rank_color(rank: int, K: int) -> str:
    """``#rrggbb`` of ``rank`` on RdYlGn_r over ``[0, K - 1]`` (green = best)."""
    r, g, b = map_values(np.array(rank / max(K - 1, 1)), "RdYlGn_r", 0.0, 1.0)
    return f"#{r:02x}{g:02x}{b:02x}"


def generate_and_score(model, model_args, npz_path, K, reward_config, sampler_config):
    """Generate K trajectories and score each one with full breakdown."""
    device = torch.device(DEVICE)
    data = load_npz_data(npz_path, device)
    es = data.get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es

    trajs = generate_diverse_group_batched(model, model_args, data, sampler_config, device)

    # Compute GT path length for underprogress context
    gt_path_len = 0.0
    if "ego_agent_future" in data:
        gt = data["ego_agent_future"]
        if gt.dim() == 3:
            gt = gt[0]
        gt_xy = gt[:, :2]
        gt_valid = gt_xy.abs().sum(dim=-1) > 0.1
        if gt_valid.sum() >= 10:
            gt_path_len = float(torch.diff(gt_xy[gt_valid], dim=0).norm(dim=-1).sum().item())

    results = []
    for k_i in range(trajs.shape[0]):
        traj_t = trajs[k_i : k_i + 1]
        r = compute_reward_batch(traj_t, data, reward_config)[0]
        gate, near, _, _, _ = compute_lane_departure_penalty(traj_t, ego_shape, data)

        # Compute path length
        traj_np = trajs[k_i].cpu().numpy()
        path_len = float(np.linalg.norm(np.diff(traj_np[:, :2], axis=0), axis=1).sum())

        results.append(
            {
                "k": k_i,
                "traj": traj_np,
                "total": r.total,
                "progress": r.progress,
                "smoothness": r.smoothness,
                "centerline": r.centerline,
                "safety": r.safety,
                "lane_crossing": r.lane_crossing,
                "lane_near": r.lane_near_frac,
                "rb_crossing": r.rb_crossing,
                "rb_near": r.rb_near_penalty,
                "path_len": path_len,
            }
        )

    results.sort(key=lambda x: -x["total"])
    return results, data, gt_path_len


def draw_grpo_scene(fig, results, npz_path, scene_idx, gt_path_len, tag):
    """Draw one scene with all K trajectories + reward breakdown."""
    npz = np.load(npz_path)
    K = len(results)

    # Layout: trajectory plot on left (60%), table on right (40%)
    ax_map = fig.panel((0.02, 0.05, 0.55, 0.88), equal=True)
    ax_tbl = fig.panel((0.60, 0.05, 0.38, 0.88), xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax_tbl.axis_off()

    wb, length, width = 2.75, 4.34, 1.70
    es = npz.get("ego_shape", None)
    if es is not None and len(es) >= 3:
        wb, length, width = float(es[0]), float(es[1]), float(es[2])
    ro = (length - wb) / 2

    # Draw lane boundaries (grey)
    lanes = npz["lanes"]
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        if lane.shape[1] > 7:
            pts = lane[:, :2]
            lb, rb = lane[:, 4:6], lane[:, 6:8]
            v = np.abs(pts).sum(axis=1) > 0.1
            if v.sum() > 1:
                ax_map.lines([(pts + lb)[v]], "#bbbbbb", width=0.7, alpha=0.5)
                ax_map.lines([(pts + rb)[v]], "#bbbbbb", width=0.7, alpha=0.5)

    # Draw road borders (red)
    ls = npz["line_strings"]
    for i in range(ls.shape[0]):
        line = ls[i]
        if np.abs(line[:, :2]).sum() < 1e-6:
            continue
        if ls.shape[-1] >= 4 and line[:, 3].max() > 0.5:
            valid = (line[:, 3] > 0.5) & (np.abs(line[:, :2]).sum(axis=1) > 0.01)
            if valid.sum() > 1:
                ax_map.lines([line[valid, :2]], "#ff0000", width=3, alpha=0.7, z=4)

    # Draw GT (green dashed)
    gt = npz["ego_agent_future"]
    ax_map.lines(
        [gt[:, :2]],
        "#008000",
        width=2.5,
        alpha=0.6,
        dash="dash",
        z=5,
        label=f"GT ({gt_path_len:.1f}m)",
    )

    # Ego box at origin
    ax_map.polygons(
        [oriented_box(0.0, 0.0, 0.0, length, width, ro)],
        fill="#3366cc",
        edge="#000000",
        width=2,
        alpha=0.9,
        z=20,
    )

    # Color map: rank 1=dark green, rank K=dark red
    all_pts = [gt[:, :2], np.array([[0, 0]])]

    for rank, r in enumerate(results):
        traj = r["traj"]
        color = _rank_color(rank, K)
        alpha = 0.8 if rank < 3 else (0.5 if rank < 8 else 0.3)
        lw = 2.5 if rank < 3 else 1.2

        lc = "OUT" if r["lane_crossing"] else "IN"
        rb = "RB!" if r["rb_crossing"] else ""
        label = f"#{rank + 1} {r['total']:+.0f} {lc}{rb}" if rank < 5 else None

        ax_map.lines([traj[:, :2]], color, width=lw, alpha=alpha, z=10 - rank * 0.1)
        if rank < 5:
            ax_map.markers(traj[-1, :2], "o", size=6, color=color, alpha=0.9, z=15, label=label)

        # Endpoint footprint for top 3
        if rank < 3:
            t_end = len(traj) - 1
            cx, cy = traj[t_end, 0], traj[t_end, 1]
            cos_h, sin_h = traj[t_end, 2], traj[t_end, 3]
            hn = np.sqrt(cos_h**2 + sin_h**2)
            if hn > 0.01:
                heading = np.arctan2(sin_h / hn, cos_h / hn)
                ax_map.polygons(
                    [oriented_box(float(cx), float(cy), float(heading), length, width, ro)],
                    fill=color,
                    edge=color,
                    width=1,
                    alpha=0.25,
                    z=8,
                )

        all_pts.append(traj[:, :2])

    # Auto-zoom
    all_pts = np.vstack(all_pts)
    cx, cy = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])
    half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 8
    ax_map.set_xlim(cx - half, cx + half)
    ax_map.set_ylim(cy - half, cy + half)
    ax_map.grid(0.15)
    ax_map.legend("upper left")

    # Count in-lane and stopped
    n_in = sum(1 for r in results if not r["lane_crossing"])
    n_stopped = sum(1 for r in results if r["path_len"] < 1.0)
    in_totals = [r["total"] for r in results if not r["lane_crossing"]]
    out_totals = [r["total"] for r in results if r["lane_crossing"] and not r["rb_crossing"]]
    in_mean = np.mean(in_totals) if in_totals else float("nan")
    out_mean = np.mean(out_totals) if out_totals else float("nan")

    ax_map.title(
        f"Scene {scene_idx} [{tag}] — {n_in}/{K} in-lane, {n_stopped} stopped\n"
        f"IN={in_mean:+.1f} vs OUT={out_mean:+.1f}  GT={gt_path_len:.1f}m",
        10,
    )

    # Reward breakdown table
    headers = ["Rk", "Total", "Prog", "Smth", "CL", "Safe", "Lane", "LN%", "RB", "Path"]
    col_widths = [0.06, 0.10, 0.09, 0.09, 0.09, 0.09, 0.08, 0.09, 0.06, 0.09]

    # Header
    y = 0.97
    for j, h in enumerate(headers):
        x = sum(col_widths[:j]) + col_widths[j] / 2
        ax_tbl.text((x, y), h, 7, ha="center", va="top", weight="bold")
    y -= 0.02
    ax_tbl.lines([np.array([[0.0, y], [1.0, y]])], "#000000", width=0.5)

    for rank, r in enumerate(results):
        y -= 0.055
        if y < 0.0:
            break

        color = _rank_color(rank, K)
        lc = "OUT" if r["lane_crossing"] else "IN"
        rb = "X" if r["rb_crossing"] else ""
        stopped = r["path_len"] < 1.0

        row = [
            f"{rank + 1}",
            f"{r['total']:+.1f}",
            f"{r['progress']:.1f}",
            f"{r['smoothness']:.2f}",
            f"{r['centerline']:.2f}",
            f"{r['safety']:.2f}",
            f"{lc}",
            f"{r['lane_near']:.0%}",
            f"{rb}",
            f"{r['path_len']:.1f}m",
        ]

        # Highlight: green bg for in-lane advancing, red for stopped, orange for out-of-lane
        if stopped:
            bg_color = "#ffcccc"  # light red for stopped
        elif not r["lane_crossing"]:
            bg_color = "#ccffcc"  # light green for in-lane
        else:
            bg_color = "#fff3cc"  # light yellow for out-of-lane

        # Draw background row highlight
        ax_tbl.fill_between(np.array([0.0, 1.0]), y - 0.025, y + 0.03, color=bg_color, alpha=0.5)

        for j, val in enumerate(row):
            x = sum(col_widths[:j]) + col_widths[j] / 2
            ax_tbl.text(
                (x, y),
                val,
                6.5,
                color if rank < 5 else "#555555",
                ha="center",
                va="center",
                weight="bold" if rank < 3 else "normal",
            )

    # Legend at bottom of table
    y_leg = max(y - 0.06, 0.01)
    ax_tbl.text(
        (0.0, y_leg),
        "Green=in-lane  Yellow=out-of-lane  Red=stopped (<1m)",
        6,
        "#666666",
    )


def main():
    parser = argparse.ArgumentParser(
        description="GRPO trajectory visualization with reward breakdown"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--n_scenes", type=int, default=5)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--tag", type=str, default="model")
    # Reward config
    parser.add_argument("--survival", action="store_true")
    parser.add_argument("--enable_lane", action="store_true")
    parser.add_argument("--lane_gate", action="store_true")
    parser.add_argument("--w_progress", type=float, default=3.0)
    parser.add_argument("--rb_near_scale", type=float, default=5.0)
    parser.add_argument("--rb_wide_scale", type=float, default=0.5)
    parser.add_argument("--lane_near_scale", type=float, default=30.0)
    parser.add_argument("--lane_wide_scale", type=float, default=10.0)
    parser.add_argument("--lane_cont_scale", type=float, default=5.0)
    parser.add_argument("--stopped_penalty", type=float, default=50.0)
    parser.add_argument("--underprogress_penalty", type=float, default=100.0)
    parser.add_argument("--underprogress_threshold", type=float, default=0.5)
    parser.add_argument("--progress_norm_scale", type=float, default=20.0)
    parser.add_argument("--overprogress_margin", type=float, default=1.0)
    parser.add_argument("--overprogress_penalty", type=float, default=3.0)
    args = parser.parse_args()

    device = torch.device(DEVICE)
    model_dir = Path(args.model_path).parent
    args_path = model_dir / "args.json"
    if not args_path.exists():
        args_path = model_dir.parent / "args.json"
    model_args = Config(str(args_path))
    model = Diffusion_Planner(model_args)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    if args.lora_path:
        model = load_lora_checkpoint(model, args.lora_path)
    model.eval()

    rcfg = RewardConfig(
        enable_lane_departure=args.enable_lane,
        lane_gate_enabled=args.lane_gate,
        w_progress=args.w_progress,
        rb_near_scale=args.rb_near_scale,
        rb_wide_scale=args.rb_wide_scale,
        lane_near_scale=args.lane_near_scale,
        lane_wide_scale=args.lane_wide_scale,
        lane_cont_scale=args.lane_cont_scale,
        reward_mode="survival" if args.survival else "gate",
        enable_overprogress=True,
        overprogress_margin=args.overprogress_margin,
        overprogress_penalty=args.overprogress_penalty,
        stopped_penalty=args.stopped_penalty,
        underprogress_penalty=args.underprogress_penalty,
        underprogress_threshold=args.underprogress_threshold,
        progress_norm_scale=args.progress_norm_scale,
    )

    sampler_cfg = SamplerConfig(
        n_trajectories=args.K,
        enable_guidance=True,
        enable_centerline=True,
        enable_lane_keeping=True,
        enable_road_border=True,
        enable_speed=True,
        guidance_prob=0.7,
    )

    with open(args.scenes) as f:
        scenes = json.load(f)

    if args.indices:
        scene_indices = args.indices
    else:
        scene_indices = list(range(min(args.n_scenes, len(scenes))))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for si in scene_indices:
        print(f"Processing scene {si}...")
        results, data, gt_path_len = generate_and_score(
            model, model_args, scenes[si], args.K, rcfg, sampler_cfg
        )

        fig = Figure((18, 10), 150)
        draw_grpo_scene(fig, results, scenes[si], si, gt_path_len, args.tag)

        fname = out / f"scene_{si:03d}.png"
        fig.save(fname)
        print(f"  Saved: {fname}")

        # Print quick summary
        n_in = sum(1 for r in results if not r["lane_crossing"])
        n_stopped = sum(1 for r in results if r["path_len"] < 1.0)
        print(
            f"  {n_in}/{args.K} in-lane, {n_stopped} stopped, "
            f"best={results[0]['total']:+.1f}, worst={results[-1]['total']:+.1f}"
        )

    print(f"\nDone. {len(scene_indices)} figures saved to {out}")


if __name__ == "__main__":
    main()
