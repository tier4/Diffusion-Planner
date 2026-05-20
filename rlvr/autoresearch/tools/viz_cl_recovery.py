#!/usr/bin/env python3
"""Visualize centerline-recovery trajectories: best-CL traj of K per scene.

For each scene in --indices, regenerates K trajectories using the training
generation_variant (rsft_v2 by default), picks the single trajectory with the
highest centerline score, and plots it alongside the deterministic model
prediction and GT on a lane+border map.

Purpose: see what "recovery" actually looks like on scenes where some of the
16 GRPO samples meaningfully improve centerline over the starting state.

Usage:
    python -m rlvr.autoresearch.tools.viz_cl_recovery \
        --model_path /path/to/base.pth \
        --lora_path /path/to/lora_dir \
        --scenes /path/to/scenes.json \
        --config /path/to/grpo_config.json \
        --indices 196 197 193 200 ... \
        --output_dir /path/out
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from matplotlib.patches import Rectangle

from guidance_gui.generate_samples import generate_samples
from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.grpo_trainer_batched import (
    _normalize_batch,
    _stack_scene_data,
    generate_all_scenes_batched,
    get_generation_config_labels_for_variant,
)
from rlvr.reward import RewardConfig, compute_centerline_score_batch, compute_reward_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def draw_scene_base(ax, npz_path):
    """Lane centerlines (grey dashed) + lane boundaries (blue) + road borders (black)
    + route_lanes centerline (bold orange — the scene's planned path, what CL reward scores against)."""
    npz = np.load(npz_path, allow_pickle=True)
    lanes = npz["lanes"]
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        pts = lane[:, :2]
        if lane.shape[1] > 7:
            lb, rb = lane[:, 4:6], lane[:, 6:8]
            ax.plot(pts[:, 0] + lb[:, 0], pts[:, 1] + lb[:, 1], "-", color="#6a9ec9", lw=0.6, alpha=0.6)
            ax.plot(pts[:, 0] + rb[:, 0], pts[:, 1] + rb[:, 1], "-", color="#6a9ec9", lw=0.6, alpha=0.6)
        ax.plot(pts[:, 0], pts[:, 1], "--", color="grey", lw=0.4, alpha=0.5)

    # Route_lanes: the scene's planned path — this is what `compute_centerline_score_batch` scores against
    if "route_lanes" in npz.files:
        rl = npz["route_lanes"]
        for i in range(rl.shape[0]):
            lane = rl[i]
            if np.abs(lane[:, :2]).sum() < 1e-6:
                continue
            pts = lane[:, :2]
            valid = np.linalg.norm(pts, axis=-1) > 1e-3
            if valid.sum() >= 2:
                ax.plot(pts[valid, 0], pts[valid, 1], "-", color="#ff7f0e", lw=2.0, alpha=0.85,
                        label="ROUTE centerline" if i == 0 else None, zorder=8)

    if "line_strings" in npz.files:
        ls = npz["line_strings"]
        for i in range(ls.shape[0]):
            ch = ls[i]
            if ch.shape[-1] < 4:
                continue
            pts = ch[:, :2]
            ch3 = ch[:, 3]
            mask = ch3 > 0.5
            if mask.sum() >= 2:
                ax.plot(pts[mask, 0], pts[mask, 1], "-", color="black", lw=1.1, alpha=0.9)

    if "neighbor_agents_past" in npz.files and "neighbor_agents_future" in npz.files:
        nb_past = npz["neighbor_agents_past"]
        if nb_past.ndim == 4:
            nb_past = nb_past[0]
        nb_fut = npz["neighbor_agents_future"]
        if nb_fut.ndim == 4:
            nb_fut = nb_fut[0]
        for i in range(nb_past.shape[0]):
            xy0 = nb_past[i, -1, :2]
            if abs(xy0[0]) + abs(xy0[1]) < 1e-6:
                continue
            fut_xy = nb_fut[i, :, :2]
            fut_valid = np.abs(fut_xy).sum(axis=-1) > 1e-6
            if fut_valid.sum() < 2:
                disp = 0.0
            else:
                disp = float(np.linalg.norm(fut_xy[fut_valid].max(axis=0)
                                             - fut_xy[fut_valid].min(axis=0)))
            if disp >= 0.5:
                continue
            cos_h = nb_past[i, -1, 2]; sin_h = nb_past[i, -1, 3]
            heading = math.atan2(sin_h, cos_h)
            width = float(nb_past[i, -1, 6])
            length = float(nb_past[i, -1, 7])
            if width < 0.1 or length < 0.1:
                continue
            t_rot = mtransforms.Affine2D().rotate(heading).translate(xy0[0], xy0[1]) + ax.transData
            ax.add_patch(Rectangle(
                (-length / 2, -width / 2), length, width,
                lw=1.2, ec="#cc6600", fc="#ffb366", alpha=0.75,
                zorder=14, transform=t_rot,
            ))


def draw_traj(ax, traj, label, color, npz_path, with_footprints=True):
    pl = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()
    ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=2, alpha=0.75, zorder=11)
    ax.plot(traj[::5, 0], traj[::5, 1], "o", color=color, ms=3, alpha=0.9, mew=0, zorder=12,
            label=f"{label} ({pl:.1f}m)")
    if not with_footprints:
        return
    npz = np.load(npz_path, allow_pickle=True)
    es = npz.get("ego_shape", None)
    wb = float(es[0]) if es is not None and len(es) >= 1 else 4.76
    length = float(es[1]) if es is not None and len(es) >= 2 else 7.24
    width = float(es[2]) if es is not None and len(es) >= 3 else 2.29
    ro = length - wb
    for ts in [0, 20, 40, 60, len(traj) - 1]:
        if ts >= len(traj): continue
        cx, cy = traj[ts, 0], traj[ts, 1]
        cos_h, sin_h = traj[ts, 2], traj[ts, 3]
        hn = np.sqrt(cos_h ** 2 + sin_h ** 2)
        if hn <= 0.01: continue
        heading = np.arctan2(sin_h / hn, cos_h / hn)
        t_rot = mtransforms.Affine2D().rotate(heading).translate(cx, cy) + ax.transData
        alpha = 0.35 if ts == len(traj) - 1 else 0.15
        lw = 1.2 if ts == len(traj) - 1 else 0.5
        ax.add_patch(Rectangle((-ro, -width / 2), length, width, lw=lw,
                               ec=color, fc=color, alpha=alpha, zorder=11, transform=t_rot))


def draw_gt(ax, npz_path):
    npz = np.load(npz_path, allow_pickle=True)
    gt = npz["ego_agent_future"][:, :2]
    valid = np.abs(gt).sum(axis=-1) > 0.01
    if valid.sum() < 2: return
    gt_v = gt[valid]
    ax.plot(gt_v[:, 0], gt_v[:, 1], "-", color="black", lw=1.5, alpha=0.5, label=f"GT ({np.linalg.norm(np.diff(gt_v, axis=0), axis=1).sum():.1f}m)")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True,
                        help="GRPO training config JSON (reward weights + generation_variant)")
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--generation_variant", type=str, default=None)
    parser.add_argument("--noise_min", type=float, default=0.5)
    parser.add_argument("--noise_max", type=float, default=2.0)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0,
                        help="Set torch RNG seed for reproducible generation.")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    model.to(device).eval()
    if args.lora_path:
        model = load_lora_checkpoint(model, args.lora_path)
        model.eval()

    rcfg = load_reward_config(args.config)
    variant = args.generation_variant
    if variant is None:
        with open(args.config) as f:
            variant = json.load(f).get("generation_variant", "default")
    slot_labels = get_generation_config_labels_for_variant(variant, args.K)

    with open(args.scenes) as f:
        scenes = json.load(f)

    n = len(args.indices)
    rows = (n + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(9 * args.cols, 9 * rows))
    if rows == 1 and args.cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes[None, :]
    elif args.cols == 1: axes = axes[:, None]
    axes_flat = axes.flatten()

    for plot_idx, si in enumerate(args.indices):
        if plot_idx >= len(axes_flat): break
        ax = axes_flat[plot_idx]
        path = scenes[si]
        data = load_npz_data(path, device)
        batch = _stack_scene_data([data], device)
        norm_batch = _normalize_batch(batch, model_args)

        # GT speed for speed guidance
        if "ego_agent_future" in data:
            gt = data["ego_agent_future"]
            if gt.dim() == 3: gt = gt[0]
            gt_np = gt.cpu().numpy()
            valid = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
            gt_v_high = float(np.linalg.norm(np.diff(gt_np[valid][:, :2], axis=0) / 0.1, axis=-1).max()) if valid.sum() >= 5 else 3.0
        else:
            gt_v_high = 3.0

        # t=0 centerline (ego at origin)
        es = data.get("ego_shape")
        ego_shape = es[0] if es is not None and es.dim() > 1 else es
        traj0 = torch.zeros(1, 1, 4, device=device); traj0[0, 0, 2] = 1.0
        t0_cl = compute_centerline_score_batch(traj0, ego_shape, data)[0].item()

        # Deterministic prediction
        det_norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        det_norm = model_args.observation_normalizer(det_norm)
        det_traj = generate_samples(model, model_args, det_norm, 0.0, 1, None, device)[0]
        r_det = compute_reward_batch(torch.tensor(det_traj[None], device=device, dtype=torch.float32), data, rcfg)[0]

        # K trajectories
        trajs = generate_all_scenes_batched(
            model, model_args, norm_batch,
            K=args.K, noise_range=(args.noise_min, args.noise_max),
            device=device, gen_chunk_size=args.K,
            gt_max_speed=gt_v_high, generation_variant=variant,
        )[0]  # [K, T, 4]

        # Score all K trajs — centerline reward is uncapped.
        cl_scores_k = compute_centerline_score_batch(
            trajs, ego_shape, data,
        ).cpu().tolist()
        per_k = []
        for k_i in range(trajs.shape[0]):
            tr = trajs[k_i:k_i+1]
            r = compute_reward_batch(tr, data, rcfg)[0]
            # r.centerline already comes from the same uncapped function — keep it.
            new_total = r.total
            # Mark trajectory as valid (no gate violation) for BestCL filtering
            valid = (
                not r.rb_crossing
                and not r.lane_crossing
                and not r.kinematic_violated
                and (r.collision_step is None or r.collision_step < 0)
            )
            per_k.append({"k": k_i, "total": new_total, "cl": cl_scores_k[k_i], "valid": valid})
        # Also include DET as a candidate for BestCL (not just among K generated)
        det_traj_tensor = torch.tensor(det_traj[None], device=device, dtype=torch.float32)
        det_cl = compute_centerline_score_batch(
            det_traj_tensor, ego_shape, data,
        )[0].item()
        det_valid = (
            not r_det.rb_crossing
            and not r_det.lane_crossing
            and not r_det.kinematic_violated
            and (r_det.collision_step is None or r_det.collision_step < 0)
        )
        per_k_with_det = per_k + [{"k": -1, "total": None, "cl": det_cl,
                                    "is_det": True, "valid": det_valid}]
        top1 = max(per_k, key=lambda x: x["total"])
        # BestCL: only consider valid trajectories (no gate violations)
        valid_candidates = [p for p in per_k_with_det if p["valid"]]
        if valid_candidates:
            best_cl_e = max(valid_candidates, key=lambda x: x["cl"])
            best_cl_label_prefix = ""
        else:
            best_cl_e = max(per_k_with_det, key=lambda x: x["cl"])
            best_cl_label_prefix = "[NO VALID] "
        top1_traj = trajs[top1["k"]].cpu().numpy()
        best_traj = det_traj if best_cl_e.get("is_det") else trajs[best_cl_e["k"]].cpu().numpy()
        best_cl_label = "DET" if best_cl_e.get("is_det") else slot_labels[best_cl_e["k"]][:14]

        # Draw
        draw_scene_base(ax, path)
        draw_gt(ax, path)
        draw_traj(ax, det_traj, f"Det (cl={r_det.centerline:+.2f})", "#1f77b4", path)
        draw_traj(ax, top1_traj,
                  f"Top1 k={top1['k']} {slot_labels[top1['k']][:14]} (tot={top1['total']:+.1f} cl={top1['cl']:+.2f})",
                  "#d62728", path)
        draw_traj(ax, best_traj,
                  f"{best_cl_label_prefix}BestCL[valid] {best_cl_label} (cl={best_cl_e['cl']:+.2f})",
                  "#2ca02c", path)

        # Frame on trajectory area
        all_pts = np.vstack([det_traj[:, :2], top1_traj[:, :2], best_traj[:, :2], np.load(path)["ego_agent_future"][:, :2], [[0, 0]]])
        cx, cy = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])
        half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 6
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper left")
        delta = best_cl_e["cl"] - t0_cl
        ax.set_title(
            f"[{si}] t0_cl={t0_cl:+.2f}  top1_cl={top1['cl']:+.2f}  best_cl={best_cl_e['cl']:+.2f}  Δ_best={delta:+.2f}",
            fontsize=9,
        )

    for j in range(len(args.indices), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    out = Path(args.output_dir) / "cl_recovery.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
