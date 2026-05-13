#!/usr/bin/env python3
"""Per-prediction route-centerline deviation heatmap.

Sibling to ``heatmap_route_deviation``. Same route polyline + projection +
arc-binning machinery (from ``_heatmap_common``) but the projected points are
the **predicted-future trajectory** at each frame, not the realised ego pose.

For each NPZ in --npz_dir we:
  1. Recover the ego world pose from its ``goal_pose``.
  2. Take the 80-step predicted future in ego frame:
       * GT  := NPZ's ``ego_agent_future``  (autonomous DP that drove the bag)
       * Det := one forward pass through ``--model_path`` (no guidance, no K=8)
  3. Transform each of the 80 (dx, dy) into the world frame using the ego pose.
  4. Project all 80 world points onto the route polyline → (arc, signed_lat).
  5. Pool across frames and aggregate ``mean(|lat|)`` per arc-length bin.

The result answers: *for predictions that LAND on this part of the route,
how far off the centerline do they sit?* Hot bins = "predictions in this
region are lacking by N metres" — independent of which frame they came
from. No closed-loop error compounding.

Usage:
    python -m scenario_generation.tools.heatmap_prediction_deviation \\
        --route   /path/to/sg_route.pkl \\
        --npz_dir /path/to/npz_session2/ \\
        --model_path /path/to/best_model.pth \\
        --output  /path/to/heatmap.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from rlvr.autoresearch.tools.viz_p4_recovery import (
    _normalize_batch,
    _stack_scene_data,
)
from scenario_generation.tools._heatmap_common import (
    bin_by_arc,
    build_route_polyline,
    deviation_series,
    load_route,
    plot_route_heatmap,
    recover_ego_world_pose_from_goal,
    segments_from_polyline,
)


def _ego_relative_future_world(
    rel_xy: np.ndarray, ego_xy: tuple[float, float], ego_yaw: float
) -> np.ndarray:
    """Transform (T, 2) ego-frame relative future into (T, 2) world points."""
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (rel_xy @ R.T) + np.array(ego_xy, dtype=np.float64)


def _predicted_future_for_one(model, model_args, data: dict, device) -> np.ndarray:
    """Det forward on a single scene. Returns (T, 2) ego-frame xy of pred."""
    batch = _stack_scene_data([data], device)
    norm_batch = _normalize_batch(batch, model_args)
    decoder = model.module.decoder if hasattr(model, "module") else model.decoder
    saved_fn = decoder._guidance_fn
    decoder._guidance_fn = None
    try:
        P = 1 + model_args.predicted_neighbor_num
        future_len = model_args.future_len
        norm_batch_d = {k: v for k, v in norm_batch.items()}
        norm_batch_d["sampled_trajectories"] = torch.zeros(
            1, P, future_len + 1, 4, device=device
        )
        with torch.no_grad():
            _, det_out = model(norm_batch_d)
        det = det_out["prediction"][0, 0].detach().cpu().numpy()
    finally:
        decoder._guidance_fn = saved_fn
    return det[:, :2]


def _gt_future(data: dict) -> np.ndarray | None:
    gt = data.get("ego_agent_future")
    if gt is None:
        return None
    gt = gt.detach().cpu().numpy()
    if gt.ndim == 3:
        gt = gt[0]
    if gt.shape[-1] < 2 or np.abs(gt[:, :2]).sum() < 0.1:
        return None
    return gt[:, :2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--route", type=Path, required=True)
    p.add_argument("--npz_dir", type=Path, required=True,
                   help="Directory of sequential NPZ frames from one bag/session.")
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label_gt", default="GT (bag future)")
    p.add_argument("--label_det", default="Model det")
    p.add_argument("--bin_m", type=float, default=5.0)
    p.add_argument("--clip_max_m", type=float, default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1,
                   help="Sub-sample frames: use every Nth NPZ. Default 1.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading route {args.route}")
    route = load_route(args.route)
    pts, s = build_route_polyline(route)
    s_max = float(s[-1])
    print(f"Route polyline: {len(pts)} pts, arc length {s_max:.1f} m")

    npzs = sorted(args.npz_dir.glob("*.npz"))
    if args.stride > 1:
        npzs = npzs[:: args.stride]
    if args.max_frames:
        npzs = npzs[: args.max_frames]
    if not npzs:
        raise SystemExit(f"No NPZs under {args.npz_dir}")
    print(f"Processing {len(npzs)} frames (stride={args.stride})")

    print(f"Loading model {args.model_path}")
    model_dir = args.model_path.parent
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

    gt_world_pts: list[np.ndarray] = []
    det_world_pts: list[np.ndarray] = []
    ego_arcs: list[float] = []
    skipped = 0

    for fi, npz_path in enumerate(npzs):
        try:
            data = load_npz_data(str(npz_path), device)
        except Exception as e:
            print(f"  [skip {npz_path.name}] {e}")
            skipped += 1
            continue
        gp = data.get("goal_pose")
        if gp is None:
            skipped += 1
            continue
        gp = gp.detach().cpu().numpy().reshape(-1)
        ex, ey, eyaw = recover_ego_world_pose_from_goal(gp, route)

        gt_xy = _gt_future(data)
        det_xy = _predicted_future_for_one(model, model_args, data, device)

        if gt_xy is not None:
            gt_world_pts.append(_ego_relative_future_world(gt_xy, (ex, ey), eyaw))
        det_world_pts.append(_ego_relative_future_world(det_xy, (ex, ey), eyaw))

        # Also record where the ego currently is on the route (for an
        # informational "ego progressed to here" overlay).
        ego_arcs.append(deviation_series(np.array([[ex, ey]]), pts, s)[0, 0])

        if (fi + 1) % 25 == 0:
            print(f"  {fi+1}/{len(npzs)}  ego_arc={ego_arcs[-1]:.1f}m")

    print(f"Done. Skipped {skipped} frames.")

    gt_all = np.concatenate(gt_world_pts, axis=0) if gt_world_pts else np.zeros((0, 2))
    det_all = np.concatenate(det_world_pts, axis=0)
    print(f"Projecting {len(gt_all)} GT + {len(det_all)} det points to polyline")

    dev_gt = deviation_series(gt_all, pts, s) if len(gt_all) else np.zeros((0, 3))
    dev_det = deviation_series(det_all, pts, s)

    bin_m = float(args.bin_m)
    bs_mid, mean_gt, max_gt = bin_by_arc(dev_gt, s_max, bin_m)
    _, mean_det, max_det = bin_by_arc(dev_det, s_max, bin_m)
    n_bins = len(bs_mid)
    bin_segments = segments_from_polyline(pts, s, bin_m, n_bins)

    # color scale
    if args.clip_max_m is not None:
        vmax = float(args.clip_max_m)
    else:
        concat = np.concatenate([
            mean_gt[~np.isnan(mean_gt)], mean_det[~np.isnan(mean_det)]
        ])
        vmax = float(np.percentile(concat, 95)) if len(concat) else 1.0
        vmax = max(vmax, 0.25)
    print(f"Color scale: 0 → {vmax:.2f} m")

    diff = mean_det - mean_gt
    diff_abs = np.nanmax(np.abs(diff)) if np.any(~np.isnan(diff)) else 1.0
    diff_abs = max(float(diff_abs), 0.1)

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.6, 1.6, 1.6, 1.0])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a, sharey=ax_a)
    ax_d = fig.add_subplot(gs[2], sharex=ax_a, sharey=ax_a)
    ax_line = fig.add_subplot(gs[3])

    plot_route_heatmap(ax_a, pts, bin_segments, mean_gt,
                       f"{args.label_gt}: mean |lat| of 80-step predictions per "
                       f"{bin_m:.0f} m bin",
                       0.0, vmax, "viridis")
    plot_route_heatmap(ax_b, pts, bin_segments, mean_det,
                       f"{args.label_det}: mean |lat| of 80-step predictions per "
                       f"{bin_m:.0f} m bin",
                       0.0, vmax, "viridis")
    plot_route_heatmap(ax_d, pts, bin_segments, diff,
                       f"{args.label_det} − {args.label_gt}  "
                       f"(positive = model worse than GT)",
                       -diff_abs, diff_abs, "RdBu_r")

    for ax, vmin_, vmax_, cm in [
        (ax_a, 0.0, vmax, "viridis"),
        (ax_b, 0.0, vmax, "viridis"),
        (ax_d, -diff_abs, diff_abs, "RdBu_r"),
    ]:
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin_, vmax=vmax_))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("m", fontsize=8)

    ax_line.plot(bs_mid, mean_gt, label=f"{args.label_gt} mean |lat|", color="C0", lw=1.4)
    ax_line.plot(bs_mid, mean_det, label=f"{args.label_det} mean |lat|", color="C1", lw=1.4)
    ax_line.plot(bs_mid, max_gt, label=f"{args.label_gt} max |lat|", color="C0", lw=0.8, alpha=0.5)
    ax_line.plot(bs_mid, max_det, label=f"{args.label_det} max |lat|", color="C1", lw=0.8, alpha=0.5)
    ax_line.axhline(0.25, color="#888", lw=0.5, ls="--")
    ax_line.set_xlabel("Route arc length (m)")
    ax_line.set_ylabel("Prediction deviation from centerline (m)")
    ax_line.legend(fontsize=8, ncol=2)
    ax_line.grid(alpha=0.3)

    fig.suptitle(
        f"Per-prediction route deviation: {args.label_gt} vs {args.label_det}  "
        f"({len(npzs)} frames × 80 pred-steps, route {s_max:.0f} m, "
        f"{n_bins} bins × {bin_m:.0f} m)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"saved {args.output}")

    np.savez(args.output.with_suffix(".npz"),
             route_pts=pts, route_s=s, bin_m=bin_m, bin_s_mid=bs_mid,
             mean_abs_gt=mean_gt, mean_abs_det=mean_det,
             max_abs_gt=max_gt, max_abs_det=max_det,
             dev_series_gt=dev_gt, dev_series_det=dev_det,
             ego_arc_per_frame=np.array(ego_arcs))
    print(f"saved {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
