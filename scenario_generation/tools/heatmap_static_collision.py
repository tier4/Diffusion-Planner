#!/usr/bin/env python3
"""Static-collision clearance heatmap along a route.

Scores each sim step's model prediction against stopped neighbours via
``rlvr.reward.compute_static_collision_penalty`` and projects the
per-step min OBB clearance onto the route arc-length to produce a heatmap.

Supports single-run and two-run (A vs B) comparison modes.

Input: dumped NPZs from ``scenario_generation.replay`` with
``parked_vehicles_yaml`` or ``static_npc_count`` enabled, plus a model
checkpoint and reward config.

Usage:
    python -m scenario_generation.tools.heatmap_static_collision \\
        --route /path/to/route.pkl \\
        --run_a /path/to/run_a_dir \\
        --model_a /path/to/model_a.pth \\
        --config /path/to/reward_config.json \\
        --output /path/to/heatmap.png

    # Two-run comparison:
    python -m scenario_generation.tools.heatmap_static_collision \\
        --route /path/to/route.pkl \\
        --run_a /path/to/run_a_dir --model_a /path/to/model_a.pth \\
        --run_b /path/to/run_b_dir --model_b /path/to/model_b.pth \\
        --config /path/to/reward_config.json \\
        --output /path/to/heatmap.png
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scenario_generation.tools._heatmap_common import (
    bin_scalar_by_arc,
    build_route_polyline,
    load_route,
    plot_route_heatmap,
    recover_ego_world_pose_from_goal,
    segments_from_polyline,
)

_NPZ_RE = re.compile(r"replay_step_(\d+)\.npz$")


def _score_run_ego_actual(
    run_dir: Path,
    route,
    pts: np.ndarray,
    s: np.ndarray,
    stride: int = 1,
    max_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score based on actual ego pose vs parked neighbors. No model needed."""
    from diffusion_planner.model.guidance.collision import batch_signed_distance_rect

    from rlvr.reward import _closest_points_between_rects
    from scenario_generation.tools._heatmap_common import project_to_polyline

    def _build_obb_corners(cx, cy, cos_h, sin_h, length, width, wheelbase):
        """Build (1, 4, 2) OBB corners from center pose + dims."""
        rear_overhang = (length - wheelbase) / 2.0
        x0, x1 = -rear_overhang, length - rear_overhang
        y0, y1 = -width / 2.0, width / 2.0
        local = torch.tensor([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=torch.float32)
        R = torch.tensor([[cos_h, -sin_h], [sin_h, cos_h]], dtype=torch.float32)
        world = (R @ local.T).T + torch.tensor([cx, cy], dtype=torch.float32)
        return world.unsqueeze(0)

    npz_dir = run_dir / "npz"
    if not npz_dir.exists():
        npz_dir = run_dir
    npz_paths = sorted(
        p for p in npz_dir.glob("*.npz") if _NPZ_RE.search(p.name)
    )
    if stride > 1:
        npz_paths = npz_paths[::stride]
    if max_steps:
        npz_paths = npz_paths[:max_steps]
    if not npz_paths:
        raise SystemExit(f"No replay_step_*.npz under {npz_dir}")
    print(f"  {len(npz_paths)} steps to score (ego actual)")

    arc_positions = []
    min_clearances = []

    for i, path in enumerate(npz_paths):
        with np.load(path, allow_pickle=True) as raw:
            data_np = {k: raw[k] for k in raw.files if k != "version"}

        ex, ey, _ = recover_ego_world_pose_from_goal(data_np["goal_pose"], route)
        s_arc, _, _ = project_to_polyline(np.array([ex, ey]), pts, s)

        es = data_np["ego_shape"]
        if es.ndim == 2:
            es = es[0]
        wb, ego_len, ego_w = float(es[0]), float(es[1]), float(es[2])

        nb_past = data_np["neighbor_agents_past"]
        if nb_past.ndim == 3:
            nb_last = nb_past[:, -1, :]
        else:
            nb_last = nb_past[0, :, -1, :]
        valid = np.abs(nb_last[:, :2]).sum(axis=-1) > 1e-6
        if not valid.any():
            arc_positions.append(s_arc)
            min_clearances.append(99.0)
            continue

        nb_xy = nb_last[valid, :2]
        nb_cos = nb_last[valid, 2]
        nb_sin = nb_last[valid, 3]
        if nb_last.shape[-1] < 8:
            raise ValueError(
                f"neighbor_agents_past has {nb_last.shape[-1]} columns, "
                "expected >= 8 (x, y, cos, sin, vx, vy, width, length)"
            )
        nb_w = nb_last[valid, 6]
        nb_l = nb_last[valid, 7]

        ego_corners = _build_obb_corners(0.0, 0.0, 1.0, 0.0, ego_len, ego_w, wb)

        best_d = 99.0
        for j in range(len(nb_xy)):
            nx, ny = float(nb_xy[j, 0]), float(nb_xy[j, 1])
            nc, ns_ = float(nb_cos[j]), float(nb_sin[j])
            nw, nl = float(nb_w[j]), float(nb_l[j])
            if nw < 0.1 or nl < 0.1:
                raise ValueError(
                    f"Neighbor {j} has invalid dimensions (w={nw}, l={nl})"
                )
            npc_corners = _build_obb_corners(nx, ny, nc, ns_, nl, nw, nl * 0.65)

            pt1, pt2 = _closest_points_between_rects(ego_corners, npc_corners)
            d_val = float(torch.norm(pt1[0] - pt2[0]))

            sd = batch_signed_distance_rect(ego_corners, npc_corners)
            if float(sd[0]) < 0:
                d_val = float(sd[0])

            if d_val < best_d:
                best_d = d_val

        arc_positions.append(s_arc)
        min_clearances.append(best_d)

        if (i + 1) % 200 == 0:
            print(f"    scored {i+1}/{len(npz_paths)}")

    return np.array(arc_positions), np.array(min_clearances)


def _score_run(
    run_dir: Path,
    model_path: Path,
    route,
    reward_config_path: Path,
    pts: np.ndarray,
    s: np.ndarray,
    device: str,
    stride: int = 1,
    max_steps: int | None = None,
    inference_delay: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a run. Returns (arc_positions (M,), min_clearance (M,))."""
    from rlvr.autoresearch.tools.audit_static_collision import (
        _score_prediction,
    )
    from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
    from scenario_generation.npz_loader import from_npz
    from scenario_generation.simulate import _predict_batch, load_model
    from scenario_generation.tools._heatmap_common import project_to_polyline

    reward_cfg = load_reward_config(str(reward_config_path))
    if not reward_cfg.static_collision_enabled:
        raise SystemExit(
            f"reward config {reward_config_path} has static_collision_enabled=false. "
            "Set it to true."
        )

    npz_dir = run_dir / "npz"
    if not npz_dir.exists():
        npz_dir = run_dir
    npz_paths = sorted(
        p for p in npz_dir.glob("*.npz")
        if _NPZ_RE.search(p.name)
    )
    if stride > 1:
        npz_paths = npz_paths[::stride]
    if max_steps:
        npz_paths = npz_paths[:max_steps]
    if not npz_paths:
        raise SystemExit(f"No replay_step_*.npz under {npz_dir}")
    print(f"  {len(npz_paths)} steps to score")

    model, model_args = load_model(str(model_path), device)

    arc_positions = []
    min_clearances = []

    for i, path in enumerate(npz_paths):
        with np.load(path, allow_pickle=True) as raw:
            data_np = {k: raw[k] for k in raw.files if k != "version"}

        ex, ey, eyaw = recover_ego_world_pose_from_goal(data_np["goal_pose"], route)

        scene = from_npz(str(path))
        preds = _predict_batch(
            model, model_args, scene, [scene.ego_agent_id], device,
            inference_delay=inference_delay,
        )
        ego_pred = preds.get(scene.ego_agent_id)
        if ego_pred is None:
            continue

        result = _score_prediction(data_np, ego_pred, reward_cfg, device)

        s_arc, _, _ = project_to_polyline(np.array([ex, ey]), pts, s)
        arc_positions.append(s_arc)
        min_clearances.append(result["sc_min_dist"])

        if (i + 1) % 100 == 0:
            print(f"    scored {i+1}/{len(npz_paths)}")

    return np.array(arc_positions), np.array(min_clearances)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--route", type=Path, required=True)
    p.add_argument("--run_a", type=Path, required=True)
    p.add_argument("--model_a", type=Path, default=None)
    p.add_argument("--run_b", type=Path, default=None)
    p.add_argument("--model_b", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None,
                   help="Reward config JSON (required for --mode predicted)")
    p.add_argument("--mode", choices=["predicted", "ego_actual"], default="ego_actual",
                   help="'ego_actual' scores the real ego pose at each step (no model). "
                        "'predicted' scores the model's 80-step prediction (needs --model_a/b + --config).")
    p.add_argument("--label_a", default="A")
    p.add_argument("--label_b", default="B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bin_m", type=float, default=5.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--clip_max_m", type=float, default=5.0,
                   help="Clamp color scale at this clearance (m). Default 5.")
    p.add_argument("--min_arc_m", type=float, default=None)
    p.add_argument("--max_arc_m", type=float, default=None)
    p.add_argument("--inference_delay", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading route {args.route}")
    route = load_route(args.route)
    pts, s = build_route_polyline(route)
    s_max = float(s[-1])
    print(f"Route polyline: {len(pts)} pts, arc length {s_max:.1f} m")

    if args.mode == "predicted":
        if args.model_a is None or args.config is None:
            raise SystemExit("--mode predicted requires --model_a and --config")

    print(f"[{args.label_a}] scoring {args.run_a} (mode={args.mode})")
    if args.mode == "ego_actual":
        arc_a, clr_a = _score_run_ego_actual(
            args.run_a, route, pts, s, args.stride, args.max_steps,
        )
    else:
        arc_a, clr_a = _score_run(
            args.run_a, args.model_a, route, args.config,
            pts, s, device, args.stride, args.max_steps, args.inference_delay,
        )
    print(f"  {len(arc_a)} scored steps, clearance min={clr_a.min():.3f} "
          f"mean={clr_a.mean():.3f} p5={np.percentile(clr_a, 5):.3f}")

    has_b = args.run_b is not None
    if has_b:
        print(f"[{args.label_b}] scoring {args.run_b} (mode={args.mode})")
        if args.mode == "ego_actual":
            arc_b, clr_b = _score_run_ego_actual(
                args.run_b, route, pts, s, args.stride, args.max_steps,
            )
        else:
            if args.model_b is None:
                raise SystemExit("--mode predicted with --run_b requires --model_b")
            arc_b, clr_b = _score_run(
                args.run_b, args.model_b, route, args.config,
                pts, s, device, args.stride, args.max_steps, args.inference_delay,
            )
        print(f"  {len(arc_b)} scored steps, clearance min={clr_b.min():.3f} "
              f"mean={clr_b.mean():.3f} p5={np.percentile(clr_b, 5):.3f}")

    if args.min_arc_m is not None:
        mask_a = arc_a >= args.min_arc_m
        arc_a, clr_a = arc_a[mask_a], clr_a[mask_a]
        if has_b:
            mask_b = arc_b >= args.min_arc_m
            arc_b, clr_b = arc_b[mask_b], clr_b[mask_b]

    if args.max_arc_m is not None:
        mask_a = arc_a <= args.max_arc_m
        arc_a, clr_a = arc_a[mask_a], clr_a[mask_a]
        s_max = min(s_max, args.max_arc_m)
        if has_b:
            mask_b = arc_b <= args.max_arc_m
            arc_b, clr_b = arc_b[mask_b], clr_b[mask_b]

    bin_m = args.bin_m
    bs_mid_a, mean_a, min_a = bin_scalar_by_arc(arc_a, clr_a, s_max, bin_m)
    n_bins = len(bs_mid_a)
    bin_segments = segments_from_polyline(pts, s, bin_m, n_bins)

    vmax = args.clip_max_m

    if has_b:
        bs_mid_b, mean_b, min_b = bin_scalar_by_arc(arc_b, clr_b, s_max, bin_m)

        diff = mean_a - mean_b
        diff_abs = float(np.nanmax(np.abs(diff))) if np.any(~np.isnan(diff)) else 1.0
        diff_abs = max(diff_abs, 0.1)

        fig = plt.figure(figsize=(22, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.8, 1.0],
                              width_ratios=[1, 1, 1], hspace=0.25, wspace=0.15)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1], sharex=ax_a, sharey=ax_a)
        ax_d = fig.add_subplot(gs[0, 2], sharex=ax_a, sharey=ax_a)
        ax_line = fig.add_subplot(gs[1, :])

        plot_route_heatmap(ax_a, pts, bin_segments, mean_a,
                           f"{args.label_a}: mean clearance (m)", 0.0, vmax, "RdYlGn")
        plot_route_heatmap(ax_b, pts, bin_segments, mean_b,
                           f"{args.label_b}: mean clearance (m)", 0.0, vmax, "RdYlGn")
        plot_route_heatmap(ax_d, pts, bin_segments, diff,
                           f"A − B  (red = A closer, blue = B closer)",
                           -diff_abs, diff_abs, "RdBu")

        for ax, vmin_, vmax_, cm in [
            (ax_a, 0.0, vmax, "RdYlGn"),
            (ax_b, 0.0, vmax, "RdYlGn"),
            (ax_d, -diff_abs, diff_abs, "RdBu"),
        ]:
            sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin_, vmax=vmax_))
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.04, aspect=30)
            cb.ax.tick_params(labelsize=8)
            cb.set_label("m", fontsize=8)

        ax_line.plot(bs_mid_a, mean_a, label=f"{args.label_a} mean clr", color="C0", lw=1.4)
        ax_line.plot(bs_mid_b, mean_b, label=f"{args.label_b} mean clr", color="C1", lw=1.4)
        ax_line.plot(bs_mid_a, min_a, label=f"{args.label_a} min clr", color="C0", lw=0.8, alpha=0.5)
        ax_line.plot(bs_mid_b, min_b, label=f"{args.label_b} min clr", color="C1", lw=0.8, alpha=0.5)
        ax_line.axhline(0.2, color="#cc0000", lw=0.5, ls="--", label="cross (0.2m)")
        ax_line.axhline(0.4, color="#ff8800", lw=0.5, ls="--", label="near (0.4m)")
        ax_line.set_xlabel("Route arc length (m)")
        ax_line.set_ylabel("Min OBB clearance to stopped neighbor (m)")
        ax_line.legend(fontsize=8, ncol=3)
        ax_line.grid(alpha=0.3)

        fig.suptitle(
            f"Static collision clearance: {args.label_a} vs {args.label_b}  "
            f"({len(arc_a)} / {len(arc_b)} steps, route {s_max:.0f} m)",
            fontsize=11,
        )
    else:
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.25)
        ax_map = fig.add_subplot(gs[0])
        ax_line = fig.add_subplot(gs[1])

        plot_route_heatmap(ax_map, pts, bin_segments, mean_a,
                           f"{args.label_a}: mean clearance (m)", 0.0, vmax, "RdYlGn")
        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0.0, vmax=vmax))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax_map, pad=0.02, fraction=0.04, aspect=30)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("m", fontsize=8)

        ax_line.plot(bs_mid_a, mean_a, label="mean clr", color="C0", lw=1.4)
        ax_line.plot(bs_mid_a, min_a, label="min clr", color="C0", lw=0.8, alpha=0.5)
        ax_line.axhline(0.2, color="#cc0000", lw=0.5, ls="--", label="cross (0.2m)")
        ax_line.axhline(0.4, color="#ff8800", lw=0.5, ls="--", label="near (0.4m)")
        ax_line.set_xlabel("Route arc length (m)")
        ax_line.set_ylabel("Min OBB clearance to stopped neighbor (m)")
        ax_line.legend(fontsize=8)
        ax_line.grid(alpha=0.3)

        fig.suptitle(
            f"Static collision clearance: {args.label_a}  "
            f"({len(arc_a)} steps, route {s_max:.0f} m)",
            fontsize=11,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Saved {args.output}")

    np.savez(
        args.output.with_suffix(".npz"),
        arc_a=arc_a, clr_a=clr_a,
        **({"arc_b": arc_b, "clr_b": clr_b} if has_b else {}),
        route_pts=pts, route_s=s,
        bin_m=bin_m, bin_s_mid=bs_mid_a,
        mean_a=mean_a, min_a=min_a,
        **({"mean_b": mean_b, "min_b": min_b} if has_b else {}),
    )
    print(f"Saved {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
