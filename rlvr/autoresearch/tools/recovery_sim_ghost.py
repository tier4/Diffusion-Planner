#!/usr/bin/env python3
"""Ghost-overlay closed-loop sim: baseline + PRiSM on the same perturbed scene.

For one source NPZ + perturbation (kind/magnitude/side), runs the 8-second
closed-loop rollout under TWO different models (typically LoRA-less baseline
vs PRiSM-trained LoRA), and writes per-step PNGs that show:

  * lane network, road borders, route polylines (from the perturbed t=0 frame)
  * BOTH ego footprints at the current step, in different colors, plus their
    cumulative trails
  * each model's predicted 80-step plan ahead, in matching faded colors
  * title with per-step lateral offsets for both models

Optional --make_webm assembles the per-step PNGs into a WebM clip.

Usage:
    python -m rlvr.autoresearch.tools.recovery_sim_ghost \\
        --scene /path/to/replay_step_NNNN.npz \\
        --kind parallel --magnitude 1.0 --side - \\
        --baseline_model /path/to/<baseline>/best_model.pth \\
        --prism_model    /path/to/<prism>/best_model.pth \\
        --output_dir /path/out --steps 80 [--make_webm]
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import load_npz_data

# Reuse all the heavy lifting from recovery_sim
from rlvr.autoresearch.tools.recovery_sim import (
    _build_segments,
    _draw_agent_box,
    _ego_obb_corners,
    _lane_polylines,
    _nearest_border_point,
    _point_to_segments_dist,
    _road_border_polylines,
    _route_polylines,
    _apply_perturbation,
    _LANE_BORDER_COLOR,
    _LANE_COLOR,
    _ROAD_BORDER_COLOR,
    _ROUTE_COLOR,
    _VIEW_HALF_M,
    closed_loop_rollout_with_plans,
)
from rlvr.autoresearch.tools.recovery_test import get_tangent_at_origin


_BASELINE_COLOR = "#1f77b4"   # blue
_PRISM_COLOR    = "#d62728"   # red


def _load_model(model_path: str, lora_path: str | None, device):
    model_dir = Path(model_path).parent
    args_path = model_dir / "args.json"
    if not args_path.exists():
        args_path = model_dir.parent / "args.json"
    margs = Config(str(args_path))
    model = Diffusion_Planner(margs)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    if lora_path:
        model = load_lora_checkpoint(model, lora_path)
        model.eval()
    return model, margs


def _render_ghost_step(
    output_path: Path, step: int, n_steps: int,
    bl_pose: np.ndarray, bl_speed: float, bl_plan: np.ndarray | None,
    pr_pose: np.ndarray, pr_speed: float, pr_plan: np.ndarray | None,
    centerlines, lefts, rights, border_polylines, route_polylines,
    centerline_segments,
    ego_length: float, ego_width: float,
    perturbation_label: str, init_lateral: float,
    view_half_m: float = _VIEW_HALF_M,
) -> None:
    bx, by, bh = float(bl_pose[0]), float(bl_pose[1]), float(bl_pose[2])
    px, py, ph = float(pr_pose[0]), float(pr_pose[1]), float(pr_pose[2])
    cx, cy = (bx + px) / 2, (by + py) / 2

    fig = Figure(figsize=(11, 11))
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("#f8f8f8")

    # Lane network
    if centerlines:
        ax.add_collection(LineCollection(
            centerlines, colors=_LANE_COLOR, linewidths=0.6, alpha=0.28, zorder=1))
    if lefts:
        ax.add_collection(LineCollection(
            lefts, colors=_LANE_BORDER_COLOR, linewidths=1.1, alpha=0.7, zorder=2))
    if rights:
        ax.add_collection(LineCollection(
            rights, colors=_LANE_BORDER_COLOR, linewidths=1.1, alpha=0.7, zorder=2))

    # Road borders (AABB-filtered)
    half = view_half_m * 1.5
    flt_borders = [pl for pl in border_polylines
                   if pl.shape[0] >= 2 and (
                       (pl[:, 0] >= cx - half) & (pl[:, 0] <= cx + half)
                       & (pl[:, 1] >= cy - half) & (pl[:, 1] <= cy + half)).any()]
    if flt_borders:
        ax.add_collection(LineCollection(
            flt_borders, colors=_ROAD_BORDER_COLOR, linewidths=2.0, alpha=0.9, zorder=5))

    # Route polylines
    for pl in route_polylines:
        if pl.shape[0] >= 2:
            ax.plot(pl[:, 0], pl[:, 1], "-", color=_ROUTE_COLOR,
                    lw=2.5, alpha=0.55, zorder=3)

    # Plans (faded thin)
    if bl_plan is not None and bl_plan.shape[0] > 1:
        ax.plot(bl_plan[:, 0], bl_plan[:, 1], "-",
                color=_BASELINE_COLOR, lw=1.4, alpha=0.45, zorder=24)
    if pr_plan is not None and pr_plan.shape[0] > 1:
        ax.plot(pr_plan[:, 0], pr_plan[:, 1], "-",
                color=_PRISM_COLOR, lw=1.4, alpha=0.45, zorder=24)

    # Ego footprints + arrows
    _draw_agent_box(ax, bx, by, bh, ego_length, ego_width,
                    _BASELINE_COLOR, alpha=0.78, lw=2, zorder=20)
    _draw_agent_box(ax, px, py, ph, ego_length, ego_width,
                    _PRISM_COLOR, alpha=0.78, lw=2, zorder=21)
    al = max(ego_length, 2.5)
    ax.annotate("", xy=(bx + al * math.cos(bh), by + al * math.sin(bh)),
                xytext=(bx, by),
                arrowprops=dict(arrowstyle="-|>", color=_BASELINE_COLOR,
                                lw=1.2, mutation_scale=10), zorder=22)
    ax.annotate("", xy=(px + al * math.cos(ph), py + al * math.sin(ph)),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color=_PRISM_COLOR,
                                lw=1.2, mutation_scale=10), zorder=23)

    # Per-model lateral offsets
    cur_lat_b = float(_point_to_segments_dist(
        np.array([[bx, by]]), centerline_segments)[0]) if centerline_segments.shape[0] else float("nan")
    cur_lat_p = float(_point_to_segments_dist(
        np.array([[px, py]]), centerline_segments)[0]) if centerline_segments.shape[0] else float("nan")

    # Legend
    ax.plot([], [], "-", color=_BASELINE_COLOR, lw=2,
            label=f"baseline (LoRA-less)  v={bl_speed:.1f} m/s  lat={cur_lat_b:.2f}m")
    ax.plot([], [], "-", color=_PRISM_COLOR, lw=2,
            label=f"PRiSM  v={pr_speed:.1f} m/s  lat={cur_lat_p:.2f}m")
    ax.legend(fontsize=9, loc="upper left")

    ax.set_xlim(cx - view_half_m, cx + view_half_m)
    ax.set_ylim(cy - view_half_m, cy + view_half_m)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("X (m, initial ego frame)")
    ax.set_ylabel("Y (m, initial ego frame)")
    ax.set_title(
        f"Step {step:04d}/{n_steps}  t={step * 0.1:.1f}s  perturb={perturbation_label}  init lat={init_lateral:.2f}m\n"
        f"baseline lat={cur_lat_b:.2f}m   PRiSM lat={cur_lat_p:.2f}m   "
        f"Δ={cur_lat_b - cur_lat_p:+.2f}m (positive = PRiSM closer to centerline)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    fig.clf()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--kind", type=str, default="parallel",
                        choices=["parallel", "yaw", "velocity", "combined"])
    parser.add_argument("--magnitude", type=float, default=1.0)
    parser.add_argument("--side", type=str, default="-", choices=["+", "-"])
    parser.add_argument("--combined_yaw_deg", type=float, default=5.0)
    parser.add_argument("--baseline_model", type=str, required=True)
    parser.add_argument("--baseline_lora", type=str, default=None)
    parser.add_argument("--prism_model", type=str, required=True)
    parser.add_argument("--prism_lora", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--advance_k", type=int, default=0)
    parser.add_argument("--ego_length", type=float, default=7.2369)
    parser.add_argument("--ego_width", type=float, default=2.29156)
    parser.add_argument("--view_half_m", type=float, default=_VIEW_HALF_M)
    parser.add_argument("--make_webm", action="store_true")
    parser.add_argument("--webm_fps", type=int, default=10)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ghost-sim] loading 2 models...")
    bl_model, bl_args = _load_model(args.baseline_model, args.baseline_lora, device)
    pr_model, pr_args = _load_model(args.prism_model,    args.prism_lora,    device)

    # Load + perturb
    data = load_npz_data(args.scene, device)
    t_unit, n_unit = get_tangent_at_origin(data["route_lanes"])
    side_val = +1.0 if args.side == "+" else -1.0
    perturbed, label = _apply_perturbation(
        data, n_unit, args.kind, args.magnitude, side_val,
        combined_yaw_deg=args.combined_yaw_deg,
    )
    init_lat = float(_point_to_segments_dist(
        np.array([[0.0, 0.0]]), _build_segments(perturbed["route_lanes"]))[0])
    perturb_label = label
    print(f"  perturbation: {perturb_label}  init_lat={init_lat:.2f}m")

    # Two rollouts
    print(f"[ghost-sim] rollout (baseline)...")
    bl = closed_loop_rollout_with_plans(
        bl_model, bl_args, perturbed,
        n_steps=args.steps, advance_k=args.advance_k,
    )
    print(f"[ghost-sim] rollout (PRiSM)...")
    pr = closed_loop_rollout_with_plans(
        pr_model, pr_args, perturbed,
        n_steps=args.steps, advance_k=args.advance_k,
    )

    # Pre-compute scene polylines from perturbed (matches recovery_sim)
    rl = perturbed["route_lanes"]
    if rl.dim() == 4: rl = rl[0]
    lanes = perturbed.get("lanes")
    if lanes is not None and lanes.dim() == 4: lanes = lanes[0]
    line_strings = perturbed.get("line_strings")
    if line_strings is not None and line_strings.dim() == 4: line_strings = line_strings[0]
    centerlines, lefts, rights = _lane_polylines(lanes.cpu().numpy() if lanes is not None else None)
    border_polylines = _road_border_polylines(line_strings.cpu().numpy() if line_strings is not None else None)
    route_polylines = _route_polylines(rl.cpu().numpy())
    centerline_segments = _build_segments(perturbed["route_lanes"])

    # Render per-step PNGs
    n = args.steps
    print(f"[ghost-sim] rendering {n + 1} frames...")
    for step_i in range(n + 1):
        bl_plan = bl["plans_world"][step_i] if step_i < len(bl["plans_world"]) else None
        pr_plan = pr["plans_world"][step_i] if step_i < len(pr["plans_world"]) else None
        _render_ghost_step(
            out / f"ghost_step_{step_i:04d}.png",
            step=step_i, n_steps=n,
            bl_pose=bl["positions"][step_i],   bl_speed=float(bl["velocities"][step_i]),
            bl_plan=bl_plan,
            pr_pose=pr["positions"][step_i],   pr_speed=float(pr["velocities"][step_i]),
            pr_plan=pr_plan,
            centerlines=centerlines, lefts=lefts, rights=rights,
            border_polylines=border_polylines, route_polylines=route_polylines,
            centerline_segments=centerline_segments,
            ego_length=args.ego_length, ego_width=args.ego_width,
            perturbation_label=perturb_label, init_lateral=init_lat,
            view_half_m=args.view_half_m,
        )

    if args.make_webm:
        webm = out / "ghost_sim.webm"
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.webm_fps),
            "-i", str(out / "ghost_step_%04d.png"),
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p",
            "-b:v", "0", "-crf", "30",
            str(webm),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[ghost-sim] WebM: {webm}")
        except Exception as e:
            print(f"[ghost-sim] ffmpeg failed: {e}")

    print(f"\nDone — {out} ({n + 1} frames)")


if __name__ == "__main__":
    main()
