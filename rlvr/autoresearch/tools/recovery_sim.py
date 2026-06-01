#!/usr/bin/env python3
"""Recovery sim: render sim-style PNGs of a closed-loop lateral-recovery rollout.

Mirrors the rendering of ``scenario_generation.replay`` (`save_step_figure`):
lane network, road borders, ego oriented box, predicted trajectory ahead,
and a body-to-border distance overlay. Operates in the **initial ego frame**
(the frame stored in the NPZ) so we don't need an OSM / lanelet2 lookup —
``data["lanes"]``, ``data["line_strings"]`` and ``data["route_lanes"]`` are
already in this frame.

Usage:

    python -m rlvr.autoresearch.tools.recovery_sim \\
        --scene .../replay_step_0681.npz \\
        --kind parallel --magnitude 0.5 --side + \\
        --model_path .../merged.pth \\
        --output_dir .../recovery_sim_imgs/scene_xxx_par+0.50 \\
        [--lora_path PATH] [--steps 80] [--make_webm]

Output:
    output_dir/step_0000.png ... step_NNNN.png
    output_dir/rollout.webm  (if --make_webm)
    output_dir/meta.json     (per-step ego pose, lateral offset, etc.)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import load_npz_data
from scenario_generation.visualize import draw_agent_box as _viz_draw_agent_box
from rlvr.autoresearch.tools.recovery_test import (
    _build_segments,
    _point_to_segments_dist,
    apply_combined_perturbation,
    apply_lateral_shift,
    apply_velocity_perturbation,
    apply_yaw_perturbation,
    deterministic_predict,
    get_tangent_at_origin,
    transform_to_new_ego_frame,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Match scenario_generation.replay color palette so PNGs look identical.
_LANE_COLOR = "#bbbbbb"
_LANE_BORDER_COLOR = "#888888"
_ROAD_BORDER_COLOR = "#dd2222"
_EGO_COLOR = "#3366cc"
_ROUTE_COLOR = "#3366cc"
_PRED_COLOR = "#3366cc"
_VIEW_HALF_M = 50.0


# ---------------------------------------------------------------------------
# Geometry helpers (mirror scenario_generation.replay & .visualize)
# ---------------------------------------------------------------------------


def _draw_agent_box(ax, x, y, heading, length, width, color,
                    alpha=0.85, lw=1.5, zorder=20, wheelbase=None) -> None:
    """OBB footprint at world (x, y, heading). Delegates to the central
    ``scenario_generation.visualize.draw_agent_box``: pass ``wheelbase`` for the
    ego (rear-axle convention, box offset forward); leave None for neighbors
    (centroid convention).
    """
    _viz_draw_agent_box(ax, x, y, heading, length, width, color,
                        alpha=alpha, lw=lw, zorder=zorder, wheelbase=wheelbase)


def _ego_obb_corners(ex, ey, heading, length, width) -> np.ndarray:
    """Four OBB corners (world frame) — used to attach the border-distance line."""
    rear_overhang = (length - length * 0.65) / 2
    x0, x1 = -rear_overhang, length - rear_overhang
    y0, y1 = -width / 2, width / 2
    local = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=np.float64)
    c, s = math.cos(heading), math.sin(heading)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (R @ local.T).T + np.array([ex, ey], dtype=np.float64)


def _lane_polylines(lanes: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Centerline + left/right border polylines from a [S, P, >=8] tensor."""
    centerlines, lefts, rights = [], [], []
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        pts = lane[:, :2]
        valid = np.abs(pts).sum(axis=1) > 0.1
        if valid.sum() < 2:
            continue
        centerlines.append(pts[valid])
        if lane.shape[1] > 7:
            lefts.append((pts + lane[:, 4:6])[valid])
            rights.append((pts + lane[:, 6:8])[valid])
    return centerlines, lefts, rights


def _road_border_polylines(line_strings: np.ndarray) -> list[np.ndarray]:
    """Extract road-border polylines (channel 3 = road_border one-hot) in
    initial ego frame from the NPZ ``line_strings`` tensor.
    """
    polylines = []
    for i in range(line_strings.shape[0]):
        ls = line_strings[i]  # [P, 4]
        if ls.shape[1] < 4:
            continue
        is_border = ls[:, 3] > 0.5
        coords = ls[:, :2]
        valid = (np.abs(coords).sum(axis=1) > 1e-3) & is_border
        if valid.sum() < 2:
            continue
        polylines.append(coords[valid])
    return polylines


def _route_polylines(route_lanes: np.ndarray) -> list[np.ndarray]:
    """Centerline polylines for the on-route lanes (highlighted)."""
    out = []
    for i in range(route_lanes.shape[0]):
        rl = route_lanes[i]  # [P, 33]
        pts = rl[:, :2]
        valid = np.abs(pts).sum(axis=1) > 0.1
        if valid.sum() >= 2:
            out.append(pts[valid])
    return out


def _nearest_border_point(
    probe_xy: np.ndarray, border_polylines: list[np.ndarray]
) -> np.ndarray | None:
    """Closest point on any border polyline to probe_xy (segment-wise)."""
    if not border_polylines:
        return None
    best_pt = None
    best_d = float("inf")
    px, py = float(probe_xy[0]), float(probe_xy[1])
    for pl in border_polylines:
        if pl.shape[0] < 2:
            continue
        p1 = pl[:-1].astype(np.float64)
        p2 = pl[1:].astype(np.float64)
        seg = p2 - p1
        seg_len2 = (seg * seg).sum(axis=1)
        seg_len2[seg_len2 < 1e-9] = 1e-9
        t = (((px - p1[:, 0]) * seg[:, 0] + (py - p1[:, 1]) * seg[:, 1])
             / seg_len2)
        t = np.clip(t, 0.0, 1.0)
        closest = p1 + t[:, None] * seg
        dists = np.hypot(closest[:, 0] - px, closest[:, 1] - py)
        idx = int(dists.argmin())
        if dists[idx] < best_d:
            best_d = float(dists[idx])
            best_pt = closest[idx]
    return best_pt


# ---------------------------------------------------------------------------
# Closed-loop rollout that ALSO records, at every step, the model's full
# 80-step plan re-expressed in the INITIAL ego frame ("world" for our viz).
# ---------------------------------------------------------------------------


def closed_loop_rollout_with_plans(
    model,
    model_args,
    init_data: dict[str, torch.Tensor],
    n_steps: int = 80,
    advance_k: int = 0,
    dt: float = 0.1,
) -> dict:
    """Run the closed-loop rollout and capture per-step predictions in the
    initial ego frame. Direct extension of recovery_test.closed_loop_rollout —
    forked because we need extra outputs (yaw + per-step world-frame plans).

    Returns dict:
        positions: [n_steps + 1, 3] (x, y, yaw) in initial ego frame
        plans_world: list of [T, 3] arrays (predicted (x, y, yaw) in initial frame)
                     length == n_steps. plans_world[i] is taken at step i.
        velocities: [n_steps + 1] speed in m/s estimated from finite diffs
    """
    data = {k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in init_data.items()}

    # Cumulative pose of "current" ego in the INITIAL frame
    cum_x, cum_y = 0.0, 0.0
    cum_cos, cum_sin = 1.0, 0.0

    positions = [np.array([0.0, 0.0, 0.0])]
    velocities = [0.0]
    plans_world: list[np.ndarray] = []

    # Initial speed from ego_current_state (channels 4=vx, 5=vy in current frame)
    ecs0 = data["ego_current_state"]
    if ecs0.dim() == 2:
        ecs0 = ecs0[0]
    init_speed = float(torch.linalg.vector_norm(ecs0[4:6]).item())
    velocities[0] = init_speed

    for step_i in range(n_steps):
        pred = deterministic_predict(model, model_args, data)  # [T, 4] in CURRENT frame

        # Re-express the ENTIRE plan in the INITIAL frame for visualization.
        # current frame -> initial frame: p_init = (cum_x, cum_y) + R(cum) @ p_cur
        cur_xy = pred[:, :2].astype(np.float64)
        wx = cum_x + cum_cos * cur_xy[:, 0] - cum_sin * cur_xy[:, 1]
        wy = cum_y + cum_sin * cur_xy[:, 0] + cum_cos * cur_xy[:, 1]
        # Heading in current frame -> initial frame
        cur_h = np.arctan2(pred[:, 3], pred[:, 2])
        cum_yaw = math.atan2(cum_sin, cum_cos)
        wh = (cur_h + cum_yaw).astype(np.float64)
        plans_world.append(np.stack([wx, wy, wh], axis=-1))

        if pred.shape[0] <= advance_k:
            advance_k = pred.shape[0] - 1

        nx_loc = float(pred[advance_k, 0])
        ny_loc = float(pred[advance_k, 1])
        ncos_loc = float(pred[advance_k, 2])
        nsin_loc = float(pred[advance_k, 3])
        norm = float(np.hypot(ncos_loc, nsin_loc)) or 1.0
        ncos_loc /= norm
        nsin_loc /= norm

        # Velocity estimate (used to keep ego_current_state.v fresh across steps).
        if advance_k + 1 < pred.shape[0]:
            dvx_loc = float(pred[advance_k + 1, 0] - pred[advance_k, 0]) / dt
            dvy_loc = float(pred[advance_k + 1, 1] - pred[advance_k, 1]) / dt
        else:
            dvx_loc = float(pred[advance_k, 0]) / dt
            dvy_loc = float(pred[advance_k, 1]) / dt
        new_vx = ncos_loc * dvx_loc + nsin_loc * dvy_loc
        new_vy = -nsin_loc * dvx_loc + ncos_loc * dvy_loc

        # Cumulative compose
        new_world_x = cum_x + cum_cos * nx_loc - cum_sin * ny_loc
        new_world_y = cum_y + cum_sin * nx_loc + cum_cos * ny_loc
        new_cum_cos = cum_cos * ncos_loc - cum_sin * nsin_loc
        new_cum_sin = cum_sin * ncos_loc + cum_cos * nsin_loc

        positions.append(np.array([new_world_x, new_world_y,
                                   math.atan2(new_cum_sin, new_cum_cos)]))
        velocities.append(float(np.hypot(new_vx, new_vy)))

        cum_x, cum_y, cum_cos, cum_sin = (
            new_world_x, new_world_y, new_cum_cos, new_cum_sin
        )

        # Roll past, append old origin so the model history stays consistent
        if "ego_agent_past" in data:
            eap = data["ego_agent_past"].clone()
            old_origin = torch.tensor([0.0, 0.0, 1.0, 0.0],
                                      dtype=eap.dtype, device=eap.device)
            T = eap.shape[1]
            eap = torch.cat([eap[:, 1:T],
                             old_origin.view(1, 1, 4).expand(eap.shape[0], 1, 4)], dim=1)
            data["ego_agent_past"] = eap

        # Re-express world in the new ego frame
        data = transform_to_new_ego_frame(data, nx_loc, ny_loc, ncos_loc, nsin_loc)
        if "ego_current_state" in data:
            ecs = data["ego_current_state"]
            ecs[..., 0] = 0.0
            ecs[..., 1] = 0.0
            ecs[..., 2] = 1.0
            ecs[..., 3] = 0.0
            ecs[..., 4] = float(new_vx)
            ecs[..., 5] = float(new_vy)
            data["ego_current_state"] = ecs

    return {
        "positions": np.stack(positions, axis=0),  # [N+1, 3]
        "plans_world": plans_world,                # len N
        "velocities": np.array(velocities),
    }


# ---------------------------------------------------------------------------
# Per-step rendering (mirrors scenario_generation.replay.save_step_figure)
# ---------------------------------------------------------------------------


def _render_step(
    output_path: Path,
    *,
    step: int,
    n_steps: int,
    ego_pose: np.ndarray,           # (3,) x, y, yaw in initial frame
    ego_speed: float,
    plan_world: np.ndarray,         # (T, 3) predicted future in initial frame
    centerlines: list[np.ndarray],
    lefts: list[np.ndarray],
    rights: list[np.ndarray],
    border_polylines: list[np.ndarray],
    route_polylines: list[np.ndarray],
    centerline_segments: np.ndarray,    # (N_seg, 2, 2) for nearest-distance label
    ego_length: float,
    ego_width: float,
    perturbation_label: str,
    init_lateral: float,
    view_half_m: float = _VIEW_HALF_M,
) -> None:
    """Render and save one step's overview PNG."""
    ex, ey, eh = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])

    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("#f8f8f8")

    # 1) Lane network (centerline + left/right markings, gray)
    if centerlines:
        ax.add_collection(LineCollection(
            centerlines, colors=_LANE_COLOR, linewidths=0.6,
            alpha=0.7 * 0.4, zorder=1,
        ))
    if lefts:
        ax.add_collection(LineCollection(
            lefts, colors=_LANE_BORDER_COLOR, linewidths=1.1,
            alpha=0.7, zorder=2,
        ))
    if rights:
        ax.add_collection(LineCollection(
            rights, colors=_LANE_BORDER_COLOR, linewidths=1.1,
            alpha=0.7, zorder=2,
        ))

    # 1b) Road borders (red) — AABB-filtered to the viewport
    half = view_half_m * 1.5
    filtered_borders = []
    for pl in border_polylines:
        if pl.shape[0] < 2:
            continue
        in_view = (
            (pl[:, 0] >= ex - half) & (pl[:, 0] <= ex + half)
            & (pl[:, 1] >= ey - half) & (pl[:, 1] <= ey + half)
        )
        if in_view.any():
            filtered_borders.append(pl)
    if filtered_borders:
        ax.add_collection(LineCollection(
            filtered_borders, colors=_ROAD_BORDER_COLOR, linewidths=2.0,
            alpha=0.9, zorder=5,
        ))

    # 2) Route polylines (blue)
    for pl in route_polylines:
        if pl.shape[0] >= 2:
            ax.plot(pl[:, 0], pl[:, 1], "-", color=_ROUTE_COLOR,
                    lw=2.5, alpha=0.6, zorder=3)

    # 3) Ego footprint + heading arrow
    _draw_agent_box(ax, ex, ey, eh, ego_length, ego_width, _EGO_COLOR,
                    alpha=0.85, lw=2, zorder=20, wheelbase=ego_length * 0.65)
    arrow_len = max(ego_length, 2.5)
    ax.annotate(
        "",
        xy=(ex + arrow_len * math.cos(eh), ey + arrow_len * math.sin(eh)),
        xytext=(ex, ey),
        arrowprops=dict(arrowstyle="-|>", color=_EGO_COLOR, lw=1.5,
                        mutation_scale=12),
        zorder=21,
    )

    # 4) Predicted plan ahead
    if plan_world is not None and plan_world.shape[0] > 1:
        ax.plot(plan_world[:, 0], plan_world[:, 1], "-",
                color=_PRED_COLOR, lw=1.8, alpha=0.6, zorder=25)
        ax.plot(plan_world[::3, 0], plan_world[::3, 1], "o",
                color=_PRED_COLOR, ms=2.5, alpha=0.8, mew=0, zorder=26)
        # End-of-plan footprint (faint)
        _draw_agent_box(
            ax, plan_world[-1, 0], plan_world[-1, 1], plan_world[-1, 2],
            ego_length, ego_width, _PRED_COLOR, alpha=0.25, lw=1.0, zorder=24,
            wheelbase=ego_length * 0.65,
        )

    # 5) Body-to-border distance overlay
    border_pt = _nearest_border_point(np.array([ex, ey]), border_polylines)
    if border_pt is not None:
        corners = _ego_obb_corners(ex, ey, eh, ego_length, ego_width)
        d_corner = np.hypot(corners[:, 0] - border_pt[0],
                            corners[:, 1] - border_pt[1])
        start = corners[int(d_corner.argmin())]
        body_d = float(np.hypot(start[0] - border_pt[0], start[1] - border_pt[1]))
        ax.plot([start[0], border_pt[0]], [start[1], border_pt[1]],
                "k--", linewidth=1.3, alpha=0.7, zorder=29)
        ax.plot(border_pt[0], border_pt[1], "ko", markersize=6, zorder=30,
                markeredgecolor="white", markeredgewidth=0.8)
        mx, my = (start[0] + border_pt[0]) / 2, (start[1] + border_pt[1]) / 2
        ax.annotate(
            f"{body_d:.2f} m",
            xy=(mx, my), fontsize=8, color="black",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="black", alpha=0.7),
            zorder=31,
        )

    # 6) Lateral offset to centerline
    cur_lateral = float(_point_to_segments_dist(
        np.array([[ex, ey]]), centerline_segments
    )[0]) if centerline_segments.shape[0] else float("nan")

    # 7) Viewport
    ax.set_xlim(ex - view_half_m, ex + view_half_m)
    ax.set_ylim(ey - view_half_m, ey + view_half_m)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("X (m, initial ego frame)")
    ax.set_ylabel("Y (m, initial ego frame)")

    title = (
        f"Step {step:04d}/{n_steps}  t={step * 0.1:.1f}s  "
        f"perturb={perturbation_label}\n"
        f"ego  v={ego_speed:.1f} m/s ({ego_speed * 3.6:.0f} km/h)  "
        f"yaw={math.degrees(eh):+.1f}°  "
        f"lat_off={cur_lateral:.2f} m  (init {init_lateral:.2f} m)"
    )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    fig.clf()


# ---------------------------------------------------------------------------
# Perturbation dispatch
# ---------------------------------------------------------------------------


def _apply_perturbation(
    data: dict[str, torch.Tensor],
    n_unit: np.ndarray,
    kind: str,
    magnitude: float,
    side: float,
    combined_yaw_deg: float = 5.0,
) -> tuple[dict[str, torch.Tensor], str]:
    """Apply the requested perturbation; returns (perturbed_data, label)."""
    if kind == "parallel":
        signed = side * magnitude
        return apply_lateral_shift(data, n_unit, signed), \
               f"parallel {signed:+.2f} m"
    if kind == "yaw":
        signed = side * magnitude
        return apply_yaw_perturbation(data, np.deg2rad(signed)), \
               f"yaw {signed:+.1f}°"
    if kind == "velocity":
        scale = 1.0 + (side * magnitude / 100.0)
        return apply_velocity_perturbation(data, scale), \
               f"velocity x{scale:.2f}"
    if kind == "combined":
        signed_off = side * magnitude
        signed_yaw = side * combined_yaw_deg
        return apply_combined_perturbation(
            data, n_unit, signed_off, np.deg2rad(signed_yaw)
        ), f"combined {signed_off:+.2f} m & {signed_yaw:+.1f}°"
    raise ValueError(f"Unknown perturbation kind: {kind}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to a single NPZ scene")
    parser.add_argument("--kind", type=str, default="parallel",
                        choices=["parallel", "yaw", "velocity", "combined"])
    parser.add_argument("--magnitude", type=float, default=0.5,
                        help="Magnitude (m for parallel/combined-offset, deg for yaw, "
                             "pct for velocity)")
    parser.add_argument("--side", type=str, default="+", choices=["+", "-"])
    parser.add_argument("--combined_yaw_deg", type=float, default=5.0)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--advance_k", type=int, default=0)
    parser.add_argument("--ego_length", type=float, default=7.2369)
    parser.add_argument("--ego_width", type=float, default=2.29156)
    parser.add_argument("--view_half_m", type=float, default=_VIEW_HALF_M)
    parser.add_argument("--make_webm", action="store_true",
                        help="Encode the PNG sequence as a vp9 WebM at 10 fps")
    parser.add_argument("--webm_fps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(DEVICE)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model + optional LoRA ----
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

    # ---- Load scene ----
    data = load_npz_data(args.scene, device)

    # Tangent / normal at origin so we know the lateral direction for the shift
    t_unit, n_unit = get_tangent_at_origin(data["route_lanes"])

    # Apply the perturbation
    side_val = +1.0 if args.side == "+" else -1.0
    shifted, perturbation_label = _apply_perturbation(
        data, n_unit, args.kind, args.magnitude, side_val,
        combined_yaw_deg=args.combined_yaw_deg,
    )

    # Run the rollout (capturing per-step plans in the initial frame)
    rollout = closed_loop_rollout_with_plans(
        model, model_args, shifted,
        n_steps=args.steps, advance_k=args.advance_k,
    )
    positions = rollout["positions"]            # [N+1, 3]
    plans_world = rollout["plans_world"]        # len N
    velocities = rollout["velocities"]          # [N+1]

    # Render-time scene geometry pulled from the SHIFTED data so the
    # perturbation is visible (lanes/borders translate with the shift). For
    # parallel/yaw/velocity the rendered map is centered on the perturbed
    # origin, which is exactly what the model "sees".
    def _to_np(t: torch.Tensor) -> np.ndarray:
        x = t
        if x.dim() == 4:
            x = x[0]
        elif x.dim() == 3:
            x = x[0]
        return x.detach().cpu().numpy()

    # NB: shifted "lanes"/"line_strings"/"route_lanes" stay rank-3/4 with batch
    # 0; squeeze before rendering.
    lanes_np = shifted["lanes"][0].detach().cpu().numpy()
    line_strings_np = shifted["line_strings"][0].detach().cpu().numpy()
    route_lanes_np = shifted["route_lanes"][0].detach().cpu().numpy()

    centerlines, lefts, rights = _lane_polylines(lanes_np)
    border_polylines = _road_border_polylines(line_strings_np)
    route_polylines = _route_polylines(route_lanes_np)

    # Build centerline-segment geometry once for fast per-step lateral lookup.
    # Re-uses recovery_test._build_segments which scans route_lanes [..., 0:2].
    centerline_segments = _build_segments(shifted["route_lanes"])

    # Initial lateral (t=0) — should match |magnitude| for parallel kinds.
    init_lateral = float(_point_to_segments_dist(
        np.array([[0.0, 0.0]]), centerline_segments
    )[0]) if centerline_segments.shape[0] else float("nan")

    # ---- Render every step ----
    n_render = len(plans_world)
    for i in range(n_render):
        out_path = out_dir / f"step_{i:04d}.png"
        _render_step(
            out_path,
            step=i,
            n_steps=n_render,
            ego_pose=positions[i],
            ego_speed=float(velocities[i]),
            plan_world=plans_world[i],
            centerlines=centerlines,
            lefts=lefts,
            rights=rights,
            border_polylines=border_polylines,
            route_polylines=route_polylines,
            centerline_segments=centerline_segments,
            ego_length=args.ego_length,
            ego_width=args.ego_width,
            perturbation_label=perturbation_label,
            init_lateral=init_lateral,
            view_half_m=args.view_half_m,
        )
        if (i + 1) % 10 == 0 or i == 0:
            ex, ey, _ = positions[i]
            print(f"  step {i:04d}/{n_render}  ego=({ex:.2f},{ey:.2f})  "
                  f"v={velocities[i]:.2f} m/s")

    # ---- Meta JSON ----
    lat_per_step = _point_to_segments_dist(
        positions[:, :2], centerline_segments
    ) if centerline_segments.shape[0] else np.full(positions.shape[0], np.nan)
    meta = {
        "scene": str(args.scene),
        "model_path": str(args.model_path),
        "lora_path": str(args.lora_path) if args.lora_path else None,
        "perturbation": {
            "kind": args.kind,
            "magnitude": float(args.magnitude),
            "side": args.side,
            "combined_yaw_deg": float(args.combined_yaw_deg),
            "label": perturbation_label,
        },
        "tangent_unit": [float(t_unit[0]), float(t_unit[1])],
        "normal_unit": [float(n_unit[0]), float(n_unit[1])],
        "init_lateral": float(init_lateral),
        "n_steps": int(args.steps),
        "advance_k": int(args.advance_k),
        "ego_dims": {
            "length": float(args.ego_length),
            "width": float(args.ego_width),
        },
        "lateral_per_step": [float(x) for x in lat_per_step.tolist()],
        "trajectory_xy_yaw": [
            [float(p[0]), float(p[1]), float(p[2])] for p in positions.tolist()
        ],
        "velocity_per_step": [float(v) for v in velocities.tolist()],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- WebM (vp9) ----
    if args.make_webm:
        webm_path = out_dir / "rollout.webm"
        # libvpx-vp9 with crf 32 — same recipe used elsewhere in our pipelines.
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.webm_fps),
            "-i", str(out_dir / "step_%04d.png"),
            "-c:v", "libvpx-vp9",
            "-b:v", "0", "-crf", "32",
            "-pix_fmt", "yuv420p",
            str(webm_path),
        ]
        print(f"\n[recovery_sim] encoding webm: {webm_path}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  wrote {webm_path}  ({webm_path.stat().st_size / 1024:.0f} KiB)")
        except subprocess.CalledProcessError as e:
            print(f"  ffmpeg failed: {e.stderr.decode(errors='replace')[:500]}")

    final_lat = float(lat_per_step[-1]) if lat_per_step.size else float("nan")
    print(
        f"\n[recovery_sim] {Path(args.scene).stem}  "
        f"perturb={perturbation_label}  "
        f"init_lat={init_lateral:.2f} m  final_lat={final_lat:.2f} m  "
        f"frames={n_render}  -> {out_dir}"
    )


if __name__ == "__main__":
    main()
