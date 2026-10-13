"""Shared library for dual-model ghost overlay closed-loop simulation.

Provides model loading, neighbor extraction, scene polyline extraction,
per-step rendering, and webm assembly. Used by recovery_sim_ghost.py
(PRiSM lane-keeping) and compare_models_ghost.py (generic A/B comparison).
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

from preference_optimization.lora_utils import load_lora_checkpoint
from rlvr.autoresearch.tools.recovery_sim import (
    _LANE_BORDER_COLOR,
    _LANE_COLOR,
    _ROAD_BORDER_COLOR,
    _ROUTE_COLOR,
    _VIEW_HALF_M,
    _build_segments,
    _draw_agent_box,
    _lane_polylines,
    _point_to_segments_dist,
    _road_border_polylines,
    _route_polylines,
    closed_loop_rollout_with_plans,
)
from scenario_generation.canvas import Figure

MODEL_A_COLOR = "#1f77b4"  # blue
MODEL_B_COLOR = "#d62728"  # red
_NB_COLOR = "#cc6600"


@dataclass
class GhostSimConfig:
    model_a_label: str = "model A"
    model_b_label: str = "model B"
    model_a_color: str = MODEL_A_COLOR
    model_b_color: str = MODEL_B_COLOR
    view_half_m: float = _VIEW_HALF_M
    ego_length: float = 7.2369
    ego_width: float = 2.29156
    ego_wheelbase: float = 4.76
    steps: int = 80
    advance_k: int = 0
    webm_fps: int = 10
    # Render this many recorded-history frames (gray ego from ego_agent_past)
    # BEFORE the closed-loop frames — the model-context preamble.
    hist_steps: int = 0
    subtitle: str = ""
    show_lateral: bool = True


def load_model(model_path: str, lora_path: str | None, device):
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


def extract_stopped_neighbors(npz_path: str) -> list[tuple[float, float, float, float, float]]:
    """Return list of (x, y, heading, length, width) for stopped neighbors."""
    data_np = dict(np.load(npz_path, allow_pickle=True))
    boxes = []
    if "neighbor_agents_past" not in data_np or "neighbor_agents_future" not in data_np:
        return boxes
    nb_past = data_np["neighbor_agents_past"]
    if nb_past.ndim == 4:
        nb_past = nb_past[0]
    nb_fut = data_np["neighbor_agents_future"]
    if nb_fut.ndim == 4:
        nb_fut = nb_fut[0]
    for i in range(nb_past.shape[0]):
        xy0 = nb_past[i, -1, :2]
        if abs(xy0[0]) + abs(xy0[1]) < 1e-6:
            continue
        fut_xy = nb_fut[i, :, :2]
        fut_valid = np.abs(fut_xy).sum(axis=-1) > 1e-6
        disp = (
            0.0
            if fut_valid.sum() < 2
            else float(np.linalg.norm(fut_xy[fut_valid].max(0) - fut_xy[fut_valid].min(0)))
        )
        if disp >= 0.5:
            continue
        w = float(nb_past[i, -1, 6])
        length = float(nb_past[i, -1, 7])
        if w < 0.1 or length < 0.1:
            continue
        h = float(math.atan2(nb_past[i, -1, 3], nb_past[i, -1, 2]))
        boxes.append((float(xy0[0]), float(xy0[1]), h, length, w))
    return boxes


def extract_scene_polylines(data: dict[str, torch.Tensor]):
    """Extract lane, border, route polylines and centerline segments from scene data."""
    rl = data["route_lanes"]
    if rl.dim() == 4:
        rl = rl[0]
    lanes = data.get("lanes")
    if lanes is not None and lanes.dim() == 4:
        lanes = lanes[0]
    line_strings = data.get("line_strings")
    if line_strings is not None and line_strings.dim() == 4:
        line_strings = line_strings[0]
    if lanes is None:
        raise ValueError("Scene data missing 'lanes' — NPZ is incomplete")
    if line_strings is None:
        raise ValueError("Scene data missing 'line_strings' — NPZ is incomplete")
    centerlines, lefts, rights = _lane_polylines(lanes.cpu().numpy())
    border_polylines = _road_border_polylines(line_strings.cpu().numpy())
    route_polylines = _route_polylines(rl.cpu().numpy())
    centerline_segments = _build_segments(data["route_lanes"])
    return centerlines, lefts, rights, border_polylines, route_polylines, centerline_segments


def render_ghost_step(
    output_path: Path,
    step: int,
    n_steps: int,
    a_pose: np.ndarray,
    a_speed: float,
    a_plan: np.ndarray | None,
    b_pose: np.ndarray,
    b_speed: float,
    b_plan: np.ndarray | None,
    centerlines,
    lefts,
    rights,
    border_polylines,
    route_polylines,
    centerline_segments,
    cfg: GhostSimConfig,
    neighbor_boxes: list[tuple[float, float, float, float, float]] | None = None,
    extra_title: str = "",
    history_mode: bool = False,
    history_trail: np.ndarray | None = None,
    fan_plans: list[np.ndarray] | None = None,
) -> None:
    ax_val, ay_val, ah_val = float(a_pose[0]), float(a_pose[1]), float(a_pose[2])
    bx_val, by_val, bh_val = float(b_pose[0]), float(b_pose[1]), float(b_pose[2])
    cx, cy = (ax_val + bx_val) / 2, (ay_val + by_val) / 2

    fig = Figure((11, 11), 100, facecolor="#f8f8f8")
    ax = fig.panel(
        (0.08, 0.06, 0.9, 0.85),
        xlim=(cx - cfg.view_half_m, cx + cfg.view_half_m),
        ylim=(cy - cfg.view_half_m, cy + cfg.view_half_m),
        equal=True,
    )

    if centerlines:
        ax.lines(centerlines, _LANE_COLOR, width=0.6, alpha=0.28, z=1)
    if lefts:
        ax.lines(lefts, _LANE_BORDER_COLOR, width=1.1, alpha=0.7, z=2)
    if rights:
        ax.lines(rights, _LANE_BORDER_COLOR, width=1.1, alpha=0.7, z=2)

    half = cfg.view_half_m * 1.5
    flt_borders = [
        pl
        for pl in border_polylines
        if pl.shape[0] >= 2
        and (
            (pl[:, 0] >= cx - half)
            & (pl[:, 0] <= cx + half)
            & (pl[:, 1] >= cy - half)
            & (pl[:, 1] <= cy + half)
        ).any()
    ]
    if flt_borders:
        ax.lines(flt_borders, _ROAD_BORDER_COLOR, width=2.0, alpha=0.9, z=5)

    for pl in route_polylines:
        if pl.shape[0] >= 2:
            ax.lines([pl[:, :2]], _ROUTE_COLOR, width=2.5, alpha=0.55, z=3)

    if neighbor_boxes:
        for nx, ny, nh, nl, nw in neighbor_boxes:
            _draw_agent_box(ax, nx, ny, nh, nl, nw, _NB_COLOR, alpha=0.75, lw=1.5, zorder=14)

    if history_mode:
        # Single gray recorded-history ego (model context preamble)
        if history_trail is not None and history_trail.shape[0] > 1:
            ax.lines([history_trail[:, :2]], "#555555", width=1.6, alpha=0.7, z=18)
        _draw_agent_box(
            ax,
            ax_val,
            ay_val,
            ah_val,
            cfg.ego_length,
            cfg.ego_width,
            "#555555",
            alpha=0.8,
            lw=2,
            zorder=20,
            wheelbase=cfg.ego_wheelbase,
        )
        ax.lines([], "#555555", width=2, label="ego history (recorded)")
        ax.legend("upper left")
        ax.grid(0.15)
        title = cfg.subtitle or ""
        if extra_title:
            title = f"{title}\n{extra_title}" if title else extra_title
        ax.title(title, 11)
        fig.save(output_path)
        return

    if fan_plans:
        for fi, fp in enumerate(fan_plans):
            ax.lines(
                [fp[:, :2]],
                cfg.model_b_color,
                width=0.8,
                alpha=0.16,
                z=22,
                label=(f"{len(fan_plans)} candidates from policy dist" if fi == 0 else None),
            )
    if a_plan is not None and a_plan.shape[0] > 1:
        ax.lines([a_plan[:, :2]], cfg.model_a_color, width=1.4, alpha=0.45, z=24)
    if b_plan is not None and b_plan.shape[0] > 1:
        ax.lines([b_plan[:, :2]], cfg.model_b_color, width=1.4, alpha=0.45, z=24)

    _draw_agent_box(
        ax,
        ax_val,
        ay_val,
        ah_val,
        cfg.ego_length,
        cfg.ego_width,
        cfg.model_a_color,
        alpha=0.78,
        lw=2,
        zorder=20,
        wheelbase=cfg.ego_wheelbase,
    )
    _draw_agent_box(
        ax,
        bx_val,
        by_val,
        bh_val,
        cfg.ego_length,
        cfg.ego_width,
        cfg.model_b_color,
        alpha=0.78,
        lw=2,
        zorder=21,
        wheelbase=cfg.ego_wheelbase,
    )
    arrow_len = max(cfg.ego_length, 2.5)
    ax.arrows(
        np.array([ax_val, ay_val]),
        np.array([ax_val + arrow_len * math.cos(ah_val), ay_val + arrow_len * math.sin(ah_val)]),
        cfg.model_a_color,
        width=1.2,
        head=10,
        z=22,
    )
    ax.arrows(
        np.array([bx_val, by_val]),
        np.array([bx_val + arrow_len * math.cos(bh_val), by_val + arrow_len * math.sin(bh_val)]),
        cfg.model_b_color,
        width=1.2,
        head=10,
        z=23,
    )

    label_a = f"{cfg.model_a_label}  v={a_speed:.1f} m/s"
    label_b = f"{cfg.model_b_label}  v={b_speed:.1f} m/s"
    if cfg.show_lateral and centerline_segments.shape[0]:
        lat_a = float(_point_to_segments_dist(np.array([[ax_val, ay_val]]), centerline_segments)[0])
        lat_b = float(_point_to_segments_dist(np.array([[bx_val, by_val]]), centerline_segments)[0])
        label_a += f"  lat={lat_a:.2f}m"
        label_b += f"  lat={lat_b:.2f}m"

    ax.lines([], cfg.model_a_color, width=2, label=label_a)
    ax.lines([], cfg.model_b_color, width=2, label=label_b)
    ax.legend("upper left")

    ax.grid(0.15)
    ax.xlabel("X (m, initial ego frame)")
    ax.ylabel("Y (m, initial ego frame)")

    title = f"Step {step:04d}/{n_steps}  t={step * 0.1:.1f}s"
    if cfg.subtitle:
        title = f"{cfg.subtitle}\n{title}"
    if extra_title:
        title += f"\n{extra_title}"
    ax.title(title, 10)
    fig.save(output_path)


def run_ghost_sim(
    scene_path: str,
    model_a,
    model_a_args,
    model_b,
    model_b_args,
    scene_data: dict[str, torch.Tensor],
    output_dir: Path,
    cfg: GhostSimConfig,
    neighbor_boxes: list[tuple[float, float, float, float, float]] | None = None,
    make_webm: bool = True,
    extra_title_fn=None,
    predict_fn_a=None,
    predict_fn_b=None,
    sg_smooth: bool = True,
    rollout_a: dict | None = None,
    rollout_b: dict | None = None,
):
    """Run dual-model ghost sim and render per-step PNGs + optional webm.

    extra_title_fn: optional callable(step, a_pose, b_pose) -> str for per-step subtitle.
    predict_fn_a / predict_fn_b: optional per-step planner-call overrides for
        the rollouts (e.g. exploration-policy guided generation);
        same signature as recovery_sim.deterministic_predict.
    sg_smooth: SG-filter every per-step plan before tracking (sim convention).
    rollout_a / rollout_b: precomputed rollout dicts (closed_loop_rollout_with_plans
        format) — render-only mode, models may be None (no GPU needed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if rollout_a is None:
        print(f"[ghost-sim] rollout ({cfg.model_a_label})...")
        rollout_a = closed_loop_rollout_with_plans(
            model_a,
            model_a_args,
            scene_data,
            n_steps=cfg.steps,
            advance_k=cfg.advance_k,
            predict_fn=predict_fn_a,
            sg_smooth=sg_smooth,
        )
    if rollout_b is None:
        print(f"[ghost-sim] rollout ({cfg.model_b_label})...")
        rollout_b = closed_loop_rollout_with_plans(
            model_b,
            model_b_args,
            scene_data,
            n_steps=cfg.steps,
            advance_k=cfg.advance_k,
            predict_fn=predict_fn_b,
            sg_smooth=sg_smooth,
        )

    centerlines, lefts, rights, border_polylines, route_polylines, cl_segments = (
        extract_scene_polylines(scene_data)
    )

    if neighbor_boxes is None:
        neighbor_boxes = extract_stopped_neighbors(scene_path)

    n = cfg.steps
    H = 0
    if cfg.hist_steps > 0 and "ego_agent_past" in scene_data:
        past = scene_data["ego_agent_past"]
        past = past[0] if past.dim() == 3 else past
        past = past.cpu().numpy()
        H = min(cfg.hist_steps, past.shape[0])
        hist = past[past.shape[0] - H :]
        print(f"[ghost-sim] rendering {H} history frames...")
        trail = []
        for hi in range(H):
            row = hist[hi]
            hh = float(np.arctan2(row[3], row[2])) if row.shape[-1] >= 4 else float(row[2])
            pose = np.array([float(row[0]), float(row[1]), hh])
            trail.append(pose[:2])
            render_ghost_step(
                output_dir / f"ghost_step_{hi:04d}.png",
                step=hi - H,
                n_steps=n,
                a_pose=pose,
                a_speed=0.0,
                a_plan=None,
                b_pose=pose,
                b_speed=0.0,
                b_plan=None,
                centerlines=centerlines,
                lefts=lefts,
                rights=rights,
                border_polylines=border_polylines,
                route_polylines=route_polylines,
                centerline_segments=cl_segments,
                cfg=cfg,
                neighbor_boxes=neighbor_boxes,
                extra_title=f"  HISTORY t={(hi - H) * 0.1:+.1f}s (model context)",
                history_mode=True,
                history_trail=np.array(trail),
            )
    print(f"[ghost-sim] rendering {n + 1} frames...")
    for step_i in range(n + 1):
        a_plan = (
            rollout_a["plans_world"][step_i] if step_i < len(rollout_a["plans_world"]) else None
        )
        b_plan = (
            rollout_b["plans_world"][step_i] if step_i < len(rollout_b["plans_world"]) else None
        )
        et = ""
        if extra_title_fn:
            et = extra_title_fn(
                step_i, rollout_a["positions"][step_i], rollout_b["positions"][step_i]
            )
        render_ghost_step(
            output_dir / f"ghost_step_{step_i + H:04d}.png",
            step=step_i,
            n_steps=n,
            a_pose=rollout_a["positions"][step_i],
            a_speed=float(rollout_a["velocities"][step_i]),
            a_plan=a_plan,
            b_pose=rollout_b["positions"][step_i],
            b_speed=float(rollout_b["velocities"][step_i]),
            b_plan=b_plan,
            centerlines=centerlines,
            lefts=lefts,
            rights=rights,
            border_polylines=border_polylines,
            route_polylines=route_polylines,
            centerline_segments=cl_segments,
            cfg=cfg,
            neighbor_boxes=neighbor_boxes,
            extra_title=et,
            fan_plans=(
                rollout_b.get("extra_plans_world", [[]])[step_i]
                if step_i < len(rollout_b.get("extra_plans_world", []))
                else None
            ),
        )

    webm_path = None
    if make_webm:
        webm_path = output_dir / "ghost_sim.webm"
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(cfg.webm_fps),
            "-i",
            str(output_dir / "ghost_step_%04d.png"),
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "0",
            "-crf",
            "32",
            "-row-mt",
            "1",
            str(webm_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(
                f"[ghost-sim] ffmpeg failed (rc={result.returncode}): {result.stderr.decode()[-200:]}"
            )
        else:
            print(f"[ghost-sim] WebM: {webm_path}")

    print(f"\nDone — {output_dir} ({n + 1} frames)")
    return rollout_a, rollout_b, webm_path
