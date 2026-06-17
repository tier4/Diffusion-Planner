#!/usr/bin/env python3
"""Audit a scenario_generation.replay run for model-predicted static collisions.

For each dumped NPZ step, re-runs the model (``--model_path``), scores the
PREDICTED 80-step ego trajectory against stopped neighbours via
``rlvr.reward.compute_static_collision_penalty``, and emits:

* ``<output_dir>/summary.json`` — per-sim-step scalar metrics (sc_min_dist,
  static_crossing, sc_n_stopped, first_crossing_step, winning-neighbour idx,
  closest-pair points) for ALL sim steps, regardless of threshold.

* ``<output_dir>/step_NNNN.png`` — overlay visualisation of the ego's
  80-step prediction + stopped-neighbour OBBs + red line between the
  closest-pair points at the argmin-clearance timestep. Drawn only for
  sim steps where the predicted min clearance falls below ``--viz_threshold``
  (default 2 m).

Purpose: the live sim PNGs already show the ego's CURRENT-pose line to the
nearest stopped NPC. This tool answers the complementary question — "did
the model *plan* a trajectory that drives too close to a stopped NPC?" —
by scoring the 80-step prediction.

The dumped NPZ has ``neighbor_agents_future`` zero-filled (live sim has no
GT future for prepopulated static NPCs). We reconstruct a stopped-neighbour
future by broadcasting ``neighbor_agents_past[:, -1, :]`` forward for all
80 timesteps. The reward primitive's stopped mask (|v0|<vel_thresh AND
total displacement<disp_thresh) correctly flags them.

Usage (example — run after ``scenario_generation.replay`` finishes):

    source /opt/ros/humble/setup.bash && source .venv/bin/activate
    python -m rlvr.autoresearch.tools.audit_static_collision \\
        --run_dir /media/.../mpc_gen_bigcurve_static_npcs_<ts>/ \\
        --model_path /media/.../x2_model_base/best_model.pth \\
        --config /path/to/reward_config.json \\
        --output_dir /media/.../mpc_gen_bigcurve_static_npcs_<ts>/static_audit \\
        --viz_threshold 2.0

The reward config JSON MUST set ``static_collision_enabled: true``. The tool
refuses to run otherwise (no silent default — ``feedback_no_silent_defaults``).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.reward import (
    RewardConfig,
    _build_ego_bbox_corners,
    compute_reward_batch,
    compute_static_collision_penalty,
)
from scenario_generation.replay import SpawnConfig

_NPZ_RE = re.compile(r"replay_step_(\d+)\.npz$")


def _synthesize_stopped_futures(nb_past: np.ndarray, future_len: int) -> np.ndarray:
    """Broadcast each neighbor's current (x, y, yaw) forward for ``future_len`` steps.

    Returns (N_nb, future_len, 3) — the dump format expected by
    ``compute_reward_batch``. For truly-stopped NPCs this matches the
    ground truth; for moving ones it's a fiction, but the reward's
    stopped mask filters them out anyway.

    ``nb_past`` layout: (N_nb, T_past, 11) with
    [x, y, cos_h, sin_h, vx, vy, width, length, type_onehot(3)].
    """
    N_nb = nb_past.shape[0]
    out = np.zeros((N_nb, future_len, 3), dtype=np.float32)
    x = nb_past[:, -1, 0]
    y = nb_past[:, -1, 1]
    cos_h = nb_past[:, -1, 2]
    sin_h = nb_past[:, -1, 3]
    yaw = np.arctan2(sin_h, cos_h)
    valid = np.abs(nb_past[:, -1, :2]).sum(axis=-1) > 1e-6
    out[valid, :, 0] = x[valid, None]
    out[valid, :, 1] = y[valid, None]
    out[valid, :, 2] = yaw[valid, None]
    return out


@torch.no_grad()
def _score_prediction(
    data_np: dict[str, np.ndarray],
    ego_pred: np.ndarray,
    reward_cfg: RewardConfig,
    device: str,
) -> dict:
    """Score one step's ego prediction with static_collision_enabled=true.

    Returns a flat dict with scalar sc_* fields and numpy arrays for the
    per-timestep min clearance + closest-pair points (used by the viz).
    """
    # Build torch tensors for the primitive.
    ego_traj = torch.from_numpy(ego_pred.astype(np.float32)).to(device).unsqueeze(0)
    T = ego_traj.shape[1]

    es = np.asarray(data_np["ego_shape"])
    if es.ndim == 2:
        es = es[0]
    ego_shape = torch.from_numpy(es[:3].astype(np.float32)).to(device)

    # Synthesize stopped-NPC futures from past[:, -1, :3_pose].
    nb_past = np.asarray(data_np["neighbor_agents_past"])
    nb_fut = _synthesize_stopped_futures(nb_past, future_len=T)

    # Build (N_nb, T, 4) [x, y, cos, sin] from (N_nb, T, 3) [x, y, yaw].
    yaw = nb_fut[..., 2:3]
    nb_fut_4 = np.concatenate([nb_fut[..., :2], np.cos(yaw), np.sin(yaw)], axis=-1).astype(
        np.float32
    )
    neighbor_futures = torch.from_numpy(nb_fut_4).to(device)
    slot_valid = neighbor_futures[:, :, :2].abs().sum(dim=(1, 2)) > 1e-6
    if slot_valid.any():
        neighbor_futures = neighbor_futures[slot_valid]
        neighbor_valid = neighbor_futures[:, :, :2].abs().sum(dim=-1) > 1e-6
        # neighbor_agents_past cols 6 (width), 7 (length).
        nb_past_valid = nb_past[slot_valid.cpu().numpy()]
        shapes_np = nb_past_valid[:, -1, [6, 7]].astype(np.float32)
        neighbor_shapes = torch.from_numpy(shapes_np).to(device)
        # Zero-shape guard (matches compute_reward_batch).
        zero = neighbor_shapes.abs().sum(dim=-1) < 1e-3
        if zero.any():
            neighbor_shapes[zero] = torch.tensor([2.0, 4.5], device=device)
    else:
        neighbor_futures = torch.zeros(0, T, 4, device=device)
        neighbor_shapes = torch.zeros(0, 2, device=device)
        neighbor_valid = torch.zeros(0, T, dtype=torch.bool, device=device)

    sc = compute_static_collision_penalty(
        ego_traj,
        ego_shape,
        neighbor_futures,
        neighbor_shapes,
        neighbor_valid,
        reward_cfg,
    )

    per_ts_min = sc["per_timestep_min"][0].cpu().numpy()  # (T,)
    ego_pts = sc["ego_closest_pt"][0].cpu().numpy()  # (T, 2)
    npc_pts = sc["npc_closest_pt"][0].cpu().numpy()  # (T, 2)
    argmin_nb = sc["argmin_neighbor"][0].cpu().numpy()  # (T,)
    first_cross = sc["first_crossing_steps"][0]

    # Min across t>=1 (t=0 isn't model-controllable).
    t_valid = slice(1, T)
    if T > 1:
        argmin_t = int(np.argmin(per_ts_min[t_valid])) + 1
        sc_min_dist = float(per_ts_min[argmin_t])
    else:
        argmin_t = 0
        sc_min_dist = float(per_ts_min[0])

    # Viz anchor: prefer the FIRST crossing timestep over the global argmin
    # when the plan contains more than one violation. Survival-mode reward
    # is driven by first-terminal-step, so the first-crossing footprint is
    # the actionable one — a later, deeper penetration could come long
    # after the trajectory would already have been floored.
    if first_cross is not None:
        viz_t = int(first_cross)
        viz_d = float(per_ts_min[viz_t])
    else:
        viz_t = argmin_t
        viz_d = sc_min_dist

    return {
        "sc_min_dist": sc_min_dist,
        "static_crossing": bool(sc["crossing_gate"][0].item() < 0.5),
        "sc_n_stopped": int(sc["stopped_mask"].sum().item()),
        "first_crossing_step": first_cross,
        "argmin_t": argmin_t,
        "argmin_neighbor_idx": int(argmin_nb[argmin_t]),
        # viz_t / viz_d is where the overlay PNG draws the ego footprint —
        # first crossing if one exists, else argmin over time.
        "viz_t": viz_t,
        "viz_d": viz_d,
        "viz_neighbor_idx": int(argmin_nb[viz_t]),
        "ego_closest_pt": ego_pts[viz_t].tolist(),
        "npc_closest_pt": npc_pts[viz_t].tolist(),
        "per_ts_min": per_ts_min.tolist(),
        # Kept in numpy form for the viz (not JSON-serialised by caller).
        "_arrays": {
            "ego_pts": ego_pts,
            "npc_pts": npc_pts,
            "argmin_nb": argmin_nb,
            "per_ts_min": per_ts_min,
            "ego_traj": ego_pred,
            "ego_corners": _build_ego_bbox_corners(
                ego_traj,
                ego_shape,
            )[0]
            .cpu()
            .numpy(),  # (T, 4, 2)
            "nb_fut_4": nb_fut_4,  # (N_nb, T, 4)
            "neighbor_shapes_all": np.asarray(data_np["neighbor_agents_past"])[:, -1, [6, 7]],
        },
    }


_ZONE_COLORS = {
    "cross": "#cc0000",  # <0.2 m — visually a collision
    "near": "#ff8800",  # 0.2..0.5 m
    "wide": "#e5c200",  # 0.5..1.0 m
    "safe": "#3366cc",  # >=1.0 m
}

_LANE_CL_COLOR = "#bbbbbb"
_LANE_BORDER_COLOR = "#888888"
_ROAD_BORDER_COLOR = "#dd2222"
_EGO_NOW_COLOR = "#3366cc"


def _draw_lane_network_from_tensor(ax, lanes: np.ndarray, alpha: float = 0.7) -> None:
    """Draw lane centerlines + left/right boundaries from the 33-dim lane tensor.

    Mirrors :func:`scenario_generation.replay._draw_lane_network` but takes the
    raw ndarray straight from an NPZ (``data_np["lanes"]``). Borders come from
    ``centerline + lane[:, 4:6]`` (left) and ``+ lane[:, 6:8]`` (right).
    """
    from matplotlib.collections import LineCollection

    if lanes is None:
        return
    if lanes.ndim == 4:
        lanes = lanes[0]
    centerlines, lefts, rights = [], [], []
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        pts = lane[:, :2]
        if np.abs(pts).sum() < 1e-6:
            continue
        valid = np.abs(pts).sum(axis=1) > 0.1
        if valid.sum() < 2:
            continue
        centerlines.append(pts[valid])
        if lane.shape[1] > 7:
            lefts.append((pts + lane[:, 4:6])[valid])
            rights.append((pts + lane[:, 6:8])[valid])

    if centerlines:
        ax.add_collection(
            LineCollection(
                centerlines,
                colors=_LANE_CL_COLOR,
                linewidths=0.6,
                alpha=alpha * 0.4,
                zorder=1,
            )
        )
    if lefts:
        ax.add_collection(
            LineCollection(
                lefts,
                colors=_LANE_BORDER_COLOR,
                linewidths=1.0,
                alpha=alpha,
                zorder=2,
            )
        )
    if rights:
        ax.add_collection(
            LineCollection(
                rights,
                colors=_LANE_BORDER_COLOR,
                linewidths=1.0,
                alpha=alpha,
                zorder=2,
            )
        )


def _zone_for(d: float, cross_t: float = 0.2, near_t: float = 0.5, wide_t: float = 1.0) -> str:
    if d < cross_t:
        return "cross"
    if d < near_t:
        return "near"
    if d < wide_t:
        return "wide"
    return "safe"


def _draw_audit_figure(
    data_np: dict[str, np.ndarray],
    step: int,
    arrays: dict,
    sc_min_dist: float,
    argmin_t: int,
    argmin_nb_global: int,
    output_path: Path,
    viz_threshold_m: float = 2.0,
    dt: float = 0.1,
) -> None:
    """2-panel audit figure for one sim step.

    Top (spatial): predicted ego trajectory polyline (8 s / 80 steps) +
    the ego's FOOTPRINT AT argmin_t rendered prominently + the offending
    stopped NPC's OBB + distance line between the closest points at
    argmin_t, labeled ``X.XX m @t=N (Y.Ys)``.

    Bottom (time series): per-timestep min clearance to any stopped
    neighbour across the 80-step plan, with zone shading (cross<0.2,
    near<0.5, wide<1.0). Makes it immediately visible WHEN within the
    8 s plan the ego enters each danger band — essential for survival-
    mode reward interpretation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(11, 12))
    ax = fig.add_subplot(2, 1, 1)
    ax_ts = fig.add_subplot(2, 1, 2)

    # --- Top: spatial view ---

    # Lane network (centerlines + left/right markings) from the map tensor —
    # needed for real-data NPZs where the lanelet context is the only way
    # to read the scene ("is this a highway / intersection / shoulder?").
    lanes = np.asarray(data_np["lanes"]) if "lanes" in data_np else None
    _draw_lane_network_from_tensor(ax, lanes, alpha=0.7)

    # Road borders (line_strings channel 3). May be sparse in real data.
    ls = np.asarray(data_np["line_strings"]) if "line_strings" in data_np else None
    if ls is not None:
        if ls.ndim == 4:
            ls = ls[0]
        if ls.shape[-1] >= 4:
            for i in range(ls.shape[0]):
                line = ls[i]
                valid = (line[:, 3] > 0.5) & (np.abs(line[:, :2]).sum(axis=-1) > 0.01)
                if valid.sum() > 1:
                    ax.plot(
                        line[valid, 0],
                        line[valid, 1],
                        color=_ROAD_BORDER_COLOR,
                        lw=2.0,
                        alpha=0.5,
                        zorder=3,
                    )

    # Predicted ego trajectory polyline (thin, grey).
    ego_xy = arrays["ego_traj"][:, :2]
    T = arrays["ego_traj"].shape[0]
    per_ts_min = arrays["per_ts_min"]  # (T,)
    ax.plot(
        ego_xy[:, 0],
        ego_xy[:, 1],
        "-",
        color="#555",
        lw=1.2,
        alpha=0.8,
        zorder=10,
        label="predicted path (0..7.9 s)",
    )

    es = np.asarray(data_np["ego_shape"])
    if es.ndim == 2:
        es = es[0]
    ego_wb, ego_L, ego_W = float(es[0]), float(es[1]), float(es[2])

    # Ego OBB at t=0 (the CURRENT pose) drawn in blue. Note ego_trajs are
    # stored at the rear-axle origin (x, y); the box centre-of-geometry is
    # at +wb/2 along the heading. Same convention as _build_ego_bbox_corners.
    cos0, sin0 = arrays["ego_traj"][0, 2], arrays["ego_traj"][0, 3]
    h0 = math.atan2(
        sin0 / max(math.hypot(cos0, sin0), 1e-6), cos0 / max(math.hypot(cos0, sin0), 1e-6)
    )
    cx0, cy0 = ego_xy[0]
    cog0 = (cx0 + math.cos(h0) * ego_wb / 2.0, cy0 + math.sin(h0) * ego_wb / 2.0)
    t_rot0 = mtransforms.Affine2D().rotate(h0).translate(cog0[0], cog0[1]) + ax.transData
    ax.add_patch(
        Rectangle(
            (-ego_L / 2, -ego_W / 2),
            ego_L,
            ego_W,
            lw=2.0,
            ec=_EGO_NOW_COLOR,
            fc=_EGO_NOW_COLOR,
            alpha=0.30,
            zorder=18,
            transform=t_rot0,
            label="ego @ t=0 (now)",
        )
    )

    # Ego footprint at argmin_t (or first-crossing, whichever the caller
    # passed in). This is the predicted pose where the plan comes closest
    # to the offending stopped NPC — the actionable failure point.
    cos_h = arrays["ego_traj"][argmin_t, 2]
    sin_h = arrays["ego_traj"][argmin_t, 3]
    hn = math.hypot(cos_h, sin_h)
    heading = math.atan2(sin_h / max(hn, 1e-6), cos_h / max(hn, 1e-6))
    cx, cy = ego_xy[argmin_t]
    cog = (cx + math.cos(heading) * ego_wb / 2.0, cy + math.sin(heading) * ego_wb / 2.0)
    zone = _zone_for(sc_min_dist)
    ego_color = _ZONE_COLORS[zone]
    t_rot = mtransforms.Affine2D().rotate(heading).translate(cog[0], cog[1]) + ax.transData
    ax.add_patch(
        Rectangle(
            (-ego_L / 2, -ego_W / 2),
            ego_L,
            ego_W,
            lw=2.0,
            ec=ego_color,
            fc=ego_color,
            alpha=0.35,
            zorder=20,
            transform=t_rot,
            label=f"ego @ t={argmin_t}",
        )
    )

    # All stopped-neighbour OBBs (faint so the offender stands out).
    # nb_shapes_all is neighbor_agents_past[:, -1, [6, 7]] == [width, length]
    # (tensor_converter.py L202-203). Unpack explicitly — this file used to
    # have the two swapped, causing visibly wrong OBB aspect ratios.
    nb_fut_4 = arrays["nb_fut_4"]
    nb_shapes_all = arrays["neighbor_shapes_all"]
    for i in range(nb_fut_4.shape[0]):
        x, y = nb_fut_4[i, 0, 0], nb_fut_4[i, 0, 1]
        if abs(x) + abs(y) < 1e-6:
            continue
        cos_h_i = nb_fut_4[i, 0, 2]
        sin_h_i = nb_fut_4[i, 0, 3]
        heading_i = math.atan2(sin_h_i, cos_h_i)
        width = float(nb_shapes_all[i, 0])  # tensor col 6 = width
        length = float(nb_shapes_all[i, 1])  # tensor col 7 = length
        is_offender = i == argmin_nb_global
        t_rot = mtransforms.Affine2D().rotate(heading_i).translate(x, y) + ax.transData
        ax.add_patch(
            Rectangle(
                (-length / 2, -width / 2),
                length,
                width,
                lw=1.6 if is_offender else 0.8,
                ec="#cc6600" if is_offender else "#cc9966",
                fc="#ffb366" if is_offender else "#ffd9a6",
                alpha=0.85 if is_offender else 0.45,
                zorder=15 if is_offender else 13,
                transform=t_rot,
            )
        )

    # Closest-pair line at argmin_t.
    pt_e = arrays["ego_pts"][argmin_t]
    pt_n = arrays["npc_pts"][argmin_t]
    ax.plot(
        [pt_e[0], pt_n[0]], [pt_e[1], pt_n[1]], "-", color=ego_color, lw=2.2, alpha=0.95, zorder=30
    )
    ax.plot(
        [pt_e[0], pt_n[0]],
        [pt_e[1], pt_n[1]],
        "o",
        color=ego_color,
        ms=6,
        zorder=31,
        markeredgecolor="white",
        markeredgewidth=0.7,
    )
    mx, my = (pt_e[0] + pt_n[0]) / 2, (pt_e[1] + pt_n[1]) / 2
    dx, dy = pt_n[0] - pt_e[0], pt_n[1] - pt_e[1]
    seg_len = math.hypot(dx, dy)
    nx_, ny_ = (-dy / seg_len, dx / seg_len) if seg_len > 1e-6 else (0.0, 1.0)
    ax.annotate(
        f"{sc_min_dist:.2f} m",
        xy=(mx + nx_ * 1.2, my + ny_ * 1.2),
        fontsize=7,
        color=ego_color,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor=ego_color, alpha=0.9),
        zorder=32,
    )

    # Frame the view around (ego-at-argmin, offender).
    all_x = np.array([cog[0], pt_n[0], ego_xy[0, 0], ego_xy[-1, 0]])
    all_y = np.array([cog[1], pt_n[1], ego_xy[0, 1], ego_xy[-1, 1]])
    cx_v = float((all_x.min() + all_x.max()) * 0.5)
    cy_v = float((all_y.min() + all_y.max()) * 0.5)
    half = max(float(all_x.max() - all_x.min()), float(all_y.max() - all_y.min())) * 0.6 + 6.0
    ax.set_xlim(cx_v - half, cx_v + half)
    ax.set_ylim(cy_v - half, cy_v + half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(
        f"sim step {step:04d}  predicted sc_min_dist={sc_min_dist:.2f} m "
        f"@t={argmin_t}/{T - 1} ({argmin_t * dt:.1f}s)  offender_idx={argmin_nb_global}",
        fontsize=10,
    )

    # --- Bottom: per-timestep clearance time series ---

    t_axis = np.arange(T) * dt  # seconds into the 8 s plan
    ax_ts.axhspan(0, 0.2, color=_ZONE_COLORS["cross"], alpha=0.10, label="cross <0.2 m")
    ax_ts.axhspan(0.2, 0.5, color=_ZONE_COLORS["near"], alpha=0.08, label="near <0.5 m")
    ax_ts.axhspan(0.5, 1.0, color=_ZONE_COLORS["wide"], alpha=0.06, label="wide <1 m")

    ax_ts.plot(
        t_axis,
        per_ts_min,
        "-",
        color="#333",
        lw=1.3,
        alpha=0.9,
        label="min clearance to any stopped NPC",
    )
    ax_ts.plot(
        t_axis[argmin_t],
        per_ts_min[argmin_t],
        "o",
        color=ego_color,
        ms=7,
        zorder=10,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label=f"argmin @ t={argmin_t}",
    )
    ax_ts.axvline(t_axis[argmin_t], color=ego_color, ls="--", lw=0.8, alpha=0.5)
    ax_ts.axhline(viz_threshold_m, color="#777", lw=0.8, ls="--", alpha=0.6)

    ax_ts.set_xlabel("prediction time (s)  →  plan horizon")
    ax_ts.set_ylabel("clearance (m)")
    ax_ts.set_xlim(0, t_axis[-1])
    ax_ts.set_ylim(
        min(-0.2, float(per_ts_min.min()) - 0.1),
        max(viz_threshold_m + 0.5, float(per_ts_min.max()) + 0.1),
    )
    ax_ts.grid(True, alpha=0.2)
    ax_ts.legend(loc="upper right", fontsize=7)
    ax_ts.set_title(
        "Zones entered over the 8 s plan — key for survival-mode reward",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _draw_timeline_figure(summary: list[dict], output_path: Path, viz_threshold_m: float) -> None:
    """One-shot whole-sim view: predicted sc_min_dist vs sim step.

    Shows at a glance which parts of the run produced dangerous plans.
    Crossings (sc_min_dist < 0.2 m by default) are highlighted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not summary:
        return

    steps = np.array([r["step"] for r in summary])
    dmin = np.array([r["sc_min_dist"] for r in summary])
    n_stopped = np.array([r["sc_n_stopped"] for r in summary])
    crossed = np.array([r["static_crossing"] for r in summary], dtype=bool)

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 1, 1)

    # Background shading for danger zones.
    ax.axhspan(0, 0.2, color="#cc0000", alpha=0.10, label="_cross")
    ax.axhspan(0.2, 0.5, color="#ff8800", alpha=0.08, label="_near")
    ax.axhspan(0.5, 1.0, color="#e5c200", alpha=0.06, label="_wide")

    ax.plot(steps, dmin, "-", color="#333", lw=1.0, alpha=0.8, label="predicted sc_min_dist")
    ax.plot(
        steps[crossed],
        dmin[crossed],
        "o",
        color="#cc0000",
        ms=4,
        label=f"crossing (d<0.2 m): {int(crossed.sum())}",
    )
    ax.axhline(
        viz_threshold_m,
        color="#777",
        lw=0.8,
        ls="--",
        alpha=0.6,
        label=f"viz_threshold ({viz_threshold_m:.1f} m)",
    )

    ax.set_xlabel("sim step")
    ax.set_ylabel("predicted sc_min_dist (m)")
    ax.set_ylim(0, max(viz_threshold_m + 1.0, float(dmin.max()) + 0.2))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    n_total = len(summary)
    pct_cross = 100.0 * crossed.sum() / n_total if n_total else 0.0
    ax.set_title(
        f"Predicted static-collision timeline — {n_total} sim steps  "
        f"({crossed.sum()} crossings, {pct_cross:.1f}%)  "
        f"max stopped NPCs visible/step: {int(n_stopped.max()) if n_total else 0}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Replay run directory (must contain npz/ and spawn_config.json).",
    )
    parser.add_argument(
        "--model_path", type=Path, required=True, help="Model checkpoint used for inference."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Reward config JSON. MUST have static_collision_enabled=true.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write summary.json + step_NNNN.png (default: <run_dir>/static_audit/).",
    )
    parser.add_argument(
        "--viz_threshold",
        type=float,
        default=2.0,
        help="Render overlay PNG only when predicted sc_min_dist < this (metres). Default 2.0.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--stride", type=int, default=1, help="Process every Nth sim step. Default 1 (all)."
    )
    args = parser.parse_args()

    reward_cfg = load_reward_config(args.config)
    if not getattr(reward_cfg, "static_collision_enabled", False):
        raise SystemExit(
            f"{args.config} must set static_collision_enabled=true — "
            f"otherwise the audit would silently report all zeros. "
            f"Add 'static_collision_enabled: true' and at least "
            f"'sc_near_scale: 1.0' to the config JSON."
        )

    run_dir = args.run_dir
    npz_dir = run_dir / "npz"
    if not npz_dir.is_dir():
        raise SystemExit(f"{npz_dir} missing")
    spawn_cfg_path = run_dir / "spawn_config.json"
    if not spawn_cfg_path.exists():
        raise SystemExit(f"{spawn_cfg_path} missing")
    spawn_cfg = SpawnConfig.from_json(spawn_cfg_path)

    out_dir = args.output_dir or (run_dir / "static_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(npz_dir.glob("replay_step_*.npz"))
    if not npz_paths:
        raise SystemExit(f"no replay_step_*.npz in {npz_dir}")
    if args.stride > 1:
        npz_paths = npz_paths[:: args.stride]

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # Lazy model load — avoids importing scenario_generation.simulate
    # (ROS-dependent) in contexts where it isn't needed.
    from scenario_generation.npz_loader import from_npz
    from scenario_generation.simulate import _predict_batch, load_model

    print(f"Loading model {args.model_path}")
    model, model_args = load_model(str(args.model_path), device)

    summary: list[dict] = []
    n_viz = 0
    for i, path in enumerate(npz_paths):
        m = _NPZ_RE.search(path.name)
        if not m:
            continue
        step = int(m.group(1))

        with np.load(path, allow_pickle=True) as raw:
            data_np = {k: raw[k] for k in raw.files if k != "version"}

        scene = from_npz(str(path))
        preds = _predict_batch(
            model,
            model_args,
            scene,
            [scene.ego_agent_id],
            device,
            inference_delay=spawn_cfg.inference_delay,
        )
        ego_pred = preds.get(scene.ego_agent_id)
        if ego_pred is None:
            print(f"  [skip] step {step}: no ego prediction")
            continue

        result = _score_prediction(data_np, ego_pred, reward_cfg, device)

        # Viz gate + draw. Anchor at the FIRST crossing if present (what
        # survival-mode reward cares about), else at the global argmin.
        if result["sc_min_dist"] < args.viz_threshold:
            _draw_audit_figure(
                data_np,
                step,
                result["_arrays"],
                result["viz_d"],
                result["viz_t"],
                result["viz_neighbor_idx"],
                out_dir / f"step_{step:04d}.png",
                viz_threshold_m=args.viz_threshold,
            )
            n_viz += 1

        # Strip numpy arrays before serialising.
        result.pop("_arrays", None)
        result["step"] = step
        result["npz"] = path.name
        summary.append(result)

        if (i + 1) % 50 == 0:
            print(f"  Scored {i + 1}/{len(npz_paths)} (viz so far: {n_viz})")

    summary_path = out_dir / "summary.json"
    payload = {
        "run_dir": str(run_dir),
        "model_path": str(args.model_path),
        "reward_config_path": str(args.config),
        "viz_threshold_m": float(args.viz_threshold),
        "stride": int(args.stride),
        "n_steps_scored": len(summary),
        "n_overlay_pngs": n_viz,
        "steps": summary,
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {summary_path} ({len(summary)} rows, {n_viz} viz PNGs)")

    # Whole-sim timeline: single PNG summarising every predicted step.
    timeline_path = out_dir / "timeline.png"
    _draw_timeline_figure(summary, timeline_path, args.viz_threshold)
    print(f"Wrote {timeline_path}")


if __name__ == "__main__":
    main()
