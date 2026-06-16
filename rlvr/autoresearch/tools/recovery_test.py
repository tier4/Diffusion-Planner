#!/usr/bin/env python3
"""Recovery test harness: does the planner close a perturbation back to centerline?

Diagnoses the failure mode where the model runs parallel-offset to centerline
without recovering. Supports three perturbation types:

  - parallel : lateral position shift (history rigidly translated)
  - yaw      : heading rotation (history rigidly rotated about current pose)
  - velocity : speed scaling (history + current velocity scaled)
  - combined : parallel 0.5m + yaw 5 deg

For each (scene, perturbation_kind, magnitude, side):

  1. Load NPZ, compute centerline tangent at the ego's current pose from
     route_lanes (channels 0-1 = x/y, 2-3 = direction).
  2. Apply perturbation to ego's current pose AND past history.
  3. Run ONE-SHOT deterministic inference -> 80-step trajectory.
  4. (Optional, --closed_loop) Run a CLOSED-LOOP rollout: re-predict at every
     step, advance ego pose to pred[k_advance], rebuild past, repeat for
     `closed_loop_steps` (default 80 = 8 sec at 0.1s).
  5. Compute per-step lateral offset to nearest route_lane centerline at
     t=0, t=40 (4 s) and t=79 (8 s). Recovery rate = (|t0|-|t79|)/|t0|.
  6. Run K=8 generation with the training generation_variant; report best-K.

Usage:
    python -m rlvr.autoresearch.tools.recovery_test \
        --model_path /path/to/base.pth \
        --lora_path /path/to/lora_dir \
        --scenes /path/to/scenes.json \
        --config /path/to/grpo_config.json \
        --output /path/to/out.json \
        [--offsets 0.25,0.5,0.75,1.0] [--K 8] \
        [--perturbation_kinds parallel,yaw,velocity,combined] \
        [--yaw_degs 2,5,10] [--vel_pcts 30,50] \
        [--closed_loop] [--closed_loop_steps 80] [--closed_loop_advance_k 1]
"""

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import load_npz_data
from rlvr.grpo_trainer_batched import (
    _normalize_batch,
    _stack_scene_data,
    generate_all_scenes_batched,
    get_generation_config_labels_for_variant,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Centerline geometry helpers (operate on the raw, un-normalized data dict)
# ---------------------------------------------------------------------------


def _flatten_lane_points(route_lanes: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centers [M,2], dirs [M,2], valid [M]) from route_lanes [1,S,P,33]."""
    rl = route_lanes
    if rl.dim() == 4:
        rl = rl[0]
    rl = rl.detach().cpu().numpy()  # [S, P, 33]
    S, P, _ = rl.shape
    centers = rl[..., 0:2].reshape(S * P, 2)
    dirs = rl[..., 2:4].reshape(S * P, 2)
    valid = np.linalg.norm(centers, axis=-1) > 1e-3
    return centers, dirs, valid


def get_tangent_at_origin(route_lanes: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (tangent_unit [2], normal_unit [2]) at ego origin (0,0).

    Picks the closest valid lane centerline point to the origin and uses its
    direction channel (route_lanes[..., 2:4]). Normal is left-of-tangent
    (rotate tangent +90 deg: (-dy, dx)).
    """
    centers, dirs, valid = _flatten_lane_points(route_lanes)
    if not valid.any():
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])

    d = np.linalg.norm(centers, axis=-1)
    d = np.where(valid, d, 1e9)
    idx = int(np.argmin(d))
    t = dirs[idx].astype(np.float64)
    n_t = float(np.linalg.norm(t))
    if n_t < 1e-6:
        # fallback: ego is already at origin with cos=1,sin=0 ego heading
        t = np.array([1.0, 0.0])
        n_t = 1.0
    t = t / n_t
    n = np.array([-t[1], t[0]])  # +90° (left side positive)
    return t, n


def _build_segments(route_lanes: torch.Tensor) -> np.ndarray:
    """Build [N_seg, 2, 2] array of consecutive valid centerline segments per
    lane. Skips a segment if either endpoint is invalid.
    """
    rl = route_lanes
    if rl.dim() == 4:
        rl = rl[0]
    rl = rl.detach().cpu().numpy()  # [S, P, 33]
    S, P, _ = rl.shape
    segs = []
    for s in range(S):
        for p in range(P - 1):
            a = rl[s, p, 0:2]
            b = rl[s, p + 1, 0:2]
            if np.linalg.norm(a) < 1e-3 or np.linalg.norm(b) < 1e-3:
                continue
            segs.append([a, b])
    if not segs:
        return np.zeros((0, 2, 2), dtype=np.float64)
    return np.array(segs, dtype=np.float64)


def _point_to_segments_dist(points: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """For each point in points [T, 2] and segments [N, 2, 2], return the min
    perpendicular distance from each point to any segment (clamped to segment
    endpoints when the foot of perpendicular is outside).
    """
    if segs.shape[0] == 0:
        return np.full((points.shape[0],), np.nan)
    a = segs[:, 0, :]  # [N, 2]
    b = segs[:, 1, :]  # [N, 2]
    ab = b - a  # [N, 2]
    ab_len_sq = np.sum(ab * ab, axis=-1).clip(min=1e-9)  # [N]
    # broadcast: points [T, 1, 2] - a [1, N, 2] = [T, N, 2]
    ap = points[:, None, :] - a[None, :, :]
    dot = np.sum(ap * ab[None, :, :], axis=-1)  # [T, N]
    t = (dot / ab_len_sq[None, :]).clip(0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * ab[None, :, :]  # [T, N, 2]
    diff = points[:, None, :] - proj  # [T, N, 2]
    dist = np.linalg.norm(diff, axis=-1)  # [T, N]
    return dist.min(axis=-1)


def per_step_lateral_distance(traj_xy: np.ndarray, route_lanes: torch.Tensor) -> np.ndarray:
    """For each [T,2] trajectory point, return absolute perpendicular distance
    to the nearest route_lanes centerline (treating consecutive valid
    centerline points as polyline segments). This is more accurate than a
    nearest-point lookup because lane centerline points are sparse along the
    lane direction (often >2m apart).
    """
    segs = _build_segments(route_lanes)
    return _point_to_segments_dist(traj_xy, segs)


# ---------------------------------------------------------------------------
# Lateral shift application (in-place on a fresh copy of the data dict)
# ---------------------------------------------------------------------------


def apply_lateral_shift(
    data: dict[str, torch.Tensor], normal_unit: np.ndarray, offset_m: float
) -> dict[str, torch.Tensor]:
    """Return a new data dict with ego current state and past history shifted
    by offset_m * normal_unit. Headings/velocities untouched.

    This shifts the EGO COORDINATES inside the ego-centric scene. It is a
    cheap proxy for placing the ego at a lateral offset relative to the
    centerline: the rest of the world (lanes, neighbors, line strings) stays
    fixed in ego-frame, so the centerline now appears offset by -offset_m
    on the perpendicular axis. After the shift, recomputing per-step
    lateral distance to lane centerline gives ~|offset_m| at t=0.
    """
    out = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    nx = float(normal_unit[0])
    ny = float(normal_unit[1])
    dx = offset_m * nx
    dy = offset_m * ny

    # ego_current_state [B, 10]: channels 0,1 are x,y
    if "ego_current_state" in out:
        ecs = out["ego_current_state"].clone()
        ecs[..., 0] = ecs[..., 0] + dx
        ecs[..., 1] = ecs[..., 1] + dy
        out["ego_current_state"] = ecs

    # ego_agent_past [B, T, 4] after heading_to_cos_sin: channels 0,1 = x,y
    if "ego_agent_past" in out:
        eap = out["ego_agent_past"].clone()
        eap[..., 0] = eap[..., 0] + dx
        eap[..., 1] = eap[..., 1] + dy
        out["ego_agent_past"] = eap

    return out


def apply_yaw_perturbation(
    data: dict[str, torch.Tensor], yaw_rad: float
) -> dict[str, torch.Tensor]:
    """Return a new data dict with ego heading rotated by `yaw_rad` (radians).

    The current ego pose stays at its position (we rotate about the current
    (x, y)). Past history is rigidly rotated about the same pivot so the
    relative motion remains consistent. Velocity vector (vx, vy) and
    acceleration vector (ax, ay) are also rotated. Heading-encoding (cos, sin)
    fields rotate by `yaw_rad`.

    Pivot = ego current state (x, y). For each historical position p_t:
        p_t' = pivot + R(yaw_rad) @ (p_t - pivot)
    For headings (cos, sin), apply 2D rotation by `yaw_rad`.
    """
    out = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))

    # ego_current_state [B, 10]: 0,1=x,y; 2,3=cos,sin; 4,5=vx,vy; 6,7=ax,ay; 9=yaw_rate
    if "ego_current_state" in out:
        ecs = out["ego_current_state"].clone()
        # Pivot is the current ego (x, y). Rotating about the pivot leaves
        # (x, y) invariant — we only need to rotate the orientation/vel/acc.
        cos_old = ecs[..., 2].clone()
        sin_old = ecs[..., 3].clone()
        ecs[..., 2] = c * cos_old - s * sin_old
        ecs[..., 3] = s * cos_old + c * sin_old
        vx_old = ecs[..., 4].clone()
        vy_old = ecs[..., 5].clone()
        ecs[..., 4] = c * vx_old - s * vy_old
        ecs[..., 5] = s * vx_old + c * vy_old
        ax_old = ecs[..., 6].clone()
        ay_old = ecs[..., 7].clone()
        ecs[..., 6] = c * ax_old - s * ay_old
        ecs[..., 7] = s * ax_old + c * ay_old
        out["ego_current_state"] = ecs
        pivot_x = float(ecs[0, 0].item())
        pivot_y = float(ecs[0, 1].item())
    else:
        pivot_x, pivot_y = 0.0, 0.0

    # ego_agent_past [B, T, 4]: 0,1=x,y; 2,3=cos,sin (after heading_to_cos_sin)
    if "ego_agent_past" in out:
        eap = out["ego_agent_past"].clone()
        x_old = eap[..., 0].clone() - pivot_x
        y_old = eap[..., 1].clone() - pivot_y
        eap[..., 0] = pivot_x + c * x_old - s * y_old
        eap[..., 1] = pivot_y + s * x_old + c * y_old
        cos_old = eap[..., 2].clone()
        sin_old = eap[..., 3].clone()
        eap[..., 2] = c * cos_old - s * sin_old
        eap[..., 3] = s * cos_old + c * sin_old
        out["ego_agent_past"] = eap

    return out


def apply_velocity_perturbation(
    data: dict[str, torch.Tensor], vel_scale: float
) -> dict[str, torch.Tensor]:
    """Return a new data dict with ego velocity (vx, vy) scaled by `vel_scale`.

    Past trajectory positions are also rescaled relative to current ego pose
    so that the implied velocity (delta_pos / dt) matches the new speed:
        p_t' = pivot + vel_scale * (p_t - pivot)
    where pivot = current ego (x, y).

    Acceleration (ax, ay) is scaled by `vel_scale` to keep dynamics
    self-consistent. cos/sin headings are unchanged.
    """
    out = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    pivot_x, pivot_y = 0.0, 0.0
    if "ego_current_state" in out:
        ecs = out["ego_current_state"].clone()
        pivot_x = float(ecs[0, 0].item())
        pivot_y = float(ecs[0, 1].item())
        ecs[..., 4] = ecs[..., 4] * vel_scale  # vx
        ecs[..., 5] = ecs[..., 5] * vel_scale  # vy
        ecs[..., 6] = ecs[..., 6] * vel_scale  # ax
        ecs[..., 7] = ecs[..., 7] * vel_scale  # ay
        out["ego_current_state"] = ecs

    if "ego_agent_past" in out:
        eap = out["ego_agent_past"].clone()
        x_off = eap[..., 0].clone() - pivot_x
        y_off = eap[..., 1].clone() - pivot_y
        eap[..., 0] = pivot_x + vel_scale * x_off
        eap[..., 1] = pivot_y + vel_scale * y_off
        out["ego_agent_past"] = eap

    return out


def apply_combined_perturbation(
    data: dict[str, torch.Tensor],
    normal_unit: np.ndarray,
    offset_m: float,
    yaw_rad: float,
) -> dict[str, torch.Tensor]:
    """Combined: lateral shift then yaw rotation about the SHIFTED current pose."""
    out = apply_lateral_shift(data, normal_unit, offset_m)
    out = apply_yaw_perturbation(out, yaw_rad)
    return out


# ---------------------------------------------------------------------------
# Closed-loop world-frame transform: advance ego by one model-predicted step
# ---------------------------------------------------------------------------


def _rotate_xy_about(
    x: torch.Tensor, y: torch.Tensor, c: float, s: float, ox: float = 0.0, oy: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate (x, y) by (c, s) = (cos a, sin a) about pivot (ox, oy)."""
    xs = x - ox
    ys = y - oy
    xn = c * xs - s * ys + ox
    yn = s * xs + c * ys + oy
    return xn, yn


def transform_to_new_ego_frame(
    data: dict[str, torch.Tensor],
    new_ego_x: float,
    new_ego_y: float,
    new_ego_cos: float,
    new_ego_sin: float,
) -> dict[str, torch.Tensor]:
    """Transform every world feature so the new ego pose is at origin.

    `new_ego_*` are in the CURRENT ego frame. After this transform:
      - the new ego pose is at (0, 0) facing +x
      - everything else (route_lanes, neighbors, line_strings, polygons, ...)
        is re-expressed in the new ego frame.

    Mirrors ``StatePerturbation.centric_transform`` but operates on raw
    (un-normalized) data and uses an explicit (x, y, cos, sin) center.

    Velocity/acceleration channels of ego_current_state stay UNCHANGED in
    magnitude — they only get rotated by `-new_yaw`.
    """
    out = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    # We want to apply the inverse transform: translate by -(nx, ny), then
    # rotate by -new_yaw. Rotation matrix R(-yaw) = [[cos, sin], [-sin, cos]].
    cs = float(new_ego_cos)
    sn = float(new_ego_sin)
    # forward rotation by -yaw: (x', y') = (cs * x + sn * y, -sn * x + cs * y)
    # We'll express via "rotate by (c=cs, s=-sn)" to reuse helpers.
    rc = cs
    rs = -sn

    def _xform_pos(t: torch.Tensor, x_idx: int = 0, y_idx: int = 1) -> torch.Tensor:
        x = t[..., x_idx] - new_ego_x
        y = t[..., y_idx] - new_ego_y
        t[..., x_idx] = rc * x - rs * y
        t[..., y_idx] = rs * x + rc * y
        return t

    def _xform_dir(t: torch.Tensor, cx_idx: int, cy_idx: int) -> torch.Tensor:
        cx = t[..., cx_idx]
        cy = t[..., cy_idx]
        t[..., cx_idx] = rc * cx - rs * cy
        t[..., cy_idx] = rs * cx + rc * cy
        return t

    # ego_current_state [B, 10]: 0,1 x,y; 2,3 cos,sin; 4,5 vx,vy; 6,7 ax,ay; 8,9 steer,yaw_rate
    if "ego_current_state" in out:
        ecs = out["ego_current_state"]
        ecs = _xform_pos(ecs, 0, 1)
        ecs = _xform_dir(ecs, 2, 3)
        ecs = _xform_dir(ecs, 4, 5)
        ecs = _xform_dir(ecs, 6, 7)
        out["ego_current_state"] = ecs

    # ego_agent_past [B, T, 4]: 0,1 x,y; 2,3 cos,sin
    if "ego_agent_past" in out:
        eap = out["ego_agent_past"]
        eap = _xform_pos(eap, 0, 1)
        eap = _xform_dir(eap, 2, 3)
        out["ego_agent_past"] = eap

    # neighbor_agents_past [B, N, T, 11]: 0,1 x,y; 2,3 cos,sin; 4,5 vx,vy
    if "neighbor_agents_past" in out:
        nap = out["neighbor_agents_past"]
        if nap.numel() > 0:
            mask = torch.sum(torch.ne(nap[..., :6], 0), dim=-1) == 0
            nap = _xform_pos(nap, 0, 1)
            nap = _xform_dir(nap, 2, 3)
            nap = _xform_dir(nap, 4, 5)
            nap[mask] = 0.0
            out["neighbor_agents_past"] = nap

    # lanes [B, S, P, 12]: 0,1 center; 2,3 dir; 4,5 left-offset (vector); 6,7 right-offset (vector)
    # NOTE: 4-7 are RELATIVE direction vectors (rotate only), per
    # data_augmentation.centric_transform — see vector_transform() with no center.
    if "lanes" in out:
        ln = out["lanes"]
        if ln.numel() > 0:
            mask = torch.sum(torch.ne(ln[..., :8], 0), dim=-1) == 0
            ln = _xform_pos(ln, 0, 1)
            ln = _xform_dir(ln, 2, 3)
            ln = _xform_dir(ln, 4, 5)
            ln = _xform_dir(ln, 6, 7)
            ln[mask] = 0.0
            out["lanes"] = ln

    # route_lanes [B, S, P, 33]: 0,1 center; 2,3 dir; 4,5 left-offset; 6,7 right-offset
    if "route_lanes" in out:
        rl = out["route_lanes"]
        if rl.numel() > 0:
            mask = torch.sum(torch.ne(rl[..., :8], 0), dim=-1) == 0
            rl = _xform_pos(rl, 0, 1)
            rl = _xform_dir(rl, 2, 3)
            rl = _xform_dir(rl, 4, 5)
            rl = _xform_dir(rl, 6, 7)
            rl[mask] = 0.0
            out["route_lanes"] = rl

    # polygons [B, P, V, 2]: 0,1 x,y
    if "polygons" in out:
        pg = out["polygons"]
        if pg.numel() > 0:
            mask = torch.sum(torch.ne(pg, 0), dim=-1) == 0
            pg = _xform_pos(pg, 0, 1)
            pg[mask] = 0.0
            out["polygons"] = pg

    # line_strings [B, L, V, ?]: 0,1 x,y
    if "line_strings" in out:
        ls = out["line_strings"]
        if ls.numel() > 0:
            mask = torch.sum(torch.ne(ls, 0), dim=-1) == 0
            ls = _xform_pos(ls, 0, 1)
            ls[mask] = 0.0
            out["line_strings"] = ls

    # static_objects [B, N, 10]: 0,1 x,y; 2,3 cos,sin
    if "static_objects" in out:
        so = out["static_objects"]
        if so.numel() > 0:
            mask = torch.sum(torch.ne(so[..., :10], 0), dim=-1) == 0
            so = _xform_pos(so, 0, 1)
            so = _xform_dir(so, 2, 3)
            so[mask] = 0.0
            out["static_objects"] = so

    # goal_pose [B, 4]: 0,1 x,y; 2,3 cos,sin
    if "goal_pose" in out:
        gp = out["goal_pose"]
        if gp.numel() > 0:
            gp = _xform_pos(gp, 0, 1)
            gp = _xform_dir(gp, 2, 3)
            out["goal_pose"] = gp

    return out


def closed_loop_rollout(
    model,
    model_args,
    init_data: dict[str, torch.Tensor],
    init_route_lanes: torch.Tensor,
    n_steps: int = 80,
    advance_k: int = 1,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out a closed-loop trajectory.

    At each step:
      1. Run the model -> pred[80, 4] in the current ego frame.
      2. Take pred[advance_k] as the new ego pose (in current ego frame).
      3. Update ego_current_state, ego_agent_past, and re-express world
         features in the new ego frame so the new ego is at origin.
      4. Track the cumulative pose in the INITIAL ego frame.

    Args:
        model: Diffusion_Planner (with optional LoRA already merged/loaded).
        model_args: Config object.
        init_data: Raw (un-normalized) perturbed scene dict (B=1).
        init_route_lanes: route_lanes tensor in the INITIAL ego frame, used
            to score the rolled-out positions against the original centerline.
        n_steps: Number of closed-loop steps (8 sec at dt=0.1 -> 80).
        advance_k: Which prediction step to advance to per loop (1 = 0.1 s
            ahead). Larger k = coarser look-ahead, fewer inferences. We always
            run n_steps inferences regardless.
        dt: timestep in seconds (informational only).

    Returns:
        positions: [n_steps + 1, 2] cumulative ego (x, y) in INITIAL frame.
        lateral_dists: [n_steps + 1] perpendicular distance to nearest
            init_route_lanes segment, including step 0.
    """
    data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in init_data.items()}

    positions = [np.array([0.0, 0.0])]
    # Cumulative transform: world point in INITIAL frame can be reconstructed
    # as cum_pos + R(cum_yaw) @ point_in_current_frame.
    cum_x = 0.0
    cum_y = 0.0
    cum_cos = 1.0
    cum_sin = 0.0

    for _ in range(n_steps):
        pred = deterministic_predict(model, model_args, data)  # [T, 4] in current frame
        if pred.shape[0] <= advance_k:
            advance_k = pred.shape[0] - 1
        nx_loc = float(pred[advance_k, 0])
        ny_loc = float(pred[advance_k, 1])
        ncos_loc = float(pred[advance_k, 2])
        nsin_loc = float(pred[advance_k, 3])
        # Normalize (cos, sin) — the model output may not be unit length.
        norm = float(np.hypot(ncos_loc, nsin_loc)) or 1.0
        ncos_loc /= norm
        nsin_loc /= norm

        # Estimate the new ego velocity from the predicted finite-difference.
        # We compare pred[advance_k+1] to pred[advance_k] when available, else
        # compare pred[advance_k] to the previous origin (0, 0).
        if advance_k + 1 < pred.shape[0]:
            dvx_loc = float(pred[advance_k + 1, 0] - pred[advance_k, 0]) / dt
            dvy_loc = float(pred[advance_k + 1, 1] - pred[advance_k, 1]) / dt
        else:
            dvx_loc = float(pred[advance_k, 0]) / dt
            dvy_loc = float(pred[advance_k, 1]) / dt
        # Rotate (dvx_loc, dvy_loc) into the NEW ego frame: rotate by -new_yaw,
        # i.e. multiply by (ncos, -nsin).
        new_vx = ncos_loc * dvx_loc + nsin_loc * dvy_loc
        new_vy = -nsin_loc * dvx_loc + ncos_loc * dvy_loc

        # Compose with cumulative transform: new pos in initial frame =
        # cum + R(cum_cos, cum_sin) @ (nx_loc, ny_loc)
        new_world_x = cum_x + cum_cos * nx_loc - cum_sin * ny_loc
        new_world_y = cum_y + cum_sin * nx_loc + cum_cos * ny_loc
        # New cumulative yaw (cos, sin) = (cum_cos + i*cum_sin) * (ncos + i*nsin)
        new_cum_cos = cum_cos * ncos_loc - cum_sin * nsin_loc
        new_cum_sin = cum_sin * ncos_loc + cum_cos * nsin_loc

        positions.append(np.array([new_world_x, new_world_y]))
        cum_x, cum_y, cum_cos, cum_sin = (new_world_x, new_world_y, new_cum_cos, new_cum_sin)

        # Build the next ego_agent_past: shift the existing past by 1 slot
        # and append the OLD ego pose (which is at -pred[advance_k] expressed
        # in the new ego frame, i.e. (0,0) in the OLD frame -> rotated/translated).
        # We do this BEFORE transforming the rest of the world.
        if "ego_agent_past" in data:
            eap = data["ego_agent_past"].clone()  # [B, T, 4]
            old_origin = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=eap.dtype, device=eap.device)
            T = eap.shape[1]
            # Drop oldest, shift, append old origin (in OLD ego frame).
            eap = torch.cat(
                [eap[:, 1:T], old_origin.view(1, 1, 4).expand(eap.shape[0], 1, 4)], dim=1
            )
            data["ego_agent_past"] = eap

        # Now transform to the new ego frame.
        data = transform_to_new_ego_frame(
            data,
            nx_loc,
            ny_loc,
            ncos_loc,
            nsin_loc,
        )
        # Force ego_current_state to be at origin facing +x in new frame, and
        # update vx/vy from the model's predicted next motion. Without this,
        # ego_current_state[4:6] is stale (still rotated initial velocity)
        # and the model never gets the signal that it's slowing/speeding up.
        if "ego_current_state" in data:
            ecs = data["ego_current_state"]
            ecs[..., 0] = 0.0
            ecs[..., 1] = 0.0
            ecs[..., 2] = 1.0
            ecs[..., 3] = 0.0
            ecs[..., 4] = float(new_vx)
            ecs[..., 5] = float(new_vy)
            # acceleration / steer / yaw_rate left as-is (rotated by transform)
            data["ego_current_state"] = ecs

    pos_arr = np.stack(positions, axis=0)  # [n+1, 2]
    lateral = per_step_lateral_distance(pos_arr, init_route_lanes)
    return pos_arr, lateral


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def deterministic_predict(model, model_args, data: dict[str, torch.Tensor]) -> np.ndarray:
    """Single deterministic forward pass. Returns ego trajectory [T, 4]."""
    device = next(model.parameters()).device
    batch = _stack_scene_data([data], device)
    norm_batch = _normalize_batch(batch, model_args)

    B = norm_batch["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    norm_batch["sampled_trajectories"] = torch.zeros(B, P, future_len + 1, 4, device=device)

    # Disable any pre-existing guidance fn for the deterministic baseline.
    decoder = model.module.decoder if hasattr(model, "module") else model.decoder
    saved_fn = decoder._guidance_fn
    decoder._guidance_fn = None
    try:
        with torch.no_grad():
            _, outputs = model(norm_batch)
    finally:
        decoder._guidance_fn = saved_fn
    return outputs["prediction"][0, 0].detach().cpu().numpy()  # [T, 4]


def k_predict(
    model,
    model_args,
    data: dict[str, torch.Tensor],
    K: int,
    variant: str,
    noise_range: tuple[float, float],
    gt_max_speed: float,
    use_route_cl: bool,
) -> np.ndarray:
    """Generate K trajectories using the training variant. Returns [K, T, 4]."""
    device = next(model.parameters()).device
    batch = _stack_scene_data([data], device)
    norm_batch = _normalize_batch(batch, model_args)
    with torch.no_grad():
        trajs = generate_all_scenes_batched(
            model,
            model_args,
            norm_batch,
            K=K,
            noise_range=noise_range,
            device=device,
            gen_chunk_size=K,
            gt_max_speed=gt_max_speed,
            generation_variant=variant,
            use_route_cl_guidance=use_route_cl,
        )
    return trajs[0].detach().cpu().numpy()  # [K, T, 4]


def gt_max_speed_from_data(data: dict[str, torch.Tensor]) -> float:
    """Compute GT max speed for speed-guidance scaling, mirroring the trainer."""
    if "ego_agent_future" in data:
        gt = data["ego_agent_future"]
        if gt.dim() == 3:
            gt = gt[0]
        gt_np = gt.detach().cpu().numpy()
        valid = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
        if valid.sum() >= 5:
            vel = np.diff(gt_np[valid][:, :2], axis=0) / 0.1
            return float(np.linalg.norm(vel, axis=-1).max())
    # Fallback: use ego_current_state speed magnitude (channels 4,5 = vx,vy)
    ecs = data["ego_current_state"]
    if ecs.dim() == 2:
        ecs = ecs[0]
    return float(torch.linalg.vector_norm(ecs[4:6]).item()) or 3.0


# ---------------------------------------------------------------------------
# Trial builder
# ---------------------------------------------------------------------------


def _build_trials(
    data: dict[str, torch.Tensor],
    n_unit: np.ndarray,
    kinds: list[str],
    offsets: list[float],
    yaw_degs: list[float],
    vel_pcts: list[float],
    sides: list[float],
    combined_offset: float,
    combined_yaw_deg: float,
) -> list[dict]:
    """Generate the list of (kind, magnitude, side, perturbed_data) trials."""
    out = []
    for kind in kinds:
        if kind == "parallel":
            for off in offsets:
                for s in sides:
                    out.append(
                        {
                            "kind": "parallel",
                            "magnitude": float(off),
                            "side_str": "+" if s > 0 else "-",
                            "data": apply_lateral_shift(data, n_unit, s * off),
                            "ref_magnitude": float(abs(off)),
                        }
                    )
        elif kind == "yaw":
            for deg in yaw_degs:
                for s in sides:
                    yaw_rad = np.deg2rad(s * deg)
                    out.append(
                        {
                            "kind": "yaw",
                            "magnitude": float(deg),
                            "side_str": "+" if s > 0 else "-",
                            "data": apply_yaw_perturbation(data, yaw_rad),
                            # Yaw has no direct lateral magnitude. Use 0.0 to
                            # signal "fall back to t0 distance" in the caller —
                            # recovery rate becomes (t0 - t79)/t0, i.e. the
                            # fraction of initial lateral error that was closed.
                            "ref_magnitude": 0.0,
                        }
                    )
        elif kind == "velocity":
            for pct in vel_pcts:
                for s in sides:
                    scale = 1.0 + (s * pct / 100.0)  # +50% -> 1.5, -50% -> 0.5
                    out.append(
                        {
                            "kind": "velocity",
                            "magnitude": float(pct),
                            "side_str": "+" if s > 0 else "-",
                            "data": apply_velocity_perturbation(data, scale),
                            # Velocity has no direct lateral magnitude. Same
                            # convention as yaw — fall back to t0 so the rate
                            # is "fraction of initial lateral error closed".
                            "ref_magnitude": 0.0,
                        }
                    )
        elif kind == "combined":
            for s in sides:
                yaw_rad = np.deg2rad(s * combined_yaw_deg)
                out.append(
                    {
                        "kind": "combined",
                        "magnitude": float(combined_yaw_deg),  # report yaw as primary
                        "side_str": "+" if s > 0 else "-",
                        "data": apply_combined_perturbation(
                            data, n_unit, s * combined_offset, yaw_rad
                        ),
                        "ref_magnitude": float(abs(combined_offset)),
                    }
                )
        else:
            print(f"  [warn] unknown perturbation kind '{kind}' — skipped")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--scenes", type=str, required=True, help="JSON list of NPZ scene paths")
    parser.add_argument(
        "--offsets",
        type=str,
        default="0.25,0.50,0.75,1.00",
        help="Comma-separated lateral offsets (meters) for parallel + combined",
    )
    parser.add_argument(
        "--yaw_degs",
        type=str,
        default="2,5,10",
        help="Comma-separated yaw perturbations in degrees (each tested + and -)",
    )
    parser.add_argument(
        "--vel_pcts",
        type=str,
        default="30,50",
        help="Comma-separated velocity scale percentages (each tested as +pct/-pct)",
    )
    parser.add_argument(
        "--combined_offset",
        type=float,
        default=0.5,
        help="Lateral offset (m) used for the 'combined' kind",
    )
    parser.add_argument(
        "--combined_yaw_deg",
        type=float,
        default=5.0,
        help="Yaw perturbation (deg) used for the 'combined' kind",
    )
    parser.add_argument(
        "--perturbation_kinds",
        type=str,
        default="parallel",
        help="Comma-separated perturbation kinds: parallel, yaw, velocity, combined",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="GRPO config JSON (variant + reward weights)"
    )
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--candidate_name", type=str, default="model")
    parser.add_argument("--noise_min", type=float, default=0.5)
    parser.add_argument("--noise_max", type=float, default=2.0)
    parser.add_argument(
        "--closed_loop",
        action="store_true",
        help="Also run a closed-loop rollout (re-predict every step)",
    )
    parser.add_argument(
        "--closed_loop_steps",
        type=int,
        default=80,
        help="Number of closed-loop steps (80 = 8 sec at 0.1 s)",
    )
    parser.add_argument(
        "--closed_loop_advance_k",
        type=int,
        default=0,
        help="Which prediction step to advance to per loop. "
        "0 = pred[0] (0.1 s ahead), 1 = pred[1] (0.2 s), etc. "
        "n_steps * (advance_k+1) * 0.1 = total simulated seconds",
    )
    parser.add_argument(
        "--skip_K", action="store_true", help="Skip the K-sample generation (faster diagnostics)"
    )
    args = parser.parse_args()

    device = torch.device(DEVICE)

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

    # ---- Load config to get generation_variant + use_route_cl_guidance ----
    with open(args.config) as f:
        cfg_json = json.load(f)
    variant = cfg_json.get("generation_variant", "default")
    use_route_cl = bool(cfg_json.get("use_route_cl_guidance", False))
    slot_labels = get_generation_config_labels_for_variant(variant, args.K)

    # ---- Scenes + offsets ----
    with open(args.scenes) as f:
        scene_paths = json.load(f)
    offsets_pos = [float(x) for x in args.offsets.split(",") if x.strip()]
    yaw_degs = [float(x) for x in args.yaw_degs.split(",") if x.strip()]
    vel_pcts = [float(x) for x in args.vel_pcts.split(",") if x.strip()]
    perturbation_kinds = [x.strip() for x in args.perturbation_kinds.split(",") if x.strip()]
    sides = [+1.0, -1.0]

    print(f"[recovery_test] candidate={args.candidate_name}")
    print(f"  model_path={args.model_path}")
    print(f"  lora_path={args.lora_path}")
    print(f"  variant={variant}  use_route_cl_guidance={use_route_cl}")
    print(f"  scenes={len(scene_paths)}  K={args.K}  closed_loop={args.closed_loop}")
    print(f"  perturbation_kinds={perturbation_kinds}")
    print(f"  offsets={offsets_pos}  yaw_degs={yaw_degs}  vel_pcts={vel_pcts}")

    results = []
    det_recovery_pool = []
    bestK_recovery_pool = []
    cl_recovery_pool: list[float] = []

    for scene_path in scene_paths:
        try:
            data = load_npz_data(scene_path, device)
        except Exception as e:
            print(f"  [skip] {Path(scene_path).name}: {e}")
            continue

        # Tangent + normal at origin (ego frame)
        t_unit, n_unit = get_tangent_at_origin(data["route_lanes"])
        v_high = gt_max_speed_from_data(data)

        # Baseline t=0 distance from ego origin to nearest route_lane segment.
        origin_dist = float(
            per_step_lateral_distance(np.array([[0.0, 0.0]]), data["route_lanes"])[0]
        )

        # Build (kind, magnitude_label, signed_value, perturbed_data) trial list
        trials = _build_trials(
            data=data,
            n_unit=n_unit,
            kinds=perturbation_kinds,
            offsets=offsets_pos,
            yaw_degs=yaw_degs,
            vel_pcts=vel_pcts,
            sides=sides,
            combined_offset=args.combined_offset,
            combined_yaw_deg=args.combined_yaw_deg,
        )

        for trial in trials:
            kind = trial["kind"]
            mag = trial["magnitude"]
            side_str = trial["side_str"]
            shifted = trial["data"]
            # Reference perturbation magnitude — used as denominator of recovery
            # rate. For lateral perturbations this is meters. For yaw it's the
            # initial heading-induced lateral drift, but since our t0 distance
            # ALREADY includes the rotated past, we use a fallback of t0_dist.
            ref_mag = trial["ref_magnitude"]

            # 1. ONE-SHOT deterministic
            det_traj = deterministic_predict(model, model_args, shifted)
            det_d = per_step_lateral_distance(det_traj[:, :2], data["route_lanes"])
            det_t0 = float(det_d[0])
            det_t40 = float(det_d[40]) if det_d.shape[0] > 40 else float("nan")
            det_t79 = float(det_d[-1])
            denom = max(ref_mag if ref_mag > 1e-3 else det_t0, 1e-3)
            det_recovery = (det_t0 - det_t79) / denom

            row = {
                "scene": Path(scene_path).name,
                "kind": kind,
                "magnitude": mag,
                "side": side_str,
                "tangent_unit": [float(t_unit[0]), float(t_unit[1])],
                "normal_unit": [float(n_unit[0]), float(n_unit[1])],
                "origin_dist": origin_dist,
                "ref_magnitude": ref_mag,
                "det": {
                    "t0": det_t0,
                    "t40": det_t40,
                    "t79": det_t79,
                    "recovery_rate": det_recovery,
                },
            }

            # 2. K-sample best (skipped when --skip_K)
            if not args.skip_K:
                k_trajs = k_predict(
                    model,
                    model_args,
                    shifted,
                    K=args.K,
                    variant=variant,
                    noise_range=(args.noise_min, args.noise_max),
                    gt_max_speed=v_high,
                    use_route_cl=use_route_cl,
                )  # [K, T, 4]

                k_t79 = []
                for ki in range(k_trajs.shape[0]):
                    d_k = per_step_lateral_distance(k_trajs[ki, :, :2], data["route_lanes"])
                    k_t79.append(float(d_k[-1]))
                k_t79_arr = np.array(k_t79)
                best_k = int(np.argmin(np.abs(k_t79_arr)))
                best_t79 = float(k_t79_arr[best_k])
                best_slot = slot_labels[best_k] if best_k < len(slot_labels) else f"slot_{best_k}"
                d_best = per_step_lateral_distance(k_trajs[best_k, :, :2], data["route_lanes"])
                best_t0 = float(d_best[0])
                best_t40 = float(d_best[40]) if d_best.shape[0] > 40 else float("nan")
                best_recovery = (best_t0 - best_t79) / denom

                row["best_K"] = {
                    "t0": best_t0,
                    "t40": best_t40,
                    "t79": best_t79,
                    "recovery_rate": best_recovery,
                    "winning_slot": best_slot,
                    "winning_k": best_k,
                    "k_t79_all": [float(x) for x in k_t79_arr.tolist()],
                }
                bestK_recovery_pool.append(best_recovery)

            # 3. Closed-loop rollout (optional)
            cl_recovery = None
            if args.closed_loop:
                pos_cl, lat_cl = closed_loop_rollout(
                    model,
                    model_args,
                    shifted,
                    init_route_lanes=data["route_lanes"],
                    n_steps=args.closed_loop_steps,
                    advance_k=args.closed_loop_advance_k,
                )
                cl_t0 = float(lat_cl[0])
                cl_t40 = float(lat_cl[40]) if lat_cl.shape[0] > 40 else float("nan")
                cl_t79 = float(lat_cl[-1])
                cl_recovery = (cl_t0 - cl_t79) / denom
                row["closed_loop"] = {
                    "t0": cl_t0,
                    "t40": cl_t40,
                    "t79": cl_t79,
                    "recovery_rate": cl_recovery,
                    "n_steps": int(args.closed_loop_steps),
                    "advance_k": int(args.closed_loop_advance_k),
                    "lateral_series": [float(x) for x in lat_cl.tolist()],
                    "trajectory_xy": [[float(p[0]), float(p[1])] for p in pos_cl.tolist()],
                }
                cl_recovery_pool.append(cl_recovery)

            det_recovery_pool.append(det_recovery)

            cl_str = ""
            if cl_recovery is not None:
                cl_str = f"  cl t0={row['closed_loop']['t0']:.2f} t79={row['closed_loop']['t79']:.2f} rec={cl_recovery:+.2f}"
            best_str = ""
            if "best_K" in row:
                best_str = (
                    f"  best t79={row['best_K']['t79']:.2f} rec={row['best_K']['recovery_rate']:+.2f}"
                    f" slot={row['best_K']['winning_slot']}"
                )
            print(
                f"  {Path(scene_path).stem[-22:]} {kind:9s} {mag:+.2f}{side_str}  "
                f"det t0={det_t0:.2f} t79={det_t79:.2f} rec={det_recovery:+.2f}"
                f"{best_str}{cl_str}"
            )
            results.append(row)

    det_mean = float(np.mean(det_recovery_pool)) if det_recovery_pool else 0.0
    best_mean = float(np.mean(bestK_recovery_pool)) if bestK_recovery_pool else 0.0
    cl_mean = float(np.mean(cl_recovery_pool)) if cl_recovery_pool else 0.0
    aggregate = {
        "det_mean_recovery": det_mean,
        "best_K_mean_recovery": best_mean,
        "closed_loop_mean_recovery": cl_mean,
        "guidance_helps": best_mean > det_mean + 0.05 if bestK_recovery_pool else False,
        "n_results": len(results),
        "n_closed_loop": len(cl_recovery_pool),
    }

    # Per-kind breakdown — useful when mixing parallel/yaw/velocity in one run.
    by_kind: dict[str, dict] = {}
    for kind in perturbation_kinds:
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            continue
        by_kind[kind] = {
            "n": len(rows),
            "det_mean_recovery": float(np.mean([r["det"]["recovery_rate"] for r in rows])),
        }
        if any("best_K" in r for r in rows):
            by_kind[kind]["best_K_mean_recovery"] = float(
                np.mean([r["best_K"]["recovery_rate"] for r in rows if "best_K" in r])
            )
        if any("closed_loop" in r for r in rows):
            by_kind[kind]["closed_loop_mean_recovery"] = float(
                np.mean([r["closed_loop"]["recovery_rate"] for r in rows if "closed_loop" in r])
            )
    aggregate["by_kind"] = by_kind

    out = {
        "candidate_name": args.candidate_name,
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "variant": variant,
        "use_route_cl_guidance": use_route_cl,
        "K": args.K,
        "scenes": [str(p) for p in scene_paths],
        "offsets": offsets_pos,
        "yaw_degs": yaw_degs,
        "vel_pcts": vel_pcts,
        "perturbation_kinds": perturbation_kinds,
        "closed_loop": bool(args.closed_loop),
        "closed_loop_steps": int(args.closed_loop_steps),
        "closed_loop_advance_k": int(args.closed_loop_advance_k),
        "results": results,
        "aggregate": aggregate,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[recovery_test] Aggregate over {aggregate['n_results']} trials:")
    print(f"  det mean recovery:         {det_mean:+.3f}")
    if bestK_recovery_pool:
        print(f"  best-K mean recovery:      {best_mean:+.3f}")
    if cl_recovery_pool:
        print(f"  closed-loop mean recovery: {cl_mean:+.3f}  (n={len(cl_recovery_pool)})")
    if by_kind:
        print("  by kind:")
        for kind, st in by_kind.items():
            extra = ""
            if "best_K_mean_recovery" in st:
                extra += f"  bestK={st['best_K_mean_recovery']:+.3f}"
            if "closed_loop_mean_recovery" in st:
                extra += f"  cl={st['closed_loop_mean_recovery']:+.3f}"
            print(f"    {kind:9s} n={st['n']:3d}  det={st['det_mean_recovery']:+.3f}{extra}")
    if bestK_recovery_pool:
        print(f"  guidance_helps:       {aggregate['guidance_helps']}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
