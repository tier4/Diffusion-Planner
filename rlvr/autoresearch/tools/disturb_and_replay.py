#!/usr/bin/env python3
"""Manufacture training scenes by perturbing existing warm scenes, then replay
the BASELINE MODEL's predicted future from the perturbed pose to act as the
new ``ego_agent_future`` GT.

Three perturbation kinds are applied per scene:

  (1) Subtle pose perturbation (Friday rule, |x|,|y| <= 1.5m, |yaw| <= 10 deg).
      Implemented as: shift ``ego_current_state[0:4]`` to encode a non-zero
      pose. ``ego_agent_past`` is also rigid-shifted by the same transform so
      the past traces lead consistently into the perturbed current pose.

  (2) Parallel-offset history shift. Find the route centerline tangent at
      ego's current arc_s (use the closest valid route_lanes point as the
      anchor) and compute the perpendicular vector. Apply a lateral offset:
      shift ``ego_current_state[0:2]`` and every step of
      ``ego_agent_past[:, 0:2]`` by ``offset * perp``. Heading stays parallel.

  (3) Random pose+history jitter combo. Same as (1) plus per-step noise on
      ``ego_agent_past`` to simulate noisy history estimates.

Per scene we produce 1 baseline (no perturbation) + 2 kind-1 + 2 kind-2
variants (matching the apply order in
``project_disturb_replay_perturbation_specs.md``).

For every kept variant (including ``base``), the BASELINE MODEL is run in
deterministic inference mode on the perturbed observation. The 80-step
prediction (in ego frame, ``(x, y, cos, sin)``) is converted to
``(x, y, heading_rad)`` and stored as the new ``ego_agent_future``. This
gives a recovery-style target that closes the perturbation gap via the
baseline's learned policy — exactly what we want RSFT to imitate.

Variants are rejected when the perturbed pose is out-of-lane at t=0 (we use
``compute_lane_departure_penalty`` so the geometry exactly matches training).

Usage::

    python -m rlvr.autoresearch.tools.disturb_and_replay \
        --scenes /path/to/warm.json \
        --output_dir /path/to/aug_dir \
        --output_scene_list /path/to/aug_list.json \
        --base_model /path/to/x2_model_base/best_model.pth \
        [--n_per_scene 5] [--offsets 0.3,0.5,0.8] \
        [--reject_out_of_lane] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from preference_optimization.utils import load_npz_data
from rlvr.grpo_trainer_batched import _normalize_batch, _stack_scene_data
from rlvr.reward import compute_lane_departure_penalty


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _wrap_angle(rad: float) -> float:
    """Wrap radians to (-pi, pi]."""
    return float((rad + math.pi) % (2 * math.pi) - math.pi)


def _rotate_past_about_pivot(
    past: np.ndarray, pivot_x: float, pivot_y: float, yaw_rad: float
) -> np.ndarray:
    """Rotate ``ego_agent_past`` (T, 3) about (pivot_x, pivot_y) by ``yaw_rad``.

    Mirrors ``recovery_test.apply_yaw_perturbation``. Past format is
    ``(x, y, heading_rad)`` so heading is bumped by ``yaw_rad``.
    """
    out = past.copy()
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    x_off = out[:, 0] - pivot_x
    y_off = out[:, 1] - pivot_y
    out[:, 0] = pivot_x + c * x_off - s * y_off
    out[:, 1] = pivot_y + s * x_off + c * y_off
    out[:, 2] = np.array(
        [_wrap_angle(float(h) + yaw_rad) for h in out[:, 2]], dtype=np.float32
    )
    return out


def _apply_yaw_to_state(
    state: np.ndarray, yaw_rad: float, set_position: tuple[float, float] | None = None
) -> np.ndarray:
    """Rotate ego_current_state heading + velocity + acceleration by ``yaw_rad``.

    Layout: [x, y, cos, sin, vx, vy, ax, ay, steering, yaw_rate].

    If ``set_position`` is given (e.g. for combined parallel+yaw), the (x, y)
    channels are overwritten BEFORE the rotation is applied. Otherwise the
    position is left untouched (yaw rotates about the current ego position).
    """
    out = state.copy()
    if set_position is not None:
        out[0] = float(set_position[0])
        out[1] = float(set_position[1])
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    cos_old, sin_old = float(out[2]), float(out[3])
    out[2] = c * cos_old - s * sin_old
    out[3] = s * cos_old + c * sin_old
    vx_old, vy_old = float(out[4]), float(out[5])
    out[4] = c * vx_old - s * vy_old
    out[5] = s * vx_old + c * vy_old
    ax_old, ay_old = float(out[6]), float(out[7])
    out[6] = c * ax_old - s * ay_old
    out[7] = s * ax_old + c * ay_old
    return out


def _route_tangent_at_origin(route_lanes: np.ndarray) -> np.ndarray:
    """Return the route centerline unit tangent vector closest to the ego origin.

    Falls back to ``(1, 0)`` (ego forward) when no valid route point is found.
    Route lane channels [0:2] are positions and [2:4] are tangent dx/dy in
    ego frame (see ``scenario_generation/tensor_converter.py::_build_lanes``).
    """
    # route_lanes: (N_lanes, P, D)
    pts = route_lanes[..., :2]  # (N, P, 2)
    tan = route_lanes[..., 2:4]  # (N, P, 2)
    valid = np.abs(route_lanes[..., :8]).sum(axis=-1) > 1e-3
    if not valid.any():
        return np.array([1.0, 0.0], dtype=np.float32)
    flat_pts = pts.reshape(-1, 2)
    flat_tan = tan.reshape(-1, 2)
    flat_valid = valid.reshape(-1)
    flat_pts = flat_pts[flat_valid]
    flat_tan = flat_tan[flat_valid]
    if flat_pts.size == 0:
        return np.array([1.0, 0.0], dtype=np.float32)
    dists = np.linalg.norm(flat_pts, axis=1)
    j = int(np.argmin(dists))
    t = flat_tan[j]
    n = float(np.linalg.norm(t))
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (t / n).astype(np.float32)


def _build_segments(route_lanes: np.ndarray) -> np.ndarray:
    """Build [N_seg, 2, 2] consecutive valid centerline segments per lane.

    Used for measuring lateral distance from a point to the nearest route
    centerline polyline (same convention as ``recovery_test``).
    """
    if route_lanes.ndim == 2:
        route_lanes = route_lanes[None]
    rl = route_lanes  # [S, P, D]
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
    """Per-point min perpendicular distance to any of the supplied segments.

    Args:
        points: [T, 2]
        segs: [N, 2, 2]
    Returns:
        [T] distances; ``nan`` if ``segs`` is empty.
    """
    if segs.shape[0] == 0:
        return np.full((points.shape[0],), np.nan)
    a = segs[:, 0, :]  # [N, 2]
    b = segs[:, 1, :]  # [N, 2]
    ab = b - a
    ab_len_sq = np.sum(ab * ab, axis=-1).clip(min=1e-9)
    ap = points[:, None, :] - a[None, :, :]
    dot = np.sum(ap * ab[None, :, :], axis=-1)
    t = (dot / ab_len_sq[None, :]).clip(0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * ab[None, :, :]
    diff = points[:, None, :] - proj
    dist = np.linalg.norm(diff, axis=-1)
    return dist.min(axis=-1)


def _apply_rigid_to_past(
    past: np.ndarray, dx: float, dy: float, dtheta: float
) -> np.ndarray:
    """Shift ``ego_agent_past`` so past steps lead into the perturbed current pose.

    Past stays rigidly attached to the ego: every (px, py, ph) gets rotated by
    ``dtheta`` around the new origin and translated by (dx, dy). Heading is
    bumped by ``dtheta`` (then wrapped).

    Args:
        past: (T, 3) array of (x, y, heading_rad).
    Returns:
        Same shape, perturbed.
    """
    out = past.copy()
    c, s = math.cos(dtheta), math.sin(dtheta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = out[:, :2] @ rot.T  # rotate around origin
    xy[:, 0] += dx
    xy[:, 1] += dy
    out[:, :2] = xy
    out[:, 2] = np.array(
        [_wrap_angle(float(h) + dtheta) for h in out[:, 2]], dtype=np.float32
    )
    return out


def _set_ego_current_state(
    state: np.ndarray, dx: float, dy: float, dtheta: float, dv: float = 0.0
) -> np.ndarray:
    """Encode a perturbed pose in ``ego_current_state``.

    Layout: [x, y, cos, sin, vx, vy, ax, ay, steering, yaw_rate].
    We modify only [0:4] and [4] (longitudinal velocity).
    """
    out = state.copy()
    out[0] = dx
    out[1] = dy
    out[2] = math.cos(dtheta)
    out[3] = math.sin(dtheta)
    if dv != 0.0:
        out[4] = max(0.0, float(out[4]) + dv)
    return out


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    name: str  # short tag, e.g. "base", "subtle_L", "parallel_R_0.5", "yaw_L_10"
    dx: float
    dy: float
    dtheta_deg: float
    dv: float = 0.0
    history_jitter_std: float = 0.0  # per-step xy gaussian std (m), 0 = none
    # Mode selects how the perturbation is applied. Options:
    #   "shift" - existing kind 1/2 behavior: set position to (dx,dy), set
    #             heading to (cos(dtheta), sin(dtheta)) (overwrites). Velocity
    #             and acceleration are NOT rotated. Past is rigid-shifted then
    #             yaw-rotated about the new origin.
    #   "yaw"   - pure rotation about the current ego pose (0,0). dx/dy are
    #             ignored (must be 0). Velocity/accel rotate by dtheta.
    #             Past is rotated about (0,0) by dtheta.
    #   "combo" - parallel shift then yaw rotation about the SHIFTED pose.
    #             Position becomes (dx,dy); heading/vel/accel rotate by dtheta;
    #             past is rigid-shifted by (dx,dy) then yaw-rotated about
    #             (dx, dy).
    mode: str = "shift"


def _make_subtle_variant(
    rng: np.random.Generator, sign: float, side: str,
    perp: np.ndarray, tangent: np.ndarray,
) -> Variant:
    trans_mag = float(rng.uniform(0.5, 1.5))
    yaw_mag = float(rng.uniform(5.0, 10.0))
    long_frac = float(rng.uniform(-0.5, 0.5))
    d = sign * trans_mag * perp + long_frac * tangent
    dv = float(rng.uniform(-0.5, 0.5))
    return Variant(
        name=f"subtle_{side}",
        dx=float(d[0]),
        dy=float(d[1]),
        dtheta_deg=sign * yaw_mag,
        dv=dv,
        mode="shift",
    )


def _make_parallel_variant(
    rng: np.random.Generator, sign: float, side: str,
    perp: np.ndarray, offset_choices: list[float],
) -> Variant:
    mag = float(rng.choice(offset_choices))
    d = sign * mag * perp
    return Variant(
        name=f"parallel_{side}_{mag:.1f}",
        dx=float(d[0]),
        dy=float(d[1]),
        dtheta_deg=0.0,
        mode="shift",
    )


def _make_yaw_variant(
    rng: np.random.Generator, sign: float, side: str,
    yaw_choices: list[float],
) -> Variant:
    yaw_mag = float(rng.choice(yaw_choices))
    return Variant(
        name=f"yaw_{side}_{yaw_mag:.1f}",
        dx=0.0,
        dy=0.0,
        dtheta_deg=sign * yaw_mag,
        mode="yaw",
    )


def _make_combined_variant(
    rng: np.random.Generator, sign: float, side: str,
    perp: np.ndarray, offset_choices: list[float], yaw_choices: list[float],
) -> Variant:
    mag = float(rng.choice(offset_choices))
    yaw_mag = float(rng.choice(yaw_choices))
    d = sign * mag * perp
    return Variant(
        name=f"combo_{side}_{mag:.1f}_{yaw_mag:.1f}",
        dx=float(d[0]),
        dy=float(d[1]),
        dtheta_deg=sign * yaw_mag,
        mode="combo",
    )


def _build_variants(
    rng: np.random.Generator, offsets: list[float], n_per_scene: int,
    tangent: np.ndarray, kind: str = "default",
    yaw_degs: list[float] | None = None,
) -> list[Variant]:
    """Return a fixed-order list of variants for one scene.

    ``kind`` selects the recipe:

      - ``default``: 1 baseline + 2 subtle (shift) + 2 parallel (shift); extra
        slots become jitter combos. Original tool behavior.
      - ``parallel_only``: 1 baseline + (n-1) parallel variants split L/R
        across the supplied ``offsets``. Used for v4_highmag.
      - ``yaw_only``: 1 baseline + (n-1) yaw variants split L/R across
        ``yaw_degs``. Used for v5_yaw.
      - ``combined``: 1 baseline + 2 subtle + 2 parallel + 2 yaw + 2 combo
        (parallel+yaw). Used for v6_combined.

    For ``parallel_only`` and ``yaw_only``, when ``n_per_scene-1`` does not
    divide evenly across L/R variants, magnitudes are cycled so the order is
    deterministic given the seed.
    """
    perp = np.array([-tangent[1], tangent[0]], dtype=np.float32)  # left of tangent
    offset_choices = list(offsets) if offsets else [0.5]
    yaw_choices = list(yaw_degs) if yaw_degs else [5.0, 10.0, 15.0]

    variants: list[Variant] = [Variant("base", 0.0, 0.0, 0.0)]

    if kind == "default":
        for sign, side in ((+1.0, "L"), (-1.0, "R")):
            variants.append(_make_subtle_variant(rng, sign, side, perp, tangent))
        for sign, side in ((+1.0, "L"), (-1.0, "R")):
            variants.append(_make_parallel_variant(rng, sign, side, perp, offset_choices))

        # If n_per_scene > 5 we add kind-3 jitter combos.
        extra = max(0, n_per_scene - len(variants))
        for k in range(extra):
            sign = +1.0 if (k % 2 == 0) else -1.0
            side = "L" if sign > 0 else "R"
            trans_mag = float(rng.uniform(0.4, 1.0))
            yaw_mag = float(rng.uniform(3.0, 8.0))
            d = sign * trans_mag * perp
            variants.append(
                Variant(
                    name=f"jitter_{side}_{k}",
                    dx=float(d[0]),
                    dy=float(d[1]),
                    dtheta_deg=sign * yaw_mag,
                    history_jitter_std=0.10,
                )
            )

    elif kind == "parallel_only":
        # Distribute (n-1) variants L/R, cycling through magnitudes. For
        # n_per_scene=4 with offsets=[1.0, 1.5] this yields:
        #   L_1.0, R_1.0, L_1.5, R_1.5
        slots = max(0, n_per_scene - 1)
        for i in range(slots):
            mag_idx = i // 2
            mag = offset_choices[mag_idx % len(offset_choices)]
            sign = +1.0 if (i % 2 == 0) else -1.0
            side = "L" if sign > 0 else "R"
            d = sign * mag * perp
            variants.append(
                Variant(
                    name=f"parallel_{side}_{mag:.1f}",
                    dx=float(d[0]),
                    dy=float(d[1]),
                    dtheta_deg=0.0,
                    mode="shift",
                )
            )

    elif kind == "yaw_only":
        slots = max(0, n_per_scene - 1)
        for i in range(slots):
            yaw_idx = i // 2
            yaw_mag = yaw_choices[yaw_idx % len(yaw_choices)]
            sign = +1.0 if (i % 2 == 0) else -1.0
            side = "L" if sign > 0 else "R"
            variants.append(
                Variant(
                    name=f"yaw_{side}_{yaw_mag:.1f}",
                    dx=0.0,
                    dy=0.0,
                    dtheta_deg=sign * yaw_mag,
                    mode="yaw",
                )
            )

    elif kind == "combined":
        # 2 subtle + 2 parallel + 2 yaw + 2 combo, capped to n_per_scene-1.
        recipe = []
        for sign, side in ((+1.0, "L"), (-1.0, "R")):
            recipe.append(("subtle", sign, side))
        for sign, side in ((+1.0, "L"), (-1.0, "R")):
            recipe.append(("parallel", sign, side))
        for sign, side in ((+1.0, "L"), (-1.0, "R")):
            recipe.append(("yaw", sign, side))
        for sign, side in ((+1.0, "L"), (-1.0, "R")):
            recipe.append(("combo", sign, side))

        slots = max(0, n_per_scene - 1)
        for tag, sign, side in recipe[:slots]:
            if tag == "subtle":
                variants.append(_make_subtle_variant(rng, sign, side, perp, tangent))
            elif tag == "parallel":
                variants.append(_make_parallel_variant(rng, sign, side, perp, offset_choices))
            elif tag == "yaw":
                variants.append(_make_yaw_variant(rng, sign, side, yaw_choices))
            elif tag == "combo":
                variants.append(
                    _make_combined_variant(rng, sign, side, perp, offset_choices, yaw_choices)
                )

    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    if len(variants) > n_per_scene:
        variants = variants[:n_per_scene]
    return variants


def _apply_variant(
    npz: dict[str, np.ndarray], variant: Variant, rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return a perturbed copy of ``npz`` per ``variant``.

    NOTE: ``ego_agent_future`` is INTENTIONALLY NOT modified here. It will be
    overwritten later with the baseline model's prediction.
    """
    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in npz.items()}

    if variant.name == "base":
        return out

    dtheta = math.radians(variant.dtheta_deg)
    mode = getattr(variant, "mode", "shift")

    if mode == "yaw":
        # Pure rotation about (0,0). Velocity/accel rotate too.
        out["ego_current_state"] = _apply_yaw_to_state(
            out["ego_current_state"], dtheta, set_position=None
        )
        # Past rotates about ego pose (which is at the origin in ego frame).
        out["ego_agent_past"] = _rotate_past_about_pivot(
            out["ego_agent_past"], pivot_x=0.0, pivot_y=0.0, yaw_rad=dtheta
        )

    elif mode == "combo":
        # Parallel shift first, then yaw rotation about the SHIFTED pose.
        # ego_current_state: set position + rotate orientation/vel/accel.
        out["ego_current_state"] = _apply_yaw_to_state(
            out["ego_current_state"], dtheta, set_position=(variant.dx, variant.dy)
        )
        if variant.dv != 0.0:
            ecs = out["ego_current_state"].copy()
            ecs[4] = max(0.0, float(ecs[4]) + variant.dv)
            out["ego_current_state"] = ecs
        # Past: rigid translate by (dx, dy), then rotate about new pivot (dx, dy).
        past = out["ego_agent_past"].copy()
        past[:, 0] = past[:, 0] + variant.dx
        past[:, 1] = past[:, 1] + variant.dy
        past = _rotate_past_about_pivot(past, variant.dx, variant.dy, dtheta)
        out["ego_agent_past"] = past

    else:  # mode == "shift" (existing kind 1 / kind 2 behavior)
        out["ego_current_state"] = _set_ego_current_state(
            out["ego_current_state"], variant.dx, variant.dy, dtheta, variant.dv
        )
        past = _apply_rigid_to_past(
            out["ego_agent_past"], variant.dx, variant.dy, dtheta
        )
        if variant.history_jitter_std > 0.0:
            noise = rng.normal(
                loc=0.0, scale=variant.history_jitter_std, size=(past.shape[0], 2)
            ).astype(np.float32)
            past[:, :2] = past[:, :2] + noise
        out["ego_agent_past"] = past

    return out


# ---------------------------------------------------------------------------
# Lane membership filter
# ---------------------------------------------------------------------------

@torch.no_grad()
def _is_in_lane(
    perturbed_npz_path: Path, dx: float, dy: float, dtheta: float,
    device: torch.device, threshold: float = 0.15,
) -> tuple[bool, float]:
    """Run ``compute_lane_departure_penalty`` on a 2-step traj at the perturbed pose."""
    data = load_npz_data(str(perturbed_npz_path), device)
    es = data.get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es
    if ego_shape is None:
        ego_shape = torch.tensor([2.75, 4.34, 1.70], device=device)

    traj = torch.zeros(1, 2, 4, device=device)
    traj[0, :, 0] = dx
    traj[0, :, 1] = dy
    traj[0, :, 2] = math.cos(dtheta)
    traj[0, :, 3] = math.sin(dtheta)

    gate, near, wide, _, _ = compute_lane_departure_penalty(traj, ego_shape, data)
    if gate.item() < 0.5:
        return False, 0.0
    if near.item() > 0:
        clearance = 0.10
    elif wide.item() > 0:
        clearance = 0.30
    else:
        clearance = 0.50
    return clearance >= threshold, clearance


# ---------------------------------------------------------------------------
# Baseline model inference
# ---------------------------------------------------------------------------

def _load_base_model(
    base_model_path: Path, device: torch.device,
) -> tuple[Diffusion_Planner, Config]:
    """Load the LoRA-less base model + its training config.

    Mirrors the pattern in ``recovery_test._load_model``.
    """
    model_dir = base_model_path.parent
    args_path = model_dir / "args.json"
    if not args_path.exists():
        args_path = model_dir.parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(
            f"Could not locate args.json next to {base_model_path}"
        )

    model_args = Config(str(args_path))
    model_args.device = device
    model = Diffusion_Planner(model_args)
    ckpt = torch.load(str(base_model_path), map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, model_args


@torch.no_grad()
def _baseline_predict_future(
    model: Diffusion_Planner,
    model_args: Config,
    perturbed_npz_path: Path,
    device: torch.device,
) -> np.ndarray:
    """Run a single deterministic forward pass and return the predicted future.

    Returns:
        ``[T, 3]`` numpy array of ``(x, y, heading_rad)`` in ego frame, suitable
        for direct assignment to ``ego_agent_future``.
    """
    data = load_npz_data(str(perturbed_npz_path), device)
    batch = _stack_scene_data([data], device)
    norm_batch = _normalize_batch(batch, model_args)

    B = norm_batch["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    norm_batch["sampled_trajectories"] = torch.zeros(
        B, P, future_len + 1, 4, device=device
    )

    decoder = model.module.decoder if hasattr(model, "module") else model.decoder
    saved_fn = getattr(decoder, "_guidance_fn", None)
    decoder._guidance_fn = None
    try:
        _, outputs = model(norm_batch)
    finally:
        decoder._guidance_fn = saved_fn

    pred = outputs["prediction"][0, 0].detach().cpu().numpy()  # [T, 4] (x,y,cos,sin)
    if pred.shape[0] != future_len:
        # Defensive: clip / pad to expected length
        T = future_len
        out = np.zeros((T, 4), dtype=np.float32)
        n = min(T, pred.shape[0])
        out[:n] = pred[:n]
        pred = out

    # Convert (x, y, cos, sin) -> (x, y, heading_rad)
    fut = np.zeros((pred.shape[0], 3), dtype=np.float32)
    fut[:, 0] = pred[:, 0]
    fut[:, 1] = pred[:, 1]
    fut[:, 2] = np.arctan2(pred[:, 3], pred[:, 2]).astype(np.float32)
    return fut


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_offsets(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True, help="JSON list of input warm NPZ paths")
    parser.add_argument("--output_dir", required=True, help="Directory for augmented NPZs")
    parser.add_argument(
        "--output_scene_list",
        required=True,
        help="JSON output: list of resulting NPZ paths",
    )
    parser.add_argument(
        "--base_model",
        required=True,
        help="Path to the LoRA-less base model checkpoint (.pth) used to "
             "generate the recovery future for every kept variant.",
    )
    parser.add_argument("--n_per_scene", type=int, default=5)
    parser.add_argument(
        "--kind",
        type=str,
        default="default",
        choices=["default", "parallel_only", "yaw_only", "combined"],
        help="Recipe for variants per scene. 'default' = 1 base + 2 subtle + "
             "2 parallel (existing). 'parallel_only' = 1 base + (n-1) parallel "
             "split L/R cycling --offsets. 'yaw_only' = 1 base + (n-1) yaw "
             "perturbations cycling --yaw_degs. 'combined' = 1 base + 2 subtle "
             "+ 2 parallel + 2 yaw + 2 (parallel+yaw).",
    )
    parser.add_argument(
        "--offsets",
        type=str,
        default="0.3,0.5,0.8",
        help="Comma-separated lateral offsets for parallel-shift kinds (m).",
    )
    parser.add_argument(
        "--yaw_degs",
        type=str,
        default="5,10,15",
        help="Comma-separated yaw magnitudes (deg) for yaw_only / combined kinds.",
    )
    parser.add_argument(
        "--reject_out_of_lane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop variants whose perturbed pose is outside (or within "
             "--reject_threshold of) the lane boundary.",
    )
    parser.add_argument("--reject_threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_list_path = Path(args.output_scene_list)
    out_list_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.scenes) as f:
        scenes = json.load(f)
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"--scenes must be a non-empty JSON list; got {type(scenes)}")

    offsets = _parse_offsets(args.offsets)
    yaw_degs = _parse_offsets(args.yaw_degs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print(f"[disturb_and_replay] device={device} base_model={args.base_model}")
    base_model, base_args = _load_base_model(Path(args.base_model), device)

    written: list[str] = []
    n_attempted = 0
    n_rejected = 0
    n_inference_failed = 0
    rejection_log: list[dict] = []
    by_kind: dict[str, dict[str, int]] = {}

    # Sanity tracking — measure if the predicted future actually closes the
    # perturbation gap across the dataset.
    recovery_diag: list[dict] = []

    for scene_idx, scene_path in enumerate(scenes):
        scene_path = str(scene_path)
        try:
            with np.load(scene_path) as raw:
                npz = {k: raw[k].copy() for k in raw.files}
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {scene_idx} {scene_path}: load failed ({e})")
            continue

        if "route_lanes" not in npz:
            print(f"[skip] {scene_idx} {scene_path}: no route_lanes")
            continue
        tangent = _route_tangent_at_origin(npz["route_lanes"])
        # Pre-compute centerline segments once per scene for diag.
        scene_segs = _build_segments(npz["route_lanes"])

        scene_stem = Path(scene_path).stem
        variants = _build_variants(
            rng, offsets, args.n_per_scene, tangent,
            kind=args.kind, yaw_degs=yaw_degs,
        )

        for var_idx, variant in enumerate(variants):
            n_attempted += 1
            kind = variant.name.split("_")[0]
            by_kind.setdefault(kind, {"attempted": 0, "kept": 0, "rejected": 0})
            by_kind[kind]["attempted"] += 1

            perturbed = _apply_variant(npz, variant, rng)

            out_path = output_dir / f"{scene_stem}_var{var_idx:02d}.npz"
            np.savez(out_path, **perturbed)

            if args.reject_out_of_lane and variant.name != "base":
                ok, clearance = _is_in_lane(
                    out_path,
                    variant.dx,
                    variant.dy,
                    math.radians(variant.dtheta_deg),
                    device,
                    args.reject_threshold,
                )
                if not ok:
                    n_rejected += 1
                    by_kind[kind]["rejected"] += 1
                    rejection_log.append(
                        {
                            "scene": scene_path,
                            "variant": variant.name,
                            "dx": variant.dx,
                            "dy": variant.dy,
                            "dtheta_deg": variant.dtheta_deg,
                            "clearance": clearance,
                        }
                    )
                    out_path.unlink(missing_ok=True)
                    continue

            # Run baseline inference -> overwrite ego_agent_future
            try:
                fut = _baseline_predict_future(base_model, base_args, out_path, device)
            except Exception as e:  # noqa: BLE001
                print(
                    f"[infer-fail] {scene_idx} {variant.name}: {e}; "
                    f"dropping variant."
                )
                n_inference_failed += 1
                by_kind[kind]["rejected"] += 1
                out_path.unlink(missing_ok=True)
                continue

            # Sanity: lateral distance to centerline at t=0 (perturbed pose) vs t=79 (end of recovery).
            try:
                # t=0 = perturbed origin in ego frame
                pt0 = np.array([[variant.dx, variant.dy]], dtype=np.float64)
                d0 = float(_point_to_segments_dist(pt0, scene_segs)[0])
                pt79 = np.array([fut[-1, :2]], dtype=np.float64)
                d79 = float(_point_to_segments_dist(pt79, scene_segs)[0])
                recovery_diag.append({
                    "scene": Path(scene_path).stem,
                    "variant": variant.name,
                    "lat_t0": d0,
                    "lat_t79": d79,
                    "delta": d0 - d79,
                })
            except Exception:
                pass

            perturbed["ego_agent_future"] = fut.astype(np.float32)
            np.savez(out_path, **perturbed)

            written.append(str(out_path))
            by_kind[kind]["kept"] += 1

        if (scene_idx + 1) % 10 == 0:
            print(
                f"  Processed {scene_idx + 1}/{len(scenes)} scenes — "
                f"{len(written)} kept / {n_rejected} rejected / "
                f"{n_inference_failed} infer-failed so far"
            )

    with open(out_list_path, "w") as f:
        json.dump(written, f, indent=2)

    # Recovery diagnostic summary
    if recovery_diag:
        deltas = np.array([r["delta"] for r in recovery_diag])
        recovery_summary = {
            "n_diag": len(recovery_diag),
            "mean_lat_t0": float(np.mean([r["lat_t0"] for r in recovery_diag])),
            "mean_lat_t79": float(np.mean([r["lat_t79"] for r in recovery_diag])),
            "mean_delta": float(np.mean(deltas)),
            "p50_delta": float(np.median(deltas)),
            "p95_delta": float(np.percentile(deltas, 95)),
            "frac_recovered": float((deltas > 0.0).mean()),
        }
    else:
        recovery_summary = {}

    summary = {
        "n_input_scenes": len(scenes),
        "n_per_scene_target": args.n_per_scene,
        "n_attempted": n_attempted,
        "n_kept": len(written),
        "n_rejected": n_rejected,
        "n_inference_failed": n_inference_failed,
        "by_kind": by_kind,
        "offsets": offsets,
        "reject_out_of_lane": bool(args.reject_out_of_lane),
        "reject_threshold": args.reject_threshold,
        "seed": args.seed,
        "base_model": str(args.base_model),
        "output_dir": str(output_dir),
        "output_scene_list": str(out_list_path),
        "recovery_summary": recovery_summary,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    rej_path = output_dir / "rejections.json"
    with open(rej_path, "w") as f:
        json.dump(rejection_log, f, indent=2)
    diag_path = output_dir / "recovery_diag.json"
    with open(diag_path, "w") as f:
        json.dump(recovery_diag, f, indent=2)

    print(
        f"\nDone. {len(written)}/{n_attempted} variants kept "
        f"({n_rejected} rejected, {n_inference_failed} infer-failed) "
        f"from {len(scenes)} scenes."
    )
    print("Per-kind:")
    for k, v in by_kind.items():
        print(f"  {k:>10}: kept {v['kept']:>4d} / rejected {v['rejected']:>4d}")
    if recovery_summary:
        rs = recovery_summary
        print(
            "Recovery diagnostic: "
            f"lat_t0={rs['mean_lat_t0']:.3f}m  lat_t79={rs['mean_lat_t79']:.3f}m  "
            f"mean_delta={rs['mean_delta']:+.3f}m  "
            f"frac_recovered={rs['frac_recovered']:.2f}"
        )
    print(f"Wrote scene list: {out_list_path}")
    print(f"Wrote summary  : {summary_path}")


if __name__ == "__main__":
    main()
