#!/usr/bin/env python3
"""Neighbor-position perturbation variants for avoidance scene NPZs.

`disturb_and_replay` perturbs the EGO frame (every field moves rigidly with
the new ego pose), so the ego-to-obstacle geometry of a scene only varies
through ego offsets. This tool varies the OTHER side: it shifts the STOPPED
(parked / blocking) neighbors themselves — laterally and longitudinally in
each neighbor's own heading frame — producing new ego-to-obstacle geometries
from the same base scene.

By default only stopped neighbors are moved (the avoidance-relevant obstacles,
same detection convention as ghost_sim_common.extract_stopped_neighbors:
non-empty slot, future displacement < 0.5 m, valid box dims). With
``--include_moving`` the MOVING neighbors are also shifted — e.g. the colliding
NPC in a perception-reproducer moving collision — by the SAME rigid translation
(a constant world offset applied to every past+future timestep, which preserves
the track's shape, per-step headings and velocities). Map, ego past/future and
all other fields are copied verbatim. Zero rows are padding and never touched.

Safety screens (canonical reward-path geometry, no hand-rolled OBB math):
  - t0-clean: the ego pose at t=0 (origin of the ego frame) must clear the
    SHIFTED boxes by >= --min_t0_clearance, else the variant is dropped
    (a scene already violating at t=0 cannot teach recovery);
  - the GT ego_agent_future clearance vs the shifted boxes is recorded in
    the manifest and counted loudly when it crosses (downstream distillation
    overwrites the future, but the count must be visible).

Usage:
    python -m rlvr.autoresearch.tools.perturb_neighbors_npz \
        --scenes <list.json> --out_dir <dir> --out_list <out.json> \
        --ego_shape WB,L,W --lat_range 0.3,0.8 --lon_range 2.0,5.0 \
        --n_per_scene 4 --min_t0_clearance 0.05 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from scenario_generation.explorer_runner import plan_static_clearance

_MOVING_DISP_THRESH = 0.5  # m of future displacement separating stopped vs moving


def _neighbor_future_disp(nb_fut_i: np.ndarray) -> float:
    """Bounding-box future displacement (m) of one neighbor's future track."""
    fut_xy = nb_fut_i[:, :2]
    fut_valid = np.abs(fut_xy).sum(axis=-1) > 1e-6
    if fut_valid.sum() < 2:
        return 0.0
    return float(np.linalg.norm(fut_xy[fut_valid].max(0) - fut_xy[fut_valid].min(0)))


def _selected_neighbor_indices(
    nb_past: np.ndarray, nb_fut: np.ndarray, include_moving: bool
) -> tuple[list[int], set[int]]:
    """Indices of perturbable neighbors + the subset classified as moving.

    Always returns STOPPED neighbors (future displacement < 0.5 m — the
    avoidance-relevant parked/blocking obstacles, same convention as
    extract_stopped_neighbors). When ``include_moving`` is set, MOVING neighbors
    (displacement >= 0.5 m, e.g. the colliding NPC in a reproducer moving
    collision) are also selected. Empty (padding) slots and degenerate boxes are
    excluded. Returns (selected_indices, moving_index_set).
    """
    selected: list[int] = []
    moving: set[int] = set()
    for i in range(nb_past.shape[0]):
        xy0 = nb_past[i, -1, :2]
        if abs(float(xy0[0])) + abs(float(xy0[1])) < 1e-6:
            continue  # empty slot (padding)
        w = float(nb_past[i, -1, 6])
        length = float(nb_past[i, -1, 7])
        if w < 0.1 or length < 0.1:
            continue
        is_moving = _neighbor_future_disp(nb_fut[i]) >= _MOVING_DISP_THRESH
        if is_moving and not include_moving:
            continue  # moving traffic — skipped unless --include_moving
        selected.append(i)
        if is_moving:
            moving.add(i)
    return selected, moving


def _boxes_from_arrays(nb_past: np.ndarray, idxs: list[int]):
    """(x, y, heading, length, width) boxes for the given neighbor indices."""
    boxes = []
    for i in idxs:
        x, y = float(nb_past[i, -1, 0]), float(nb_past[i, -1, 1])
        h = math.atan2(float(nb_past[i, -1, 3]), float(nb_past[i, -1, 2]))
        boxes.append((x, y, h, float(nb_past[i, -1, 7]), float(nb_past[i, -1, 6])))
    return boxes


def _shift_neighbor(
    nb_past: np.ndarray, nb_fut: np.ndarray, i: int, dlat: float, dlon: float
) -> None:
    """Rigid in-place shift of neighbor i in its own heading frame.

    Zero rows are padding and stay zero; only valid rows move.
    """
    c = float(nb_past[i, -1, 2])
    s = float(nb_past[i, -1, 3])
    dx = dlon * c - dlat * s
    dy = dlon * s + dlat * c
    past_valid = np.abs(nb_past[i, :, :2]).sum(axis=-1) > 1e-6
    nb_past[i, past_valid, 0] += dx
    nb_past[i, past_valid, 1] += dy
    fut_valid = np.abs(nb_fut[i, :, :2]).sum(axis=-1) > 1e-6
    nb_fut[i, fut_valid, 0] += dx
    nb_fut[i, fut_valid, 1] += dy


def _future_as_plan(fut: np.ndarray) -> np.ndarray:
    """ego_agent_future (T,3 yaw | T,4 cos/sin) -> (T,4) [x,y,cos,sin]."""
    if fut.shape[-1] == 4:
        return fut.astype(np.float32)
    if fut.shape[-1] == 3:
        return np.stack(
            [fut[:, 0], fut[:, 1], np.cos(fut[:, 2]), np.sin(fut[:, 2])], axis=-1
        ).astype(np.float32)
    raise ValueError(f"unsupported ego_agent_future width {fut.shape[-1]}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenes", required=True, help="JSON list of NPZ paths")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_list", required=True)
    parser.add_argument(
        "--ego_shape", required=True, help="WB,L,W — no default, must match the platform"
    )
    parser.add_argument(
        "--lat_range",
        required=True,
        help="min,max lateral |offset| in m (neighbor frame), sign randomized",
    )
    parser.add_argument(
        "--lon_range",
        required=True,
        help="min,max longitudinal |offset| in m (neighbor frame), sign randomized",
    )
    parser.add_argument("--n_per_scene", type=int, required=True)
    parser.add_argument(
        "--include_moving",
        action="store_true",
        help="also perturb MOVING neighbors (future displacement >= 0.5 m), e.g. the "
        "colliding NPC in a reproducer moving collision. Default: stopped neighbors only.",
    )
    parser.add_argument(
        "--min_t0_clearance",
        type=float,
        required=True,
        help="variant dropped if ego t=0 pose clears shifted boxes by less",
    )
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ego_shape = tuple(float(x) for x in args.ego_shape.split(","))
    lat_lo, lat_hi = (float(x) for x in args.lat_range.split(","))
    lon_lo, lon_hi = (float(x) for x in args.lon_range.split(","))
    rng = np.random.default_rng(args.seed)

    with open(args.scenes) as f:
        paths = json.load(f)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, manifest = [], []
    n_t0_drop = n_no_stopped = n_gt_cross = 0
    # 2 identical rows: the canonical collision fn needs >=2 timesteps.
    t0_pose = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    for sp in paths:
        with np.load(sp, allow_pickle=True) as _z:
            base = dict(_z)
        if "neighbor_agents_past" not in base or "neighbor_agents_future" not in base:
            raise ValueError(f"{sp}: missing neighbor arrays")
        nb_past0 = base["neighbor_agents_past"]
        nb_fut0 = base["neighbor_agents_future"]
        if nb_past0.ndim == 4:
            raise ValueError(f"{sp}: batched neighbor array — expected unbatched NPZ")
        idxs, moving_idxs = _selected_neighbor_indices(nb_past0, nb_fut0, args.include_moving)
        if not idxs:
            n_no_stopped += 1
            kind = "stopped/moving" if args.include_moving else "stopped"
            print(f"  [skip] {Path(sp).name}: no {kind} neighbors to perturb")
            continue
        gt_plan = _future_as_plan(base["ego_agent_future"])
        pool = Path(sp).parent.name
        for v in range(args.n_per_scene):
            nb_past = nb_past0.copy()
            nb_fut = nb_fut0.copy()
            offsets = {}
            for i in idxs:
                dlat = float(rng.uniform(lat_lo, lat_hi)) * float(rng.choice([-1.0, 1.0]))
                dlon = float(rng.uniform(lon_lo, lon_hi)) * float(rng.choice([-1.0, 1.0]))
                _shift_neighbor(nb_past, nb_fut, i, dlat, dlon)
                offsets[int(i)] = {"dlat": round(dlat, 3), "dlon": round(dlon, 3)}
            # t0-clean: ego at origin must clear EVERY shifted box (stopped or
            # moving — the box is each neighbor's t=0 pose, so this is valid for both).
            boxes = _boxes_from_arrays(nb_past, idxs)
            t0_clr = float(plan_static_clearance(t0_pose, boxes, ego_shape, device))
            if t0_clr < args.min_t0_clearance:
                n_t0_drop += 1
                continue
            # GT-future-cross warning is a STATIC-obstacle concept (the GT future
            # shouldn't pass through a parked car). It is meaningless against a
            # moving box, so score it only over the STOPPED subset; None if all moving.
            stopped_boxes = _boxes_from_arrays(nb_past, [i for i in idxs if i not in moving_idxs])
            gt_clr = (
                float(plan_static_clearance(gt_plan, stopped_boxes, ego_shape, device))
                if stopped_boxes
                else None
            )
            if gt_clr is not None and gt_clr < 0.0:
                n_gt_cross += 1
            out = dict(base)
            out["neighbor_agents_past"] = nb_past
            out["neighbor_agents_future"] = nb_fut
            out_path = out_dir / f"{pool}__{Path(sp).stem}_nbr{v:02d}.npz"
            np.savez(out_path, **out)
            written.append(str(out_path))
            manifest.append(
                {
                    "source": sp,
                    "variant": v,
                    "offsets": offsets,
                    "moving_slots": sorted(moving_idxs),
                    "t0_clearance": round(t0_clr, 3),
                    "gt_future_clearance": (round(gt_clr, 3) if gt_clr is not None else None),
                    "out": str(out_path),
                }
            )

    with open(args.out_list, "w") as f:
        json.dump(written, f, indent=1)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n[perturb_nbr] {len(written)} variants from {len(paths)} scenes -> {args.out_list}")
    kind = "stopped/moving" if args.include_moving else "stopped"
    print(f"  dropped: {n_t0_drop} t0-violating; skipped: {n_no_stopped} scenes w/o {kind} nbrs")
    print(
        f"  WARNING: {n_gt_cross} variants have GT future crossing the shifted obstacle "
        "(future must be replaced by distillation before curated SFT)"
    )


if __name__ == "__main__":
    main()
