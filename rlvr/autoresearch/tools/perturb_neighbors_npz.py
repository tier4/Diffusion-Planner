#!/usr/bin/env python3
"""Neighbor-position perturbation variants for avoidance scene NPZs.

`disturb_and_replay` perturbs the EGO frame (every field moves rigidly with
the new ego pose), so the ego-to-obstacle geometry of a scene only varies
through ego offsets. This tool varies the OTHER side: it shifts the STOPPED
(parked / blocking) neighbors themselves — laterally and longitudinally in
each neighbor's own heading frame — producing new ego-to-obstacle geometries
from the same base scene.

Only stopped neighbors are moved (the avoidance-relevant obstacles, same
detection convention as ghost_sim_common.extract_stopped_neighbors: non-empty
slot, future displacement < 0.5 m, valid box dims). Moving traffic, map,
ego past/future and all other fields are copied verbatim. Zero rows are
padding and are never touched.

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


def _stopped_neighbor_indices(nb_past: np.ndarray, nb_fut: np.ndarray) -> list[int]:
    """Indices of stopped neighbors (same criteria as extract_stopped_neighbors)."""
    idxs = []
    for i in range(nb_past.shape[0]):
        xy0 = nb_past[i, -1, :2]
        if abs(float(xy0[0])) + abs(float(xy0[1])) < 1e-6:
            continue  # empty slot (padding)
        fut_xy = nb_fut[i, :, :2]
        fut_valid = np.abs(fut_xy).sum(axis=-1) > 1e-6
        disp = (
            0.0
            if fut_valid.sum() < 2
            else float(np.linalg.norm(fut_xy[fut_valid].max(0) - fut_xy[fut_valid].min(0)))
        )
        if disp >= 0.5:
            continue  # moving traffic — not an avoidance obstacle
        w = float(nb_past[i, -1, 6])
        length = float(nb_past[i, -1, 7])
        if w < 0.1 or length < 0.1:
            continue
        idxs.append(i)
    return idxs


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
        base = dict(np.load(sp, allow_pickle=True))
        if "neighbor_agents_past" not in base or "neighbor_agents_future" not in base:
            raise ValueError(f"{sp}: missing neighbor arrays")
        nb_past0 = base["neighbor_agents_past"]
        nb_fut0 = base["neighbor_agents_future"]
        if nb_past0.ndim == 4:
            raise ValueError(f"{sp}: batched neighbor array — expected unbatched NPZ")
        idxs = _stopped_neighbor_indices(nb_past0, nb_fut0)
        if not idxs:
            n_no_stopped += 1
            print(f"  [skip] {Path(sp).name}: no stopped neighbors to perturb")
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
            boxes = _boxes_from_arrays(nb_past, idxs)
            t0_clr = float(plan_static_clearance(t0_pose, boxes, ego_shape, device))
            if t0_clr < args.min_t0_clearance:
                n_t0_drop += 1
                continue
            gt_clr = float(plan_static_clearance(gt_plan, boxes, ego_shape, device))
            if gt_clr < 0.0:
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
                    "t0_clearance": round(t0_clr, 3),
                    "gt_future_clearance": round(gt_clr, 3),
                    "out": str(out_path),
                }
            )

    with open(args.out_list, "w") as f:
        json.dump(written, f, indent=1)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n[perturb_nbr] {len(written)} variants from {len(paths)} scenes -> {args.out_list}")
    print(f"  dropped: {n_t0_drop} t0-violating; skipped: {n_no_stopped} scenes w/o stopped nbrs")
    print(
        f"  WARNING: {n_gt_cross} variants have GT future crossing the shifted obstacle "
        "(future must be replaced by distillation before curated SFT)"
    )


if __name__ == "__main__":
    main()
