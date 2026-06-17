#!/usr/bin/env python3
"""Add far-from-path STOPPED distractor vehicles to avoidance scene NPZs.

Motivation (false-detection reduction for the guidance explorer): the policy
must learn to avoid only the stopped vehicles that are actually IN the ego's
way, not every stopped vehicle in the scene. The counterfactual `strip` tool
teaches "no neighbor -> eta 0", but that overshoots into domain blindness
(the policy learns to key on EMPTY road rather than on proximity). This tool
instead places NEW stopped vehicles at locations that are deliberately FAR
from the baseline ego trajectory, so they are NOT avoidance candidates:

  - in an avoidance scene, the real close obstacle stays, plus decoys that
    must be ignored;
  - in a neighbor-stripped anchor scene, the only neighbors are decoys, so
    the correct output is still eta 0 even though stopped cars are present.

Together these teach the discriminative target "avoid the close one, ignore
the distant ones" rather than "avoid iff any stopped car exists".

Placement (ego frame): pick an anchor point along the ego_agent_future path,
offset it laterally by a large signed magnitude (so it sits beside / off the
corridor), orient it along the local path heading. A placed vehicle is KEPT
only if the whole baseline ego plan clears it by >= --min_path_clearance
(canonical plan_static_clearance OBB, no hand-rolled geometry) AND the t=0 ego
pose clears it by >= --min_t0_clearance. Distractors are written into empty
neighbor slots (320-cap); existing agents are copied verbatim.

Usage:
    python -m rlvr.autoresearch.tools.add_distractor_neighbors_npz \
        --scenes <list.json> --out_dir <dir> --out_list <out.json> \
        --ego_shape 4.76,7.24,2.29 \
        --n_per_scene 1 --n_distractors 1,3 \
        --lat_range 6.0,16.0 --min_path_clearance 3.0 --min_t0_clearance 1.0 \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from scenario_generation.explorer_runner import plan_static_clearance

# Default distractor footprint (a typical parked sedan/van; metres).
DEFAULT_VEH_WIDTH = 2.0
DEFAULT_VEH_LENGTH = 4.7


def _future_as_plan(fut: np.ndarray) -> np.ndarray:
    """ego_agent_future (T,3 yaw | T,4 cos/sin) -> (T,4) [x,y,cos,sin]."""
    if fut.shape[-1] == 4:
        return fut.astype(np.float32)
    if fut.shape[-1] == 3:
        return np.stack(
            [fut[:, 0], fut[:, 1], np.cos(fut[:, 2]), np.sin(fut[:, 2])], axis=-1
        ).astype(np.float32)
    raise ValueError(f"unsupported ego_agent_future width {fut.shape[-1]}")


def _empty_slots(nb_past: np.ndarray) -> list[int]:
    """Indices of padding rows (xy ~ 0 at the current/last past timestep)."""
    xy = nb_past[:, -1, :2]
    mask = np.abs(xy).sum(axis=-1) < 1e-6
    return [int(i) for i in np.where(mask)[0]]


def _existing_boxes(nb_past: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    """(x,y,heading,length,width) of every non-empty neighbor (used to reject
    distractors that would overlap a real agent)."""
    boxes = []
    for i in range(nb_past.shape[0]):
        xy = nb_past[i, -1, :2]
        if abs(float(xy[0])) + abs(float(xy[1])) < 1e-6:
            continue
        h = math.atan2(float(nb_past[i, -1, 3]), float(nb_past[i, -1, 2]))
        boxes.append(
            (float(xy[0]), float(xy[1]), h, float(nb_past[i, -1, 7]), float(nb_past[i, -1, 6]))
        )
    return boxes


def _valid_path_indices(plan: np.ndarray) -> np.ndarray:
    """Indices of the ego plan with a non-zero (valid) waypoint."""
    return np.where(np.abs(plan[:, :2]).sum(axis=-1) > 1e-6)[0]


def _sample_distractor(
    plan: np.ndarray, valid_idx: np.ndarray, lat_lo: float, lat_hi: float, rng
) -> tuple[float, float, float, float, float]:
    """Sample one (x,y,heading,length,width) beside the ego path."""
    t = int(rng.choice(valid_idx))
    px, py = float(plan[t, 0]), float(plan[t, 1])
    ph = math.atan2(float(plan[t, 3]), float(plan[t, 2]))
    # perpendicular to the local path heading
    nperp = (-math.sin(ph), math.cos(ph))
    d = float(rng.uniform(lat_lo, lat_hi)) * float(rng.choice([-1.0, 1.0]))
    x = px + d * nperp[0]
    y = py + d * nperp[1]
    # parked along the road (parallel), occasionally facing the other way
    h = ph + (math.pi if rng.random() < 0.5 else 0.0)
    return (x, y, h, DEFAULT_VEH_LENGTH, DEFAULT_VEH_WIDTH)


def _write_distractor(nb_past: np.ndarray, nb_fut: np.ndarray, slot: int, box) -> None:
    """Fill neighbor `slot` with a static (stopped) vehicle, in-place."""
    x, y, h, length, width = box
    c, s = math.cos(h), math.sin(h)
    # past: repeated static pose, zero velocity, vehicle one-hot
    nb_past[slot, :, :] = 0.0
    nb_past[slot, :, 0] = x
    nb_past[slot, :, 1] = y
    nb_past[slot, :, 2] = c
    nb_past[slot, :, 3] = s
    nb_past[slot, :, 6] = width
    nb_past[slot, :, 7] = length
    nb_past[slot, :, 8] = 1.0  # vehicle one-hot
    # future: same static pose (parked car), matching the future column width
    w = nb_fut.shape[-1]
    nb_fut[slot, :, :] = 0.0
    nb_fut[slot, :, 0] = x
    nb_fut[slot, :, 1] = y
    if w == 4:
        nb_fut[slot, :, 2] = c
        nb_fut[slot, :, 3] = s
    elif w == 3:
        nb_fut[slot, :, 2] = h
    else:
        raise ValueError(f"unsupported neighbor_agents_future width {w}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenes", required=True, help="JSON list of NPZ paths")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_list", required=True)
    parser.add_argument("--ego_shape", required=True, help="WB,L,W — must match the platform")
    parser.add_argument("--n_per_scene", type=int, required=True)
    parser.add_argument(
        "--n_distractors", required=True, help="min,max distractors per output scene"
    )
    parser.add_argument(
        "--lat_range", required=True, help="min,max lateral |offset| from the ego path (m)"
    )
    parser.add_argument(
        "--min_path_clearance",
        type=float,
        required=True,
        help="distractor rejected if the ego plan clears it by less (keeps it a non-candidate)",
    )
    parser.add_argument(
        "--min_t0_clearance",
        type=float,
        required=True,
        help="distractor rejected if the t=0 ego pose clears it by less",
    )
    parser.add_argument(
        "--min_neighbor_clearance",
        type=float,
        default=0.5,
        help="reject a distractor whose OBB clears existing neighbors / already-placed "
        "distractors by less than this (avoids overlapping/physically-impossible scenes)",
    )
    parser.add_argument(
        "--max_tries", type=int, default=20, help="resample attempts per distractor"
    )
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ego_shape = tuple(float(x) for x in args.ego_shape.split(","))
    lat_lo, lat_hi = (float(x) for x in args.lat_range.split(","))
    nd_lo, nd_hi = (int(x) for x in args.n_distractors.split(","))
    # Fail loud on reversed ranges (no silent reversed-sampling / empty draw).
    if not (0.0 <= lat_lo <= lat_hi):
        raise SystemExit(f"--lat_range must be 0 <= min <= max; got {args.lat_range!r}")
    if not (1 <= nd_lo <= nd_hi):
        raise SystemExit(f"--n_distractors must be 1 <= min <= max; got {args.n_distractors!r}")
    rng = np.random.default_rng(args.seed)

    with open(args.scenes) as f:
        paths = json.load(f)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0_pose = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    written, manifest = [], []
    n_no_slot = n_no_path = n_short = 0
    total_placed = 0
    for sp in paths:
        base = dict(np.load(sp, allow_pickle=True))
        if "neighbor_agents_past" not in base or "neighbor_agents_future" not in base:
            raise ValueError(f"{sp}: missing neighbor arrays")
        nb_past0 = base["neighbor_agents_past"]
        nb_fut0 = base["neighbor_agents_future"]
        if nb_past0.ndim == 4:
            raise ValueError(f"{sp}: batched neighbor array — expected unbatched NPZ")
        plan = _future_as_plan(base["ego_agent_future"])
        valid_idx = _valid_path_indices(plan)
        # ego_agent_future is zero-padded on invalid steps; those rows would put a
        # phantom ego at the origin and skew the clearance. Use ONLY valid waypoints,
        # and require >=2 (plan_static_clearance reads per_timestep_min[:,1:]).
        if valid_idx.size < 2:
            n_no_path += 1
            print(f"  [skip] {Path(sp).name}: ego_agent_future has <2 valid waypoints")
            continue
        plan_valid = plan[valid_idx]
        pool = Path(sp).parent.name
        for v in range(args.n_per_scene):
            nb_past = nb_past0.copy()
            nb_fut = nb_fut0.copy()
            slots = _empty_slots(nb_past)
            if not slots:
                n_no_slot += 1
                continue
            n_target = int(rng.integers(nd_lo, nd_hi + 1))
            placed = []
            placed_boxes = []  # (x,y,h,l,w) of distractors placed in THIS variant
            existing = _existing_boxes(nb_past)  # real agents already in the scene
            for _ in range(min(n_target, len(slots))):
                box = None
                for _try in range(args.max_tries):
                    cand = _sample_distractor(plan, valid_idx, lat_lo, lat_hi, rng)
                    path_clr = float(plan_static_clearance(plan_valid, [cand], ego_shape, device))
                    t0_clr = float(plan_static_clearance(t0_pose, [cand], ego_shape, device))
                    if path_clr < args.min_path_clearance or t0_clr < args.min_t0_clearance:
                        continue
                    # Reject if it overlaps a real agent or an already-placed distractor.
                    # Canonical OBB (no centroid-to-centroid): the candidate's own pose as
                    # a 2-step plan, ego footprint as a conservative (larger) proxy box.
                    others = existing + placed_boxes
                    cand_pose = np.array(
                        [[cand[0], cand[1], math.cos(cand[2]), math.sin(cand[2])]] * 2,
                        dtype=np.float32,
                    )
                    if others and (
                        float(plan_static_clearance(cand_pose, others, ego_shape, device))
                        < args.min_neighbor_clearance
                    ):
                        continue
                    box = cand
                    break
                if box is None:
                    continue  # could not find a safe far placement this try-budget
                slot = slots.pop()
                _write_distractor(nb_past, nb_fut, slot, box)
                placed_boxes.append(box)
                placed.append(
                    {
                        "slot": slot,
                        "x": round(box[0], 2),
                        "y": round(box[1], 2),
                        "heading": round(box[2], 3),
                    }
                )
            if not placed:
                n_short += 1
                continue
            out = dict(base)
            out["neighbor_agents_past"] = nb_past
            out["neighbor_agents_future"] = nb_fut
            out_path = out_dir / f"{pool}__{Path(sp).stem}_dist{v:02d}.npz"
            np.savez(out_path, **out)
            written.append(str(out_path))
            total_placed += len(placed)
            manifest.append(
                {"source": sp, "variant": v, "distractors": placed, "out": str(out_path)}
            )

    with open(args.out_list, "w") as f:
        json.dump(written, f, indent=1)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(
        f"\n[add_distractor] {len(written)} scenes ({total_placed} distractors) "
        f"from {len(paths)} bases -> {args.out_list}"
    )
    print(f"  skipped: {n_no_path} no-path, {n_no_slot} no-empty-slot, {n_short} no-safe-placement")


if __name__ == "__main__":
    main()
