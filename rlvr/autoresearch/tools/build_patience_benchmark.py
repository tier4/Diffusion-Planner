#!/usr/bin/env python3
"""Build a frozen patience benchmark from an NPZ dataset list.

Scans a scene list for *waits-during-interaction* scenes — the recorded ego is
quasi-stationary for >= ``min_wait_s`` somewhere in its future while there is a
plausible interaction reason in the observation (a pedestrian within
``ped_near_m`` of the GT path, or a red traffic light on a route lane) — then
event-deduplicates them (consecutive NPZs sample the same wait at multiple
offsets) keeping ONE scene per event, preferring the wait-ONSET offset (ego
still moving at t0, GT stops within the horizon): onset is the discriminative
test of the braking DECISION, staying-stopped is trivially passed.

Outputs:
  --out       frozen benchmark list (deduped, onset-preferred, capped at --max_scenes)
  --out_pool  optional: the FULL undeduped waits-interaction pool (anchor-slice
              ``waits_scene_list`` input for the R2LPL lifelong runner)

Channel conventions (see diffusion_planner_ros lanelet_converter): lane feature
channels 8-12 are the traffic-light one-hot [green, yellow, red, unknown, none];
neighbor one-hot [vehicle, pedestrian, bicycle] occupies columns 8-10 of
``neighbor_agents_past``.

Usage:
    python -m rlvr.autoresearch.tools.build_patience_benchmark \
        --scenes <dataset_list.json> --out <benchmark.json> [--out_pool <pool.json>]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

_DT = 0.1
_LANE_TL_RED_CH = 10
_NEIGHBOR_PED_COL = 9
_FRAME_RE = re.compile(r"^(?P<bag>.+)_(?P<frame>\d+)$")


def scan_scene(
    path: str,
    *,
    wait_speed_mps: float,
    min_wait_steps: int,
    ped_near_m: float,
    onset_speed_mps: float = 2.0,
) -> dict | None:
    """Classify one NPZ. Returns per-scene flags, or {"error": ...} on load failure."""
    try:
        with np.load(path) as d:
            route = d["route_lanes"]
            nb_now = d["neighbor_agents_past"][:, -1, :]
            fut = d["ego_agent_future"][:, :2]
    except Exception as exc:  # noqa: BLE001 - report, caller decides
        return {"error": str(exc)}

    red_route = bool((route[..., _LANE_TL_RED_CH] > 0.5).any())

    valid = np.abs(nb_now[:, :4]).sum(axis=1) > 0
    is_ped = (nb_now[:, _NEIGHBOR_PED_COL] > 0.5) & valid
    ped_near = False
    if is_ped.any():
        dists = np.linalg.norm(nb_now[is_ped, :2][:, None, :] - fut[None, :, :], axis=-1)
        ped_near = bool((dists.min(axis=1) < ped_near_m).any())

    step_disp = np.linalg.norm(np.diff(fut, axis=0), axis=-1)
    slow = step_disp < wait_speed_mps * _DT
    run = best = 0
    for s in slow:
        run = run + 1 if s else 0
        best = max(best, run)
    gt_waits = best >= min_wait_steps

    v0 = float(step_disp[0] / _DT) if step_disp.shape[0] else 0.0
    # Onset: ego moving at t0, GT reaches a stop within the horizon.
    is_onset = v0 > onset_speed_mps and bool((step_disp / _DT < wait_speed_mps).any())

    return {
        "gt_waits_interact": gt_waits and (ped_near or red_route),
        "is_onset": is_onset,
        "v0": v0,
        "ped_near": ped_near,
        "red_route": red_route,
    }


def _event_key(path: str, frame_gap: int) -> tuple[str, int] | None:
    """(bag_id, frame//gap) — scenes of one bag within ``frame_gap`` frames = one event."""
    stem = Path(path).stem
    m = _FRAME_RE.match(stem)
    if m is None:
        return None
    return (str(Path(path).parent / m.group("bag")), int(m.group("frame")) // frame_gap)


def build_benchmark(
    scene_paths: list[str],
    *,
    wait_speed_mps: float,
    min_wait_steps: int,
    ped_near_m: float,
    event_frame_gap: int,
    max_scenes: int,
    onset_speed_mps: float = 2.0,
) -> tuple[list[str], list[str], dict]:
    pool: list[tuple[str, dict]] = []
    n_err = 0
    for p in scene_paths:
        r = scan_scene(
            p,
            wait_speed_mps=wait_speed_mps,
            min_wait_steps=min_wait_steps,
            ped_near_m=ped_near_m,
            onset_speed_mps=onset_speed_mps,
        )
        if r is None or "error" in r:
            n_err += 1
            continue
        if r["gt_waits_interact"]:
            pool.append((p, r))

    # Event dedup: one scene per (bag, frame-bin), onset-preferred, then highest v0
    # (the offset where the braking decision is hardest).
    events: dict[tuple[str, int], tuple[str, dict]] = {}
    unkeyed: list[tuple[str, dict]] = []
    for p, r in pool:
        key = _event_key(p, event_frame_gap)
        if key is None:
            unkeyed.append((p, r))
            continue
        prev = events.get(key)
        if prev is None or (r["is_onset"], r["v0"]) > (prev[1]["is_onset"], prev[1]["v0"]):
            events[key] = (p, r)

    deduped = [*events.values(), *unkeyed]
    deduped.sort(key=lambda pr: (not pr[1]["is_onset"], -pr[1]["v0"], pr[0]))
    benchmark = [p for p, _ in deduped[:max_scenes]]
    stats = {
        "scanned": len(scene_paths),
        "errors": n_err,
        "pool": len(pool),
        "events": len(deduped),
        "benchmark": len(benchmark),
        "onset_in_benchmark": sum(1 for p, r in deduped[:max_scenes] if r["is_onset"]),
    }
    return benchmark, [p for p, _ in pool], stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", required=True, help="JSON list of NPZ paths to scan")
    ap.add_argument("--out", required=True, help="frozen benchmark JSON (deduped)")
    ap.add_argument("--out_pool", help="optional full waits-interaction pool JSON")
    ap.add_argument("--max_scenes", type=int, default=150)
    ap.add_argument("--wait_speed_mps", type=float, default=0.5)
    ap.add_argument("--min_wait_s", type=float, default=1.0)
    ap.add_argument("--ped_near_m", type=float, default=5.0)
    ap.add_argument(
        "--onset_speed_mps",
        type=float,
        default=2.0,
        help="minimum t0 speed for a scene to count as a wait-ONSET offset",
    )
    ap.add_argument(
        "--event_frame_gap",
        type=int,
        default=100,
        help="frames (10 Hz) binning consecutive scenes into one wait event",
    )
    args = ap.parse_args()

    scene_paths = [str(p) for p in json.load(open(args.scenes))]
    benchmark, full_pool, stats = build_benchmark(
        scene_paths,
        wait_speed_mps=args.wait_speed_mps,
        min_wait_steps=int(round(args.min_wait_s / _DT)),
        ped_near_m=args.ped_near_m,
        event_frame_gap=args.event_frame_gap,
        max_scenes=args.max_scenes,
        onset_speed_mps=args.onset_speed_mps,
    )
    if not benchmark:
        raise RuntimeError(
            "no waits-interaction scenes found; refusing to write an empty benchmark"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(benchmark, indent=2))
    if args.out_pool:
        Path(args.out_pool).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_pool).write_text(json.dumps(full_pool, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"benchmark -> {args.out}" + (f"\npool -> {args.out_pool}" if args.out_pool else ""))


if __name__ == "__main__":
    main()
