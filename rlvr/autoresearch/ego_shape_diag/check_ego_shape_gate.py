#!/usr/bin/env python3
"""ACCEPTANCE GATE: does the planned distance depend on the ego-dimension token? It must not.

``ego_shape`` describes the vehicle's geometry. A healthy model plans the same distance
whatever it says — geometry decides how a manoeuvre is shaped, not whether the vehicle
moves. When a training set contains only a couple of distinct ``ego_shape`` values and
those groups differ in how fast they drive, the token becomes the cheapest available
predictor of speed, and the model learns to read it as "which speed distribution do I
imitate". The failure then appears only at one end of the range: at some wheelbase the
plan collapses to a standstill on geometry the model otherwise handles.

Sweep the token and compare. A flat sweep passes; a cliff means the token is acting as a
speed prior. Pair this with ``check_training_data`` (the confound in the data) and
``check_tl_gate`` (whether the standstill is actually correct at that moment).

    python -m rlvr.autoresearch.ego_shape_diag.check_ego_shape_gate \
        --model <best_model.pth> --scenes <dir-or-glob> \
        [--goal X,Y] [--wheelbases 2.75,3.5,4.0,4.5,4.76] [--min_span 15] [--limit 5]

Seconds of compute. Open-loop trajectory error does not detect this failure.
"""

from __future__ import annotations

import argparse

import torch

from rlvr.autoresearch.ego_shape_diag._common import (
    find_scenes,
    load_deployable_model,
    load_scene,
    plan_span,
    set_shape,
)


def sweep_scene(model, margs, d: dict, wheelbases, length, width, device) -> list[float]:
    """Plan span at each wheelbase, holding the other dimensions fixed."""
    return [plan_span(model, margs, set_shape(d, wb, length, width), device) for wb in wheelbases]


def report_scene(path: str, wheelbases: list[float], sweep: list[float], min_span: float) -> bool:
    """Print one scene's sweep; returns True when it fails the gate."""
    bad = [s for s in sweep if s < min_span]
    print(path.split("/")[-1])
    print("   " + "  ".join(f"{w}:{s:.1f}" for w, s in zip(wheelbases, sweep)))
    print(
        f"   spread across wheelbase: {max(sweep) - min(sweep):6.2f} m"
        f"   {'FAIL' if bad else 'pass'}"
    )
    return bool(bad)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="deployable checkpoint (args.json alongside)")
    ap.add_argument("--scenes", required=True, help="scene dir, glob, or single .npz")
    ap.add_argument("--goal", default=None, help="x,y ego-frame goal override")
    ap.add_argument("--wheelbases", default="2.75,3.5,4.0,4.5,4.76")
    ap.add_argument("--length", type=float, default=4.34, help="held fixed across the sweep")
    ap.add_argument("--width", type=float, default=1.84, help="held fixed across the sweep")
    ap.add_argument("--min_span", type=float, default=15.0, help="gate threshold in metres")
    ap.add_argument("--limit", type=int, default=5, help="scenes to test")
    return ap.parse_args()


def main() -> int:
    a = _parse_args()
    goal = tuple(float(x) for x in a.goal.split(",")) if a.goal else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_deployable_model(a.model, device)
    wheelbases = [float(x) for x in a.wheelbases.split(",")]
    scenes = find_scenes(a.scenes)[: a.limit]

    print(f"model  : {a.model}")
    print(f"scenes : {len(scenes)}   gate: span >= {a.min_span} m at every wheelbase\n")
    failures = 0
    for path in scenes:
        d = load_scene(path, device, goal)
        sweep = sweep_scene(model, margs, d, wheelbases, a.length, a.width, device)
        failures += report_scene(path, wheelbases, sweep, a.min_span)
    print(
        f"\n{'FAIL' if failures else 'PASS'}: {failures}/{len(scenes)} scenes below {a.min_span} m"
    )
    print("A large spread means the ego-dimension token is acting as a speed prior.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
