#!/usr/bin/env python3
"""ACCEPTANCE GATE, traffic-light stratified: does the model go on green and hold on red?

A stop is only a failure if the light says go. Scenes captured across a light cycle
therefore cannot be graded with a single plan-span threshold: an unstratified gate marks
the red frames as stalls and — worse — scores a model that *runs* red lights as healthy.
Group the scenes by the traffic-light state their own route lanes carry, then require:

    green scenes  ->  mean plan span >= --min_green_m   (the model commits to go)
    red scenes    ->  mean plan span <= --max_red_m     (the model holds)

Amber is reported but not gated: easing and proceeding are both defensible, so it is a
behaviour difference between checkpoints rather than a pass or a fail.

    python -m rlvr.autoresearch.ego_shape_diag.check_tl_gate \
        --model <best_model.pth> --scenes <dir-or-glob> \
        [--goal X,Y] [--shape wheelbase,length,width] \
        [--min_green_m 15] [--max_red_m 2]

Seconds of compute and no closed-loop simulation. Worth running on every checkpoint,
because open-loop trajectory error does not detect this failure: a model can hold a
standstill plan at signalised geometry while scoring normally on displacement metrics.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from diffusion_planner.dimensions import (
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_RED,
    TRAFFIC_LIGHT_YELLOW,
)

from rlvr.autoresearch.ego_shape_diag._common import (
    find_scenes,
    load_deployable_model,
    load_scene,
    parse_shape,
    plan_span,
    set_shape,
)

_TL_SLICE = slice(TRAFFIC_LIGHT_GREEN, TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT + 1)
_CLASSES = {
    TRAFFIC_LIGHT_GREEN - TRAFFIC_LIGHT_GREEN: "green",
    TRAFFIC_LIGHT_YELLOW - TRAFFIC_LIGHT_GREEN: "amber",
    TRAFFIC_LIGHT_RED - TRAFFIC_LIGHT_GREEN: "red",
}
_ORDER = ("green", "amber", "red", "none")


def route_tl_class(d: dict) -> str:
    """The traffic-light state carried by the scene's own route lanes.

    Red wins over amber and amber over green when several appear: a route whose lanes
    include a red signal is a hold situation whatever the other lanes show. Scenes with
    no signalled route lane are reported as ``none`` and excluded from the gate.
    """
    rl = d["route_lanes"].detach().cpu().numpy()
    if rl.ndim > 3:
        rl = rl.reshape(-1, rl.shape[-2], rl.shape[-1])
    valid = np.abs(rl).sum(axis=(1, 2)) > 0
    if not valid.any():
        return "none"
    onehot = rl[valid][:, 0, _TL_SLICE]
    hot = onehot.max(axis=1) > 0
    if not hot.any():
        return "none"
    present = {_CLASSES.get(int(c)) for c in onehot[hot].argmax(axis=1)}
    for state in ("red", "amber", "green"):
        if state in present:
            return state
    return "none"


def measure(model, margs, scenes, device, goal, shape) -> dict[str, list[float]]:
    """Plan span for every scene, bucketed by its route traffic-light state."""
    spans: dict[str, list[float]] = {k: [] for k in _ORDER}
    for path in scenes:
        d = load_scene(path, device, goal)
        spans[route_tl_class(d)].append(
            plan_span(model, margs, set_shape(d, *shape) if shape else d, device)
        )
    return spans


def verdict(spans: dict[str, list[float]], min_green_m: float, max_red_m: float) -> list[str]:
    """Gate failures, empty when the checkpoint passes."""
    if not spans["green"] and not spans["red"]:
        raise SystemExit(
            "no signalled route lanes in any scene — this gate needs scenes whose route "
            "carries a traffic-light state; refusing to report a vacuous pass"
        )
    failures = []
    if spans["green"]:
        green = float(np.mean(spans["green"]))
        if green < min_green_m:
            failures.append(f"green {green:.2f} m < {min_green_m} m (does not commit to go)")
    if spans["red"]:
        red = float(np.mean(spans["red"]))
        if red > max_red_m:
            failures.append(f"red {red:.2f} m > {max_red_m} m (runs the light)")
    return failures


def report(spans: dict[str, list[float]], model_path: str, n_scenes: int, shape) -> None:
    print(f"model  : {model_path}")
    print(f"scenes : {n_scenes}   " + "  ".join(f"{k} {len(spans[k])}" for k in _ORDER))
    print(f"shape  : {shape if shape else 'as recorded'}")
    for k in _ORDER:
        if spans[k]:
            print(f"   {k:<6} n={len(spans[k]):<3} mean span {float(np.mean(spans[k])):6.2f} m")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="deployable checkpoint (args.json alongside)")
    ap.add_argument("--scenes", required=True, help="scene dir, glob, or single .npz")
    ap.add_argument("--goal", default=None, help="x,y ego-frame goal override")
    ap.add_argument("--shape", default=None, help="wheelbase,length,width to evaluate at")
    ap.add_argument("--min_green_m", type=float, default=15.0)
    ap.add_argument("--max_red_m", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = every scene")
    return ap.parse_args()


def main() -> int:
    a = _parse_args()
    goal = tuple(float(x) for x in a.goal.split(",")) if a.goal else None
    shape = parse_shape(a.shape) if a.shape else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_deployable_model(a.model, device)
    scenes = find_scenes(a.scenes)
    if a.limit:
        scenes = scenes[: a.limit]

    spans = measure(model, margs, scenes, device, goal, shape)
    report(spans, a.model, len(scenes), shape)

    failures = verdict(spans, a.min_green_m, a.max_red_m)
    print()
    for f in failures:
        print(f"FAIL: {f}")
    print("PASS" if not failures else "FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
