#!/usr/bin/env python3
"""Route-wide A/B between two ego-dimension settings: which stops are pathological?

Along a converted route, score every scene twice — once at each of two ``ego_shape``
settings — and classify the stops:

    PATHOLOGICAL : span_a < --stop_thresh AND span_b >= --ok_thresh
                   (only the dimension token separates driving from standing still)
    LEGITIMATE   : both spans < --stop_thresh
                   (the settings agree there is a reason to stop)

The counts matter more than any single frame: a handful of pathological stops is noise,
while a steady fraction across a route means the token is driving the decision. Use
``check_tl_gate`` on the same scenes to confirm whether the stops line up with red
signals, which is the usual innocent explanation.

    python -m rlvr.autoresearch.ego_shape_diag.check_route_ab \
        --model <best_model.pth> --scenes <dir> \
        --shape_a W,L,W --shape_b W,L,W [--goal X,Y]
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import torch

from rlvr.autoresearch.ego_shape_diag._common import (
    find_scenes,
    load_deployable_model,
    load_scene,
    parse_shape,
    plan_span,
    set_shape,
)

_FRAME_RE = re.compile(r"(\d+)\.npz")


def score_route(model, margs, scenes, device, goal, shape_a, shape_b) -> list[tuple]:
    """(frame index, span at shape A, span at shape B) for every scene on the route."""
    rows: list[tuple] = []
    for path in scenes:
        found = _FRAME_RE.findall(path)
        idx = int(found[-1]) if found else len(rows)
        d = load_scene(path, device, goal)
        rows.append(
            (
                idx,
                plan_span(model, margs, set_shape(d, *shape_a), device),
                plan_span(model, margs, set_shape(d, *shape_b), device),
            )
        )
    return rows


def classify(rows: list[tuple], stop_thresh: float, ok_thresh: float) -> tuple[list, list]:
    """Split the stops into (pathological, legitimate)."""
    patho = [r for r in rows if r[1] < stop_thresh and r[2] >= ok_thresh]
    legit = [r for r in rows if r[1] < stop_thresh and r[2] < stop_thresh]
    return patho, legit


def _report_spans(rows: list[tuple], stop_thresh: float) -> None:
    print(f"scenes: {len(rows)}")
    for name, index in (("A", 1), ("B", 2)):
        spans = np.array([r[index] for r in rows])
        print(
            f"shape {name} span: mean={spans.mean():6.2f} p50={np.percentile(spans, 50):6.2f} "
            f"frac<{stop_thresh:g}m={(spans < stop_thresh).mean() * 100:5.1f}%"
        )


def _report_pathological(patho: list[tuple], dt: float) -> None:
    if not patho:
        return
    print("\npathological frames:   t(s)  shape A  shape B")
    for idx, span_a, span_b in patho:
        print(f"   {idx * dt:8.1f}  {span_a:7.2f}  {span_b:7.2f}")
    print("\nCross-check these timestamps against the traffic-light state; stops on red")
    print("are legitimate however the two settings differ.")


def report(rows: list[tuple], stop_thresh: float, ok_thresh: float, dt: float) -> None:
    patho, legit = classify(rows, stop_thresh, ok_thresh)
    _report_spans(rows, stop_thresh)
    print(
        f"\nPATHOLOGICAL (A<{stop_thresh:g} and B>={ok_thresh:g}): "
        f"{len(patho)}/{len(rows)} = {len(patho) / len(rows) * 100:.1f}% of route"
    )
    print(
        f"LEGITIMATE   (both <{stop_thresh:g}, they agree):    "
        f"{len(legit)}/{len(rows)} = {len(legit) / len(rows) * 100:.1f}%"
    )
    _report_pathological(patho, dt)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="deployable checkpoint (args.json alongside)")
    ap.add_argument("--scenes", required=True, help="dir of NPZs converted from a full run")
    ap.add_argument("--shape_a", required=True, help="wheelbase,length,width (the suspect one)")
    ap.add_argument("--shape_b", required=True, help="wheelbase,length,width (the reference)")
    ap.add_argument("--goal", default=None, help="x,y ego-frame goal override")
    ap.add_argument("--stop_thresh", type=float, default=5.0)
    ap.add_argument("--ok_thresh", type=float, default=15.0)
    ap.add_argument("--dt", type=float, default=0.1, help="seconds per frame index unit")
    return ap.parse_args()


def main() -> int:
    a = _parse_args()
    goal = tuple(float(x) for x in a.goal.split(",")) if a.goal else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_deployable_model(a.model, device)
    rows = score_route(
        model,
        margs,
        find_scenes(a.scenes),
        device,
        goal,
        parse_shape(a.shape_a),
        parse_shape(a.shape_b),
    )
    report(rows, a.stop_thresh, a.ok_thresh, a.dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
