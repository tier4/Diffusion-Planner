#!/usr/bin/env python3
"""Audit a training scene set for a correlation between ego dimensions and speed.

If a corpus mixes vehicle platforms, ``ego_shape`` can end up being the only input that
separates them — and if those groups also differ in how far they travel, the token stops
meaning geometry and starts meaning "which speed distribution do I imitate". That is a
property of the data, visible before any training run.

Reports per dimension class:
  * how many distinct ``ego_shape`` values exist (two is a binary switch: highest risk)
  * travel over the prediction horizon, the quantity that must NOT be class-dependent
  * how often the ego is already stopped, and how often the target is stationary
  * goal-distance distribution, as a sanity check on the conversion

    python -m rlvr.autoresearch.ego_shape_diag.check_training_data \
        --scenes <dir-or-json-list> [--wheelbase_split 4.0] [--sample 800]

Healthy result: the travel distributions overlap. If one class is markedly slower, either
decorrelate the token (jitter, dropout, rebalance) or drop the off-platform data — and
check the resulting model with ``check_ego_shape_gate``.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random

import numpy as np


def load_list(path: str) -> list[str]:
    """Accept a directory, a glob, or a dataset-list JSON (list, or dict of lists)."""
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
    if path.endswith(".json"):
        data = json.loads(open(path).read())
        if isinstance(data, dict):
            for key in ("path_list", "scenes", "npz_list"):
                if key in data:
                    return list(data[key])
            return list(next(iter(data.values())))
        return list(data)
    return sorted(glob.glob(path))


def collect(paths: list[str], split: float) -> tuple[collections.Counter, dict]:
    """Tally distinct ego_shape values and per-class travel / goal / stopped stats."""
    shapes: collections.Counter = collections.Counter()
    per: dict = collections.defaultdict(lambda: {"travel": [], "goal": [], "stopped": 0, "n": 0})
    for path in paths:
        try:
            z = np.load(path)
        except Exception:
            continue
        if "ego_shape" not in z:
            continue
        shape = tuple(np.round(z["ego_shape"], 4))
        shapes[shape] += 1
        entry = per[f"wheelbase > {split:g}" if shape[0] > split else f"wheelbase <= {split:g}"]
        entry["n"] += 1
        if "ego_agent_future" in z:
            future = z["ego_agent_future"][:, :2]
            entry["travel"].append(float(np.linalg.norm(future[-1] - future[0])))
        if "goal_pose" in z:
            g = z["goal_pose"]
            entry["goal"].append(float(np.hypot(g[0], g[1])))
        if "ego_current_state" in z:
            state = z["ego_current_state"]
            if len(state) > 5 and float(np.hypot(state[4], state[5])) < 0.5:
                entry["stopped"] += 1
    return shapes, per


def report_classes(per: dict) -> None:
    for name, entry in per.items():
        travel = np.array(entry["travel"])
        goal = np.array(entry["goal"])
        print(f"\n  {name}  n={entry['n']}")
        if len(travel):
            print(
                f"    horizon travel : p10={np.percentile(travel, 10):6.2f} "
                f"p50={np.percentile(travel, 50):6.2f} p90={np.percentile(travel, 90):6.2f} "
                f"mean={travel.mean():6.2f} m"
            )
            print(f"    stationary targets (<2 m): {(travel < 2).mean() * 100:5.1f}%")
        print(
            f"    ego stopped now (<0.5 m/s): {entry['stopped'] / max(entry['n'], 1) * 100:5.1f}%"
        )
        if len(goal):
            print(
                f"    goal distance: p5={np.percentile(goal, 5):7.1f} "
                f"p50={np.percentile(goal, 50):7.1f} p95={np.percentile(goal, 95):7.1f} m"
                f"   frac<20m={(goal < 20).mean() * 100:4.1f}%"
            )


def report_gap(per: dict) -> None:
    """The headline comparison, only meaningful with exactly two classes."""
    if len(per) != 2:
        return
    (name_a, entry_a), (name_b, entry_b) = list(per.items())
    travel_a, travel_b = np.array(entry_a["travel"]), np.array(entry_b["travel"])
    if not len(travel_a) or not len(travel_b):
        return
    med_a, med_b = np.percentile(travel_a, 50), np.percentile(travel_b, 50)
    slower = name_a if med_a < med_b else name_b
    print(
        f"\n>>> median horizon travel: {name_a}={med_a:.2f} m  vs  {name_b}={med_b:.2f} m"
        f"   ({abs(med_a - med_b) / max(med_a, med_b) * 100:.0f}% gap)"
    )
    print(f">>> '{slower}' is the SLOWER class. With two ego_shape values the model can use")
    print(">>> the token to select that slower distribution. Decorrelate it.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="dir, glob, or dataset-list json")
    ap.add_argument(
        "--wheelbase_split",
        type=float,
        default=4.0,
        help="split scenes into two classes at this wheelbase, in metres",
    )
    ap.add_argument("--sample", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    paths = load_list(a.scenes)
    random.seed(a.seed)
    if len(paths) > a.sample:
        paths = random.sample(paths, a.sample)
    print(f"sampled {len(paths)} scenes\n")

    shapes, per = collect(paths, a.wheelbase_split)
    if not shapes:
        raise SystemExit(
            f"no scene under {a.scenes!r} carries ego_shape — wrong dataset root or a "
            "converter that does not emit it; refusing to report an empty audit"
        )
    total = sum(shapes.values())
    binary = "   <-- BINARY SWITCH, highest risk" if len(shapes) == 2 else ""
    print(f"distinct ego_shape values: {len(shapes)}{binary}")
    for shape, count in shapes.most_common(10):
        print(f"   {shape}  ->  {count:5d}  ({count / total * 100:5.1f}%)")

    print("\nper class:")
    report_classes(per)
    report_gap(per)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
