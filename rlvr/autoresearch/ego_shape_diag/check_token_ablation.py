#!/usr/bin/env python3
"""Which input group is holding the plan back? Zero one at a time and re-measure.

When a model plans a standstill on geometry it should drive through, this says which
part of the input it is reacting to. Zero each input group in turn and watch the plan
span: a group whose removal releases the plan is the one the model is responding to.

``--split_line_strings`` goes further within that group, separating stop lines from road
borders and additionally pushing the borders outward. The distinction matters: if
dropping the borders releases the plan but moving them away does not, the model is not
failing to fit through a gap — it is holding at a boundary it has learned to respect.

    python -m rlvr.autoresearch.ego_shape_diag.check_token_ablation \
        --model <best_model.pth> --scene <npz> --shape W,L,W \
        [--goal X,Y] [--split_line_strings]

Ablation shows what the model attends to, not what is wrong with the data; confirm any
finding against the corpus with ``check_training_data``.
"""

from __future__ import annotations

import argparse

import torch

from rlvr.autoresearch.ego_shape_diag._common import (
    find_scenes,
    load_deployable_model,
    load_scene,
    parse_shape,
    plan_span,
    set_shape,
)

GROUPS = [
    "neighbor_agents_past",
    "static_objects",
    "line_strings",
    "polygons",
    "lanes",
    "route_lanes",
    "ego_agent_past",
]
_STOP_LINE_CHANNEL = 2
_ROAD_BORDER_CHANNEL = 3


def ablate_groups(model, margs, base: dict, ref: float, device) -> None:
    for key in GROUPS:
        if key not in base:
            continue
        d = dict(base)
        d[key] = torch.zeros_like(base[key])
        span = plan_span(model, margs, d, device)
        tag = "  <-- RELEASES" if span > max(ref * 3, 10) else ""
        print(f"  {'zero ' + key:32s} {span:7.2f} m{tag}")


def split_line_strings(model, margs, base: dict, device) -> None:
    """Separate stop lines from road borders, then test clearance by moving borders."""
    strings = base["line_strings"]
    is_stop = strings[..., _STOP_LINE_CHANNEL].abs().sum(-1) > 0
    is_border = strings[..., _ROAD_BORDER_CHANNEL].abs().sum(-1) > 0
    print(
        f"\n  line_strings slots: stop_line={int(is_stop.sum())} road_border={int(is_border.sum())}"
    )
    for name, mask in (("stop_line", is_stop), ("road_border", is_border)):
        d = dict(base)
        d["line_strings"] = strings * (~mask).float().unsqueeze(-1).unsqueeze(-1)
        print(f"  {'drop ' + name + ' only':32s} {plan_span(model, margs, d, device):7.2f} m")
    for push in (1.0, 4.0, 8.0):
        d = dict(base)
        moved = strings.clone()
        xy = moved[..., :2]
        radius = xy.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        selected = is_border.unsqueeze(-1).unsqueeze(-1).float()
        occupied = (xy.abs().sum(-1, keepdim=True) > 0).float()
        moved[..., :2] = xy + (xy / radius) * push * selected * occupied
        d["line_strings"] = moved
        print(
            f"  {'push road borders +%.0f m' % push:32s} "
            f"{plan_span(model, margs, d, device):7.2f} m"
        )
    print("\n  Borders pushed away but the plan unchanged => not a fit/clearance problem.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="deployable checkpoint (args.json alongside)")
    ap.add_argument("--scene", required=True, help="a single .npz (or dir; the first is used)")
    ap.add_argument(
        "--shape", required=True, help="ego_shape to hold fixed: wheelbase,length,width"
    )
    ap.add_argument("--goal", default=None, help="x,y ego-frame goal override")
    ap.add_argument(
        "--split_line_strings",
        action="store_true",
        help="also separate stop lines from road borders within that group",
    )
    a = ap.parse_args()

    goal = tuple(float(x) for x in a.goal.split(",")) if a.goal else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_deployable_model(a.model, device)
    scene = find_scenes(a.scene)[0]
    base = set_shape(load_scene(scene, device, goal), *parse_shape(a.shape))

    ref = plan_span(model, margs, base, device)
    print(f"scene: {scene.split('/')[-1]}")
    print(f"ego_shape held at {a.shape}\n")
    print(f"  {'reference (untouched)':32s} {ref:7.2f} m")
    ablate_groups(model, margs, base, ref, device)
    if a.split_line_strings and "line_strings" in base:
        split_line_strings(model, margs, base, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
