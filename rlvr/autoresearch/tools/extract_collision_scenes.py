"""Extract the 80-scene batch leading up to each mined collision.

Second pass after ``mine_collisions_reproducer``: for every segment in the hits
JSONL that had a collision, re-run the closed-loop reproducer and dump the
``pre_steps`` (default 80) scenes before the FIRST collision (<= ``collision_thresh``
m to any neighbor) as training NPZs. Each saved scene is a full model-input
snapshot (ego/neighbor history + map) plus the recorded neighbor GT future and the
realized ego future truncated at the collision — i.e. the exact context the GRPO
K=16 generator and the ARM7 guidance model need to produce candidate trajectories.

When the collision is within the first ``pre_steps`` steps, the window reaches
before the segment start; those earlier scenes are taken from the recorded NPZs of
prior frames (real GT history), which the full-route timeline still holds.

Example::

    python -m rlvr.autoresearch.tools.extract_collision_scenes \
        --npz_root  $SSD/.../npz_dir \
        --hits_jsonl /tmp/repro_hits.jsonl \
        --model_path $SSD/x2_model_base/best_model.pth \
        --out_dir /tmp/collision_batches
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from scenario_generation.reproducer_rollout import extract_collision_scenes
from scenario_generation.route_timeline import RouteTimeline, group_routes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz_root", type=Path, required=True, help="dir tree of route NPZ frames")
    p.add_argument("--sidecar_root", type=Path, default=None, help="pose/track-id sidecar tree")
    p.add_argument(
        "--hits_jsonl", type=Path, required=True, help="mine_collisions_reproducer output"
    )
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True, help="root for the per-segment batches")
    p.add_argument("--collision_thresh", type=float, default=0.2, help="m to any neighbor")
    p.add_argument("--pre_steps", type=int, default=80)
    p.add_argument("--search_radius", type=float, default=1.5)
    p.add_argument("--unstick_after", type=int, default=300)
    p.add_argument("--unstick_advance_m", type=float, default=5.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_hits", type=int, default=-1, help="limit segments processed (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from scenario_generation.simulate import load_model

    model, model_args = load_model(args.model_path, device)

    routes = group_routes(sorted(args.npz_root.rglob("*.npz")))
    rows = [json.loads(line) for line in args.hits_jsonl.read_text().splitlines() if line.strip()]
    # Only segments that actually collided are worth extracting.
    hits = [r for r in rows if r.get("n_collision_steps", 0) > 0]
    if args.max_hits > 0:
        hits = hits[: args.max_hits]
    print(f"collision segments to extract: {len(hits)} | device: {device}")

    # Group by route so each RouteTimeline is built once.
    by_route: dict[str, list] = defaultdict(list)
    for r in hits:
        by_route[r["route"]].append(tuple(r["segment"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    summary = []
    for route_key, segs in by_route.items():
        if route_key not in routes:
            print(f"  WARN route {route_key} not under --npz_root; skipping {len(segs)} segs")
            continue
        tl = RouteTimeline(routes[route_key], sidecar_dir=args.sidecar_root)
        for start, end in segs:
            od = args.out_dir / f"{route_key}_{start}_{end}"
            mani = extract_collision_scenes(
                model,
                model_args,
                tl,
                start,
                end,
                od,
                device=device,
                collision_thresh=args.collision_thresh,
                pre_steps=args.pre_steps,
                search_radius=args.search_radius,
                unstick_after=args.unstick_after,
                unstick_advance_m=args.unstick_advance_m,
            )
            if mani is None:
                print(f"  {route_key} [{start},{end}]: no collision <= {args.collision_thresh}m")
                continue
            n_ok += 1
            summary.append({"route": route_key, **mani})
            print(
                f"  {route_key} [{start},{end}]: collision@{mani['collision_step']} -> "
                f"{mani['n_scenes']} scenes ({mani['n_live']} live, {mani['n_recorded']} recorded)"
            )

    (args.out_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nextracted {n_ok} collision batches -> {args.out_dir}")


if __name__ == "__main__":
    main()
