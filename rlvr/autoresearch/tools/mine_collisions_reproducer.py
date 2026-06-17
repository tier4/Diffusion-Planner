"""Mine closed-loop collision / near-collision scenes with the Perception Reproducer.

Runs a checkpoint closed-loop over recorded routes (ego driven by the planner +
PerfectTracker; neighbors replayed from the log via the autoware-style cursor),
scores every step with the canonical avoidance OBB
(``compute_static_collision_penalty``), and writes a ranked index of the segments
where the model collides or nearly collides.

Work unit = one route (bag-prefix group) sliced into ~60 s segments. Output is one
compact JSONL row per segment; ``--dump_hits`` optionally renders flagged ones.

Example::

    python -m rlvr.autoresearch.tools.mine_collisions_reproducer \
        --npz_root  $SSD/x2-rlvr-data/npz_val_padded320 \
        --sidecar_root $SSD/x2-rlvr-data/npz_val \
        --model_path $SSD/x2_model_base/best_model.pth \
        --out /tmp/repro_hits.jsonl --seg_len 600
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import run_segments_batched
from scenario_generation.route_timeline import RouteTimeline, group_routes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz_root", type=Path, required=True, help="dir tree of route NPZ frames")
    p.add_argument(
        "--sidecar_root",
        type=Path,
        default=None,
        help="dir tree of pose JSON sidecars (if not next to the NPZ, e.g. npz_val for padded320)",
    )
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output JSONL of per-segment metrics")
    p.add_argument("--seg_len", type=int, default=600, help="frames per segment (~60s @10Hz)")
    p.add_argument("--near_miss_thresh", type=float, default=0.5)
    p.add_argument("--search_radius", type=float, default=1.5)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="segments run in lock-step per batched GPU forward (throughput lever)",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_routes", type=int, default=-1, help="limit routes (debug)")
    p.add_argument("--max_segments", type=int, default=-1, help="limit total segments (debug)")
    return p.parse_args()


def _enumerate_routes(npz_root: Path) -> dict[str, list[Path]]:
    paths = sorted(npz_root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz under {npz_root}")
    return group_routes(paths)


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from scenario_generation.simulate import load_model

    model, model_args = load_model(args.model_path, device)

    routes = _enumerate_routes(args.npz_root)
    route_keys = sorted(routes)
    if args.max_routes > 0:
        route_keys = route_keys[: args.max_routes]
    print(f"routes: {len(route_keys)} | device: {device}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    timers = Timers()
    rows: list[dict] = []
    n_seg = 0
    t0 = time.perf_counter()

    fout = open(args.out, "w")

    def _flush(buf_units, buf_keys):
        """Run a buffered batch of work units through the batched rollout."""
        nonlocal n_seg
        if not buf_units:
            return
        res_list = run_segments_batched(
            model,
            model_args,
            buf_units,
            device=device,
            batch_size=args.batch_size,
            near_miss_thresh=args.near_miss_thresh,
            search_radius=args.search_radius,
            warmup_steps=args.warmup_steps,
            timers=timers,
        )
        for key, res in zip(buf_keys, res_list):
            row = {"route": key, **res.metrics}
            rows.append(row)
            fout.write(json.dumps(row, default=float) + "\n")
            n_seg += 1
        fout.flush()

    buf_units: list[tuple] = []
    buf_keys: list[str] = []
    stop = False
    for ri, key in enumerate(route_keys):
        with timers("timeline_build"):
            tl = RouteTimeline(routes[key], sidecar_dir=args.sidecar_root, timers=timers)
        for start, end in tl.iter_segments(args.seg_len):
            buf_units.append((tl, start, end))
            buf_keys.append(key)
            if args.max_segments > 0 and (n_seg + len(buf_units)) >= args.max_segments:
                stop = True
                break
        # Flush once the buffer holds at least one full batch (keeps the GPU fed
        # without holding every route's NPZ cache in memory at once).
        if len(buf_units) >= args.batch_size or stop:
            _flush(buf_units, buf_keys)
            buf_units, buf_keys = [], []
        print(f"[{ri + 1}/{len(route_keys)}] {key}: {n_seg} segments done")
        if stop:
            break
    _flush(buf_units, buf_keys)
    fout.close()

    elapsed = time.perf_counter() - t0
    # Rank: collisions first, then tightest clearance.
    hits = sorted(rows, key=lambda r: (-r["n_collision_steps"], r["min_clearance"]))
    print(
        f"\n=== mined {n_seg} segments in {elapsed:.1f}s ({elapsed / max(1, n_seg):.2f}s/seg) ==="
    )
    print("top hits (collisions desc, clearance asc):")
    for r in hits[:10]:
        print(
            f"  {r['route']} {r['segment']}  collisions={r['n_collision_steps']:3d}  "
            f"min_clr={r['min_clearance']:.2f}  near_miss={r['n_near_miss_steps']:3d}  "
            f"term={r['terminated']}"
        )
    print("\n" + timers.report(n_seg))
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
