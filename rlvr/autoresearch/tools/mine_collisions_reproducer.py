"""Mine closed-loop collision / near-collision scenes with the Perception Reproducer.

Runs a checkpoint closed-loop over recorded routes (ego driven by the planner +
PerfectTracker; neighbors replayed from the log via the autoware-style cursor),
scores every step with a raw all-neighbor OBB overlap check (``score_step`` —
collision = the ego box overlaps ANY neighbor box, moving or static, including
rear-end hits; no stopped-only / ego-speed / direction gates), and writes a ranked
index of the segments where the model collides or nearly collides.

Work unit = one route (bag-prefix group) sliced into ~60 s segments. Output is one
compact JSONL row per segment; ``--dump_hits`` optionally renders flagged ones.

Example::

    python -m rlvr.autoresearch.tools.mine_collisions_reproducer \
        --npz_root  /path/to/npz_padded \
        --sidecar_root /path/to/npz_sidecars \
        --model_path /path/to/model.pth \
        --out /tmp/repro_hits.jsonl --seg_len 600
"""

from __future__ import annotations

import argparse
import heapq
import json
import subprocess
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
        help="dir tree of pose JSON sidecars (if not next to the NPZ, e.g. the "
        "pre-padding conversion tree when the padded NPZs dropped their sidecars)",
    )
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--lora_path", type=Path, default=None)
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
    p.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="how many top-ranked segments to keep in memory for the summary/render "
        "(every segment is still streamed to the JSONL; bounds RAM on huge corpora)",
    )
    # Throughput / coverage levers (forwarded to run_segments_batched).
    p.add_argument("--n_build_threads", type=int, default=8, help="threads for the CPU input build")
    p.add_argument(
        "--preload",
        action="store_true",
        help="decompress every recorded frame of each route into the in-RAM npz cache "
        "BEFORE rolling it out, so np.load/zlib never lands on the per-step critical path. "
        "OFF by default: it front-loads the whole route's decompress (and holds it in RAM), "
        "which is a net loss for short/few-segment jobs where prefetch already hides the I/O; "
        "turn on for large multi-pass mining where the I/O half dominates.",
    )
    p.add_argument(
        "--gpu_transform",
        action="store_true",
        help="run world_to_ego_frame on-device (one batched op) instead of per-segment numpy "
        "on the CPU build threads. OFF by default: the transform is a small, already-overlapped "
        "slice of the wall (the model forward dominates), and when --save_dir is set the saved "
        "scenes must be materialized back to CPU each step, so the gain is marginal; results are "
        "numerically equivalent to the CPU path (~1e-5, float-ordering).",
    )
    # Neighbor history is always rebuilt from the SIMULATED shown motion (track by UUID,
    # interpolate between recorded anchors, velocity = finite diff of shown positions). The old
    # "recorded" mode (copy each cursor frame's 31-step history verbatim) is removed.
    p.add_argument(
        "--prefetch_ahead",
        type=int,
        default=2,
        help="prefetch this many upcoming frames per segment during the GPU forward "
        "(overlaps npz I/O with compute; 0=off, results unchanged)",
    )
    p.add_argument(
        "--max_steps_mult",
        type=int,
        default=3,
        help="step cap = this x seg_len (the only timeout; default 3x)",
    )
    p.add_argument(
        "--unstick_after",
        type=int,
        default=300,
        help="snap the ego to the GT pose ahead after this many no-progress steps (0=off)",
    )
    p.add_argument("--unstick_advance_m", type=float, default=5.0)
    p.add_argument(
        "--dump_hits",
        type=int,
        default=0,
        help="render the top-N ranked hit segments to PNGs under <out>.renders/ (0=off)",
    )
    p.add_argument("--render_webm", action="store_true", help="assemble dumped hit PNGs into WebM")
    p.add_argument("--webm_fps", type=int, default=10)
    # One-pass collision-scene save (no second extract pass). When --save_dir is set,
    # each segment buffers its last --save_pre_steps scenes during THIS rollout and dumps
    # them to <save_dir>/<route>_<start>_<end>/ on the first step within --save_thresh of a
    # neighbor — so the saved scenes match the collision THIS run detected (reproducible).
    p.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="if set, save pre-collision scene batches in one pass (no separate extractor)",
    )
    p.add_argument(
        "--save_pre_steps", type=int, default=80, help="MIN scenes saved before each hit"
    )
    p.add_argument(
        "--save_pre_arc_m",
        type=float,
        default=1.0,
        help="extend the window past --save_pre_steps until the ego has travelled this much "
        "arc length (m) — so a slow creep into a stopped car still captures real approach, "
        "not 80 near-identical frames. Capped at --save_max_scenes.",
    )
    p.add_argument(
        "--save_max_scenes",
        type=int,
        default=160,
        help="hard cap on scenes saved per hit (bounds buffer RAM ~1.25MB/scene/segment)",
    )
    p.add_argument(
        "--save_min_post_snap_s",
        type=float,
        default=3.0,
        help="DROP a hit if an unstick teleport fired less than this many seconds before it "
        "(too little settled history; the contact is likely teleport-induced). 0=keep all.",
    )
    p.add_argument(
        "--save_thresh",
        type=float,
        default=0.2,
        help="m-to-neighbor trigger for saving a scene batch (use 0.5 to also catch near-misses)",
    )
    p.add_argument(
        "--save_min_pre_frames",
        type=int,
        default=30,
        help="skip a hit if fewer than this many LIVE frames precede the contact. The saver "
        "no longer backfills recorded frames (that spliced a discontinuous prefix onto the "
        "live rollout), so an early contact yields a shorter all-live window — and is dropped "
        "entirely below this floor.",
    )
    p.add_argument(
        "--save_min_ego_speed",
        type=float,
        default=0.5,
        help="exclude collisions where the ego is not moving: only save a hit if ego speed at "
        "the contact step exceeds this (m/s). Filters out the ego being rear-ended / sitting "
        "stopped against a neighbor — not a model-caused avoidance failure. 0 disables.",
    )
    return p.parse_args()


def _make_webm(frames_dir: Path, out_path: Path, fps: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "%05d.png"),
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "0",
            "-crf",
            "32",
            "-row-mt",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        check=True,
    )


def _enumerate_routes(npz_root: Path) -> dict[str, list[Path]]:
    # OPT-OUT of skip-filtering on purpose: the reproducer is the ONE consumer that needs
    # the converter's skip_for_training frames (red-light dwell etc.) so the timeline is
    # gap-free. We deliberately do NOT call scene_skip.filter_scene_list here — every
    # frame under npz_root is replayed (skip_filter=False is implicit).
    paths = sorted(npz_root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz under {npz_root}")
    return group_routes(paths)


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from scenario_generation.simulate import load_model

    model, model_args = load_model(args.model_path, device)
    if args.lora_path:
        from preference_optimization.lora_utils import load_lora_checkpoint

        print(f"loading LoRA: {args.lora_path}")
        model = load_lora_checkpoint(model, str(args.lora_path))
        model.eval()

    routes = _enumerate_routes(args.npz_root)
    route_keys = sorted(routes)
    if args.max_routes > 0:
        route_keys = route_keys[: args.max_routes]
    print(f"routes: {len(route_keys)} | device: {device}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    timers = Timers()
    # Every segment is streamed to the JSONL on disk; in memory we keep only a bounded
    # top-K heap for the ranked summary + optional renders, so RAM stays flat over
    # millions of segments. Heap key = (collisions, -min_clearance, seq) -> the heap's
    # smallest is the least interesting kept row and is evicted first.
    top_k = max(args.top_k, args.dump_hits)
    heap: list[tuple] = []
    n_seg = 0
    seq = 0
    t0 = time.perf_counter()

    fout = open(args.out, "w")

    def _keep(row: dict) -> None:
        nonlocal seq
        key = (row["n_collision_steps"], -row["min_clearance"], seq)
        seq += 1
        heapq.heappush(heap, (key, row))
        if len(heap) > top_k:
            heapq.heappop(heap)  # drop the least interesting

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
            unstick_after=args.unstick_after,
            unstick_advance_m=args.unstick_advance_m,
            max_steps_mult=args.max_steps_mult,
            n_build_threads=args.n_build_threads,
            prefetch_ahead=args.prefetch_ahead,
            timers=timers,
            save_dir=args.save_dir,
            save_pre_steps=args.save_pre_steps,
            save_thresh=(args.save_thresh if args.save_dir is not None else None),
            save_pre_arc_m=args.save_pre_arc_m,
            save_max_scenes=args.save_max_scenes,
            save_min_post_snap_frames=int(round(args.save_min_post_snap_s / 0.1)),
            save_min_pre_frames=args.save_min_pre_frames,
            save_min_ego_speed=args.save_min_ego_speed,
            route_keys=buf_keys,
            gpu_transform=args.gpu_transform,
            neighbor_history_mode="sim",  # always sim (recorded mode removed)
        )
        for key, res in zip(buf_keys, res_list):
            row = {"route": key, **res.metrics}
            fout.write(json.dumps(row, default=float) + "\n")
            _keep(row)
            n_seg += 1
        fout.flush()

    buf_units: list[tuple] = []
    buf_keys: list[str] = []
    stop = False
    try:
        for ri, key in enumerate(route_keys):
            with timers("timeline_build"):
                tl = RouteTimeline(routes[key], sidecar_dir=args.sidecar_root, timers=timers)
            if args.preload:
                # Warm the whole route into the npz cache up front (opt-in; see flag help).
                with timers("preload"):
                    tl.prefetch(range(len(tl)))
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
    finally:
        fout.close()  # don't leak the handle / lose buffered rows on an error mid-run

    elapsed = time.perf_counter() - t0
    # Rank the kept top-K: collisions desc, then tightest clearance (heap key desc).
    hits = [row for _key, row in sorted(heap, reverse=True)]
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
    print(f"\nwrote {n_seg} rows -> {args.out} (kept top {len(hits)} in memory for ranking)")

    # Optionally render the top-N ranked hit segments to PNGs for inspection.
    if args.dump_hits > 0:
        from scenario_generation.reproducer_rollout import render_segment

        render_root = args.out.with_suffix(".renders")
        render_root.mkdir(parents=True, exist_ok=True)
        for r in hits[: args.dump_hits]:
            if r["n_collision_steps"] == 0 and r["n_near_miss_steps"] == 0:
                continue  # nothing interesting to render
            s0, e0 = r["segment"]
            tl = RouteTimeline(routes[r["route"]], sidecar_dir=args.sidecar_root)
            od = render_root / f"{r['route']}_{s0}_{e0}"
            print(f"  rendering hit {r['route']} [{s0},{e0}] -> {od}")
            render_segment(
                model,
                model_args,
                tl,
                s0,
                e0,
                od,
                device=device,
                near_miss_thresh=args.near_miss_thresh,
                search_radius=args.search_radius,
                title_prefix=f"{r['route']}  {s0:05d}-{e0:05d}",
            )
            if args.render_webm:
                webm = od / "hit_segment.webm"
                _make_webm(od, webm, args.webm_fps)
                print(f"    webm -> {webm}")
        print(f"rendered {args.dump_hits} hit segment(s) -> {render_root}")


if __name__ == "__main__":
    main()
