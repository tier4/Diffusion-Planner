"""Closed-loop validation of a Diffusion-Planner checkpoint with the PerfectTracker.

Open-loop counterpart: ``valid_predictor.py`` runs ONE forward per sample and scores
the prediction against the recorded GT future. This script instead drives the ego in
CLOSED LOOP through ``scenario_generation``: each tick the model predicts the ego
trajectory, ``PerfectTracker`` advances the ego one step along it, the recorded
neighbors are replayed from the log via the Perception-Reproducer cursor, and the
realized ego footprint is scored against those neighbors with the canonical OBB
(``score_object_step`` -> collision / near-miss / min clearance).

A *route* = one bag-prefix group of consecutive 10 Hz NPZ frames (``RouteTimeline``);
each route is rolled out whole with ``render_segment`` (one GPU forward per tick), which BOTH
returns the route metrics AND writes a per-step PNG of the live-ego scene. Every run therefore
always produces video: one MP4 per route (``<route>.mp4``). Per-route metrics are streamed to
``segments.jsonl`` and aggregated into ``summary.json`` (both next to the checkpoint).
Clearance t-digest sketches used for multi-GPU p5 merge are written beside them as
``tdigests.jsonl`` / ``tdigests_{rank}.jsonl`` so ``segments.jsonl`` stays human-readable.

Multi-GPU: launched via ``torch.distributed.run`` (same as training). Each rank reads
``RANK``/``WORLD_SIZE`` from the environment and evaluates its route shard
(``route_keys[rank::world]``). Rank 0 merges all shards when done.
Example::

    NPZ=/path/to/closed_loop_npz_dir   # or /path/to/path_list.json
    python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone \
        valid_predictor_closed_loop.py \
        --model_path /path/to/best_model.pth \
        --npz_root ${NPZ}
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from diffusion_planner.utils import ddp


def _negative_mps2(value: str) -> float:
    """argparse type: strong-brake threshold must be negative (accel <= thresh)."""
    v = float(value)
    if v > 0.0:
        raise argparse.ArgumentTypeError(
            f"strong_brake_mps2 must be <= 0 (got {v}); a positive threshold matches nearly every step"
        )
    return v


def parse_args() -> argparse.Namespace:
    # Only the checkpoint and the NPZ dir are required; everything else is a tunable
    # knob with the closed-loop mining default. Outputs land next to the checkpoint.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="checkpoint .pth; args.json must sit next to it (e.g. epoch0001/best_model.pth)",
    )
    p.add_argument(
        "--npz_root",
        type=Path,
        required=True,
        help="dir tree of route NPZ frames (recursively globbed, grouped into routes), OR a .json "
        "path list of such dirs (one route dir per entry, like --train_set_list). Pose JSON "
        "sidecars are read from next to each .npz, falling back to its own source tree.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="output dir for segments.jsonl/summary.json/videos. Default: "
        "<model_path dir>/closed_loop/<timestamp>/",
    )
    # --- tunable knobs (default to the closed-loop mining config) ---
    p.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    p.add_argument("--near_miss_thresh", type=float, default=0.5, help="near-miss clearance (m)")
    p.add_argument(
        "--strong_brake_mps2",
        type=_negative_mps2,
        default=-2.5,
        help="strong-brake threshold (m/s^2, negative); a step counts when this and the previous frame both have tangential accel <= this",
    )
    p.add_argument(
        "--search_radius", type=float, default=1.5, help="PerceptionReproducer cursor search (m)"
    )
    p.add_argument(
        "--yaw_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pose mode only: drop recorded frames whose heading differs from live ego by "
        "more than pi/2 (threshold fixed at pi/2; --no-yaw-gate disables). No-op in clock "
        "mode, which never calls cursor.step (closed-loop/R2LPL default)",
    )
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="steps driven by the recorded GT pose before handing control to the model",
    )
    p.add_argument(
        "--unstick_after",
        type=int,
        default=300,
        help="snap the ego to the GT pose ahead after this many no-progress steps (0=off)",
    )
    p.add_argument(
        "--unstick_advance_m", type=float, default=2.5, help="how far ahead to snap when unsticking"
    )
    p.add_argument(
        "--unstick_radius_mult",
        type=float,
        default=3.0,
        help="when stuck, first widen the cursor search_radius to this x nominal so it reaches "
        "frames further ahead (model proceeds on its own); restored to nominal once the ego moves. "
        "<=1 disables this gentle stage (teleport straight away at --unstick_after)",
    )
    p.add_argument(
        "--unstick_teleport_after",
        type=int,
        default=300,
        help="if still stuck this many steps AFTER the radius was widened, fall back to the hard "
        "teleport onto the GT pose ahead (last resort)",
    )
    p.add_argument("--fps", type=int, default=10, help="output video frame rate (10 = realtime)")
    p.add_argument(
        "--replan_interval",
        type=int,
        default=4,
        help="re-run the model every N steps (1 = every step). Between inferences the cached plan "
        "is executed, re-expressed in the current ego frame each step; the ego still steps at 10Hz",
    )
    p.add_argument(
        "--timeline_progress_mode",
        choices=("pose", "clock"),
        default="pose",
        help="recorded-world progress source; world delay is defined only for clock mode",
    )
    p.add_argument(
        "--world_delay_mode",
        choices=("none", "lag_extrapolated", "lag_raw"),
        default="none",
        help="delay only the model's recorded-world input; metrics retain the true clock",
    )
    p.add_argument("--k_lag", type=int, default=0, help="world-input delay in 0.1 s ticks")
    p.add_argument(
        "--delay_step",
        type=int,
        default=0,
        help="legacy bundled knob: sets both --plan_dead_time_step and --prefix_step (0..5)",
    )
    p.add_argument(
        "--plan_dead_time_step",
        type=int,
        default=None,
        help="simulator dead time in 0.1 s ticks between the inferred ego state and the tick "
        "the plan starts executing (inference + publish + controller intake); the previous "
        "plan keeps executing meanwhile. Overrides --delay_step for this half",
    )
    p.add_argument(
        "--prefix_step",
        type=int,
        default=None,
        help="model-input setting: committed rows handed to the decoder as a fixed prefix via "
        "the delay tensor (0 = prefix conditioning off, as on the deployed ONNX). Must be "
        "<= plan dead time. Overrides --delay_step for this half",
    )
    p.add_argument(
        "--tracker_mode",
        choices=("perfect", "mpc", "delayed"),
        default="perfect",
        help="ego execution model; delayed wraps MPC with a dead-time/first-order plant",
    )
    p.add_argument(
        "--plant_parameter_set",
        choices=("official", "measured", "custom"),
        default="official",
    )
    p.add_argument("--steer_dead_time_s", type=float, default=None)
    p.add_argument("--steer_time_constant_s", type=float, default=None)
    p.add_argument("--accel_dead_time_s", type=float, default=None)
    p.add_argument("--accel_time_constant_s", type=float, default=None)
    p.add_argument(
        "--controller_compensation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="predict through pending plant commands before MPC; valid only with delayed tracker",
    )
    p.add_argument(
        "--draw_every",
        type=int,
        default=8,
        help="render a PNG only every N steps (1 = every step). PNG rendering (matplotlib) is the "
        "dominant cost; this throttles it without touching the rollout. Frames are encoded at --fps "
        "regardless, so the video also plays N x faster (shorter). For real-time use --fps 10/N",
    )
    p.add_argument(
        "--draw_workers",
        type=int,
        default=4,
        help="render the PNGs on this many worker processes (minimum 1). Output is "
        "byte-identical whatever the count; costs ~780 MB of RSS per worker, per GPU shard",
    )
    p.add_argument(
        "--abort_deviation_m",
        type=float,
        default=0.0,
        help="early-abort a segment (terminated='diverged') once GT deviation exceeds this "
        "(m) for --abort_after steps (0=disabled)",
    )
    p.add_argument("--abort_after", type=int, default=30)
    p.add_argument(
        "--abort_max_snaps",
        type=int,
        default=0,
        help="early-abort a segment after this many unstick teleports (0=disabled)",
    )
    p.add_argument(
        "--drop_objects",
        action="store_true",
        help="empty-world ablation: zero out dynamic/static objects each step (map kept)",
    )
    # Windowed evaluation (scenario_generation.eval_windows): GT warm-started windows scored
    # with gates + a graded composite instead of one whole-route rollout.
    w = p.add_argument_group("windowed evaluation")
    w.add_argument(
        "--eval_windows",
        choices=("none", "fixed", "anchor"),
        default="none",
        help="none = whole-route rollout (default); fixed = fixed-length clock windows; "
        "anchor = windows around anchors from --anchors_json",
    )
    w.add_argument("--window_len_s", type=float, default=30.0)
    w.add_argument("--window_stride_s", type=float, default=None)
    w.add_argument("--window_min_tail_s", type=float, default=5.0)
    w.add_argument("--anchor_pre_s", type=float, default=10.0)
    w.add_argument("--anchor_post_s", type=float, default=10.0)
    w.add_argument("--anchor_unit", choices=("frame", "sec"), default="frame")
    w.add_argument(
        "--anchors_json",
        type=Path,
        default=None,
        help="JSON {route key or unique substring: [anchor, ...]} for --eval_windows anchor",
    )
    w.add_argument("--coverage_abort_m", type=float, default=100.0)
    w.add_argument("--coverage_abort_after", type=int, default=30)
    w.add_argument("--progress_gate_min", type=float, default=0.2)
    w.add_argument("--min_recorded_progress_m", type=float, default=5.0)
    w.add_argument("--window_lon_tol_m", type=float, default=30.0)
    w.add_argument("--window_lat_tol_m", type=float, default=3.0)
    w.add_argument("--window_w_progress", type=float, default=0.5)
    w.add_argument("--window_w_lon", type=float, default=0.25)
    w.add_argument("--window_w_lat", type=float, default=0.25)
    w.add_argument(
        "--window_indices",
        type=int,
        nargs="*",
        default=None,
        help="evaluate only these window indices of each route (videos / spot checks)",
    )
    w.add_argument(
        "--window_draw_every",
        type=int,
        default=None,
        help="render a PNG every N ticks per window and encode <window>.mp4 (default: no video)",
    )
    w.add_argument("--window_video_fps", type=float, default=5.0)
    w.add_argument(
        "--max_windows_per_route",
        type=int,
        default=None,
        help="smoke tests only: evaluate just the first N windows of each route",
    )
    return p.parse_args()


def _window_config(args: argparse.Namespace):
    """Build the ``WindowConfig`` for ``--eval_windows`` (validated by the driver)."""
    from scenario_generation.eval_windows import WindowConfig

    anchors = {}
    if args.eval_windows == "anchor":
        if args.anchors_json is None:
            raise ValueError("--eval_windows anchor requires --anchors_json")
        anchors = json.loads(Path(args.anchors_json).read_text())
    return WindowConfig(
        mode=args.eval_windows,
        window_len_s=args.window_len_s,
        window_stride_s=args.window_stride_s,
        min_tail_s=args.window_min_tail_s,
        anchor_pre_s=args.anchor_pre_s,
        anchor_post_s=args.anchor_post_s,
        anchor_unit=args.anchor_unit,
        anchors=anchors,
        coverage_abort_m=args.coverage_abort_m,
        coverage_abort_after=args.coverage_abort_after,
        progress_gate_min=args.progress_gate_min,
        min_recorded_progress_m=args.min_recorded_progress_m,
        lon_tol_m=args.window_lon_tol_m,
        lat_tol_m=args.window_lat_tol_m,
        w_progress=args.window_w_progress,
        w_lon=args.window_w_lon,
        w_lat=args.window_w_lat,
        max_windows_per_route=args.max_windows_per_route,
        window_indices=tuple(args.window_indices) if args.window_indices else None,
        draw_every=args.window_draw_every,
        video_fps=args.window_video_fps,
    )


# ``_eval_knobs`` keys that belong to the whole-route driver, not to ``render_segment``.
_DRIVER_ONLY_KNOBS = ("fps", "draw_workers")


def _window_render_kwargs(knobs: dict) -> dict:
    return {k: v for k, v in knobs.items() if k not in _DRIVER_ONLY_KNOBS}


def _load_model(model_path: Path, device: str):
    """Load either a torch checkpoint or an exported .onnx into the same (model, model_args)
    contract. args.json must sit next to whichever file is given."""
    from scenario_generation.simulate import load_model, load_onnx_model

    if model_path.suffix == ".onnx":
        return load_onnx_model(model_path, device)
    return load_model(model_path, device)


def _eval_knobs(args: argparse.Namespace) -> dict:
    """The rollout tunables forwarded to run_closed_loop_eval (everything except model/npz/out/
    device/shard), gathered once so the sequential and per-worker calls stay in lockstep."""
    from scenario_generation.closed_loop_delay import (
        resolve_plant_parameters,
        validate_delay_options,
    )

    # Fail at the CLI boundary, before DDP setup or checkpoint loading. The
    # evaluation entry point validates again because it is also a public API.
    validate_delay_options(
        timeline_progress_mode=args.timeline_progress_mode,
        world_delay_mode=args.world_delay_mode,
        k_lag=args.k_lag,
        delay_step=args.delay_step,
        plan_dead_time_step=args.plan_dead_time_step,
        prefix_step=args.prefix_step,
        tracker_mode=args.tracker_mode,
        neighbor_history_mode="recorded",
        replan_interval=args.replan_interval,
        controller_compensation=args.controller_compensation,
    )
    resolve_plant_parameters(
        args.plant_parameter_set,
        steer_dead_time_s=args.steer_dead_time_s,
        steer_time_constant_s=args.steer_time_constant_s,
        accel_dead_time_s=args.accel_dead_time_s,
        accel_time_constant_s=args.accel_time_constant_s,
    )
    return dict(
        near_miss_thresh=args.near_miss_thresh,
        search_radius=args.search_radius,
        yaw_gate=args.yaw_gate,
        warmup_steps=args.warmup_steps,
        unstick_after=args.unstick_after,
        unstick_advance_m=args.unstick_advance_m,
        unstick_radius_mult=args.unstick_radius_mult,
        unstick_teleport_after=args.unstick_teleport_after,
        fps=args.fps,
        replan_interval=args.replan_interval,
        timeline_progress_mode=args.timeline_progress_mode,
        world_delay_mode=args.world_delay_mode,
        k_lag=args.k_lag,
        delay_step=args.delay_step,
        plan_dead_time_step=args.plan_dead_time_step,
        prefix_step=args.prefix_step,
        plant_parameter_set=args.plant_parameter_set,
        steer_dead_time_s=args.steer_dead_time_s,
        steer_time_constant_s=args.steer_time_constant_s,
        accel_dead_time_s=args.accel_dead_time_s,
        accel_time_constant_s=args.accel_time_constant_s,
        controller_compensation=args.controller_compensation,
        draw_every=args.draw_every,
        draw_workers=args.draw_workers,
        neighbor_history_mode="recorded",
        tracker_mode=args.tracker_mode,
        strong_brake_mps2=args.strong_brake_mps2,
        abort_deviation_m=args.abort_deviation_m,
        abort_after=args.abort_after,
        abort_max_snaps=args.abort_max_snaps,
        drop_objects=args.drop_objects,
    )


def _merge_shards(
    out_dir: Path, npz_root, near_miss_thresh: float, *, strong_brake_mps2: float
) -> dict:
    """Aggregate every rank's segments_{rank}.jsonl (+ tdigests sidecars) into one summary.

    Also writes a merged, human-readable ``segments.jsonl`` (same tdigest-stripped shape the
    sequential path writes) for downstream consumers.
    """
    from scenario_generation.closed_loop_eval import (
        aggregate,
        load_segment_rows_with_tdigests,
        segment_row_for_json,
    )

    rows = load_segment_rows_with_tdigests(out_dir)
    with open(out_dir / "segments.jsonl", "w") as f:
        for r in sorted(rows, key=lambda r: r["route"]):
            f.write(json.dumps(segment_row_for_json(r, route=r["route"]), default=float) + "\n")
    summary = aggregate(rows, near_miss_thresh, strong_brake_mps2=strong_brake_mps2)
    summary["npz_root"] = str(npz_root)
    summary["n_routes"] = len({r["route"] for r in rows})
    summary["video_mp4s"] = sorted(str(p) for p in out_dir.glob("*.mp4"))
    return summary


def _write_summary(out_dir: Path, summary: dict) -> None:
    with open(out_dir / "summary.json", "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "video_mp4s"}, f, indent=4)


def main() -> None:
    import torch

    from scenario_generation.closed_loop_eval import run_closed_loop_eval

    args = parse_args()

    out_dir = args.out_dir or (
        args.model_path.parent / "closed_loop" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    knobs = _eval_knobs(args)

    class _CfgShim:
        ddp = True

    _rank, _local_rank, _world = ddp.ddp_setup_universal(True, _CfgShim())
    print(f"{_rank=}, {_local_rank=}, {_world=}")
    shard = (_rank, _world) if _world > 1 else None

    if args.device.startswith("cuda"):
        torch.cuda.set_device(_local_rank)
        device = f"cuda:{_local_rank}"
    else:
        device = args.device

    model, model_args = _load_model(args.model_path, device)
    print(
        f"device: {device} | model: {args.model_path} | out: {out_dir}"
        + (f" | shard: {shard}" if shard else "")
    )

    if args.eval_windows != "none":
        from scenario_generation.eval_windows import run_windowed_eval

        if shard is not None:
            raise RuntimeError("--eval_windows runs single-process; launch without DDP")
        cfg = _window_config(args)
        t0 = time.perf_counter()
        summary = run_windowed_eval(
            model,
            model_args,
            args.npz_root,
            out_dir,
            cfg=cfg,
            render_kwargs={"device": device, **_window_render_kwargs(knobs)},
            verbose=True,
        )
        summary["model_path"] = str(args.model_path)
        with open(out_dir / "windows_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(
            f"\n=== windowed closed-loop eval: {summary['n_windows']} windows / "
            f"{summary['n_routes']} routes in {summary['elapsed_sec']:.1f}s ==="
        )
        for key in (
            "score_macro",
            "score_micro",
            "graded_mean",
            "gate_pass_rate",
            "progress_ratio_mean",
            "ade_lon_m_mean",
            "ade_lat_m_mean",
            "at_fault_windows",
            "road_border_windows",
            "diverged_windows",
        ):
            print(f"{key}: {summary.get(key)}")
        return

    t0 = time.perf_counter()
    summary = run_closed_loop_eval(
        model,
        model_args,
        args.npz_root,
        out_dir,
        device=device,
        verbose=True,
        shard=shard,
        **knobs,
    )
    summary["elapsed_sec"] = time.perf_counter() - t0

    # Barrier: wait for all ranks to finish writing their segments_{rank}.jsonl
    # Only rank 0 merges the results to avoid race conditions
    if shard is not None:
        import torch.distributed as dist

        dist.barrier()
        if _rank == 0:
            summary = _merge_shards(
                out_dir,
                args.npz_root,
                args.near_miss_thresh,
                strong_brake_mps2=args.strong_brake_mps2,
            )
            summary["elapsed_sec"] = time.perf_counter() - t0

    summary["model_path"] = str(args.model_path)
    _write_summary(out_dir, summary)

    from scenario_generation.closed_loop_eval import format_summary_lines

    n_seg = summary["n_segments"]
    print(f"\n=== closed-loop validation: {n_seg} segments in {summary['elapsed_sec']:.1f}s ===")
    for line in format_summary_lines(summary):
        print(line)
    print(f"videos: one <route>.mp4 per route in {out_dir}")


if __name__ == "__main__":
    main()
