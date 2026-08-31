"""Render one perturbed-start rollout as a video, so a table row can be watched.

Same protocol as :mod:`rlvr.autoresearch.tools.eval_recovery_route` — the ego pose AND
its world-frame history are rigidly shifted sideways at t=0, then the model drives
closed loop with the perfect tracker — but the harness's real per-step renderer runs
instead of the silent scorer, so the rollout becomes PNGs and a WebM.

The per-step lane usage is recorded on the way through **with the same scorer the
tables use**, so the caption numbers cannot drift from the numbers they illustrate.
Pass two ``--models`` and ``--hstack`` to get the side-by-side clip: one start, one
perturbation, two arms, which is the only form in which a clip is evidence.

Pick the start with :mod:`rlvr.autoresearch.tools.find_disagreeing_starts` rather than
by eye — a clip of a start where both arms behave the same shows nothing.

Usage:
    python -m rlvr.autoresearch.tools.render_recovery_clip \\
        --models incumbent:$OUT_A/epoch0200_ema/best_model.pth \\
                 treated:$OUT_B/epoch0200_ema/best_model.pth \\
        --npz_root <route NPZ dir> --start 1206 --offset 1.0 \\
        --drop_objects --hstack --out_dir clips/
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import planner_metrics.subscores as subscores  # noqa: I001  (kept last: heavy import)
from rlvr.autoresearch.tools.eval_recovery_route import (
    _OFFSET,
    ROLLOUT_SETTINGS,
    _offset_ego_state,
    centerline_inputs,
    collect_usage,
    in_process_draw_pool,
    require_deployable_checkpoint,
    route_distance_m,
    summarize_rollout,
)
from scenario_generation import reproducer_rollout
from scenario_generation.closed_loop_eval import enumerate_routes
from scenario_generation.route_timeline import RouteTimeline
from scenario_generation.simulate import load_model

_CL_SCORES: dict[int, float] = {}
_ROUTE_DIST: dict[int, float] = {}
_ORIG_DRAW = reproducer_rollout._draw_step


def _scoring_draw_step(np_dict, pred, ego_shape, path, step=0, **kwargs):
    """Score the realized pose, then hand off to the harness's real renderer."""
    import torch

    data, ego_traj = centerline_inputs(np_dict)
    score = subscores.compute_centerline_score_batch(
        ego_traj,
        torch.as_tensor(np.asarray(ego_shape), dtype=torch.float32),
        data,
        usage_mode="baselink",
    )
    _CL_SCORES[int(step)] = float(score[0])
    _ROUTE_DIST[int(step)] = route_distance_m(data)
    return _ORIG_DRAW(np_dict, pred, ego_shape, path, step=step, **kwargs)


def encode_webm(png_dir: Path, out_webm: Path, fps: int) -> None:
    """PNG sequence -> VP9 WebM, the format these write-ups use."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            str(png_dir / "*.png"),
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
            str(out_webm),
        ],
        check=True,
    )


def hstack_webm(left: Path, right: Path, out_webm: Path) -> None:
    """Two clips side by side. Each pane's camera follows its own vehicle, so the
    thing to compare is the vehicles' motion, not the backgrounds."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            "hstack=inputs=2",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "0",
            "-crf",
            "32",
            "-row-mt",
            "1",
            str(out_webm),
        ],
        check=True,
    )


def render_one(label, model_path, tl, args, out_root, stem, draw_pool) -> tuple[Path, dict]:
    """One arm's clip plus the verdict for its caption."""
    model, model_args = load_model(model_path, args.device)
    png_dir = out_root / f"{stem}_{label}"
    png_dir.mkdir(parents=True, exist_ok=True)
    _OFFSET["v"] = args.offset
    _CL_SCORES.clear()
    _ROUTE_DIST.clear()

    settings = dict(ROLLOUT_SETTINGS)
    sign = "L" if args.offset > 0 else "R"
    settings["title_prefix"] = (
        f"{label}  |  start {args.start}  {sign}{abs(args.offset):g}m"
        f"  |  {'no traffic' if args.drop_objects else 'with traffic'}"
    )
    reproducer_rollout.render_segment(
        model,
        model_args,
        tl,
        args.start,
        args.start + args.steps,
        png_dir,
        device=args.device,
        replan_interval=args.replan_interval,
        max_steps=args.steps,
        drop_objects=args.drop_objects,
        draw_pool=draw_pool,
        **settings,
    )
    rec = collect_usage(_CL_SCORES, _ROUTE_DIST)
    if rec is None:
        raise SystemExit(
            f"{label}: rollout was too short to score ({len(_CL_SCORES)} steps); "
            "pick a start further from the end of the route"
        )
    usage, route_dist = rec
    webm = out_root / f"{stem}_{label}.webm"
    encode_webm(png_dir, webm, args.fps)
    verdict = summarize_rollout(usage, route_dist)
    verdict["per_step"] = [round(float(x), 4) for x in usage]
    return webm, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True, help="label:model.pth (1 or 2)")
    ap.add_argument("--npz_root", required=True, help="recorded route NPZ dir (with sidecars)")
    ap.add_argument("--start", type=int, required=True, help="start frame")
    ap.add_argument("--offset", type=float, required=True, help="lateral shift, m (sign matters)")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--replan_interval", type=int, default=1)
    ap.add_argument("--drop_objects", action="store_true")
    ap.add_argument("--fps", type=int, default=10, help="10 = real time at draw_every 1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hstack", action="store_true", help="also emit the side-by-side clip")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    reproducer_rollout._ego_state_from_frame = _offset_ego_state
    reproducer_rollout._draw_step = _scoring_draw_step

    routes = enumerate_routes(Path(args.npz_root))
    key = sorted(routes)[0]
    tl = RouteTimeline(routes[key], sidecar_dir=Path(args.npz_root))
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    sign = "L" if args.offset > 0 else "R"
    world = "notraffic" if args.drop_objects else "traffic"
    stem = f"start{args.start}_{sign}{abs(args.offset):g}m_{world}"

    draw_pool = in_process_draw_pool()
    clips, report = [], {}
    for spec in args.models:
        label, model_path = spec.split(":", 1)
        require_deployable_checkpoint(model_path)
        webm, verdict = render_one(label, model_path, tl, args, out_root, stem, draw_pool)
        clips.append(webm)
        report[label] = verdict
        print(
            f"{label}: {'LOST' if verdict['lost'] else 'recovered' if verdict['recovered'] else 'settled short'}"
            f"  settle={verdict['usage_settle']}  -> {webm.name}",
            flush=True,
        )
    draw_pool.shutdown(wait=True)

    if args.hstack:
        if len(clips) != 2:
            raise SystemExit("--hstack needs exactly two --models")
        pair = out_root / f"COMPARE_{stem}.webm"
        hstack_webm(clips[0], clips[1], pair)
        print("pair:", pair)

    (out_root / f"{stem}_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
