"""Perturbed-start closed-loop recovery eval on a recorded route.

The question it answers: **put the ego somewhere it should not be, and does the
model drive back to the lane centre?** Open-loop validation loss cannot answer it
-- it measures agreement with a recorded future from an on-distribution state,
which is a different question -- so this is the headline metric for judging a
training-recipe A/B (augmentation, LR schedule, loss weights, data mix).

Per (model, start frame, lateral offset): rigidly shift the initial ego pose AND
its world-frame history sideways -- a pure-lateral perturbation with no kinematic
cue, so nothing in the state hints that a correction is under way -- then roll
``--steps`` (default 80 = 8 s) closed loop with the perfect tracker, replanning
every step, and score every REALIZED pose with the canonical
``compute_centerline_score_batch(usage_mode="baselink")``.

Reported per model: ``recovered_rate``, ``lost_rate``, and the settle
distribution. **Both rates are needed, and that is not decoration.** Because
these tools score ONE pose per call, ``compute_centerline_score_batch`` takes its
coverage-gap branch and returns exactly 0 for a pose further than
``subscores._PROXIMITY`` from any valid route-lane centre -- so a rollout that
wanders off the map scores as PERFECTLY centred. This tool therefore records each
pose's distance to the route alongside the score and gates every usage statistic
on it, so "recovered to the centreline" and "left the route" cannot be confused.
Never read a settle number without its ``lost_rate`` beside it.

Notes on use:

* Point ``--models`` at a **deployable checkpoint**: one whose weights are the
  EMA copy, not the raw optimizer iterates. A training milestone holds both, and
  ``simulate.load_model`` would silently take the raw ones, so this tool refuses
  a checkpoint that still carries ``ema_state_dict`` and tells you how to
  convert it.
* Shard by ``--start_begin`` / ``--start_end`` and run several shards in
  parallel; a rollout needs little GPU memory. Aggregate the shard JSONs and
  report only complete rows: a single shard is not a result.
* ``--drop_objects`` gives the empty-world ablation (no other traffic, map
  intact), which separates "reacts badly to traffic" from "cannot follow the
  route".

Implementation: the rollout driver is used unmodified; two module-level hooks in
``scenario_generation.reproducer_rollout`` are replaced for the process --
``_ego_state_from_frame`` (inject the offset) and ``_draw_step`` (score instead
of writing a PNG). They are not restored, so import this module for its helpers,
never for a side-effect-free call.

Alongside ``--out_json`` the tool writes ``<out_json>.meta.json`` recording the
argv, the resolved checkpoints, the rollout settings and how many rollouts were
skipped -- so a number in a write-up can always be traced back to its run.

Usage:
    python -m rlvr.autoresearch.tools.eval_recovery_route \\
        --models base:/path/base.pth treated:/path/treated.pth \\
        --npz_root <dir with the recorded route NPZ + sidecars> \\
        --drop_objects --start_stride 4 --offsets 1.0 \\
        --out_json out/recovery.json
"""

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import planner_metrics.subscores as subscores
from scenario_generation import reproducer_rollout
from scenario_generation.closed_loop_eval import enumerate_routes
from scenario_generation.route_timeline import RouteTimeline
from scenario_generation.simulate import load_model

# Per-step records of the rollout in flight, keyed by step index so the result is
# independent of the order the render pool happens to run the hook in.
_CL_SCORES: dict[int, float] = {}
_ROUTE_DIST: dict[int, float] = {}
_OFFSET = {"v": 0.0}

# The radius inside which the centerline scorer actually scores. Mirrors the local
# `_PROXIMITY` in subscores.compute_centerline_score_batch, which is not importable;
# if that value ever changes, this must change with it or the gate and the score
# will disagree about which poses count.
ROUTE_COVERAGE_RADIUS_M = 5.0
_ORIG_EGO_STATE = reproducer_rollout._ego_state_from_frame


def _offset_ego_state(tl, idx):
    """Shift the start pose and the whole ego history sideways by ``_OFFSET``."""
    pose, ego_hist, dyn = _ORIG_EGO_STATE(tl, idx)
    off = _OFFSET["v"]
    if off != 0.0:
        nx, ny = -math.sin(pose[2]), math.cos(pose[2])
        pose[0] += off * nx
        pose[1] += off * ny
        ego_hist[:, 0] += off * nx
        ego_hist[:, 1] += off * ny
    return pose, ego_hist, dyn


# `render_segment` takes every knob explicitly -- commit 6c490ec0c (2026-08-21) removed
# its defaults on purpose, so a caller cannot inherit a rollout setting it did not choose.
# These are the values this eval is defined by; the ones that matter for the measurement are
# the perfect tracker, recorded neighbour history, replanning every step, and no unsticking
# (a rollout that fails must be recorded as failing, not teleported back onto the route).
ROLLOUT_SETTINGS = dict(
    near_miss_thresh=0.5,
    search_radius=1.5,
    warmup_steps=0,
    unstick_after=0,
    unstick_advance_m=5.0,
    unstick_radius_mult=3.0,
    unstick_teleport_after=300,
    draw_every=1,  # the scoring hook runs per drawn step, so score every step
    tracker_mode="perfect",
    neighbor_history_mode="recorded",
    yaw_gate=True,
    strong_brake_mps2=-2.5,
    abort_deviation_m=0.0,  # never abort early: an off-route rollout is a result
    abort_after=30,
    abort_max_snaps=0,
    goal_mode="segment",
    title_prefix=None,
    distance_label_offset_m=1.2,
    view_half_m=50.0,
    max_stuck_steps=0,
    goal_reach_m=5.0,
    interpolate=True,
    color_by_uuid=True,
    window=None,
    timeline_progress_mode="pose",
)


def in_process_draw_pool() -> ThreadPoolExecutor:
    """The executor the per-step hook MUST run on: one worker, in this process.

    Passing ``draw_pool=None`` makes ``render_segment`` build a **spawn
    ProcessPoolExecutor** -- right for matplotlib, fatal here. The hook would be
    pickled by module+qualname, re-imported in a child interpreter, and would
    record its scores in that child's copy of the records, so the parent would
    collect nothing at all (``collect_usage`` raises on exactly that). One
    in-process worker also keeps the hook in step order.
    """
    return ThreadPoolExecutor(max_workers=1)


def require_deployable_checkpoint(model_path: str) -> None:
    """Refuse a training milestone that still carries raw optimizer iterates.

    ``simulate.load_model`` does ``ckpt.get("model", ckpt)``, and a milestone saved
    by the trainer holds BOTH ``model`` (the raw iterates, which at a live learning
    rate are one arbitrary sample of a random walk) and ``ema_state_dict`` (the
    deployable copy). Loading the raw ones produces a plausible-looking but wrong
    verdict with no warning at all, so this fails loudly instead.
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        raise SystemExit(
            f"{model_path} is a training milestone carrying both raw and EMA weights; "
            "load_model would silently use the RAW iterates. Convert it first:\n"
            '  ck = torch.load(src, map_location="cpu", weights_only=False)\n'
            '  ema = {k.removeprefix("module."): v for k, v in ck["ema_state_dict"].items()}\n'
            '  torch.save({"model": ema}, dst)   # and copy args.json next to dst'
        )


def route_distance_m(data: dict) -> float:
    """Distance from the ego (at the origin of this frame) to the nearest route point.

    ``route_lanes`` with a ``lanes`` fallback mirrors what the scorer itself does
    (``subscores.compute_centerline_score_batch``); the gate has to be computed off
    the same geometry the score was, or the two halves of the metric disagree. A
    frame carrying neither is not a route frame, and raises.
    """
    lanes = data.get("route_lanes", data.get("lanes"))
    if lanes is None:
        raise KeyError(
            "frame carries neither 'route_lanes' nor 'lanes'; the route-distance gate "
            "cannot be computed, and without it every rollout would score as lost"
        )
    if lanes.dim() == 4:
        lanes = lanes[0]
    centers = lanes[..., :2].reshape(-1, 2)
    valid = centers.norm(dim=-1) > 1e-3
    if not bool(valid.any()):
        return float("inf")
    return float(centers[valid].norm(dim=-1).min())


def centerline_inputs(np_dict: dict) -> tuple[dict, torch.Tensor]:
    """The scorer's inputs for the realized pose: map tensors + the ego at the origin."""
    data = {
        k: torch.as_tensor(np.asarray(v))
        for k, v in np_dict.items()
        if k in ("route_lanes", "lanes")
    }
    ego_traj = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)
    return data, ego_traj


def _scoring_draw_step(np_dict, pred, ego_shape, path, step=0, **kwargs):
    """Score the realized pose and record its distance to the route (no PNG)."""
    data, ego_traj = centerline_inputs(np_dict)
    score = subscores.compute_centerline_score_batch(
        ego_traj,
        torch.as_tensor(np.asarray(ego_shape), dtype=torch.float32),
        data,
        usage_mode="baselink",
    )
    _CL_SCORES[int(step)] = float(score[0])
    _ROUTE_DIST[int(step)] = route_distance_m(data)


def summarize_rollout(usage: np.ndarray, route_dist: np.ndarray) -> dict:
    """Turn one rollout's per-step records into its verdict.

    ``near`` gates every usage statistic on the pose actually being within the
    scorer's 5 m coverage radius, so an off-route rollout cannot borrow the
    coverage-gap zero and read as perfectly centred.
    """
    near = route_dist <= ROUTE_COVERAGE_RADIUS_M
    settle_near = usage[-10:][near[-10:]]
    off_frac = float((~near).mean())
    lost = bool(off_frac > 0.5 or not near[-10:].any())
    return dict(
        n_steps=len(usage),
        usage_t0=float(usage[0]),
        usage_mean=float(usage[near].mean()) if near.any() else None,
        usage_settle=float(settle_near.mean()) if settle_near.size else None,
        usage_max=float(usage[near].max()) if near.any() else None,
        frac_off_route=off_frac,
        lost=lost,
        # recovered = ended the rollout back on the route and inside half a lane
        # half-width of its centre
        recovered=bool((not lost) and settle_near.size and settle_near.mean() <= 0.5),
    )


def expand_offsets(spec: str) -> list[float]:
    """``"1.0,0.5"`` -> ``[1.0, -1.0, 0.5, -0.5]``; 0 stays a single unperturbed run."""
    offsets: list[float] = []
    for tok in spec.split(","):
        v = float(tok)
        for cand in (0.0,) if v == 0.0 else (v, -v):
            if cand not in offsets:
                offsets.append(cand)
    return offsets


def summarize_model(label: str, rows: list[dict]) -> dict:
    """Per-model summary. Lost rollouts are excluded from the settle statistics --
    their score is the coverage-gap zero, not a recovery -- and reported separately
    as ``lost_rate``, so a model can only look good on settle by staying near the
    route."""
    kept = [r for r in rows if not r["lost"] and r["usage_settle"] is not None]
    settle = np.array([r["usage_settle"] for r in kept]) if kept else np.array([np.nan])
    return dict(
        label=label,
        n_rollouts=len(rows),
        lost_rate=float(np.mean([r["lost"] for r in rows])) if rows else None,
        recovered_rate=float(np.mean([r["recovered"] for r in rows])) if rows else None,
        settle_mean=float(settle.mean()),
        settle_p50=float(np.percentile(settle, 50)),
        settle_p95=float(np.percentile(settle, 95)),
        n_scored=len(kept),
    )


def add_common_args(ap: argparse.ArgumentParser, *, start_stride: int, offsets: str) -> None:
    """The arguments both route evals share. Only the two defaults differ between them."""
    ap.add_argument("--models", nargs="+", required=True, help="label:model.pth (args.json beside)")
    ap.add_argument("--npz_root", required=True, help="recorded route NPZ dir (with sidecars)")
    ap.add_argument(
        "--start_stride",
        type=int,
        default=start_stride,
        help="test a start every N frames; the defaults are the protocol these tools were "
        "calibrated on, so changing one changes the sample size and the noise floor with it",
    )
    ap.add_argument("--min_speed", type=float, default=4.0, help="skip starts slower than this m/s")
    ap.add_argument(
        "--offsets",
        default=offsets,
        help="lateral offsets in metres; each non-zero value is expanded to +v and -v",
    )
    ap.add_argument("--steps", type=int, default=80, help="rollout length (0.1 s per step)")
    ap.add_argument("--start_begin", type=int, default=0, help="starts >= this frame (shards work)")
    ap.add_argument(
        "--start_end", type=int, default=10**9, help="starts < this frame (shards work)"
    )
    ap.add_argument(
        "--drop_objects",
        action="store_true",
        help="empty-world ablation: no other traffic every step, map unchanged",
    )
    ap.add_argument("--replan_interval", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", required=True)


def open_routes(npz_root: str) -> dict:
    """Routes under ``npz_root``. ``enumerate_routes`` itself raises on an empty tree, so an
    eval can never quietly score zero scenes and look like a pass."""
    return enumerate_routes(Path(npz_root))


def candidate_starts(tl: RouteTimeline, args) -> list[int]:
    """Frames this shard tests: fast enough to be a driving scene, far enough from either end."""
    lo = max(50, args.start_begin)
    hi = min(len(tl) - args.steps - 5, args.start_end)
    return [i for i in range(lo, hi, args.start_stride) if float(tl.speeds[i]) >= args.min_speed]


def drive_rollout(
    model, model_args, tl: RouteTimeline, start: int, args, draw_pool, out_dir
) -> str:
    """One closed-loop rollout; the per-step hook fills the module's records.

    Returns how it ended (``goal`` / ``stuck`` / ``diverged`` / ``max_steps``): a rollout
    cut short for a reason is not the same result as one that ran its full budget, so the
    row records it.
    """
    outcome = reproducer_rollout.render_segment(
        model,
        model_args,
        tl,
        start,
        start + args.steps,
        out_dir,
        device=args.device,
        replan_interval=args.replan_interval,
        max_steps=args.steps,
        drop_objects=args.drop_objects,
        draw_pool=draw_pool,
        **ROLLOUT_SETTINGS,
    )
    return str(outcome["terminated"])


MIN_SCORED_STEPS = 10

# Rollouts dropped for being shorter than MIN_SCORED_STEPS, so the count can be reported
# rather than quietly shrinking the denominator.
SKIPPED: dict[str, int] = {"short_rollouts": 0}


def collect_usage(scores: dict[int, float], dists: dict[int, float]) -> tuple | None:
    """Per-step records in step order as ``(usage, route_dist)``, or None for a too-short rollout.

    Usage is LINEAR: the raw centerline score is a squared penalty. An empty record means the
    hook never ran in this process, which is a bug, not a short rollout -- so it raises. A
    rollout with 1..MIN_SCORED_STEPS-1 steps IS dropped (its settle window would be the
    perturbation itself) but counted in ``SKIPPED`` and reported, never silently.
    """
    steps = sorted(scores)
    if not steps:
        raise RuntimeError(
            "the scoring hook recorded nothing for a completed rollout -- it did not run in "
            "this process (see in_process_draw_pool)"
        )
    if len(steps) < MIN_SCORED_STEPS:
        SKIPPED["short_rollouts"] += 1
        return None
    usage = np.sqrt(np.abs(np.array([scores[k] for k in steps])))
    return usage, np.array([dists[k] for k in steps])


def json_safe(obj):
    """Replace every non-finite float with None.

    ``json.dumps`` emits the bare literal ``NaN``, which no strict JSON reader accepts --
    and these files exist to be aggregated across shards.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def write_results(out_json: Path, results: list[dict], args) -> None:
    """Write the rows, plus a sidecar recording what produced them.

    The rows stay a bare list so existing aggregators keep working; the provenance goes
    beside them, because a results file nobody can trace back to a checkpoint and a shard
    range is the problem this tool exists to fix.
    """
    out_json.write_text(json.dumps(json_safe(results)))
    meta = dict(
        argv=sys.argv,
        models=args.models,
        npz_root=args.npz_root,
        n_rows=len(results),
        skipped=dict(SKIPPED),
        rollout_settings=ROLLOUT_SETTINGS,
        args={k: v for k, v in vars(args).items() if k != "models"},
    )
    out_json.with_suffix(out_json.suffix + ".meta.json").write_text(
        json.dumps(json_safe(meta), indent=1, default=str)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap, start_stride=4, offsets="1.0")
    args = ap.parse_args()

    reproducer_rollout._ego_state_from_frame = _offset_ego_state
    reproducer_rollout._draw_step = _scoring_draw_step

    offsets = expand_offsets(args.offsets)
    routes = open_routes(args.npz_root)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    draw_pool = in_process_draw_pool()
    for label_path in args.models:
        label, model_path = label_path.split(":", 1)
        require_deployable_checkpoint(model_path)
        model, model_args = load_model(model_path, args.device)
        for key in sorted(routes):
            tl = RouteTimeline(routes[key], sidecar_dir=Path(args.npz_root))
            starts = candidate_starts(tl, args)
            print(f"{label} {key}: {len(starts)} starts x {len(offsets)} offsets", flush=True)
            for st in starts:
                for off in offsets:
                    _OFFSET["v"] = off
                    _CL_SCORES.clear()
                    _ROUTE_DIST.clear()
                    terminated = drive_rollout(
                        model, model_args, tl, st, args, draw_pool, out_json.parent / "noop"
                    )
                    rec = collect_usage(_CL_SCORES, _ROUTE_DIST)
                    if rec is None:
                        continue
                    usage, rd = rec
                    results.append(
                        dict(
                            label=label,
                            route=key,
                            start=st,
                            offset=off,
                            terminated=terminated,
                            **summarize_rollout(usage, rd),
                            per_step=[round(float(x), 4) for x in usage],
                            per_step_route_dist=[round(float(x), 2) for x in rd],
                        )
                    )
        summary = summarize_model(label, [r for r in results if r["label"] == label])
        summary["skipped_short_rollouts"] = SKIPPED["short_rollouts"]
        print(json.dumps(json_safe(summary)), flush=True)
    draw_pool.shutdown(wait=True)
    write_results(out_json, results, args)


if __name__ == "__main__":
    main()
