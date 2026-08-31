"""Signed in-cut eval: does the model cut the inside of bends?

Same rollout driver and scoring as :mod:`eval_recovery_route` -- closed loop,
perfect tracker, replan every step -- but it additionally records the **signed**
lateral offset per step and the **signed curvature** of each start window, so a
directional bias toward the inside of a bend becomes measurable. The recovery
harness alone cannot see it: it reports ``|ego_lat| / side_hw``, which is
identical for an in-cut of 0.4 m and an out-swing of 0.4 m. Measuring the
absolute offset and calling it in-cut invalidated a whole earlier round.

One signed curvature is taken per start WINDOW (mean over its 80 frames), so a window that
changes hand mid-rollout is labelled by whichever bend dominates, and an in-cut on the other
bend is recorded with the wrong sign. Keep ``--min_kappa`` high enough that the windows it
admits are single bends, and read a near-zero ``incut_mean_m`` as possibly diluted rather than
as evidence of neutrality. (The per-step realized pose is stored, so a per-step curvature can
be recomputed from the output without re-running.)

Sign convention (defined and unit-tested in
:mod:`rlvr.autoresearch.tools.incut_geometry`): ``incut_m > 0`` means the ego sits
toward the **inside** of the bend, i.e. cutting the corner.

The default ``--offsets 0`` is unperturbed on purpose: the question is the
model's natural line through a bend, and a typical in-cut is ~0.25 m, so a 1 m
injected displacement would swamp it. Pair with ``--min_kappa`` to keep only
curved windows -- ``0.01`` is a <=100 m radius bend, ``0.02`` a <=50 m one.
Curve starts are a small minority of a mostly-straight route, so check how many
survive the filter before trusting a per-turn-direction split, and always report
left and right turns separately: a plain left/right bias would otherwise read as
in-cut on whichever direction the route happens to favour.

This tool also records the realized world pose per step, so any other metric can
be recomputed from the output JSON afterwards. An earlier round stored no pose and
its results could not be re-analysed once the metric turned out to be the wrong
one.

Usage:
    python -m rlvr.autoresearch.tools.eval_incut_route \\
        --models base:/path/base.pth treated:/path/treated.pth \\
        --npz_root <dir with the recorded route NPZ + sidecars> \\
        --drop_objects --start_stride 4 --offsets 0 --min_kappa 0.01 \\
        --out_json out/incut.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import planner_metrics.subscores as subscores
from rlvr.autoresearch.tools.eval_centerline_metrics import lat_offset_and_naive_score
from rlvr.autoresearch.tools.eval_recovery_route import (
    _OFFSET,
    ROUTE_COVERAGE_RADIUS_M,
    _offset_ego_state,
    add_common_args,
    candidate_starts,
    centerline_inputs,
    collect_usage,
    drive_rollout,
    expand_offsets,
    in_process_draw_pool,
    json_safe,
    open_routes,
    require_deployable_checkpoint,
    route_distance_m,
    summarize_rollout,
    write_results,
)
from rlvr.autoresearch.tools.incut_geometry import (
    incut_from_signed_lat,
    signed_curvature_from_poses,
)
from scenario_generation import reproducer_rollout
from scenario_generation.route_timeline import RouteTimeline
from scenario_generation.simulate import load_model

_CL_SCORES: dict[int, float] = {}
_ROUTE_DIST: dict[int, float] = {}
_SIGNED_LAT: dict[int, float] = {}
_POSES: list[np.ndarray] = []

_ORIG_PRE_STEP = reproducer_rollout._pre_step


def _scoring_draw_step(np_dict, pred, ego_shape, path, step=0, **kwargs):
    """Record the unsigned score (comparable with the recovery eval) AND the signed offset."""
    data, ego_traj = centerline_inputs(np_dict)
    shape_t = torch.as_tensor(np.asarray(ego_shape), dtype=torch.float32)
    score = subscores.compute_centerline_score_batch(ego_traj, shape_t, data, usage_mode="baselink")
    k = int(step)
    _CL_SCORES[k] = float(score[0])
    _ROUTE_DIST[k] = route_distance_m(data)

    # signed offset, in the scorer's own convention (+ = left of the route direction)
    ego_half_w = float(np.asarray(ego_shape).reshape(-1)[-1]) / 2.0
    lat = lat_offset_and_naive_score(ego_traj[0], data, ego_half_w, usage_mode="baselink")
    _SIGNED_LAT[k] = (
        float("nan")
        if lat is None
        else float(np.asarray(lat["signed_lat_offset_m"]).reshape(-1)[0])
    )


def _pre_step_record_pose(s, gpu_transform: bool = False):
    """Record the realized world pose each step; otherwise the driver is untouched."""
    pre = _ORIG_PRE_STEP(s, gpu_transform)
    if pre is None:
        return None
    # `live_pose` is a non-optional field of the rollout state, set at seed and on every
    # advance, so it is read directly: a `getattr` default here could only hide a real bug.
    _POSES.append(np.asarray(s.live_pose, dtype=float).copy())
    return pre


def curve_starts(tl: RouteTimeline, args) -> list[tuple[int, float]]:
    """``(start, signed curvature)`` for the starts whose window is curved enough.

    ``--min_kappa 0`` keeps straights too, and a straight window has no inside, so its
    in-cut is NaN rather than a zero that would dilute a curve-only average.
    """
    out = []
    for i in candidate_starts(tl, args):
        kappa = signed_curvature_from_poses(tl.poses[i : i + args.steps])
        if abs(kappa) >= args.min_kappa:
            out.append((i, float(kappa)))
    return out


def incut_block(signed_lat: np.ndarray, route_dist: np.ndarray, kappa: float) -> dict:
    """In-cut statistics for one rollout, gated on the pose being near the route."""
    near = route_dist <= ROUTE_COVERAGE_RADIUS_M
    incut = incut_from_signed_lat(signed_lat, kappa)
    fin = incut[near][np.isfinite(incut[near])] if near.any() else np.array([])
    tail = incut[-10:][near[-10:]]
    fin_tail = tail[np.isfinite(tail)]
    return dict(
        # + = toward the inside of the bend (cutting the corner)
        incut_mean_m=float(fin.mean()) if fin.size else None,
        incut_settle_m=float(fin_tail.mean()) if fin_tail.size else None,
        incut_max_m=float(fin.max()) if fin.size else None,
    )


def _turn_side(kept: list[dict], sel) -> dict:
    """In-cut over one turn direction. Always reported per direction: a plain left/right
    bias would otherwise read as in-cut on whichever direction the route favours."""
    v = np.array([r["incut_mean_m"] for r in kept if sel(r["kappa"])])
    if not v.size:
        # Same keys either way: an aggregator indexing incut_mean_m must see None, not KeyError.
        return dict(n=0, incut_mean_m=None, frac_incut=None)
    return dict(n=int(v.size), incut_mean_m=float(v.mean()), frac_incut=float((v > 0).mean()))


def _mean_or_none(values: list) -> float | None:
    arr = np.array([v for v in values if v is not None], dtype=float)
    return float(arr.mean()) if arr.size else None


def summarize_model(label: str, rows: list[dict]) -> dict:
    """Per-model in-cut summary, stratified by turn direction.

    Lost rollouts are excluded from the in-cut and settle statistics -- an off-route pose has
    no meaningful lateral offset -- and surface instead as ``lost_rate``.
    """
    kept = [r for r in rows if not r["lost"] and r["incut_mean_m"] is not None]
    if not kept:
        # No scored curve rollout. `frac_incut` over an all-NaN array would evaluate to 0.0 and
        # read as "this model never cuts corners", which is a fabricated result, not a no-op.
        return dict(
            label=label,
            n_rollouts=len(rows),
            lost_rate=_mean_or_none([r["lost"] for r in rows]),
            recovered_rate=_mean_or_none([r["recovered"] for r in rows]),
            settle_mean=None,
            incut_mean_m=None,
            incut_p50_m=None,
            incut_p95_m=None,
            frac_incut=None,
            left_turns=dict(n=0, incut_mean_m=None, frac_incut=None),
            right_turns=dict(n=0, incut_mean_m=None, frac_incut=None),
            n_scored=0,
        )
    inc = np.array([r["incut_mean_m"] for r in kept])
    return dict(
        label=label,
        n_rollouts=len(rows),
        lost_rate=_mean_or_none([r["lost"] for r in rows]),
        recovered_rate=_mean_or_none([r["recovered"] for r in rows]),
        settle_mean=_mean_or_none([None if r["lost"] else r["usage_settle"] for r in rows]),
        # + = the model sits toward the inside of bends on average
        incut_mean_m=float(inc.mean()),
        incut_p50_m=float(np.percentile(inc, 50)),
        incut_p95_m=float(np.percentile(inc, 95)),
        frac_incut=float(np.mean(inc > 0)),
        left_turns=_turn_side(kept, lambda k: k > 0),
        right_turns=_turn_side(kept, lambda k: k < 0),
        n_scored=len(kept),
    )


def score_rollout(
    model,
    model_args,
    tl: RouteTimeline,
    st: int,
    kappa: float,
    off: float,
    args,
    draw_pool,
    out_dir,
) -> dict | None:
    """Drive one rollout and turn its per-step records into a result row (None if too short)."""
    _OFFSET["v"] = off
    for buf in (_CL_SCORES, _ROUTE_DIST, _SIGNED_LAT):
        buf.clear()
    _POSES.clear()
    terminated = drive_rollout(model, model_args, tl, st, args, draw_pool, out_dir)
    rec = collect_usage(_CL_SCORES, _ROUTE_DIST)
    if rec is None:
        return None
    usage, rd = rec
    slat = np.array([_SIGNED_LAT[k] for k in sorted(_SIGNED_LAT)])
    return dict(
        start=st,
        offset=off,
        kappa=round(kappa, 5),
        terminated=terminated,
        **summarize_rollout(usage, rd),
        **incut_block(slat, rd, kappa),
        per_step=[round(float(x), 4) for x in usage],
        per_step_route_dist=[round(float(x), 2) for x in rd],
        per_step_signed_lat_m=[round(float(x), 4) for x in slat],
        per_step_pose=[[round(float(v), 3) for v in p] for p in _POSES],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap, start_stride=4, offsets="0")
    ap.add_argument(
        "--min_kappa",
        type=float,
        default=0.0,
        help="keep only starts whose window |signed curvature| >= this (1/m). "
        "0.02 ~ a 50 m radius bend. 0 keeps everything (straights included).",
    )
    args = ap.parse_args()

    reproducer_rollout._pre_step = _pre_step_record_pose
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
            cand = curve_starts(tl, args)
            print(
                f"{label} {key}: {len(cand)} starts x {len(offsets)} offsets "
                f"(min_kappa={args.min_kappa})",
                flush=True,
            )
            for st, kappa in cand:
                for off in offsets:
                    row = score_rollout(
                        model,
                        model_args,
                        tl,
                        st,
                        kappa,
                        off,
                        args,
                        draw_pool,
                        out_json.parent / "noop",
                    )
                    if row is not None:
                        results.append(dict(label=label, route=key, **row))
        print(
            json.dumps(
                json_safe(summarize_model(label, [r for r in results if r["label"] == label]))
            ),
            flush=True,
        )
    draw_pool.shutdown(wait=True)
    write_results(out_json, results, args)


if __name__ == "__main__":
    main()
