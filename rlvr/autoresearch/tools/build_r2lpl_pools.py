"""Build the R2LPL anchor/waits training pools from an IL scene list.

Reconstructs, as one auditable tool, the pool pipeline that was previously run as
ad-hoc job scripts. Stages (each a subcommand, chainable via files):

  subsample    seeded-shuffle slice of a scene list (the scan inputs)
  moving       keep scenes whose ego future END DISPLACEMENT >= threshold
  stopgo       waits pool: build_patience_benchmark stop-events pool, then the
               same moving filter (stop-phase scenes that DO take off), with an
               optional union against a previous pool version
  dedup        greedy per-log frame dedup (min frame gap between kept scenes)
  skip-audit   drop scenes flagged by the canonical is_skipped checker
  arc-region   build a route-region point set from full drives in a route cache
               (arc-length window along the driven path)
  arc-exclude  drop scenes whose recorded pose lies within radius of a region

Every threshold is an explicit required argument — there are no defaults for
values that change the output, so a config cannot silently drift from the
recipe that produced a given pool.

Scene entries are NPZ paths; the recorded world pose of a scene is read from
its same-name ``.json`` sidecar (keys ``x``/``y``). Stages that read sidecars
or NPZs (moving, stopgo, skip-audit, arc-*) touch one small file per scene —
run them on a node with local access to the dataset, not against NFS.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_FRAME_RE = re.compile(r"_(\d+)$")


def _read_list(path: Path) -> list[str]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of scene paths")
    return [str(p) for p in data]


def _write_list(paths: list[str], out: Path) -> None:
    Path(out).write_text(json.dumps(paths))
    print(f"[build_r2lpl_pools] wrote {len(paths)} entries -> {out}")


def _frame_index(path: str) -> int:
    match = _FRAME_RE.search(Path(path).stem)
    if match is None:
        raise ValueError(f"cannot parse trailing frame index from {path}")
    return int(match.group(1))


def _log_key(path: str) -> tuple[str, str]:
    p = Path(path)
    return (str(p.parent), _FRAME_RE.sub("", p.stem))


def _sidecar_pose(path: str) -> np.ndarray | None:
    try:
        j = json.loads(Path(path).with_suffix(".json").read_text())
        return np.array([float(j["x"]), float(j["y"])])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


# --- moving -------------------------------------------------------------


def _end_displacement(path: str) -> float | None:
    try:
        d = np.load(path, allow_pickle=True)
        fut = d["ego_agent_future"][:, :2]
        cur = d["ego_current_state"][:2]
        return float(np.linalg.norm(fut[-1] - cur))
    except Exception:
        return None


def _moving_worker(args: tuple[str, float]) -> tuple[str, bool] | None:
    """Returns (path, kept); None means the scene was unreadable."""
    path, thresh = args
    disp = _end_displacement(path)
    if disp is None:
        return None
    return (path, disp >= thresh)


def cmd_subsample(args: argparse.Namespace) -> None:
    paths = _read_list(args.scene_list)
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    sliced = paths[args.start : args.end]
    if not sliced:
        raise ValueError(f"empty slice [{args.start}:{args.end}] of {len(paths)} entries")
    _write_list(sliced, args.out)


def cmd_moving(args: argparse.Namespace) -> None:
    paths = _read_list(args.scene_list)
    keep, unreadable = _moving_filter(paths, args.min_end_displacement_m, args.workers)
    print(
        f"[moving] kept {len(keep)}/{len(paths)} "
        f"(threshold {args.min_end_displacement_m} m, {unreadable} unreadable)"
    )
    _write_list(keep, args.out)


def _moving_filter(paths: list[str], thresh: float, workers: int) -> tuple[list[str], int]:
    keep: list[str] = []
    unreadable = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for result in ex.map(_moving_worker, ((p, thresh) for p in paths), chunksize=256):
            if result is None:
                unreadable += 1
            elif result[1]:
                keep.append(result[0])
    # An unreadable corpus must not silently shrink the pool to nothing: a wrong
    # dataset root or format mismatch makes EVERY scene unreadable, and an empty
    # pool would propagate into training as a quiet no-op.
    if paths and not keep:
        raise RuntimeError(
            f"moving filter kept 0 of {len(paths)} scenes ({unreadable} unreadable) — "
            "wrong dataset root, corrupt NPZs, a threshold above every displacement, "
            "or (for stopgo) a stop pool where genuinely no scene takes off"
        )
    return keep, unreadable


_TL_SLICE = slice(8, 13)  # route_lanes TL one-hot [green, yellow, red, white, none]


def _takeoff_worker(args: tuple[str, int, float, float]) -> tuple[str, bool] | None:
    path, recent_steps, max_recent_travel_m, min_future_travel_m = args
    try:
        with np.load(path) as d:
            past = d["ego_agent_past"][:, :2]
            recent = float(
                np.linalg.norm(np.diff(past[-(recent_steps + 1) :], axis=0), axis=1).sum()
            )
            if recent >= max_recent_travel_m:
                return (path, False)
            rl = d["route_lanes"]
            valid = np.abs(rl).sum(axis=(1, 2)) > 0
            if not valid.any():
                return (path, False)
            onehot = rl[valid][:, 0, _TL_SLICE]
            hot = onehot.max(axis=1) > 0
            if not bool((hot & (onehot.argmax(axis=1) == 0)).any()):  # 0 = green
                return (path, False)
            fut = d["ego_agent_future"][:, :2]
            travel = float(np.linalg.norm(np.diff(fut, axis=0), axis=1).sum())
    except Exception:
        return None
    return (path, travel >= min_future_travel_m)


def cmd_takeoff(args: argparse.Namespace) -> None:
    """Green take-off pool: stopped input + route-lane GREEN + genuinely departing future.

    These are the frames where the recorded driver commits to GO at a signal —
    the decision the failure-window corpus systematically under-teaches. The
    departure threshold must reject creeps: a slow roll-forward re-teaches the
    deceleration bias instead of cancelling it.
    """
    paths = _read_list(args.scene_list)
    keep: list[str] = []
    unreadable = 0
    jobs = (
        (p, args.recent_past_steps, args.max_recent_travel_m, args.min_future_travel_m)
        for p in paths
    )
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for result in ex.map(_takeoff_worker, jobs, chunksize=256):
            if result is None:
                unreadable += 1
            elif result[1]:
                keep.append(result[0])
    if paths and not keep:
        raise RuntimeError(
            f"takeoff filter kept 0 of {len(paths)} scenes ({unreadable} unreadable) — "
            "wrong dataset root, a corpus without route-lane TL states, or thresholds "
            "no recorded departure satisfies"
        )
    print(
        f"[takeoff] kept {len(keep)}/{len(paths)} "
        f"(recent < {args.max_recent_travel_m} m over {args.recent_past_steps} steps, "
        f"route green, future >= {args.min_future_travel_m} m; {unreadable} unreadable)"
    )
    _write_list(keep, args.out)


def cmd_stopgo(args: argparse.Namespace) -> None:
    pool = _read_list(args.stop_pool)
    keep, unreadable = _moving_filter(pool, args.min_end_displacement_m, args.workers)
    print(f"[stopgo] {len(keep)}/{len(pool)} stop-pool scenes take off ({unreadable} unreadable)")
    if args.union_with is not None:
        prev = _read_list(args.union_with)
        keep = sorted(set(prev) | set(keep))
        print(f"[stopgo] union with {len(prev)} previous -> {len(keep)}")
    _write_list(keep, args.out)


def cmd_dedup(args: argparse.Namespace) -> None:
    paths = _read_list(args.scene_list)
    # Frame-sorted greedy per log: deterministic given the SET of scenes (input
    # order does not matter) — keep a scene iff it is at least min_frame_gap
    # frames after the previously kept scene of the same log.
    by_log: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for p in paths:
        by_log.setdefault(_log_key(p), []).append((_frame_index(p), p))
    kept_set: set[str] = set()
    for entries in by_log.values():
        entries.sort()
        last = None
        for idx, p in entries:
            if last is None or idx - last >= args.min_frame_gap:
                kept_set.add(p)
                last = idx
    seen: set[str] = set()
    kept = []
    for p in paths:
        if p in kept_set and p not in seen:
            seen.add(p)
            kept.append(p)
    print(f"[dedup] kept {len(kept)}/{len(paths)} (min frame gap {args.min_frame_gap})")
    _write_list(kept, args.out)


def cmd_skip_audit(args: argparse.Namespace) -> None:
    from diffusion_planner.utils.scene_skip import is_skipped  # canonical checker

    paths = _read_list(args.scene_list)
    kept, dropped = [], 0
    for p in paths:
        if is_skipped(p):
            dropped += 1
        else:
            kept.append(p)
    print(f"[skip-audit] dropped {dropped}/{len(paths)} is_skipped scenes")
    _write_list(kept, args.out)


# --- arc region ----------------------------------------------------------


def cmd_arc_region(args: argparse.Namespace) -> None:
    import pickle

    start = np.array([args.route_start_x, args.route_start_y])
    goal = np.array([args.route_goal_x, args.route_goal_y])
    cache = pickle.load(open(args.route_cache, "rb"))
    seg_pts: list[np.ndarray] = []
    full = 0
    for paths in cache.values():
        if len(paths) < args.min_log_len:
            continue
        a, b = _sidecar_pose(paths[0]), _sidecar_pose(paths[-1])
        if a is None or b is None:
            continue
        if (
            np.linalg.norm(a - start) < args.endpoint_tol_m
            and np.linalg.norm(b - goal) < args.endpoint_tol_m
        ):
            full += 1
            poses = [_sidecar_pose(x) for x in paths]
            n_missing = sum(1 for q in poses if q is None)
            if n_missing:
                # Dropping interior points compresses the cumulative arc length and
                # silently shifts the [arc_from_m, arc_to_m] window along the route.
                raise RuntimeError(
                    f"arc-region: drive matched the endpoints but {n_missing}/{len(paths)} "
                    "pose sidecars are unreadable — arc coordinates would be distorted"
                )
            pts = np.array(poses)
            arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            mask = (arc >= args.arc_from_m) & (arc <= args.arc_to_m)
            seg_pts.append(pts[mask])
            if full >= args.max_drives:
                break
    if not seg_pts:
        raise RuntimeError("no full drives matched the route endpoints; region would be empty")
    seg = np.vstack(seg_pts)
    Path(args.out).write_text(json.dumps(seg.tolist()))
    print(
        f"[arc-region] {full} drives, {len(seg)} pts, "
        f"bbox x {seg[:, 0].min():.0f}-{seg[:, 0].max():.0f} y {seg[:, 1].min():.0f}-{seg[:, 1].max():.0f}"
    )


_REGION: np.ndarray | None = None
_RADIUS: float | None = None


def _region_init(region_path: str, radius: float) -> None:
    global _REGION, _RADIUS
    _REGION = np.array(json.loads(Path(region_path).read_text()))
    _RADIUS = radius


def _near_region(path: str) -> bool | None:
    """True/False = inside/outside the region; None = pose unavailable."""
    assert _REGION is not None and _RADIUS is not None
    xy = _sidecar_pose(path)
    if xy is None:
        return None
    return bool((np.linalg.norm(_REGION - xy, axis=1) < _RADIUS).any())


def cmd_arc_exclude(args: argparse.Namespace) -> None:
    paths = _read_list(args.scene_list)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_region_init,
        initargs=(str(args.region), args.radius_m),
    ) as ex:
        flags = list(ex.map(_near_region, paths, chunksize=256))
    missing = [p for p, f in zip(paths, flags, strict=True) if f is None]
    if missing:
        # For an EXCLUSION filter, an unverifiable pose silently defaulting to
        # "keep" would retain scenes inside the excluded region — fail loudly.
        raise RuntimeError(
            f"arc-exclude: {len(missing)}/{len(paths)} scenes have no readable pose "
            f"sidecar (first: {missing[0]}) — cannot verify region membership"
        )
    kept = [p for p, f in zip(paths, flags, strict=True) if not f]
    print(
        f"[arc-exclude] excluded {len(paths) - len(kept)}/{len(paths)} (radius {args.radius_m} m)"
    )
    _write_list(kept, args.out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("subsample", help="seeded-shuffle slice of a scene list")
    p.add_argument("--scene_list", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_subsample)

    p = sub.add_parser("moving", help="keep scenes with ego end displacement >= threshold")
    p.add_argument("--scene_list", type=Path, required=True)
    p.add_argument("--min_end_displacement_m", type=float, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_moving)

    p = sub.add_parser(
        "takeoff", help="stopped-input + route-green + departing-future pool (release evidence)"
    )
    p.add_argument("--scene_list", type=Path, required=True)
    p.add_argument("--recent_past_steps", type=int, required=True)
    p.add_argument("--max_recent_travel_m", type=float, required=True)
    p.add_argument("--min_future_travel_m", type=float, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_takeoff)

    p = sub.add_parser("stopgo", help="moving filter over a stop-events pool (+ optional union)")
    p.add_argument(
        "--stop_pool", type=Path, required=True, help="out_pool of build_patience_benchmark"
    )
    p.add_argument("--min_end_displacement_m", type=float, required=True)
    p.add_argument("--union_with", type=Path, default=None)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_stopgo)

    p = sub.add_parser("dedup", help="greedy per-log frame dedup")
    p.add_argument("--scene_list", type=Path, required=True)
    p.add_argument("--min_frame_gap", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_dedup)

    p = sub.add_parser("skip-audit", help="drop canonical is_skipped scenes")
    p.add_argument("--scene_list", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_skip_audit)

    p = sub.add_parser("arc-region", help="build region points from full drives in a route cache")
    p.add_argument("--route_cache", type=Path, required=True)
    p.add_argument("--route_start_x", type=float, required=True)
    p.add_argument("--route_start_y", type=float, required=True)
    p.add_argument("--route_goal_x", type=float, required=True)
    p.add_argument("--route_goal_y", type=float, required=True)
    p.add_argument("--endpoint_tol_m", type=float, required=True)
    p.add_argument("--arc_from_m", type=float, required=True)
    p.add_argument("--arc_to_m", type=float, required=True)
    p.add_argument("--min_log_len", type=int, required=True)
    p.add_argument("--max_drives", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_arc_region)

    p = sub.add_parser("arc-exclude", help="drop scenes whose pose is within radius of a region")
    p.add_argument("--scene_list", type=Path, required=True)
    p.add_argument("--region", type=Path, required=True)
    p.add_argument("--radius_m", type=float, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_arc_exclude)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
