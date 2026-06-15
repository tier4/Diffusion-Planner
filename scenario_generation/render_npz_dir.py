#!/usr/bin/env python3
"""Render a directory of training-style NPZ files as per-step PNGs that look
exactly like ``scenario_generation.replay`` output.

Reuses ``scenario_generation.replay.save_step_figure`` (the same function
the replay calls every step) so the viewport, agent boxes, route overlay,
lane network, road borders, etc. all match closed-loop sim renders.

Usage:
    python -m scenario_generation.render_npz_dir \\
        --npz_dir <path>/npz \\
        --output_dir <path>/render \\
        [--route_pkl <route>.pkl] \\
        [--workers 8] [--limit 100] [--stride 1]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from scenario_generation.npz_loader import from_npz
from scenario_generation.replay import save_step_figure
from scenario_generation.transforms import yaw_from_quat


def _world_to_ego_polylines(
    polylines_world: list[np.ndarray] | None,
    ego_xy_yaw_world: tuple[float, float, float],
    crop_radius_m: float = 120.0,
) -> list[np.ndarray] | None:
    """Transform a list of world-frame ``(K, 2)`` polylines into the ego
    frame at this frame and crop to those passing within ``crop_radius_m``
    of the ego origin (a soft viewport prefilter). Returns ``None`` when
    the input is ``None``."""
    if polylines_world is None:
        return None
    cx, cy, yaw = ego_xy_yaw_world
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
    out: list[np.ndarray] = []
    for pl in polylines_world:
        if pl.shape[0] < 2:
            continue
        local = (pl - np.array([cx, cy], dtype=np.float64)) @ R.T
        # Drop polylines whose AABB lies entirely outside the viewport in
        # at least one axis — covers e.g. polylines well left/right of ego
        # whose y-extent still straddles 0. The earlier `min(abs)` check
        # incorrectly only rejected polylines outside on BOTH axes.
        if (
            local[:, 0].min() > crop_radius_m
            or local[:, 0].max() < -crop_radius_m
            or local[:, 1].min() > crop_radius_m
            or local[:, 1].max() < -crop_radius_m
        ):
            continue
        out.append(local.astype(np.float32))
    return out if out else None


def _build_map_overlays(
    route_pkl_path: Path | None,
) -> tuple[list[np.ndarray] | None, list[np.ndarray] | None]:
    """Return ``(route_polylines, road_border_polylines)`` from a Route
    pickle. Both are world-frame ``(K, 2)`` arrays. Returns ``(None, None)``
    when no route is given."""
    if route_pkl_path is None:
        return None, None
    from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
    from scenario_generation.route import Route

    route = Route.load(route_pkl_path)
    builder = LaneletSceneBuilder(route.map_path)
    rp: list[np.ndarray] = []
    if route.route_lanelet_ids:
        for lid in route.route_lanelet_ids:
            if not builder.has_lanelet_id(lid):
                continue
            cl = np.asarray(builder.raw_centerline(lid), dtype=np.float64)
            # raw_centerline can be (N, 3) on z-bearing maps; we only need
            # the (x, y) projection for the 2-D viewport overlay.
            if cl.ndim == 2 and cl.shape[1] >= 2 and len(cl) >= 2:
                rp.append(cl[:, :2])
    rb = builder.road_border_polylines()
    return (rp if rp else None), (rb if rb else None)


def _ego_future_to_predictions(scene):
    """Convert the rosbag's ``ego_agent_future`` (T, 3) [x, y, yaw_rad] in
    ego frame to the (T, 4) [x, y, cos_h, sin_h] format the replay's
    ``agent_predictions`` dict expects. Returns an empty dict when the
    scene has no future trajectory (or it has an unexpected shape)."""
    ego = scene.ego_agent
    fut = getattr(ego, "future_trajectory", None)
    if fut is None or len(fut) == 0:
        return {}
    if fut.shape[-1] == 3:
        fut4 = np.concatenate([fut[:, :2], np.cos(fut[:, 2:3]), np.sin(fut[:, 2:3])], axis=-1)
    elif fut.shape[-1] >= 4:
        fut4 = fut[:, :4]
    else:
        return {}
    return {ego.id: fut4.astype(np.float32)}


def _override_ego_dims(scene, ego_dims: tuple[float, float, float] | None) -> None:
    """Override ego length / width on the loaded SceneContext when the npz
    was converted with mismatched dims (e.g. data_converter defaults vs.
    the actual vehicle the rosbag was recorded on)."""
    if ego_dims is None:
        return
    length, width, wheelbase = ego_dims
    ego = scene.ego_agent
    if ego is not None:
        ego.length = float(length)
        ego.width = float(width)
        # ``wheelbase`` is consumed by some downstream metrics; assign it
        # if the dataclass has the slot, otherwise ignore.
        if hasattr(ego, "wheelbase"):
            ego.wheelbase = float(wheelbase)


# Worker-side globals populated by ``_pool_init`` so that the route /
# road-border polyline lists (often thousands of (K, 2) numpy arrays) are
# transmitted ONCE per worker instead of duplicated into every task tuple.
# This matters most under the ``spawn`` start method where every task is
# pickled — under ``fork`` (Linux default) the parent's memory is shared
# copy-on-write, but the dispatch overhead is still lower this way.
_WORKER_ROUTE_POLYLINES: list[np.ndarray] | None = None
_WORKER_ROAD_BORDER_POLYLINES: list[np.ndarray] | None = None
_WORKER_EGO_DIMS: tuple[float, float, float] | None = None
_WORKER_N_STEPS: int = 0


def _pool_init(
    route_polylines_world: list[np.ndarray] | None,
    road_border_polylines_world: list[np.ndarray] | None,
    ego_dims: tuple[float, float, float] | None,
    n_steps: int,
) -> None:
    global _WORKER_ROUTE_POLYLINES, _WORKER_ROAD_BORDER_POLYLINES
    global _WORKER_EGO_DIMS, _WORKER_N_STEPS
    _WORKER_ROUTE_POLYLINES = route_polylines_world
    _WORKER_ROAD_BORDER_POLYLINES = road_border_polylines_world
    _WORKER_EGO_DIMS = ego_dims
    _WORKER_N_STEPS = n_steps


def _render_one(args):
    idx, npz_path, out_path = args
    try:
        scene = from_npz(npz_path)
        _override_ego_dims(scene, _WORKER_EGO_DIMS)
        # The npz stores all geometry in ego frame at t=0; the json
        # sidecar gives the world MGRS pose so we can transform any
        # world-frame overlays (route, road borders) into the same ego
        # frame.
        sidecar = npz_path.with_suffix(".json")
        ego_xy_yaw = (0.0, 0.0, 0.0)
        if sidecar.exists():
            import json as _json

            j = _json.loads(sidecar.read_text())
            ego_xy_yaw = (
                float(j["x"]),
                float(j["y"]),
                yaw_from_quat(j["qx"], j["qy"], j["qz"], j["qw"]),
            )
        route_polylines = _world_to_ego_polylines(
            _WORKER_ROUTE_POLYLINES,
            ego_xy_yaw,
        )
        road_border_polylines = _world_to_ego_polylines(
            _WORKER_ROAD_BORDER_POLYLINES,
            ego_xy_yaw,
            crop_radius_m=80.0,
        )
        agent_predictions = _ego_future_to_predictions(scene)
        save_step_figure(
            scene=scene,
            agent_predictions=agent_predictions,
            output_path=out_path,
            step=idx,
            n_steps=_WORKER_N_STEPS,
            route_polylines=route_polylines,
            road_border_polylines=road_border_polylines,
            metrics=None,
        )
    except Exception as e:
        return f"FAIL {npz_path.name}: {type(e).__name__}: {e}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument(
        "--route_pkl",
        type=Path,
        default=None,
        help="Optional Route pickle. Provides both the route "
        "polyline overlay and the road-border line strings "
        "(loaded once from the lanelet2 map referenced by the "
        "Route). Without it neither overlay is drawn — "
        "everything else (lane network, agents, predicted "
        "trajectory, HUD) still appears.",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--ego_length",
        type=float,
        default=None,
        help="Override ego length (m). Defaults to whatever's in "
        "the npz (the data_converter writes its --ego_length "
        "default unless overridden at convert time).",
    )
    ap.add_argument("--ego_width", type=float, default=None)
    ap.add_argument("--ego_wheelbase", type=float, default=None)
    args = ap.parse_args()
    ego_dims = None
    given = [args.ego_length, args.ego_width, args.ego_wheelbase]
    if any(v is not None for v in given):
        if not all(v is not None for v in given):
            ap.error(
                "--ego_length / --ego_width / --ego_wheelbase must all be "
                "set together, or all be omitted (defaults from npz)."
            )
        ego_dims = (args.ego_length, args.ego_width, args.ego_wheelbase)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.npz_dir.glob("*.npz"))
    if args.stride > 1:
        files = files[:: args.stride]
    if args.limit > 0:
        files = files[: args.limit]
    n_steps = len(files)
    print(
        f"Rendering {n_steps} files from {args.npz_dir} -> {args.output_dir} "
        f"with {args.workers} workers"
    )

    # Building the route polylines + road borders pulls in lanelet2 (heavy);
    # do it once in the parent and ship to workers via a Pool initializer
    # rather than duplicating into every task tuple.
    route_polylines, road_border_polylines = _build_map_overlays(args.route_pkl)
    if route_polylines:
        print(f"  loaded {len(route_polylines)} route lanelet centerlines")
    if road_border_polylines:
        print(f"  loaded {len(road_border_polylines)} road-border polylines")
    if ego_dims:
        print(
            f"  overriding ego dims: length={ego_dims[0]} width={ego_dims[1]} "
            f"wheelbase={ego_dims[2]}"
        )

    tasks = [(i, f, args.output_dir / f"step_{i:04d}.png") for i, f in enumerate(files)]

    with mp.Pool(
        args.workers,
        initializer=_pool_init,
        initargs=(route_polylines, road_border_polylines, ego_dims, n_steps),
    ) as pool:
        for j, err in enumerate(pool.imap_unordered(_render_one, tasks, chunksize=4)):
            if err:
                print(err)
            if (j + 1) % 100 == 0:
                print(f"  {j + 1}/{n_steps}")
    print("Done.")


if __name__ == "__main__":
    main()
