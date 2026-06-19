#!/usr/bin/env python3
"""Perfect-tracker WebM of a single scene NPZ with TIME-VARYING neighbor boxes.

The ghost-sim renderer (``ghost_sim_common.run_ghost_sim``) draws one STATIC
``neighbor_boxes`` set for every frame (it calls ``extract_stopped_neighbors``),
so it cannot show MOVING neighbors. This driver reuses the same canonical
per-step renderer (``render_ghost_step``) and the same polyline extraction, but
feeds it the neighbor poses from ``neighbor_agents_future[:, k]`` at each step k
— so a moving NPC's trajectory is drawn moving. The ego perfect-tracks its own
``ego_agent_future`` (the realized/GT path stored in the NPZ); no model needed.

Use it to eyeball reproducer collision scenes and their augmentations (ego-onset
``disturb_and_replay`` / obstacle ``perturb_neighbors_npz --include_moving``) and
confirm the moving-neighbor trajectory is preserved / relocated and headings stay
correct.

Usage:
    python -m rlvr.autoresearch.tools.viz_moving_scene \
        --npz <scene.npz> --out_dir <dir> --ego_shape WB,L,W [--label TEXT] [--fps 10]
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from rlvr.autoresearch.tools.ghost_sim_common import (
    GhostSimConfig,
    extract_scene_polylines,
    render_ghost_step,
)


def _unbatch(arr: np.ndarray) -> np.ndarray:
    return arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 and arr.ndim in (3, 4) else arr


def _pose_from_4col(row: np.ndarray) -> np.ndarray:
    """[x, y, cos, sin] -> [x, y, heading]."""
    return np.array([float(row[0]), float(row[1]), math.atan2(float(row[3]), float(row[2]))])


def _neighbor_boxes_at(nb_fut: np.ndarray, shapes: np.ndarray, k: int):
    """Time-varying boxes (x, y, heading, length, width) at future step k.

    nb_fut: (N, T, 4) [x, y, cos, sin]; shapes: (N, 2) [width, length].
    """
    boxes = []
    for i in range(nb_fut.shape[0]):
        x, y, cos, sin = nb_fut[i, k]
        if abs(float(x)) + abs(float(y)) < 1e-6:
            continue  # padding / invalid at this step
        h = math.atan2(float(sin), float(cos))
        w, length = float(shapes[i, 0]), float(shapes[i, 1])
        if w < 0.1 or length < 0.1:
            continue
        boxes.append((float(x), float(y), h, length, w))
    return boxes


def render_scene(
    npz_path: str,
    out_dir: Path,
    ego_shape: tuple[float, float, float],
    label: str = "scene",
    fps: int = 10,
    keep_pngs: bool = False,
) -> Path:
    """Render one scene NPZ as a perfect-tracker WebM with time-varying neighbors.

    The ego perfect-tracks its ego_agent_future; neighbors follow their
    neighbor_agents_future (drawn moving via per-step boxes). Returns the webm path.
    """
    wb, length, width = ego_shape
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = dict(np.load(npz_path, allow_pickle=True))
    ego_fut = _unbatch(np.asarray(data["ego_agent_future"], dtype=np.float32))
    if ego_fut.shape[-1] != 4:
        raise ValueError(f"ego_agent_future must be 4-col [x,y,cos,sin], got {ego_fut.shape}")
    nb_fut = _unbatch(np.asarray(data["neighbor_agents_future"], dtype=np.float32))
    nb_past = _unbatch(np.asarray(data["neighbor_agents_past"], dtype=np.float32))
    if nb_fut.shape[-1] != 4:
        raise ValueError(f"neighbor_agents_future must be 4-col, got {nb_fut.shape}")
    shapes = nb_past[:, -1, [6, 7]]  # (N, 2) width, length

    # Polylines via the canonical extractor (needs torch tensors).
    scene_t = {
        k: torch.from_numpy(np.asarray(v))
        for k, v in data.items()
        if k in ("route_lanes", "lanes", "line_strings")
    }
    centerlines, lefts, rights, borders, routes, cl_segs = extract_scene_polylines(scene_t)

    cfg = GhostSimConfig(
        model_a_label=label,
        model_b_label=label,
        ego_length=length,
        ego_width=width,
        ego_wheelbase=wb,
        steps=ego_fut.shape[0],
    )

    # Ego perfect-tracks ego_agent_future; only the valid (non-zero) steps.
    ego_valid = (np.abs(ego_fut[:, 0]) + np.abs(ego_fut[:, 1])) > 1e-6
    T = int(np.where(ego_valid)[0].max()) + 1 if ego_valid.any() else 0
    if T < 2:
        raise ValueError(
            f"{npz_path}: ego_agent_future has <2 valid steps — nothing to perfect-track. "
            "This is the collision frame (collision+00000, truncated at contact). Render the "
            "whole saved batch with --batch_dir <dir> (current-state renderer) instead."
        )

    for k in range(T):
        pose = _pose_from_4col(ego_fut[k])
        nxt = _pose_from_4col(ego_fut[min(k + 1, T - 1)])
        speed = float(np.hypot(nxt[0] - pose[0], nxt[1] - pose[1])) / 0.1
        plan = np.column_stack([ego_fut[k:T, 0], ego_fut[k:T, 1]])
        render_ghost_step(
            out_dir / f"step_{k:04d}.png",
            step=k,
            n_steps=T,
            a_pose=pose,
            a_speed=speed,
            a_plan=plan,
            b_pose=pose,
            b_speed=speed,
            b_plan=None,
            centerlines=centerlines,
            lefts=lefts,
            rights=rights,
            border_polylines=borders,
            route_polylines=routes,
            centerline_segments=cl_segs,
            cfg=cfg,
            neighbor_boxes=_neighbor_boxes_at(nb_fut, shapes, k),
            extra_title=f"  {label}  t={k * 0.1:.1f}s",
        )

    webm = out_dir / f"{label}.webm"
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(out_dir / "step_%04d.png"),
        "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-row-mt", "1",
        "-pix_fmt", "yuv420p", str(webm),
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr[-1500:]}")
    print(f"[viz_moving_scene] {T} frames -> {webm}")
    if not keep_pngs:
        for p in out_dir.glob("step_*.png"):
            p.unlink()
    return webm


def render_batch_dir(
    batch_dir: Path,
    out_dir: Path,
    ego_shape: tuple[float, float, float],
    label: str = "batch",
    fps: int = 10,
    keep_pngs: bool = False,
) -> Path:
    """Render a saved collision/near-miss batch dir as a WebM, one frame per snapshot.

    Each ``collision-NNNNN.npz`` is one re-centered snapshot, so the dir already IS the
    time sequence (ordered farthest->contact). Each frame is drawn with the canonical
    reproducer renderer ``reproducer_rollout._draw_step`` (-> ``replay.save_step_figure``),
    which draws the ego + its plan, neighbor boxes, route/borders, AND the closest-point
    ego<->nearest-neighbor distance LINE + value (via ``rlvr.reward._closest_points_between_rects``).
    No ego_future needed for the contact frame — its current neighbor box is drawn directly.
    """
    import glob

    from scenario_generation.reproducer_rollout import _draw_step

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Filenames are collision<offset>.npz, offset in -00080..+00000 (offset = step - t_c).
    # Sort by the NUMERIC offset so time runs farthest->contact (lexical sort would reverse it).
    files = sorted(
        glob.glob(str(Path(batch_dir) / "collision*.npz")),
        key=lambda f: int(Path(f).stem.replace("collision", "")),
    )
    if not files:
        raise ValueError(f"no collision*.npz under {batch_dir}")
    es = np.asarray(ego_shape, dtype=np.float32).reshape(-1)[:3]
    for k, fp in enumerate(files):
        data = dict(np.load(fp, allow_pickle=True))
        # _draw_step squeezes a leading batch axis, so add one to every field.
        batched = {kk: np.asarray(v)[None] for kk, v in data.items()}
        pred = np.asarray(data["ego_agent_future"], dtype=np.float32)  # ego plan overlay
        _draw_step(batched, pred, es, out_dir / f"step_{k:04d}.png", step=k, total=len(files))
    webm = out_dir / f"{label}.webm"
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(out_dir / "step_%04d.png"),
        "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-row-mt", "1",
        "-pix_fmt", "yuv420p", str(webm),
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr[-1500:]}")
    print(f"[viz_moving_scene] batch {len(files)} frames -> {webm}")
    if not keep_pngs:
        for p in out_dir.glob("step_*.png"):
            p.unlink()
    return webm


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--npz", help="single scene NPZ (perfect-track its ego_future)")
    ap.add_argument("--batch_dir", help="a saved collision batch dir (render snapshots in order)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ego_shape", required=True, help="WB,L,W")
    ap.add_argument("--label", default="scene")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--keep_pngs", action="store_true")
    args = ap.parse_args()
    ego_shape = tuple(float(x) for x in args.ego_shape.split(","))
    if bool(args.npz) == bool(args.batch_dir):
        ap.error("pass exactly one of --npz or --batch_dir")
    if args.batch_dir:
        render_batch_dir(Path(args.batch_dir), Path(args.out_dir), ego_shape, args.label,
                         args.fps, args.keep_pngs)  # fmt: skip
    else:
        render_scene(args.npz, Path(args.out_dir), ego_shape, args.label, args.fps, args.keep_pngs)


if __name__ == "__main__":
    main()
