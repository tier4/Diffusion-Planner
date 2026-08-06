"""Reusable closed-loop rollout + render + metric aggregation.

Shared by the standalone CLI (``diffusion_planner/valid_predictor_closed_loop.py``) and the
per-epoch training validation (``diffusion_planner/diffusion_planner/train.py``): both drive the
ego in CLOSED LOOP through ``reproducer_rollout.render_segment`` over the route NPZ frames under
``npz_root``, write a per-step PNG, build one MP4 per segment, and aggregate the per-segment
metrics into a single summary.

``run_closed_loop_eval`` takes an already-loaded ``(model, model_args)`` (so training can pass its
live model + ``TrainConfig`` straight in, no checkpoint reload) and returns the summary dict plus
the per-segment MP4 paths (for wandb upload).
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from scenario_generation.metrics.tdigest import TDIGEST_KEY, is_tdigest_key, merged_percentile
from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import render_segment
from scenario_generation.route_timeline import RouteTimeline, group_routes

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Clearance t-digests live in a sidecar ``tdigests.jsonl`` / ``tdigests_{rank}.jsonl``
# (not in the human-readable ``segments*.jsonl``) so multi-GPU merge can still pool p5.


def route_label(npz_path: Path, key: str) -> str:
    """Human-readable route label ``<location>_<date>_<key>`` for video/PNG names.

    Dataset routes are laid out ``.../<location>/<split>/<date>/<time>/routes/<time>_<idx>_<frame>``
    -- the bag-prefix ``key`` (``<time>_<idx>``) alone drops the depot/group and date, which makes the
    per-segment MP4 names ambiguous. This prepends ``<location>`` (the dir two levels above the
    ``YYYY-MM-DD`` date component) and ``<date>``. Falls back to bare ``key`` for any path that does
    not match that layout (e.g. a flat single-dir npz tree).
    """
    parts = npz_path.parts
    date_idx = next((i for i, p in enumerate(parts) if _DATE_RE.match(p)), None)
    if date_idx is not None and date_idx >= 2:
        return f"{parts[date_idx - 2]}_{parts[date_idx]}_{key}"
    return key


def enumerate_routes(npz_root: Path) -> dict[str, list[Path]]:
    """Group all .npz under ``npz_root`` into routes (bag-prefix groups)."""
    paths = sorted(Path(npz_root).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz under {npz_root}")
    return group_routes(paths)


def resolve_npz_roots(npz_root) -> list[Path]:
    """Resolve a closed-loop npz input into the list of root directories to enumerate.

    The input is a single directory tree of NPZ frames (globbed recursively), a ``.json`` file
    holding a list of such directory paths (one route dir per entry) -- the same "path list"
    form as ``--train_set_list`` / ``--valid_set_list`` -- or an already-resolved list of paths
    (e.g. from ``site_discovery.discover_sites_from_json``, which does its own per-group
    grouping). A directory is returned as a one-element list; a JSON list or a pre-resolved
    list is returned verbatim (each entry a ``Path``).
    """
    if isinstance(npz_root, (list, tuple)):
        return [Path(p) for p in npz_root]
    npz_root = Path(npz_root)
    if npz_root.suffix == ".json":
        entries = json.loads(npz_root.read_text())
        if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
            raise ValueError(f"{npz_root} must be a JSON list of directory paths")
        if not entries:
            raise ValueError(f"{npz_root} is an empty path list")
        return [Path(e) for e in entries]
    return [npz_root]


def enumerate_multi_root_routes(npz_root) -> tuple[dict[str, list[Path]], dict[str, Path]]:
    """Merge routes from every root in ``resolve_npz_roots(npz_root)`` into one route dict.

    Disambiguates any bag-prefix key that collides across roots (via ``route_label``, then a
    numeric suffix as a last resort) and remembers the source root of each route, so a
    downstream pose-sidecar fallback stays scoped to the correct tree. Returns
    ``(routes, route_sidecar_dir)``, both keyed by the final (disambiguated) route key.
    """
    roots = resolve_npz_roots(npz_root)
    routes: dict[str, list[Path]] = {}
    route_sidecar_dir: dict[str, Path] = {}
    for root in roots:
        for key, paths in enumerate_routes(root).items():
    # Key each route by its <location>_<date>_<time>_<idx> label so the per-segment PNG
    # dirs and MP4s carry the location + date, not just the ambiguous time-of-day bag prefix.
            label = route_label(paths[0], key)
            uniq, n = label, 1
            while uniq in routes:
                uniq, n = f"{label}#{n}", n + 1
            routes[uniq] = paths
            route_sidecar_dir[uniq] = root
    return routes, route_sidecar_dir


def _require_block(row: dict, category: str) -> dict:
    """Return nested metric block; fail fast if missing (no silent zeros)."""
    block = row.get(category)
    if not isinstance(block, dict):
        raise KeyError(
            f"segment metrics missing nested category {category!r} (got keys={sorted(row.keys())})"
        )
    return block


def _event_family_block(
    rows: list[dict],
    category: str,
    *,
    dual: bool,
    total_steps: int,
    n_seg: int,
    thresh_key: str | None = None,
    thresh_value: float | None = None,
) -> dict:
    """Roll up one nested metric category across segment rows."""
    out: dict = {}
    if thresh_key is not None and thresh_value is not None:
        out[thresh_key] = thresh_value

    if dual:
        prefixes = ("collision", "miss")
        for p in prefixes:
            steps = sum(int(_require_block(r, category)[f"{p}_steps"]) for r in rows)
            count = sum(int(_require_block(r, category)[f"{p}_count"]) for r in rows)
            segs = sum(1 for r in rows if int(_require_block(r, category)[f"{p}_steps"]) > 0)
            out[f"{p}_steps"] = steps
            out[f"{p}_count"] = count
            out[f"{p}_segments"] = segs
            out[f"{p}_step_rate"] = steps / total_steps if total_steps else 0.0
            out[f"{p}_segment_rate"] = segs / n_seg if n_seg else 0.0
    else:
        steps = sum(int(_require_block(r, category)["steps"]) for r in rows)
        count = sum(int(_require_block(r, category)["count"]) for r in rows)
        segs = sum(1 for r in rows if int(_require_block(r, category)["steps"]) > 0)
        out["steps"] = steps
        out["count"] = count
        out["segments"] = segs
        out["step_rate"] = steps / total_steps if total_steps else 0.0
        out["segment_rate"] = segs / n_seg if n_seg else 0.0

    return out


def _pool_clearance(rows: list[dict], category: str) -> dict[str, float]:
    """Global clearance min/mean/p5 from per-segment stats (+ optional t-digests)."""
    mins = []
    means = []
    weights = []
    p5s = []
    digests = []
    for r in rows:
        block = _require_block(r, category)
        mn = block["clearance_min_m"]
        mean = block["clearance_mean_m"]
        p5 = block["clearance_p5_m"]
        digest = block.get(TDIGEST_KEY)
        # Weight by finite clearance samples only (not n_steps_run, which includes inf steps).
        n = int(block.get("clearance_finite_steps", 0))
        if np.isfinite(mn):
            mins.append(float(mn))
        if np.isfinite(mean) and n > 0:
            means.append(float(mean))
            weights.append(n)
        if np.isfinite(p5):
            p5s.append(float(p5))
        if digest is not None:
            digests.append(digest)
    return {
        "clearance_min_m": float(min(mins)) if mins else float("inf"),
        "clearance_mean_m": float(np.average(means, weights=weights)) if means else float("inf"),
        # Prefer merged t-digest p5; fall back to min of segment p5s (conservative).
        "clearance_p5_m": (
            merged_percentile(digests, 5) if digests else (float(min(p5s)) if p5s else float("inf"))
        ),
    }


def metrics_for_json(metrics: dict) -> dict:
    """Numpy → JSON-friendly types; strip ``_tdigest`` (written to a sidecar instead)."""

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if not is_tdigest_key(k)}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    return _clean(metrics)


def segment_row_for_json(metrics: dict, **extra) -> dict:
    """Merge cleaned metrics with caller extras (no ``_tdigest``) for segments.jsonl.

    Shared by closed-loop eval and R2LPL mining writers so human-readable rows stay
    consistent. Digests still leave via ``tdigest_sidecar_row`` when the caller wants them.
    Caller ``extra`` keys override cleaned metrics on collision (e.g. ``route=...``).
    """
    return {**metrics_for_json(metrics), **extra}


def tdigest_sidecar_row(metrics: dict) -> dict | None:
    """Extract ``route`` + per-category ``_tdigest`` blobs for the digests sidecar file.

    Returns ``None`` when the row has no digests (nothing to persist for shard merge).
    """
    out: dict = {}
    if "route" in metrics:
        out["route"] = metrics["route"]
    for key, val in metrics.items():
        if not isinstance(val, dict):
            continue
        digest = val.get(TDIGEST_KEY)
        if digest is not None:
            out[key] = digest
    # Only ``route`` (and no digests) is not worth writing.
    return out if len(out) > (1 if "route" in out else 0) else None


def attach_tdigest_sidecars(rows: list[dict], sidecar_rows: list[dict]) -> None:
    """Mutate ``rows`` in place: attach sidecar digests under each category's ``_tdigest``."""
    by_route = {r["route"]: r for r in sidecar_rows if "route" in r}
    for row in rows:
        side = by_route.get(row.get("route"))
        if side is None:
            continue
        for cat, digest in side.items():
            if cat == "route":
                continue
            block = row.get(cat)
            if isinstance(block, dict):
                block[TDIGEST_KEY] = digest


def load_segment_rows_with_tdigests(
    out_dir: Path, *, shard_glob: str = "segments_*.jsonl"
) -> list[dict]:
    """Load human-readable segment rows and reattach digests from ``tdigests_*.jsonl`` sidecars."""
    rows: list[dict] = []
    for f in sorted(Path(out_dir).glob(shard_glob)):
        # Skip any accidental non-segment match (e.g. if naming ever collides).
        if "tdigest" in f.name:
            continue
        rows += [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]

    sidecar_rows: list[dict] = []
    for f in sorted(Path(out_dir).glob("tdigests_*.jsonl")):
        sidecar_rows += [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
    # Sequential runs use tdigests.jsonl (no rank suffix).
    plain = Path(out_dir) / "tdigests.jsonl"
    if plain.is_file():
        sidecar_rows += [json.loads(ln) for ln in plain.read_text().splitlines() if ln.strip()]

    attach_tdigest_sidecars(rows, sidecar_rows)
    return rows


def format_summary_lines(summary: dict) -> list[str]:
    """Human-readable closed-loop summary lines (shared by CLI / train print)."""
    n_seg = int(summary["n_segments"])
    obj = summary["object"]
    rb = summary["road_border"]
    red = summary["red_light_violation"]
    brake = summary["strong_brake"]
    repro = summary["reproducer"]
    lines = [
        f"object collision: {obj['collision_segments']}/{n_seg} segments "
        f"(rate {obj['collision_segment_rate']:.4f}), "
        f"{obj['collision_steps']} steps (rate {obj['collision_step_rate']:.6f}), "
        f"{obj['collision_count']} events",
        f"object miss (<= {obj['miss_thresh_m']} m): "
        f"{obj['miss_segments']}/{n_seg} segments "
        f"(rate {obj['miss_segment_rate']:.4f}), {obj['miss_steps']} steps, "
        f"{obj['miss_count']} events",
        f"road_border collision: {rb['collision_segments']}/{n_seg} segments, "
        f"{rb['collision_steps']} steps, {rb['collision_count']} events",
        f"road_border miss (<= {rb['miss_thresh_m']} m): "
        f"{rb['miss_segments']}/{n_seg} segments, {rb['miss_steps']} steps, "
        f"{rb['miss_count']} events",
        f"red_light_violation: {red['segments']}/{n_seg} segments "
        f"(rate {red['segment_rate']:.4f}), {red['steps']} steps, {red['count']} events",
        f"strong-brake (<= {brake['thresh_mps2']} m/s^2): "
        f"{brake['segments']}/{n_seg} segments "
        f"(rate {brake['segment_rate']:.4f}), {brake['steps']} steps, "
        f"{brake['count']} events, strongest_mps2={brake['strongest_mps2']:.3f}",
        f"object clearance min/mean/p5="
        f"{obj['clearance_min_m']:.3f}/{obj['clearance_mean_m']:.3f}/{obj['clearance_p5_m']:.3f} m  "
        f"road_border clearance min/mean/p5="
        f"{rb['clearance_min_m']:.3f}/{rb['clearance_mean_m']:.3f}/{rb['clearance_p5_m']:.3f} m",
        f"reproducer snap_count={repro['snap_count']} expand_count={repro['expand_count']} "
        f"repeat_step_rate={repro['repeat_step_rate']:.4f}  "
        f"terminated={summary['terminated_counts']}",
    ]
    return lines


def aggregate(
    rows: list[dict], near_miss_thresh: float, *, strong_brake_mps2: float = -2.5
) -> dict:
    """Aggregate per-segment nested metric rows into a closed-loop summary."""
    n_seg = len(rows)
    total_steps = sum(int(r["n_steps_run"]) for r in rows)

    # Graded (non-saturating) headline metrics: these improve smoothly as the model trains,
    # unlike the binary *_segment_rate / worst-moment *_min_clearance keys nested below.
    completions = [r["route_completion"] for r in rows if "route_completion" in r]
    # gt-deviation pooled across all steps (weight each segment's mean by its step count, which
    # equals that segment's gt-deviation sample count), so long routes aren't under-weighted.
    dev_num = sum(
        r["mean_gt_deviation_m"] * r["n_steps_run"]
        for r in rows
        if np.isfinite(r.get("mean_gt_deviation_m", float("inf")))
    )
    dev_den = sum(
        r["n_steps_run"] for r in rows if np.isfinite(r.get("mean_gt_deviation_m", float("inf")))
    )

    term_counts: dict[str, int] = {}
    for r in rows:
        term = r["terminated"]
        term_counts[term] = term_counts.get(term, 0) + 1

    obj = _event_family_block(
        rows,
        "object",
        dual=True,
        total_steps=total_steps,
        n_seg=n_seg,
        thresh_key="miss_thresh_m",
        thresh_value=float(near_miss_thresh),
    )
    obj.update(_pool_clearance(rows, "object"))

    rb = _event_family_block(
        rows,
        "road_border",
        dual=True,
        total_steps=total_steps,
        n_seg=n_seg,
        thresh_key="miss_thresh_m",
        thresh_value=float(near_miss_thresh),
    )
    rb.update(_pool_clearance(rows, "road_border"))

    red = _event_family_block(
        rows, "red_light_violation", dual=False, total_steps=total_steps, n_seg=n_seg
    )
    brake = _event_family_block(
        rows,
        "strong_brake",
        dual=False,
        total_steps=total_steps,
        n_seg=n_seg,
        thresh_key="thresh_mps2",
        thresh_value=float(strong_brake_mps2),
    )
    # Strongest consecutive-pair accel across segments (mask-filtered; +inf if none).
    strongest = [float(_require_block(r, "strong_brake")["strongest_mps2"]) for r in rows]
    brake["strongest_mps2"] = min(strongest) if strongest else float("inf")

    expand = sum(int(_require_block(r, "reproducer")["expand_count"]) for r in rows)
    snap = sum(int(_require_block(r, "reproducer")["snap_count"]) for r in rows)
    normal = sum(int(_require_block(r, "reproducer")["normal_steps"]) for r in rows)
    repeat = sum(int(_require_block(r, "reproducer")["repeat_steps"]) for r in rows)

    n_seg_diverged = term_counts.get("diverged", 0)
    return {
        "n_segments": n_seg,
        "total_steps": total_steps,
        "mean_route_completion": float(np.mean(completions)) if completions else 0.0,
        "mean_gt_deviation_m": float(dev_num / dev_den) if dev_den else float("inf"),
        "n_segments_diverged": n_seg_diverged,
        "diverged_segment_rate": n_seg_diverged / n_seg if n_seg else 0.0,
        "object": obj,
        "road_border": rb,
        "red_light_violation": red,
        "strong_brake": brake,
        "terminated_counts": term_counts,
        "reproducer": {
            "expand_count": expand,
            "snap_count": snap,
            "normal_steps": normal,
            "repeat_steps": repeat,
            "repeat_step_rate": repeat / total_steps if total_steps else 0.0,
        },
    }


def build_mp4(png_dir: Path, mp4_path: Path, fps: float) -> None:
    """Encode the PNG sequence in ``png_dir`` to an MP4.

    PNGs are named by step ``k`` and may be sparse (``draw_every`` skips frames), so glob the
    directory (gap-tolerant, name-sorted) instead of a contiguous ``%05d`` counter. ``fps`` is the
    raw frame rate, so a sparse sequence plays faster than real time (shorter video).
    """
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
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(mp4_path),
        ],
        check=True,
    )


def run_closed_loop_eval(
    model,
    model_args,
    npz_root,
    out_dir,
    *,
    device: str,
    near_miss_thresh: float,
    search_radius: float,
    warmup_steps: int,
    unstick_after: int,
    unstick_advance_m: float,
    fps: float,
    replan_interval: int,
    draw_every: int,
    neighbor_history_mode: str,
    unstick_radius_mult: float = 10.0,
    unstick_teleport_after: int = 300,
    tracker_mode: str = "mpc",
    strong_brake_mps2: float = -2.5,
    yaw_gate: bool = True,
    verbose: bool = True,
    shard: tuple[int, int] | None = None,
    abort_deviation_m: float = 0.0,
    abort_after: int = 30,
    abort_max_snaps: int = 0,
    drop_objects: bool = False,
) -> dict:
    """Render closed-loop rollouts over every route under ``npz_root`` and aggregate metrics.

    ``shard=(rank, world_size)`` restricts this call to the ``rank``-th slice of the sorted route
    list (``route_keys[rank::world_size]``) and writes its rows to ``segments_{rank}.jsonl`` instead
    of the merged ``segments.jsonl``/``summary.json`` -- the route-level multi-GPU parallel driver in
    ``valid_predictor_closed_loop.py`` spawns one such call per worker (route keys are globally
    unique, so all shards share ``out_dir`` for the per-route PNG dirs and MP4s without collision).
    The returned summary is then per-shard; the parent merges the ``segments_*.jsonl`` rows.

    ``model`` must be an eval-mode Diffusion-Planner (callable ``model(data) -> (_, outputs)`` with
    ``outputs["prediction"]``); ``model_args`` provides ``observation_normalizer`` /
    ``predicted_neighbor_num`` / ``future_len`` (a ``Config`` or ``TrainConfig``). Each route is
    rolled out whole (no sub-segmenting) into one PNG dir + one MP4 (``<route>.mp4``).
    ``segments.jsonl`` (one row per route, human-readable), ``tdigests.jsonl`` (clearance
    sketches for shard merge), and ``summary.json`` are written into ``out_dir``.

    Turn indicators are CLOSED-LOOP: the model's own predicted turn indicator is fed back into
    the input history each step, held across cached-plan steps when ``replan_interval`` > 1
    (see ``render_segment``).

    ``drop_objects``: empty-world ablation, forwarded to ``render_segment`` (see its docstring).

    Returns the summary dict with extra keys ``video_mp4s`` (list[Path] of every per-route MP4),
    ``segments`` (list[row]), and ``elapsed_sec``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # npz_root is either one directory tree, a JSON path list of route dirs, or an
    # already-resolved list of roots -- enumerate_multi_root_routes merges them all,
    # disambiguating any bag-prefix key that collides across roots and remembering the source
    # root of each route so its pose-sidecar fallback stays scoped to that tree.
    routes, route_sidecar_dir = enumerate_multi_root_routes(npz_root)
    route_keys = sorted(routes)
    if shard is not None:
        rank, world_size = shard
        route_keys = route_keys[rank::world_size]

    timers = Timers()
    rows: list[dict] = []
    video_mp4s: list[Path] = []
    t0 = time.perf_counter()

    segments_name = "segments.jsonl" if shard is None else f"segments_{shard[0]}.jsonl"
    digests_name = "tdigests.jsonl" if shard is None else f"tdigests_{shard[0]}.jsonl"
    fout = open(out_dir / segments_name, "w")
    fdigest = open(out_dir / digests_name, "w")
    try:
        for ri, key in enumerate(route_keys):
            tl = RouteTimeline(routes[key], sidecar_dir=route_sidecar_dir[key], timers=timers)
            # One route = one whole-route rollout = one <key>.mp4 (no sub-segmenting).
            png_dir = out_dir / key
            metrics = render_segment(
                model,
                model_args,
                tl,
                0,
                len(tl),
                png_dir,
                device=device,
                near_miss_thresh=near_miss_thresh,
                search_radius=search_radius,
                warmup_steps=warmup_steps,
                unstick_after=unstick_after,
                unstick_advance_m=unstick_advance_m,
                unstick_radius_mult=unstick_radius_mult,
                unstick_teleport_after=unstick_teleport_after,
                replan_interval=replan_interval,
                draw_every=draw_every,
                neighbor_history_mode=neighbor_history_mode,
                tracker_mode=tracker_mode,
                strong_brake_mps2=strong_brake_mps2,
                yaw_gate=yaw_gate,
                abort_deviation_m=abort_deviation_m,
                abort_after=abort_after,
                abort_max_snaps=abort_max_snaps,
                drop_objects=drop_objects,
            )
            row = {"route": key, **metrics}
            # Human-readable segments.jsonl (no _tdigest blobs). Digests go to a sidecar so
            # multi-GPU parents can still merge approximate global clearance p5.
            fout.write(json.dumps(segment_row_for_json(metrics, route=key), default=float) + "\n")
            fout.flush()
            side = tdigest_sidecar_row(row)
            if side is not None:
                fdigest.write(json.dumps(side, default=float) + "\n")
                fdigest.flush()
            rows.append(row)

            # A route that terminates at step 0 (e.g. ego starts within goal_reach_m) draws no PNG;
            # skip the empty ffmpeg call (its glob would error on an empty dir).
            if not any(png_dir.glob("*.png")):
                if verbose:
                    print(f"[{ri + 1}/{len(route_keys)}] {key} -> 0 frames, no video")
                continue
            seg_mp4 = out_dir / f"{key}.mp4"
            # Raw fps: with only every draw_every-th frame drawn, the video plays
            # draw_every x faster than real time. For real time use fps = 10 / draw_every.
            build_mp4(png_dir, seg_mp4, fps)
            video_mp4s.append(seg_mp4)
            if verbose:
                obj = metrics["object"]
                brake = metrics["strong_brake"]
                print(
                    f"[{ri + 1}/{len(route_keys)}] {key} -> {seg_mp4.name}  "
                    f"obj_coll={obj['collision_count']} "
                    f"obj_miss={obj['miss_count']} "
                    f"brake={brake['count']} "
                    f"min_clr={obj['clearance_min_m']:.3f}"
                )
    finally:
        fout.close()
        fdigest.close()

    # In-memory ``rows`` still carry digests for this process's aggregate.
    summary = aggregate(rows, near_miss_thresh, strong_brake_mps2=strong_brake_mps2)
    summary["npz_root"] = str(npz_root)
    summary["n_routes"] = len(route_keys)
    summary["elapsed_sec"] = time.perf_counter() - t0
    summary["video_mp4s"] = video_mp4s

    # A sharded worker leaves the merged summary.json to the parent driver (which aggregates every
    # shard's segments_*.jsonl); it only owns its own segments_{rank}.jsonl, written above.
    if shard is None:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(
                {k: v for k, v in summary.items() if k != "video_mp4s"},
                f,
                indent=2,
                default=float,
            )
    return summary
