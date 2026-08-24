"""Windowed closed-loop evaluation: GT warm-started windows, gates, and a composite score.

The log replay is non-reactive (recorded neighbors and signals), so a whole-route rollout
drifts away from the recorded world and its absolute metrics stop meaning anything.  This
module instead cuts every route into short windows, rolls each window out as an
*independent* ``render_segment`` seeded from the recorded ego state at the window start
(no state carried across windows), and scores each window as::

    window_score = coverage * nc * offroad * progress * graded
    graded       = w_progress * progress_ratio + w_lon * score_lon + w_lat * score_lat

Gates are 0/1:

``coverage``  the rollout was cut as ``diverged`` (``abort_deviation_m`` for ``abort_after``
              consecutive ticks) -- the ego left the map / neighbor guarantee radius.
``nc``        at least one *at-fault* collision (``metrics.at_fault``; ghost rear-end contacts
              from the replay are not at fault).
``offroad``   at least one road-border overlap tick (``road_border.collision_steps``).
``progress``  ``progress_ratio`` below ``progress_gate_min``, unless the recorded ego itself
              barely moved (``min_recorded_progress_m``: a red-light wait asks for no progress).

``progress_ratio`` and the lon/lat divergence come from projecting the live ego onto the
recorded ego polyline of the window (arc-length parameterised), the same decomposition used by
the validity check: ``ade_lon`` = mean |arc(live at clock tick k) - arc(recorded at k)|,
``ade_lat`` = mean |signed lateral distance|.  Windows run on the clock timeline so tick ``k``
of the window is recorded frame ``lo + k``.

Outputs: ``windows.jsonl`` (one row per window: the ``render_segment`` metrics without the
t-digest sketches + the ``v2`` block) and ``windows_summary.json`` (macro = mean over routes of
the per-route mean window score, micro = mean over all windows, gate pass rates, means).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from scenario_generation.closed_loop_eval import (
    build_mp4,
    enumerate_multi_root_routes,
    segment_row_for_json,
)
from scenario_generation.reproducer_rollout import DT, render_segment
from scenario_generation.route_timeline import RouteTimeline
from scenario_generation.tools._heatmap_common import project_points_to_polyline
from scenario_generation.window_metrics import (
    ScoreConfig,
    read_rollout,
    recorded_scene_tags,
    score_window_epdms,
)

WINDOW_SCHEMA = "closed_loop_windows/1.0"


@dataclass(frozen=True)
class WindowConfig:
    mode: str = "fixed"  # "fixed" | "anchor"
    window_len_s: float = 30.0
    window_stride_s: float | None = None  # None = non-overlapping
    min_tail_s: float = 5.0  # a shorter final remainder is merged into the previous window
    anchor_pre_s: float = 10.0
    anchor_post_s: float = 10.0
    anchor_unit: str = "frame"  # "frame" | "sec"
    anchors: dict[str, list[float]] = field(default_factory=dict)
    # Gates / score.
    coverage_abort_m: float = 100.0
    coverage_abort_after: int = 30
    progress_gate_min: float = 0.2
    min_recorded_progress_m: float = 5.0
    lon_tol_m: float = 30.0
    lat_tol_m: float = 3.0
    w_progress: float = 0.5
    w_lon: float = 0.25
    w_lat: float = 0.25
    polyline_margin_frames: int = 300
    max_windows_per_route: int | None = None  # smoke tests: evaluate only the first N windows
    window_indices: tuple[int, ...] | None = None  # evaluate only these window indices per route
    draw_every: int | None = None  # render a PNG every N ticks and encode <window>.mp4
    video_fps: float = 5.0
    # EPDMS-style scoring (scenario_generation.window_metrics).
    min_valid_s: float = 3.0
    truncate_on_ghost_contact: bool = True
    divergence_flag_min_m: float = 5.0
    divergence_flag_headway_s: float = 2.0
    clearance_weight: float = 0.0
    include_recorded_stop: bool = False

    def score_config(self) -> "ScoreConfig":
        return ScoreConfig(
            min_recorded_progress_m=self.min_recorded_progress_m,
            min_valid_s=self.min_valid_s,
            truncate_on_ghost_contact=self.truncate_on_ghost_contact,
            divergence_flag_min_m=self.divergence_flag_min_m,
            divergence_flag_headway_s=self.divergence_flag_headway_s,
            polyline_margin_frames=self.polyline_margin_frames,
            clearance_weight=self.clearance_weight,
        )

    def validate(self) -> "WindowConfig":
        if self.mode not in ("fixed", "anchor"):
            raise ValueError(f"mode must be 'fixed' or 'anchor', got {self.mode!r}")
        if self.window_len_s <= 0:
            raise ValueError("window_len_s must be > 0")
        if self.window_stride_s is not None and self.window_stride_s <= 0:
            raise ValueError("window_stride_s must be > 0 when given")
        if self.anchor_unit not in ("frame", "sec"):
            raise ValueError("anchor_unit must be 'frame' or 'sec'")
        if self.mode == "anchor" and not self.anchors:
            raise ValueError("anchor mode requires a non-empty anchors mapping")
        if self.coverage_abort_m <= 0:
            raise ValueError("coverage_abort_m must be > 0 (the coverage gate needs an abort)")
        weights = (self.w_progress, self.w_lon, self.w_lat)
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("graded weights must be >= 0 and not all zero")
        if self.lon_tol_m <= 0 or self.lat_tol_m <= 0:
            raise ValueError("lon_tol_m and lat_tol_m must be > 0")
        return self


# --------------------------------------------------------------------------- #
# window planning
# --------------------------------------------------------------------------- #
def plan_fixed_windows(
    n_frames: int, window_len: int, stride: int | None = None, min_tail: int = 50
) -> list[tuple[int, int]]:
    """Split ``[0, n_frames)`` into ``[lo, hi)`` windows of ``window_len`` frames.

    Non-overlapping by default; a final remainder shorter than ``min_tail`` is merged into
    the previous window (so no window is shorter than ``min_tail`` unless the route is).
    """
    if n_frames <= 0:
        return []
    window_len = max(int(window_len), 1)
    stride = window_len if stride is None else max(int(stride), 1)
    windows: list[tuple[int, int]] = []
    lo = 0
    while lo < n_frames:
        hi = min(lo + window_len, n_frames)
        windows.append((lo, hi))
        if hi >= n_frames:
            break
        lo += stride
    if len(windows) >= 2 and stride == window_len:
        lo_last, hi_last = windows[-1]
        if hi_last - lo_last < min_tail:
            lo_prev, _ = windows[-2]
            windows[-2:] = [(lo_prev, hi_last)]
    return windows


def plan_anchor_windows(
    n_frames: int, anchors: list[int], pre: int, post: int
) -> list[tuple[int, int]]:
    """``[anchor - pre, anchor + post)`` clipped to the route; anchors outside are dropped."""
    windows: list[tuple[int, int]] = []
    for a in anchors:
        a = int(a)
        if a < 0 or a >= n_frames:
            continue
        lo, hi = max(0, a - int(pre)), min(n_frames, a + int(post))
        if hi - lo >= 2:
            windows.append((lo, hi))
    return windows


def resolve_route_anchors(anchors: dict[str, list[float]], route_key: str) -> list[float]:
    """Anchors for ``route_key``: exact key first, else the unique key that is a substring."""
    if route_key in anchors:
        return list(anchors[route_key])
    hits = [k for k in anchors if k and k in route_key]
    if len(hits) > 1:
        raise ValueError(f"anchor keys {hits} all match route {route_key!r}; make them unique")
    return list(anchors[hits[0]]) if hits else []


def route_windows(cfg: WindowConfig, route_key: str, n_frames: int) -> list[tuple[int, int]]:
    if cfg.mode == "fixed":
        stride = None if cfg.window_stride_s is None else int(round(cfg.window_stride_s / DT))
        return plan_fixed_windows(
            n_frames,
            int(round(cfg.window_len_s / DT)),
            stride,
            int(round(cfg.min_tail_s / DT)),
        )
    anchors = resolve_route_anchors(cfg.anchors, route_key)
    if cfg.anchor_unit == "sec":
        anchors = [int(round(a / DT)) for a in anchors]
    return plan_anchor_windows(
        n_frames,
        [int(a) for a in anchors],
        int(round(cfg.anchor_pre_s / DT)),
        int(round(cfg.anchor_post_s / DT)),
    )


# --------------------------------------------------------------------------- #
# divergence / progress against the recorded ego
# --------------------------------------------------------------------------- #
def _arc_length(xy: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def window_divergence(
    poses: np.ndarray,
    lo: int,
    hi: int,
    live_xy: np.ndarray,
    *,
    margin_frames: int = 300,
) -> dict:
    """Project the live ego (one row per clock tick from ``lo``) onto the recorded polyline.

    ``poses`` is the route's recorded ego pose array (N, >=2); the polyline is the recorded
    path over ``[lo, hi)`` extended by ``margin_frames`` on both sides so a live ego that runs
    ahead of (or lags behind) the recorded one still projects onto real road.  Returns
    per-window means of the longitudinal (arc) and lateral (signed distance) divergence, the
    progress ratio, and the raw distances.
    """
    n = len(poses)
    a, b = max(0, lo - margin_frames), min(n, hi + margin_frames)
    pts = np.asarray(poses[a:b, :2], dtype=np.float64)
    if len(pts) < 2:
        raise ValueError(f"recorded polyline needs >= 2 points, got {len(pts)}")
    s = _arc_length(pts)
    rec_arc = s[lo - a : hi - a]
    live_xy = np.asarray(live_xy, dtype=np.float64).reshape(-1, 2)
    n_live = min(len(live_xy), hi - lo)
    if n_live == 0:
        return {
            "n_steps": 0,
            "ade_lon_m": float("inf"),
            "ade_lat_m": float("inf"),
            "realized_progress_m": 0.0,
            "recorded_progress_m": float(rec_arc[-1] - rec_arc[0]) if len(rec_arc) else 0.0,
            "progress_ratio": 0.0,
        }
    proj = project_points_to_polyline(live_xy[:n_live], pts, s)
    lon = proj[:, 0] - rec_arc[:n_live]
    lat = proj[:, 1]
    recorded = float(rec_arc[-1] - rec_arc[0])
    realized = float(np.max(proj[:, 0]) - rec_arc[0])
    ratio = float(np.clip(realized / recorded, 0.0, 1.0)) if recorded > 1e-6 else 1.0
    return {
        "n_steps": int(n_live),
        "ade_lon_m": float(np.mean(np.abs(lon))),
        "ade_lat_m": float(np.mean(np.abs(lat))),
        "lon_final_m": float(lon[-1]),
        "lat_abs_max_m": float(np.max(np.abs(lat))),
        "realized_progress_m": realized,
        "recorded_progress_m": recorded,
        "progress_ratio": ratio,
    }


def score_window(metrics: dict, divergence: dict, cfg: WindowConfig) -> dict:
    """Gates + graded composite for one window (see module docstring)."""
    coverage = 0 if metrics.get("terminated") == "diverged" else 1
    at_fault = metrics.get("at_fault")
    if at_fault is None:
        raise ValueError("window metrics lack the at_fault block; run with at_fault_scoring=True")
    nc = 0 if int(at_fault["count"]) > 0 else 1
    offroad = 0 if int(metrics["road_border"]["collision_steps"]) > 0 else 1
    low_recorded = divergence["recorded_progress_m"] < cfg.min_recorded_progress_m
    ratio = 1.0 if low_recorded else float(divergence["progress_ratio"])
    progress = 0 if ratio < cfg.progress_gate_min else 1
    score_lon = max(0.0, 1.0 - divergence["ade_lon_m"] / cfg.lon_tol_m)
    score_lat = max(0.0, 1.0 - divergence["ade_lat_m"] / cfg.lat_tol_m)
    wsum = cfg.w_progress + cfg.w_lon + cfg.w_lat
    graded = (cfg.w_progress * ratio + cfg.w_lon * score_lon + cfg.w_lat * score_lat) / wsum
    gate = coverage * nc * offroad * progress
    return {
        "gates": {"coverage": coverage, "nc": nc, "offroad": offroad, "progress": progress},
        "gate": int(gate),
        "progress_ratio": ratio,
        "low_recorded_progress": bool(low_recorded),
        "score_lon": float(score_lon),
        "score_lat": float(score_lat),
        "graded": float(graded),
        "score": float(gate * graded),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
# render_segment knobs a window run pins regardless of the caller's whole-route settings.
_WINDOW_RENDER_OVERRIDES = dict(
    draw_every=None,
    draw_pool=None,
    warmup_steps=0,
    unstick_after=0,
    max_stuck_steps=0,
    goal_mode="segment",
    goal_reach_m=0.0,
    window=None,
    timeline_progress_mode="clock",
    abort_max_snaps=0,
    at_fault_scoring=True,
    title_prefix=None,
    distance_label_offset_m=1.2,
    view_half_m=50.0,
    interpolate=True,
    color_by_uuid=True,
)


def summarize_windows(rows: list[dict], *, include_recorded_stop: bool = False) -> dict:
    """EPDMS-style aggregate: one score plus its breakdown.

    ``score_macro`` = mean over routes of the per-route mean window score, ``score_micro`` =
    mean over all VALID windows.  Invalid windows (valid span shorter than ``min_valid_s``)
    are counted, not scored.  Multiplicative zero counts and weighted-term means are the
    breakdown; the raw ``v2`` divergence block is aggregated alongside.
    """
    if not rows:
        return {"n_windows": 0, "n_routes": 0, "n_valid": 0}
    scored = [
        r for r in rows if not r["epdms"].get("invalid") and r["epdms"].get("score") is not None
    ]
    buckets: dict[str, list[dict]] = {}
    for r in scored:
        buckets.setdefault(r["epdms"].get("bucket", "main"), []).append(r)
    stop_rows = buckets.get("recorded_stop", [])
    valid = scored if include_recorded_stop else buckets.get("main", [])
    by_route: dict[str, list[float]] = {}
    for r in valid:
        by_route.setdefault(r["route"], []).append(float(r["epdms"]["score"]))
    terms = ("nc", "dac", "ddc", "tlc", "mp", "ep", "ttc", "sl", "comfort", "lk")
    term_means = {
        t: float(np.mean([r["epdms"]["terms"][t] for r in valid])) if valid else None for t in terms
    }
    zero = {
        t: int(sum(1 for r in valid if r["epdms"]["terms"][t] == 0.0))
        for t in ("nc", "dac", "ddc", "tlc", "mp", "ttc", "sl", "comfort", "lk")
    }
    cut_reasons: dict[str, int] = {}
    for r in rows:
        reason = r["epdms"]["valid_span"]["reason"]
        cut_reasons[reason] = cut_reasons.get(reason, 0) + 1
    finite = [r for r in valid if math.isfinite(r["epdms"]["detail"]["ade_lon_m"])]

    def _mean(key: str):
        vals = [r["epdms"]["detail"][key] for r in finite if r["epdms"]["detail"][key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "n_windows": int(len(rows)),
        "n_routes": int(len({r["route"] for r in rows})),
        "n_valid": int(len(valid)),
        "invalid_windows": int(len(rows) - len(scored)),
        "buckets": {
            k: {
                "n_windows": len(v),
                "score_micro": float(np.mean([r["epdms"]["score"] for r in v])),
            }
            for k, v in buckets.items()
        },
        "recorded_stop_windows": int(len(stop_rows)),
        "recorded_stop_included": bool(include_recorded_stop),
        "valid_span_reasons": cut_reasons,
        "score_macro": float(np.mean([np.mean(v) for v in by_route.values()]))
        if by_route
        else None,
        "score_micro": float(np.mean([r["epdms"]["score"] for r in valid])) if valid else None,
        "multiplicative_mean": float(np.mean([r["epdms"]["multiplicative"] for r in valid]))
        if valid
        else None,
        "weighted_mean": float(np.mean([r["epdms"]["weighted"] for r in valid])) if valid else None,
        "term_means": term_means,
        "zero_windows": zero,
        "progress_ratio_mean": _mean("progress_ratio"),
        "ade_lon_m_mean": _mean("ade_lon_m"),
        "ade_lat_m_mean": _mean("ade_lat_m"),
        "route_adherence_mean": _mean("route_adherence_frac"),
        "tl_measured_frac_mean": _mean("tl_measured_frac"),
        "divergence_flag_windows": int(
            sum(1 for r in valid if r["epdms"]["detail"]["divergence_flag_ticks"] > 0)
        ),
        "at_fault_windows": int(sum(1 for r in rows if r["at_fault"]["count"] > 0)),
        "at_fault_events": int(sum(r["at_fault"]["count"] for r in rows)),
        "rear_under_hard_brake_tracks": int(
            sum(r["at_fault"].get("rear_under_hard_brake_tracks", 0) for r in rows)
        ),
        "raw_collision_windows": int(sum(1 for r in rows if r["object"]["collision_count"] > 0)),
        "road_border_windows": int(sum(1 for r in rows if r["road_border"]["collision_steps"] > 0)),
        "diverged_windows": int(sum(1 for r in rows if r["terminated"] == "diverged")),
        "per_route": {
            k: {"n_windows": len(v), "score": float(np.mean(v))} for k, v in by_route.items()
        },
    }


def run_windowed_eval(
    model,
    model_args,
    npz_root,
    out_dir,
    *,
    cfg: WindowConfig,
    render_kwargs: dict,
    verbose: bool = True,
    route_filter=None,
) -> dict:
    """Roll out and score every window of every route under ``npz_root``.

    ``render_kwargs`` are the whole-route ``render_segment`` knobs (device, tracker, delay
    knobs, plant, ...); the window-specific ones in ``_WINDOW_RENDER_OVERRIDES`` and the
    coverage abort are pinned here.  Single process, no video.
    """
    cfg = cfg.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    routes, route_sidecar_dir = enumerate_multi_root_routes(npz_root)
    route_keys = sorted(routes)
    if route_filter is not None:
        route_keys = [k for k in route_keys if route_filter(k)]
    kwargs = dict(render_kwargs)
    kwargs.update(_WINDOW_RENDER_OVERRIDES)
    if cfg.draw_every is not None:
        kwargs["draw_every"] = int(cfg.draw_every)
    kwargs["abort_deviation_m"] = float(cfg.coverage_abort_m)
    kwargs["abort_after"] = int(cfg.coverage_abort_after)

    rows: list[dict] = []
    t0 = time.perf_counter()
    plan: dict[str, list[tuple[int, int]]] = {}
    with open(out_dir / "windows.jsonl", "w") as fout:
        for ri, key in enumerate(route_keys):
            tl = RouteTimeline(routes[key], sidecar_dir=route_sidecar_dir[key])
            windows = route_windows(cfg, key, len(tl))
            if cfg.max_windows_per_route is not None:
                windows = windows[: int(cfg.max_windows_per_route)]
            plan[key] = windows
            for wi, (lo, hi) in enumerate(windows):
                if cfg.window_indices is not None and wi not in cfg.window_indices:
                    continue
                wdir = out_dir / key / f"w{wi:03d}_{lo:05d}_{hi:05d}"
                metrics = render_segment(
                    model, model_args, tl, lo, hi, wdir, max_steps=hi - lo, **kwargs
                )
                if cfg.draw_every is not None and any(wdir.glob("*.png")):
                    build_mp4(wdir, out_dir / key / f"{wdir.name}.mp4", cfg.video_fps)
                rollout_rows = read_rollout(wdir / "rollout.jsonl")
                live_xy = np.array([r["ego"][:2] for r in rollout_rows], dtype=np.float64)
                div = window_divergence(
                    tl.poses, lo, hi, live_xy, margin_frames=cfg.polyline_margin_frames
                )
                v2 = score_window(metrics, div, cfg)
                v2["divergence"] = div
                epdms = score_window_epdms(
                    tl,
                    lo,
                    hi,
                    rollout_rows,
                    np.asarray(tl.npz(lo)["ego_shape"]).reshape(-1)[:3],
                    cfg.score_config(),
                    at_fault_block=metrics.get("at_fault"),
                    terminated=metrics["terminated"],
                )
                row = segment_row_for_json(metrics, route=key)
                row.update(
                    {
                        "window_index": wi,
                        "window": [int(lo), int(hi)],
                        "start_frame_id": int(tl.frame_indices[lo]),
                        "end_frame_id": int(tl.frame_indices[hi - 1]),
                        "v2": v2,
                        "epdms": epdms,
                        "scene": recorded_scene_tags(tl, lo, hi),
                    }
                )
                fout.write(json.dumps(row, default=float) + "\n")
                fout.flush()
                rows.append(row)
                if verbose:
                    sc = epdms.get("score")
                    terms = " ".join(f"{k}={v:.2f}" for k, v in epdms.get("terms", {}).items())
                    print(
                        f"[{ri + 1}/{len(route_keys)}] {key} w{wi:03d} [{lo},{hi}) "
                        f"score={'-' if sc is None else f'{sc:.3f}'} "
                        f"valid={epdms['valid_ticks']}/{hi - lo}({epdms['valid_span']['reason']})"
                        f"{' INVALID' if epdms.get('invalid') else ''} | {terms} | "
                        f"lon={div['ade_lon_m']:.2f} lat={div['ade_lat_m']:.2f} "
                        f"term={metrics['terminated']}",
                        flush=True,
                    )
    summary = summarize_windows(rows, include_recorded_stop=cfg.include_recorded_stop)
    summary.update(
        {
            "schema": WINDOW_SCHEMA,
            "config": asdict(cfg),
            "npz_root": str(npz_root),
            "windows_per_route": {k: [[int(a), int(b)] for a, b in v] for k, v in plan.items()},
            "elapsed_sec": time.perf_counter() - t0,
        }
    )
    with open(out_dir / "windows_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def rescore_windows(rows: list[dict], *, weights: dict | None = None) -> list[dict]:
    """Recompute every window's score from its stored terms with new weighted-term weights.

    ``weights`` maps term name -> weight (e.g. ``{"ep": 5, "ttc": 5, "sl": 4, "comfort": 2,
    "lk": 2, "clearance": 2}``); omitted terms keep the weight stored in the row.  Multiplicative
    terms are unchanged.  Returns new rows (the inputs are not modified).
    """
    out = []
    for r in rows:
        e = dict(r["epdms"])
        if e.get("score") is None:
            out.append(r)
            continue
        w = dict(e["weights"])
        if weights:
            w.update({k: float(v) for k, v in weights.items()})
        w = {k: v for k, v in w.items() if v > 0}
        weighted = sum(v * float(e["terms"][k]) for k, v in w.items()) / sum(w.values())
        e = {
            **e,
            "weights": w,
            "weighted": float(weighted),
            "score": float(e["multiplicative"] * weighted),
        }
        out.append({**r, "epdms": e})
    return out
