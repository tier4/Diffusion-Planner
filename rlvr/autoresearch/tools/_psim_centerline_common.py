"""Shared helpers for ``eval_psim_centerline`` and
``eval_psim_centerline_nway``.

Lateral-offset metric is computed by ``lat_offset_and_naive_score`` in
``eval_centerline_metrics.py`` (the same code path the GRPO training reward
uses). These helpers handle aggregation, cropping, and arc-length binning
on top of the metric.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reference-centerline construction (route.json + lanelet2 map).
# ---------------------------------------------------------------------------

def parse_lanelet_centerlines(
    osm_path: Path, wanted_ids: list[int], n_resample: int = 20
) -> dict[int, np.ndarray]:
    """Return a polyline (per-step midpoint of the left/right ways) for each
    requested lanelet id. Each polyline is resampled to ``n_resample`` points
    by arc-length so concatenation of consecutive lanelets is well-behaved."""
    import xml.etree.ElementTree as ET

    print(f"Parsing {osm_path} ...")
    tree = ET.parse(osm_path)
    root = tree.getroot()

    nodes: dict[int, tuple[float, float]] = {}
    ways: dict[int, list[int]] = {}
    relations: dict[int, dict[str, int]] = {}

    for el in root:
        if el.tag == "node":
            nid = int(el.get("id"))
            x = y = None
            for tag in el.findall("tag"):
                k, v = tag.get("k"), tag.get("v")
                if k == "local_x":
                    x = float(v)
                elif k == "local_y":
                    y = float(v)
            if x is not None and y is not None:
                nodes[nid] = (x, y)
        elif el.tag == "way":
            wid = int(el.get("id"))
            ways[wid] = [int(nd.get("ref")) for nd in el.findall("nd")]
        elif el.tag == "relation":
            rid = int(el.get("id"))
            roles = {m.get("role"): int(m.get("ref")) for m in el.findall("member")}
            relations[rid] = roles

    def resample(poly: np.ndarray, N: int) -> np.ndarray:
        d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        if cum[-1] < 1e-6:
            return poly[:1].repeat(N, axis=0)
        tgt = np.linspace(0.0, cum[-1], N)
        x = np.interp(tgt, cum, poly[:, 0])
        y = np.interp(tgt, cum, poly[:, 1])
        return np.stack([x, y], axis=1)

    out = {}
    for lid in wanted_ids:
        rel = relations.get(lid)
        if rel is None or "left" not in rel or "right" not in rel:
            continue
        L_ids = ways.get(rel["left"], [])
        R_ids = ways.get(rel["right"], [])
        L = np.array([nodes[n] for n in L_ids if n in nodes], dtype=np.float64)
        R = np.array([nodes[n] for n in R_ids if n in nodes], dtype=np.float64)
        if len(L) < 2 or len(R) < 2:
            continue
        out[lid] = 0.5 * (resample(L, n_resample) + resample(R, n_resample))
    return out


def build_route_polyline(osm_path: Path, route_json_path: Path) -> np.ndarray:
    """Concatenate per-lanelet centerlines from ``route.json``'s preferred
    primitives into a single ``(M, 2)`` world-frame polyline.

    Raises ``SystemExit`` if fewer than two centerline points were assembled
    (e.g. all referenced lanelet ids missing from the map, or the map and
    the route reference different worlds). Without this guard, downstream
    point-to-polyline projection silently produces near-zero offsets — much
    worse than failing loudly.
    """
    import json

    route = json.loads(route_json_path.read_text())
    ids = [seg["preferred_primitive"]["id"] for seg in route["segments"]]
    cls = parse_lanelet_centerlines(osm_path, ids)
    missing = [lid for lid in ids if lid not in cls]
    poly: list[list[float]] = []
    for lid in ids:
        if lid not in cls:
            continue
        seg = cls[lid]
        if poly and np.linalg.norm(seg[0] - poly[-1]) < 0.05:
            poly.extend(seg[1:].tolist())
        else:
            poly.extend(seg.tolist())
    arr = np.array(poly, dtype=np.float64)
    print(
        f"Route polyline: {len(ids)} lanelets requested, "
        f"{len(ids) - len(missing)} resolved -> {len(arr)} centerline points"
        + (f" ({len(missing)} missing: e.g. {missing[:5]})" if missing else "")
    )
    if len(arr) < 2:
        raise SystemExit(
            f"build_route_polyline: only {len(arr)} centerline point(s) were "
            f"resolved from {len(ids)} requested lanelets. The route.json and "
            f"the lanelet2 map likely reference different worlds. "
            f"First few missing lanelet ids: {missing[:10]}"
        )
    return arr


# ---------------------------------------------------------------------------
# World→ego synthetic route_lanes tensor for the lateral-offset helper.
# ---------------------------------------------------------------------------

def world_polyline_to_ego_route_lanes(
    polyline_world: np.ndarray,
    ego_xy: tuple[float, float],
    ego_yaw: float,
    points_per_seg: int = 20,
    lane_half_width: float = 1.75,
) -> torch.Tensor:
    """Transform the world-frame polyline into ego frame and pack into the
    ``route_lanes`` segment-point feature layout consumed by
    ``lat_offset_and_naive_score`` (centers at [0:2], dir at [2:4], left at
    [4:6], right at [6:8] — all other channels zero).

    Note on the synthetic left/right vectors: real ``lanes[..., 4:6]`` are
    2-D offset vectors pointing left of the centerline. We emit
    ``lat_dir * lane_half_width`` for the left channel and
    ``-lat_dir * lane_half_width`` for the right; the helper then projects
    via dot-product against ``lat_dir`` to recover ``+/-lane_half_width``
    and clamps with ``min=0.5``. Result: ``side_hw == lane_half_width``
    regardless of which side the ego is on. Don't "fix" the right-side
    sign to match left without re-checking ``lat_offset_and_naive_score``.

    Returns a tensor of shape ``(1, S, points_per_seg, 33)``.
    """
    cx, cy = ego_xy
    cos_y = math.cos(-ego_yaw)
    sin_y = math.sin(-ego_yaw)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
    pts = (polyline_world - np.array([cx, cy], dtype=np.float64)) @ R.T

    # Chunk into segments of `points_per_seg` with a 1-point overlap between
    # successive segments so direction vectors remain consistent at seams.
    M = len(pts)
    if M < 2:
        return torch.zeros(1, 1, points_per_seg, 33, dtype=torch.float32)
    step = points_per_seg - 1
    starts = list(range(0, M - 1, step))
    if starts[-1] + points_per_seg > M:
        starts[-1] = max(0, M - points_per_seg)
    starts = sorted(set(starts))
    S = len(starts)
    seg = np.zeros((S, points_per_seg, 33), dtype=np.float32)
    for s_idx, st in enumerate(starts):
        chunk = pts[st : st + points_per_seg]
        if len(chunk) < points_per_seg:
            pad = np.tile(chunk[-1:], (points_per_seg - len(chunk), 1))
            chunk = np.concatenate([chunk, pad], axis=0)
        seg[s_idx, :, 0] = chunk[:, 0]
        seg[s_idx, :, 1] = chunk[:, 1]
        d = np.diff(chunk, axis=0)
        d_norm = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.where(d_norm < 1e-6, 1.0, d_norm)
        d_full = np.concatenate([d, d[-1:]], axis=0)
        seg[s_idx, :, 2] = d_full[:, 0]
        seg[s_idx, :, 3] = d_full[:, 1]
        lat_dir = np.stack([-d_full[:, 1], d_full[:, 0]], axis=1)
        seg[s_idx, :, 4:6] = lat_dir * lane_half_width
        seg[s_idx, :, 6:8] = -lat_dir * lane_half_width
    return torch.from_numpy(seg)[None]


# ---------------------------------------------------------------------------
# Run aggregation + cropping + summary stats.
# ---------------------------------------------------------------------------

def polyline_cumulative_arclength(
    polyline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(arc_cum, segment_lengths)`` for a polyline.

    ``arc_cum`` has length ``N`` (one entry per polyline point, starts at 0);
    ``segment_lengths`` has length ``N-1`` (one entry per segment)."""
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum, seg


def project_point_to_polyline_arclength(
    polyline: np.ndarray, x: float, y: float,
    cum: np.ndarray | None = None,
) -> float:
    """Return the arc-length on ``polyline`` of the point closest to ``(x, y)``.

    Standard point-to-segment projection across all segments; picks the segment
    with the smallest perpendicular distance and adds the partial arc-length
    along that segment.

    Pass a precomputed ``cum`` array (from ``polyline_cumulative_arclength``)
    to skip per-call cumsum recomputation. This is O(N_polyline) extra work
    per frame otherwise, which adds up in long runs."""
    a = polyline[:-1]
    b = polyline[1:]
    ab = b - a
    sl2 = np.maximum(np.sum(ab * ab, axis=1), 1e-9)
    ap = np.array([x, y])[None, :] - a
    t = np.clip(np.sum(ap * ab, axis=1) / sl2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d2 = np.sum((proj - np.array([x, y])[None, :]) ** 2, axis=1)
    idx = int(np.argmin(d2))
    if cum is None:
        cum, _ = polyline_cumulative_arclength(polyline)
    return float(cum[idx] + t[idx] * np.sqrt(sl2[idx]))


def crop_run_by_offset(d: dict, max_offset_m: float) -> dict:
    """Drop frames whose ``|lateral offset|`` exceeds ``max_offset_m``.

    Keeps the dict-of-arrays layout used by ``eval_psim_centerline.collect_run``
    (``ts``, ``world_xy``, ``lat``, ``abs_lat``, ``lon``, ``speed``)."""
    m = np.abs(d["lat"]) <= max_offset_m
    return {k: v[m] for k, v in d.items()}


def stats_line(name: str, d: dict, all_d: dict) -> str:
    """One-line summary of a run's cropped vs raw lateral offsets.

    Returns ``"… all values n/a"`` when the cropped run has zero kept frames
    (e.g. when ``--max_offset_m`` excluded everything) so the caller can still
    print a complete summary table without crashing."""
    lat = d["lat"]
    kept = len(lat)
    total = len(all_d["lat"])
    if kept == 0:
        return (
            f"{name:>14s}  kept={kept:4d}/{total:4d}  "
            f"|lat| mean=n/a  median=n/a  p95=n/a  max=n/a  "
            f"std(lat)=n/a  mean(lat)=n/a"
        )
    abs_lat = np.abs(lat)
    return (
        f"{name:>14s}  kept={kept:4d}/{total:4d}  "
        f"|lat| mean={np.mean(abs_lat):.3f}m  "
        f"median={np.median(abs_lat):.3f}m  "
        f"p95={np.percentile(abs_lat,95):.3f}m  "
        f"max={np.max(abs_lat):.3f}m  "
        f"std(lat)={np.std(lat):.3f}m  "
        f"mean(lat)={np.mean(lat):+.3f}m"
    )


def arc_bin_diff(
    base_c: dict, prism_c: dict, bin_m: float = 5.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(arc_bin_edges, arc_bin_centers, prism_bin_mean − base_bin_mean)``.

    Bins both runs' ``|lat|`` along route arc-length with the given bin width
    and returns the difference of the per-bin means. NaN where a bin has no
    samples in either run. Returns three empty arrays when either input has
    zero kept frames (so callers can detect "no comparison possible" without
    a `.max()` reduction crash)."""
    if len(base_c["lon"]) == 0 or len(prism_c["lon"]) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    arc_max = max(base_c["lon"].max(), prism_c["lon"].max())
    arc_bins = np.arange(0, arc_max + bin_m, bin_m)
    if len(arc_bins) < 2:
        # Degenerate case: a single point at arc-length 0 (max==0). Build a
        # 1-bin schema so np.digitize / arange invariants hold downstream.
        arc_bins = np.array([0.0, bin_m], dtype=float)
    arc_centers = 0.5 * (arc_bins[1:] + arc_bins[:-1])

    def _bin(d: dict) -> np.ndarray:
        idx = np.clip(np.digitize(d["lon"], arc_bins) - 1, 0, len(arc_centers) - 1)
        sums = np.zeros(len(arc_centers))
        cnts = np.zeros(len(arc_centers))
        np.add.at(sums, idx, np.abs(d["lat"]))
        np.add.at(cnts, idx, 1.0)
        return np.divide(sums, cnts, out=np.full_like(sums, np.nan), where=cnts > 0)

    return arc_bins, arc_centers, _bin(prism_c) - _bin(base_c)
