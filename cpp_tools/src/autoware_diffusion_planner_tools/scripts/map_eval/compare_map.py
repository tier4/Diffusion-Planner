#!/usr/bin/env python3

# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map preprocessing evaluator for Diffusion Planner.

This module provides two entry paths:

- `eval-only`: evaluate existing `internal_map.json` and `reference.json`
- `export-eval`: run `map_exporter` first, then evaluate the exported JSON

High-level call flow:

    main
      -> parse_args
      -> (optional) run_export_stage
      -> evaluate_core
           -> compare_lane_segments
           -> compare_line_strings
           -> compare_polygons
           -> build_error_maps
           -> compute_point_errors
           -> render_html_dashboard

Design notes:
- Matching is split by entity type: lanes are ID-matched; lines/polygons are
  geometry-matched with start/end gating and chamfer-like ranking.
- Metric computation is independent from rendering.
- HTML rendering consumes precomputed metrics and point-error payloads.

JSON format notes (from `map_exporter.cpp`):
- Both map JSON files share top-level keys:
  - `lane_segments`: list
  - `line_strings`: list
  - `polygons`: list
  - `meta`: object (source path and export metadata)
- Internal map (`internal_map.json`) lane segment format:
  - `id`: lane ID
  - `centerline` / `left_boundary` / `right_boundary`: each is a polyline
    represented as `[[x, y, z], ...]`
- Reference map (`reference.json`) lane segment format:
  - `id`: original Lanelet2 lanelet ID
  - `centerline` / `left_boundary` / `right_boundary`: each is a list of
    point records with IDs:
    `[{"id": point_id, "points": [x, y, z]}, ...]`
- For `line_strings` and `polygons`, both internal and reference exports use:
  - `{"points": [[x, y, z], ...]}`
- Practical implication for this evaluator:
  - lane boundaries use different representations between internal/reference,
    so lane comparisons use dedicated converters (`points3_to_np` vs
    `points3_id_to_np`), while line/polygon comparisons use plain point arrays.
"""

import argparse
import csv
import dataclasses
import json
import subprocess
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader


@dataclasses.dataclass
class ErrorMaps:
    """Error maps for efficient lookup during visualization.

    Note: Key types differ by entity type:
    - lane: keyed by semantic lane ID (int) from the 'id' field in JSON
    - line/poly: keyed by positional index (int) in the list, NOT by semantic ID
    """

    lane: Dict[int, float]
    line: Dict[int, float]
    poly: Dict[int, float]


JsonMap = Dict[str, Any]
MetricRow = Dict[str, Any]
InternalMap = JsonMap
ReferenceMap = JsonMap
LaneMetricRow = MetricRow
LineMetricRow = MetricRow
PolyMetricRow = MetricRow
PointErrorRecord = Dict[str, Any]
LanePointErrors = Dict[int, Dict[str, List[PointErrorRecord]]]
LinePointErrors = Dict[int, List[PointErrorRecord]]


def _safe_split_match(r: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    """Safely parse match_index field, returning tuple or None on failure."""
    parts = r.get("match_index", "").split(":")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def build_error_maps(
    lane_rows: List[LaneMetricRow],
    line_rows: List[LineMetricRow],
    poly_rows: List[PolyMetricRow],
) -> ErrorMaps:
    """Build per-entity Hausdorff lookup maps from comparison rows.

    The returned maps are used by both static and interactive visualizations.
    Lane entries are keyed by semantic lane ID, while line/polygon entries are
    keyed by internal list index extracted from `match_index`.
    """
    return ErrorMaps(
        lane={
            int(r["entity_id"]): max(
                float(r["center_sym_hausdorff_like"]),
                float(r["left_sym_hausdorff_like"]),
                float(r["right_sym_hausdorff_like"]),
            )
            for r in lane_rows
        },
        line={
            pair[0]: float(r["sym_hausdorff_like"])
            for r in line_rows
            if (pair := _safe_split_match(r)) is not None
        },
        poly={
            pair[0]: float(r["sym_hausdorff_like"])
            for r in poly_rows
            if (pair := _safe_split_match(r)) is not None
        },
    )


def build_worst_k(
    lane_rows: List[Dict],
    line_rows: List[Dict],
    poly_rows: List[Dict],
    k: int,
) -> List[Dict]:
    """Build combined worst-K entities list from all entity types."""
    combined = (
        [
            {
                "entity_type": "lane_segment",
                "entity_id": r["entity_id"],
                "match_index": -1,
                "sym_hausdorff_like": r["center_sym_hausdorff_like"],
            }
            for r in lane_rows
        ]
        + [
            {
                "entity_type": "line_string",
                "entity_id": -1,
                "match_index": r["match_index"],
                "sym_hausdorff_like": r["sym_hausdorff_like"],
            }
            for r in line_rows
        ]
        + [
            {
                "entity_type": "polygon",
                "entity_id": -1,
                "match_index": r["match_index"],
                "sym_hausdorff_like": r["sym_hausdorff_like"],
            }
            for r in poly_rows
        ]
    )
    return sorted(combined, key=lambda x: x["sym_hausdorff_like"], reverse=True)[:k]


def load_json(path: Path) -> JsonMap:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def points3_to_np(points: Any) -> np.ndarray:
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def points3_id_to_np(points: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.zeros((0, 4), dtype=np.int64), np.zeros((0, 3), dtype=np.float64)
    ids = np.asarray([int(p["id"]) for p in points], dtype=np.int64)
    pts = np.asarray([p["points"] for p in points], dtype=np.float64)
    return ids, pts


def polyline_segments_xy(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(poly) < 2:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return poly[:-1, :2], poly[1:, :2]


def point_to_segments_distance_xy(
    points: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(seg_start) == 0:
        return np.full((len(points),), np.inf, dtype=np.float64)
    p = points[:, None, :2]
    a = seg_start[None, :, :]
    b = seg_end[None, :, :]
    ab = b - a
    ap = p - a
    denom = np.sum(ab * ab, axis=2)
    t = np.sum(ap * ab, axis=2) / np.maximum(denom, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, :, None] * ab
    d = np.linalg.norm(p - proj, axis=2)
    return d.min(axis=1)


def directed_polyline_distance_xy(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    s0, s1 = polyline_segments_xy(target)
    return point_to_segments_distance_xy(source, s0, s1)


def summarize(values: List[float]) -> Dict:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate angle difference between two polylines in degrees.

    Note: This function measures global start-to-end heading difference only.
    It uses the vector from the first point to the last point of each polyline,
    ignoring local path curvature. For example, an S-curve and a straight line
    with the same start/end points will both yield 0° difference.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va = a[-1, :2] - a[0, :2]
    vb = b[-1, :2] - b[0, :2]
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos_v = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_v)))


def symmetric_distance_stats(a: np.ndarray, b: np.ndarray) -> Dict:
    d_ab = directed_polyline_distance_xy(a, b)
    d_ba = directed_polyline_distance_xy(b, a)
    ab_mean = float(np.mean(d_ab)) if len(d_ab) else 0.0
    ba_mean = float(np.mean(d_ba)) if len(d_ba) else 0.0
    ab_max = float(np.max(d_ab)) if len(d_ab) else 0.0
    ba_max = float(np.max(d_ba)) if len(d_ba) else 0.0
    return {
        "i_to_e_mean": ab_mean,
        "e_to_i_mean": ba_mean,
        "i_to_e_max": ab_max,
        "e_to_i_max": ba_max,
        "symmetric_chamfer_like": ab_mean + ba_mean,
        "symmetric_hausdorff_like": max(ab_max, ba_max),
    }


def _points_to_error_dicts(
    pts: np.ndarray,
    errors: np.ndarray,
    point_ids: Optional[np.ndarray] = None,
    boundary_type: Optional[str] = None,
) -> List[Dict]:
    """Convert points and errors to list of dicts for JSON serialization.

    Args:
        pts: Nx3 array with columns [lng, lat, ...]
        errors: N-element array of error values

    Returns:
        List of dicts with lat, lng, error and optional point_id/boundary_type keys
    """
    if len(pts) == 0 or len(errors) == 0:
        return []
    if pts.shape[1] < 2:
        return []
    pts_array = np.column_stack([pts[:, 1], pts[:, 0], errors])
    valid_mask = np.isfinite(pts_array[:, 2])
    valid_points = pts_array[valid_mask]
    point_ids_valid = (
        point_ids[valid_mask] if point_ids is not None and len(point_ids) == len(pts) else None
    )
    out: List[PointErrorRecord] = []
    for idx, p in enumerate(valid_points):
        item: PointErrorRecord = {
            "lat": float(p[0]),
            "lng": float(p[1]),
            "error": float(p[2]),
        }
        if point_ids_valid is not None:
            item["point_id"] = int(point_ids_valid[idx])
        if boundary_type is not None:
            item["boundary_type"] = boundary_type
        out.append(item)
    return out


def _compute_point_errors_for_indexed_entities(
    internal_list: List[MetricRow],
    reference_list: List[MetricRow],
    rows: List[MetricRow],
    points_key: str = "points",
) -> Dict:
    """Generic helper to compute point errors for index-matched entities (lines/polygons).

    Args:
        internal_list: List of internal entities
        reference_list: List of reference entities
        rows: List of matching result rows with match_index field
        points_key: Key to extract points from each entity (default: "points")

    Returns:
        Dictionary mapping internal index to list of point error dicts
    """
    errors = {}
    for row in rows:
        match_idx = row.get("match_index", "-1:-1")
        parts = match_idx.split(":")
        if len(parts) != 2:
            continue
        i_idx = int(parts[0])
        r_idx = int(parts[1])

        if i_idx < 0 or r_idx < 0:
            continue
        if i_idx >= len(internal_list) or r_idx >= len(reference_list):
            continue

        p_i = points3_to_np(internal_list[i_idx][points_key])
        p_e = points3_to_np(reference_list[r_idx][points_key])

        if len(p_i) > 0 and len(p_e) > 0:
            # Use directed distance for point-level visualization
            # This shows where internal points deviate from reference, avoiding
            # scalar broadcast from max reverse distance inflating all errors
            d_ab = directed_polyline_distance_xy(p_i, p_e)

            errors[i_idx] = _points_to_error_dicts(p_i, d_ab)
    return errors


def compute_point_errors(
    internal: InternalMap,
    reference: ReferenceMap,
    lane_rows: List[LaneMetricRow],
    line_rows: List[LineMetricRow],
) -> Tuple[LanePointErrors, LinePointErrors]:
    """Compute point-level residual payloads used by the HTML dashboard.

    Lanes are matched by semantic lane ID from `lane_rows["entity_id"]`.
    Line strings are matched via `match_index` pairs from `line_rows`.
    Polygons are intentionally excluded from point-marker visualization.

    Returns:
        Tuple of:
        - lane point errors keyed by lane ID and boundary type
        - line point errors keyed by internal line index
    """
    # Build dictionary lookups for O(1) access
    int_lanes_by_id = {int(x["id"]): x for x in internal["lane_segments"]}
    ref_lanes_by_id = {int(x["id"]): x for x in reference["lane_segments"]}

    # Lane point errors - only for matched lanes (using O(1) dictionary lookup)
    lane_point_errors: LanePointErrors = {}
    for row in lane_rows:
        lane_id = int(row["entity_id"])
        int_lane = int_lanes_by_id.get(lane_id)
        ref_lane = ref_lanes_by_id.get(lane_id)
        if int_lane is None or ref_lane is None:
            continue

        lane_point_errors[lane_id] = {
            "centerline": [],
            "left_boundary": [],
            "right_boundary": [],
        }
        for boundary_key in ("centerline", "left_boundary", "right_boundary"):
            ref_ids, ref_pts = points3_id_to_np(ref_lane[boundary_key])
            int_pts = points3_to_np(int_lane[boundary_key])
            if len(ref_pts) == 0 or len(int_pts) == 0:
                continue
            # Compute reference -> internal errors. This is the direction with
            # meaningful residuals in current preprocessing pipeline.
            d_ref_to_int = directed_polyline_distance_xy(ref_pts, int_pts)
            lane_point_errors[lane_id][boundary_key] = _points_to_error_dicts(
                ref_pts,
                d_ref_to_int,
                point_ids=ref_ids,
                boundary_type=boundary_key,
            )

    # Use helper function for lines only (polygons don't need point error visualization)
    line_point_errors = _compute_point_errors_for_indexed_entities(
        internal["line_strings"], reference["line_strings"], line_rows, "points"
    )

    return lane_point_errors, line_point_errors


def compare_lane_segments(
    internal: InternalMap, reference: ReferenceMap
) -> Tuple[Dict[str, Any], List[LaneMetricRow]]:
    """Compare lane segments by semantic lane ID.

    For every shared lane ID, computes symmetric distance stats for centerline,
    left boundary, and right boundary. Returns aggregate lane metrics and
    per-entity rows used downstream by error-map and dashboard generation.
    """
    int_by_id = {int(x["id"]): x for x in internal["lane_segments"]}
    ref_by_id = {int(x["id"]): x for x in reference["lane_segments"]}
    matched_ids = sorted(set(int_by_id.keys()) & set(ref_by_id.keys()))

    center_haus = []
    left_haus = []
    right_haus = []
    entity_rows = []

    for lane_id in matched_ids:
        i_lane = int_by_id[lane_id]
        ref_lane = ref_by_id[lane_id]

        c_i = points3_to_np(i_lane["centerline"])
        c_e_ids, c_e = points3_id_to_np(ref_lane["centerline"])
        l_i = points3_to_np(i_lane["left_boundary"])
        l_e_ids, l_e = points3_id_to_np(ref_lane["left_boundary"])
        r_i = points3_to_np(i_lane["right_boundary"])
        r_e_ids, r_e = points3_id_to_np(ref_lane["right_boundary"])

        c_stats = symmetric_distance_stats(c_i, c_e)
        l_stats = symmetric_distance_stats(l_i, l_e)
        r_stats = symmetric_distance_stats(r_i, r_e)

        center_haus.append(c_stats["symmetric_hausdorff_like"])
        left_haus.append(l_stats["symmetric_hausdorff_like"])
        right_haus.append(r_stats["symmetric_hausdorff_like"])

        entity_rows.append(
            {
                "entity_type": "lane_segment",
                "entity_id": lane_id,
                "match_index": -1,
                "center_sym_hausdorff_like": c_stats["symmetric_hausdorff_like"],
                "left_sym_hausdorff_like": l_stats["symmetric_hausdorff_like"],
                "right_sym_hausdorff_like": r_stats["symmetric_hausdorff_like"],
            }
        )

    worst = sorted(entity_rows, key=lambda x: x["center_sym_hausdorff_like"], reverse=True)[:20]
    haus_summary = summarize(center_haus + left_haus + right_haus)
    metrics = {
        "key_metric_symmetric_hausdorff_like_m": haus_summary,
        "centerline_symmetric_hausdorff_like_m": summarize(center_haus),
        "left_boundary_symmetric_hausdorff_like_m": summarize(left_haus),
        "right_boundary_symmetric_hausdorff_like_m": summarize(right_haus),
        "pass_rate": {
            "centerline_symmetric_hausdorff_lt_0p2m": (
                float(np.mean(np.asarray(center_haus) < 0.2)) if center_haus else 0.0
            ),
            "left_boundary_symmetric_hausdorff_lt_0p2m": (
                float(np.mean(np.asarray(left_haus) < 0.2)) if left_haus else 0.0
            ),
            "right_boundary_symmetric_hausdorff_lt_0p2m": (
                float(np.mean(np.asarray(right_haus) < 0.2)) if right_haus else 0.0
            ),
        },
        "worst_k_centerline": worst,
    }
    return metrics, entity_rows


def match_geometry_only(
    int_items: List[Dict], ref_items: List[Dict], max_match_distance: float
) -> List[Tuple[int, int, float]]:
    # Preserve original indices while filtering empty-points items
    int_indexed = [(i, item) for i, item in enumerate(int_items) if item.get("points")]
    ref_indexed = [(j, ref) for j, ref in enumerate(ref_items) if ref.get("points")]
    if not int_indexed or not ref_indexed:
        return []

    int_orig_idx, int_clean = zip(*int_indexed) if int_indexed else ([], [])
    ref_orig_idx, ref_clean = zip(*ref_indexed) if ref_indexed else ([], [])

    int_clean = list(int_clean)
    ref_clean = list(ref_clean)
    int_orig_idx = list(int_orig_idx)
    ref_orig_idx = list(ref_orig_idx)

    used_ref = set()
    matches = []
    first_points_int = np.array([item["points"][0][:2] for item in int_clean])
    first_points_ref = np.array([ref["points"][0][:2] for ref in ref_clean])
    last_points_int = np.array([item["points"][-1][:2] for item in int_clean])
    last_points_ref = np.array([ref["points"][-1][:2] for ref in ref_clean])
    start_distances_matrix = np.linalg.norm(
        first_points_int[:, None] - first_points_ref[None, :], axis=2
    )
    start_distances_mask_matrix = start_distances_matrix < 0.3
    end_distances_matrix = np.linalg.norm(
        last_points_int[:, None] - last_points_ref[None, :], axis=2
    )
    end_distances_mask_matrix = end_distances_matrix < 0.3
    valid_mask_matrix = np.logical_and(start_distances_mask_matrix, end_distances_mask_matrix)
    for i, item in enumerate(int_clean):
        p_i = points3_to_np(item["points"])
        best_j = -1
        best_score = float("inf")
        ref_indices = np.where(valid_mask_matrix[i])[0]
        for j in ref_indices:
            if j in used_ref:
                continue
            p_ref = points3_to_np(ref_clean[j]["points"])
            score = symmetric_distance_stats(p_i, p_ref)["symmetric_chamfer_like"]
            if score < best_score:
                best_j = j
                best_score = score
        if best_j >= 0 and best_score <= max_match_distance:
            used_ref.add(best_j)
            # Return ORIGINAL indices, not filtered indices
            matches.append((int_orig_idx[i], ref_orig_idx[best_j], best_score))
    return matches


def compare_line_strings(
    internal: InternalMap, reference: ReferenceMap, max_match_distance: float
) -> Tuple[Dict[str, Any], List[LineMetricRow]]:
    """Compare line strings after geometry-based matching.

    Matching uses `match_geometry_only` with start/end gating and a chamfer-like
    matching score, then computes symmetric distance metrics for matched pairs.
    Returns aggregate metrics and per-pair rows.
    """
    in_lines = internal["line_strings"]
    ref_lines = reference["line_strings"]
    matches = match_geometry_only(in_lines, ref_lines, max_match_distance)

    sym, haus, orient = [], [], []
    rows = []
    for i, j, _ in matches:
        a = points3_to_np(in_lines[i]["points"])
        b = points3_to_np(ref_lines[j]["points"])
        stats = symmetric_distance_stats(a, b)
        sym.append(stats["symmetric_chamfer_like"])
        haus.append(stats["symmetric_hausdorff_like"])
        orient.append(angle_diff_deg(a, b))
        rows.append(
            {
                "entity_type": "line_string",
                "entity_id": -1,
                "match_index": f"{i}:{j}",
                "sym_chamfer_like": stats["symmetric_chamfer_like"],
                "sym_hausdorff_like": stats["symmetric_hausdorff_like"],
            }
        )
    return (
        {
            "key_metric_symmetric_hausdorff_like_m": summarize(haus),
            "symmetric_chamfer_like_m": summarize(sym),
            "symmetric_hausdorff_like_m": summarize(haus),
            "orientation_error_deg": summarize(orient),
            "pass_rate": {
                "symmetric_hausdorff_lt_0p2m": (
                    float(np.mean(np.asarray(haus) < 0.2)) if haus else 0.0
                )
            },
        },
        rows,
    )


def compare_polygons(
    internal: InternalMap, reference: ReferenceMap, max_match_distance: float
) -> Tuple[Dict[str, Any], List[PolyMetricRow]]:
    """Compare polygons after geometry-based matching.

    Uses the same matching policy as line strings and returns aggregate metrics
    plus per-pair rows for visualization/debug outputs.
    """
    in_polys = internal["polygons"]
    ref_polys = reference["polygons"]
    matches = match_geometry_only(in_polys, ref_polys, max_match_distance)

    sym, haus = [], []
    rows = []
    for i, j, _ in matches:
        a = points3_to_np(in_polys[i]["points"])
        b = points3_to_np(ref_polys[j]["points"])
        stats = symmetric_distance_stats(a, b)
        sym.append(stats["symmetric_chamfer_like"])
        haus.append(stats["symmetric_hausdorff_like"])
        rows.append(
            {
                "entity_type": "polygon",
                "entity_id": -1,
                "match_index": f"{i}:{j}",
                "sym_chamfer_like": stats["symmetric_chamfer_like"],
                "sym_hausdorff_like": stats["symmetric_hausdorff_like"],
            }
        )
    return (
        {
            "key_metric_symmetric_hausdorff_like_m": summarize(haus),
            "symmetric_chamfer_like_m": summarize(sym),
            "symmetric_hausdorff_like_m": summarize(haus),
            "pass_rate": {
                "symmetric_hausdorff_lt_0p2m": (
                    float(np.mean(np.asarray(haus) < 0.2)) if haus else 0.0
                )
            },
        },
        rows,
    )


def make_static_plots(
    internal: Dict,
    reference: Dict,
    lane_rows: List[Dict],
    line_rows: List[Dict],
    poly_rows: List[Dict],
    out_path: Path,
    error_maps: Optional[ErrorMaps] = None,
) -> None:
    if error_maps is None:
        error_maps = build_error_maps(lane_rows, line_rows, poly_rows)

    lane_error_map = error_maps.lane
    line_error_map = error_maps.line
    poly_error_map = error_maps.poly

    lane_haus = list(lane_error_map.values())
    line_haus = list(line_error_map.values())
    poly_haus = list(poly_error_map.values())
    vmax_lane = max(float(np.max(lane_haus)), 1e-6) if lane_haus else 1.0
    vmax_line = max(float(np.max(line_haus)), 1e-6) if line_haus else 1.0
    vmax_poly = max(float(np.max(poly_haus)), 1e-6) if poly_haus else 1.0
    cmap = matplotlib.cm.get_cmap("viridis")
    norm_lane = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax_lane)
    norm_line = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax_line)
    norm_poly = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax_poly)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel 1: Fused overlay (reference first, then internal)
    for lane in reference["lane_segments"]:
        c_id, c = points3_id_to_np(lane["centerline"])
        l_id, l = points3_id_to_np(lane["left_boundary"])
        r_id, r = points3_id_to_np(lane["right_boundary"])
        if len(c):
            ax1.plot(c[:, 0], c[:, 1], color="#2d5016", alpha=0.7, linewidth=1.5)
        if len(l):
            ax1.plot(l[:, 0], l[:, 1], color="#2d5016", alpha=0.7, linewidth=1.5)
        if len(r):
            ax1.plot(r[:, 0], r[:, 1], color="#2d5016", alpha=0.7, linewidth=1.5)
    for line_string in reference["line_strings"]:
        s = points3_to_np(line_string["points"])
        if len(s):
            ax1.plot(s[:, 0], s[:, 1], color="#2d5016", alpha=0.7, linewidth=1.5)
    for p in reference["polygons"]:
        poly = points3_to_np(p["points"])
        if len(poly):
            ax1.plot(poly[:, 0], poly[:, 1], color="#2d5016", alpha=0.7, linewidth=1.5)

    for lane in internal["lane_segments"]:
        c = points3_to_np(lane["centerline"])
        r = points3_to_np(lane["right_boundary"])
        l = points3_to_np(lane["left_boundary"])
        if len(c):
            ax1.plot(c[:, 0], c[:, 1], color="blue", alpha=0.55, linewidth=0.9)
        if len(r):
            ax1.plot(r[:, 0], r[:, 1], color="blue", alpha=0.55, linewidth=0.9)
        if len(l):
            ax1.plot(l[:, 0], l[:, 1], color="blue", alpha=0.55, linewidth=0.9)
    for i, line_string in enumerate(internal["line_strings"]):
        s = points3_to_np(line_string["points"])
        if len(s):
            ax1.plot(s[:, 0], s[:, 1], color="red", alpha=0.8)
    for i, polygon in enumerate(internal["polygons"]):
        poly = points3_to_np(polygon["points"])
        if len(poly):
            ax1.fill(
                poly[:, 0], poly[:, 1], color="green", alpha=0.4, edgecolor="green", linewidth=1
            )

    ax1.set_title("Fused Overlay (Reference + Internal)")
    ax1.set_aspect("equal")

    # Panel 2: Lane error heatmap (per-type scale)
    for lane in internal["lane_segments"]:
        lane_id = int(lane["id"])
        c = points3_to_np(lane["centerline"])
        l = points3_to_np(lane["left_boundary"])
        r = points3_to_np(lane["right_boundary"])
        if len(c):
            err = lane_error_map.get(lane_id, 0.0)
            ax2.plot(c[:, 0], c[:, 1], color=cmap(norm_lane(err)), alpha=0.9, linewidth=1.2)
        if len(l):
            err = lane_error_map.get(lane_id, 0.0)
            ax2.plot(l[:, 0], l[:, 1], color=cmap(norm_lane(err)), alpha=0.9, linewidth=1.2)
        if len(r):
            err = lane_error_map.get(lane_id, 0.0)
            ax2.plot(r[:, 0], r[:, 1], color=cmap(norm_lane(err)), alpha=0.9, linewidth=1.2)
    ax2.set_title("Lane Error Heatmap (Hausdorff)")
    ax2.set_aspect("equal")
    sm2 = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm_lane)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=ax2, shrink=0.8)
    cbar2.set_label("Hausdorff error (m)", fontsize=10)

    # Panel 3: Line string error heatmap (per-type scale)
    for i, l in enumerate(internal["line_strings"]):
        s = points3_to_np(l["points"])
        if len(s):
            err = line_error_map.get(i, 0.0)
            ax3.plot(s[:, 0], s[:, 1], color=cmap(norm_line(err)), alpha=0.9, linewidth=1.2)
    ax3.set_title("Line String Error Heatmap (Hausdorff)")
    ax3.set_aspect("equal")
    sm3 = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm_line)
    sm3.set_array([])
    cbar3 = fig.colorbar(sm3, ax=ax3, shrink=0.8)
    cbar3.set_label("Hausdorff error (m)", fontsize=10)

    # Panel 4: Polygon error heatmap (per-type scale)
    for i, p in enumerate(internal["polygons"]):
        poly = points3_to_np(p["points"])
        if len(poly):
            err = poly_error_map.get(i, 0.0)
            ax4.fill(
                poly[:, 0],
                poly[:, 1],
                color=cmap(norm_poly(err)),
                alpha=0.5,
                edgecolor=cmap(norm_poly(err)),
                linewidth=1,
            )
    ax4.set_title("Polygon Error Heatmap (Hausdorff)")
    ax4.set_aspect("equal")
    sm4 = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm_poly)
    sm4.set_array([])
    cbar4 = fig.colorbar(sm4, ax=ax4, shrink=0.8)
    cbar4.set_label("Hausdorff error (m)", fontsize=10)

    for ax in axes.flatten():
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _pts_to_coords(pts: np.ndarray) -> List[List[float]]:
    if len(pts) == 0:
        return []
    return pts[:, :2].tolist()


def _close_ring(coords: List[List[float]]) -> List[List[float]]:
    """Ensure polygon ring is closed (first == last point)."""
    if coords and coords[0] != coords[-1]:
        return coords + [coords[0]]
    return coords


def render_html_dashboard(
    internal: InternalMap,
    reference: ReferenceMap,
    lane_rows: List[LaneMetricRow],
    line_rows: List[LineMetricRow],
    poly_rows: List[PolyMetricRow],
    lane_point_errors: LanePointErrors,
    line_point_errors: LinePointErrors,
    out_path: Path,
    lane_threshold: float = 0.2,
    line_threshold: float = 0.2,
    poly_threshold: float = 1.0,
    error_maps: Optional[ErrorMaps] = None,
) -> None:
    """Render the interactive Leaflet dashboard HTML.

    This function is render-focused: it consumes precomputed comparison rows,
    error maps, and point-error payloads, then serializes map geometries and
    metrics into the Jinja template context and writes one HTML file.
    """
    if error_maps is None:
        error_maps = build_error_maps(lane_rows, line_rows, poly_rows)

    lane_error_map = error_maps.lane
    line_error_map = error_maps.line
    poly_error_map = error_maps.poly

    lanes_ref = []
    for lane in reference["lane_segments"]:
        _, c = points3_id_to_np(lane["centerline"])
        _, l = points3_id_to_np(lane["left_boundary"])
        _, r = points3_id_to_np(lane["right_boundary"])
        lane_id = int(lane["id"])
        lane_geometries = {
            "centerline": _pts_to_coords(c),
            "left_boundary": _pts_to_coords(l),
            "right_boundary": _pts_to_coords(r),
        }
        if (
            lane_geometries["centerline"]
            or lane_geometries["left_boundary"]
            or lane_geometries["right_boundary"]
        ):
            lanes_ref.append({"id": lane_id, "geometries": lane_geometries})
    lines_ref = []
    for line_string in reference["line_strings"]:
        s = points3_to_np(line_string["points"])
        if len(s):
            lines_ref.append({"coordinates": _pts_to_coords(s)})
    polys_ref = []
    for polygon in reference["polygons"]:
        poly = points3_to_np(polygon["points"])
        if len(poly):
            ring = _close_ring(_pts_to_coords(poly))
            polys_ref.append({"coordinates": [ring] if ring else []})

    # Point errors and lanes_int/lines_int/polys_int are computed below

    lane_haus = list(lane_error_map.values())
    line_haus = list(line_error_map.values())
    poly_haus = list(poly_error_map.values())
    vmax_lane = max(float(np.max(lane_haus)), 1e-6) if lane_haus else 1.0
    vmax_line = max(float(np.max(line_haus)), 1e-6) if line_haus else 1.0
    vmax_poly = max(float(np.max(poly_haus)), 1e-6) if poly_haus else 1.0

    # Build reference maps for zoom functionality
    ref_lanes_by_id = {int(x["id"]): x for x in reference["lane_segments"]}
    # Build match index maps for lines and polygons with safe parsing
    line_match_map = {
        i: j for r in line_rows if (pair := _safe_split_match(r)) is not None for i, j in [pair]
    }
    poly_match_map = {
        i: j for r in poly_rows if (pair := _safe_split_match(r)) is not None for i, j in [pair]
    }

    # Update lanes_int to include reference coordinates for zoom
    lanes_int = []
    for lane in internal["lane_segments"]:
        c_i = points3_to_np(lane["centerline"])
        l_i = points3_to_np(lane["left_boundary"])
        r_i = points3_to_np(lane["right_boundary"])
        lane_id = int(lane["id"])
        ref_lane = ref_lanes_by_id.get(lane_id)
        ref_geometries: Dict[str, List[List[float]]] = {
            "centerline": [],
            "left_boundary": [],
            "right_boundary": [],
        }
        if ref_lane:
            _, c_ref = points3_id_to_np(ref_lane["centerline"])
            _, l_ref = points3_id_to_np(ref_lane["left_boundary"])
            _, r_ref = points3_id_to_np(ref_lane["right_boundary"])
            ref_geometries = {
                "centerline": _pts_to_coords(c_ref),
                "left_boundary": _pts_to_coords(l_ref),
                "right_boundary": _pts_to_coords(r_ref),
            }
        int_geometries = {
            "centerline": _pts_to_coords(c_i),
            "left_boundary": _pts_to_coords(l_i),
            "right_boundary": _pts_to_coords(r_i),
        }
        if (
            int_geometries["centerline"]
            or int_geometries["left_boundary"]
            or int_geometries["right_boundary"]
        ):
            lanes_int.append(
                {
                    "geometries": int_geometries,
                    "ref_geometries": ref_geometries,
                    "id": lane_id,
                    "hausdorff": lane_error_map.get(lane_id, 0.0),
                }
            )

    # Update lines_int to include reference coordinates for zoom
    lines_int = []
    for i, line_string in enumerate(internal["line_strings"]):
        s_i = points3_to_np(line_string["points"])
        if len(s_i):
            # Find matching reference line using match_index
            ref_coords: List[List[float]] = []
            ref_idx = line_match_map.get(i)
            if ref_idx is not None and ref_idx < len(reference["line_strings"]):
                ref_line = reference["line_strings"][ref_idx]
                s_ref = points3_to_np(ref_line["points"])
                if len(s_ref) > 0:
                    ref_coords = _pts_to_coords(s_ref)
            lines_int.append(
                {
                    "coordinates": _pts_to_coords(s_i),
                    "ref_coordinates": ref_coords,
                    "index": i,
                    "hausdorff": line_error_map.get(i, 0.0),
                }
            )

    # Update polys_int to include reference coordinates for zoom
    polys_int = []
    for i, polygon in enumerate(internal["polygons"]):
        p_i = points3_to_np(polygon["points"])
        if len(p_i):
            # Find matching reference polygon using match_index
            ref_poly_coords: List[List[List[float]]] = []
            ref_idx = poly_match_map.get(i)
            if ref_idx is not None and ref_idx < len(reference["polygons"]):
                ref_poly = reference["polygons"][ref_idx]
                p_ref = points3_to_np(ref_poly["points"])
                if len(p_ref) > 0:
                    ref_poly_coords = [_close_ring(_pts_to_coords(p_ref))]
            ring = _close_ring(_pts_to_coords(p_i))
            polys_int.append(
                {
                    "coordinates": [ring] if ring else [],
                    "ref_coordinates": ref_poly_coords,
                    "index": i,
                    "hausdorff": poly_error_map.get(i, 0.0),
                }
            )

    template_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("interactive_map.html.j2")
    html = template.render(
        lanes_ref=lanes_ref,
        lanes_int=lanes_int,
        lines_ref=lines_ref,
        lines_int=lines_int,
        polys_ref=polys_ref,
        polys_int=polys_int,
        vmax_lane=vmax_lane,
        vmax_line=vmax_line,
        vmax_poly=vmax_poly,
        threshold_lane=lane_threshold,
        threshold_line=line_threshold,
        threshold_poly=poly_threshold,
        lane_point_errors=lane_point_errors,
        line_point_errors=line_point_errors,
    )
    out_path.write_text(html, encoding="utf-8")


def write_entity_csv(
    out_dir: Path, lane_rows: List[Dict], line_rows: List[Dict], poly_rows: List[Dict]
) -> None:
    rows = lane_rows + line_rows + poly_rows
    keys = sorted(set().union(*[r.keys() for r in rows])) if rows else []
    with (out_dir / "entity_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, restval="")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_worst_case_debug(
    internal: Dict,
    reference: Dict,
    lane_rows: List[Dict],
    line_rows: List[Dict],
    poly_rows: List[Dict],
    out_dir: Path,
    k: int,
) -> None:
    dbg_dir = out_dir / "worst_cases"
    dbg_dir.mkdir(exist_ok=True)
    ref_by_id = {int(x["id"]): x for x in reference["lane_segments"]}
    int_by_id = {int(x["id"]): x for x in internal["lane_segments"]}
    worst_lanes = sorted(lane_rows, key=lambda x: x["center_sym_hausdorff_like"], reverse=True)[:k]
    for row in worst_lanes:
        lane_id = int(row["entity_id"])
        i_lane = int_by_id.get(lane_id)
        ref_lane = ref_by_id.get(lane_id)
        payload = {"metrics": row, "internal_lane": i_lane, "reference_lane": ref_lane}
        with (dbg_dir / f"lane_{lane_id}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    worst_lines = sorted(line_rows, key=lambda x: x["sym_hausdorff_like"], reverse=True)[:k]
    for idx, row in enumerate(worst_lines):
        match_pair = _safe_split_match(row)
        if match_pair is None:
            continue
        i_idx, r_idx = match_pair
        if i_idx >= len(internal["line_strings"]) or r_idx >= len(reference["line_strings"]):
            continue
        payload = {
            "metrics": row,
            "internal_line": internal["line_strings"][i_idx],
            "reference_line": reference["line_strings"][r_idx],
        }
        with (dbg_dir / f"line_string_{idx}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    worst_polys = sorted(poly_rows, key=lambda x: x["sym_hausdorff_like"], reverse=True)[:k]
    for idx, row in enumerate(worst_polys):
        match_pair = _safe_split_match(row)
        if match_pair is None:
            continue
        i_idx, r_idx = match_pair
        if i_idx >= len(internal["polygons"]) or r_idx >= len(reference["polygons"]):
            continue
        payload = {
            "metrics": row,
            "internal_polygon": internal["polygons"][i_idx],
            "reference_polygon": reference["polygons"][r_idx],
        }
        with (dbg_dir / f"polygon_{idx}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--max_match_distance", type=float, default=5.0)
    parser.add_argument("--top_k_debug", type=int, default=20)
    parser.add_argument("--output_prefix", type=str, default="")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Open the interactive HTML overlay in the default browser after completion.",
    )
    parser.add_argument(
        "--lane_threshold",
        type=float,
        default=0.2,
        help="Threshold (m) for lanelet error color mapping. Errors >= threshold show max color.",
    )
    parser.add_argument(
        "--line_threshold",
        type=float,
        default=0.2,
        help="Threshold (m) for linestring error color mapping. Errors >= threshold show max color.",
    )
    parser.add_argument(
        "--poly_threshold",
        type=float,
        default=1.0,
        help="Threshold (m) for polygon error color mapping. Errors >= threshold show max color.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map evaluator (no Python interpolation).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_only = subparsers.add_parser(
        "eval-only", help="Evaluate existing internal/reference JSON files."
    )
    eval_only.add_argument("--internal_map", required=True, type=Path)
    eval_only.add_argument("--reference_map", required=True, type=Path)
    add_common_eval_args(eval_only)

    export_eval = subparsers.add_parser(
        "export-eval", help="Export maps via ros2 run map_exporter, then evaluate."
    )
    export_eval.add_argument("--map_path", required=True, type=Path)
    export_eval.add_argument("--internal_json_out", type=Path, default=None)
    export_eval.add_argument("--reference_json_out", type=Path, default=None)
    export_eval.add_argument("--skip_export", action="store_true")
    add_common_eval_args(export_eval)

    return parser.parse_args()


def resolve_output_path(out_dir: Path, output_prefix: str, filename: str) -> Path:
    return out_dir / (f"{output_prefix}_{filename}" if output_prefix else filename)


def run_export_stage(
    map_path: Path, internal_out: Path, reference_out: Path, skip_export: bool
) -> None:
    """Run the ROS2 map exporter stage unless `skip_export` is enabled.

    Side effects:
    - invokes `ros2 run autoware_diffusion_planner_tools map_exporter`
    - writes internal/reference JSON to the requested paths
    """
    if skip_export:
        return
    cmd = [
        "ros2",
        "run",
        "autoware_diffusion_planner_tools",
        "map_exporter",
        "--ros-args",
        "-p",
        f"map_path:={map_path}",
        "-p",
        f"internal_out:={internal_out}",
        "-p",
        f"reference_out:={reference_out}",
    ]
    print("[1/3] Exporting maps...", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def evaluate_core(
    internal_map_path: Path,
    reference_map_path: Path,
    out_dir: Path,
    max_match_distance: float,
    top_k_debug: int,
    output_prefix: str,
    open_web: bool = False,
    lane_threshold: float = 0.2,
    line_threshold: float = 0.2,
    poly_threshold: float = 1.0,
) -> None:
    """Execute the core evaluation pipeline from JSON input to reports.

    Flow:
    1) load internal/reference maps
    2) compute entity metrics and matching rows (lane/line/polygon)
    3) build error maps + point-error payloads
    4) write summary JSON and interactive HTML dashboard
    5) optionally open HTML in browser
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[2/3] Computing metrics...")
    internal = load_json(internal_map_path)
    reference = load_json(reference_map_path)

    print("[2.0/3] Computing Lane metrics...")
    lane_metrics, lane_rows = compare_lane_segments(internal, reference)
    print("[2.3/3] Computing Line String metrics...")
    line_metrics, line_rows = compare_line_strings(internal, reference, max_match_distance)
    print("[2.6/3] Computing Polygon metrics...")
    poly_metrics, poly_rows = compare_polygons(internal, reference, max_match_distance)

    error_maps = build_error_maps(lane_rows, line_rows, poly_rows)
    lane_point_errors, line_point_errors = compute_point_errors(
        internal, reference, lane_rows, line_rows
    )

    worst_k_by_hausdorff = build_worst_k(lane_rows, line_rows, poly_rows, 20)

    exec_summary = {
        "key_metric_symmetric_hausdorff_like_m": {
            "lanes": {
                "mean": lane_metrics["key_metric_symmetric_hausdorff_like_m"]["mean"],
                "p95": lane_metrics["key_metric_symmetric_hausdorff_like_m"]["p95"],
                "max": lane_metrics["key_metric_symmetric_hausdorff_like_m"]["max"],
            },
            "line_strings": {
                "mean": line_metrics["key_metric_symmetric_hausdorff_like_m"]["mean"],
                "p95": line_metrics["key_metric_symmetric_hausdorff_like_m"]["p95"],
                "max": line_metrics["key_metric_symmetric_hausdorff_like_m"]["max"],
            },
            "polygons": {
                "mean": poly_metrics["key_metric_symmetric_hausdorff_like_m"]["mean"],
                "p95": poly_metrics["key_metric_symmetric_hausdorff_like_m"]["p95"],
                "max": poly_metrics["key_metric_symmetric_hausdorff_like_m"]["max"],
            },
        },
        "pass_rate_hausdorff_lt_0p2m": {
            "lanes": lane_metrics["pass_rate"]["centerline_symmetric_hausdorff_lt_0p2m"],
            "line_strings": line_metrics["pass_rate"]["symmetric_hausdorff_lt_0p2m"],
            "polygons": poly_metrics["pass_rate"]["symmetric_hausdorff_lt_0p2m"],
        },
    }

    metrics = {
        "executive_summary": exec_summary,
        "meta": {"internal_map": str(internal_map_path), "reference_map": str(reference_map_path)},
        "lane_segments": lane_metrics,
        "line_strings": line_metrics,
        "polygons": poly_metrics,
        "worst_k_by_hausdorff": worst_k_by_hausdorff,
    }
    print("[3/3] Writing plots/reports...")

    metrics_json_path = resolve_output_path(out_dir, output_prefix, "metrics_summary.json")
    html_path = resolve_output_path(out_dir, output_prefix, "interactive_overlay.html")

    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _HAUSDORFF_FIRST = (
        "key_metric_symmetric_hausdorff_like_m",
        "centerline_symmetric_hausdorff_like_m",
        "symmetric_hausdorff_like_m",
    )

    render_html_dashboard(
        internal,
        reference,
        lane_rows,
        line_rows,
        poly_rows,
        lane_point_errors,
        line_point_errors,
        html_path,
        lane_threshold=lane_threshold,
        line_threshold=line_threshold,
        poly_threshold=poly_threshold,
        error_maps=error_maps,
    )
    if open_web:
        webbrowser.open(f"file://{html_path.resolve()}")

    print("Done.")
    print(f"- metrics: {metrics_json_path}")
    print(f"- interactive: {html_path}")


def main() -> None:
    """CLI entry point for `eval-only` and `export-eval` workflows."""
    args = parse_args()
    if args.command == "eval-only":
        evaluate_core(
            internal_map_path=args.internal_map,
            reference_map_path=args.reference_map,
            out_dir=args.out_dir,
            max_match_distance=args.max_match_distance,
            top_k_debug=args.top_k_debug,
            output_prefix=args.output_prefix,
            open_web=args.web,
            lane_threshold=args.lane_threshold,
            line_threshold=args.line_threshold,
            poly_threshold=args.poly_threshold,
        )
        return

    # export-eval mode
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    internal_json_out = (
        args.internal_json_out
        if args.internal_json_out is not None
        else resolve_output_path(out_dir, args.output_prefix, "internal_map.json")
    )
    reference_json_out = (
        args.reference_json_out
        if args.reference_json_out is not None
        else resolve_output_path(out_dir, args.output_prefix, "reference.json")
    )
    run_export_stage(args.map_path, internal_json_out, reference_json_out, args.skip_export)
    evaluate_core(
        internal_map_path=internal_json_out,
        reference_map_path=reference_json_out,
        out_dir=out_dir,
        max_match_distance=args.max_match_distance,
        top_k_debug=args.top_k_debug,
        output_prefix=args.output_prefix,
        open_web=args.web,
        lane_threshold=args.lane_threshold,
        line_threshold=args.line_threshold,
        poly_threshold=args.poly_threshold,
    )


if __name__ == "__main__":
    main()
