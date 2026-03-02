#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
import subprocess
import webbrowser
from typing import Dict
from typing import List
from typing import Tuple

from jinja2 import Environment
from jinja2 import FileSystemLoader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def points3_to_np(points: List[List[float]]) -> np.ndarray:
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


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


def compare_lane_segments(internal: Dict, reference: Dict) -> Tuple[Dict, List[Dict], Dict]:
    int_by_id = {int(x["id"]): x for x in internal["lane_segments"]}
    ref_by_id = {int(x["id"]): x for x in reference["lane_segments"]}
    matched_ids = sorted(set(int_by_id.keys()) & set(ref_by_id.keys()))

    center_sym = []
    left_sym = []
    right_sym = []
    center_haus = []
    entity_rows = []

    for lane_id in matched_ids:
        i_lane = int_by_id[lane_id]
        ref_lane = ref_by_id[lane_id]

        c_i = points3_to_np(i_lane["centerline"])
        c_e = points3_to_np(ref_lane["centerline"])
        l_i = points3_to_np(i_lane["left_boundary"])
        l_e = points3_to_np(ref_lane["left_boundary"])
        r_i = points3_to_np(i_lane["right_boundary"])
        r_e = points3_to_np(ref_lane["right_boundary"])

        c_stats = symmetric_distance_stats(c_i, c_e)
        l_stats = symmetric_distance_stats(l_i, l_e)
        r_stats = symmetric_distance_stats(r_i, r_e)

        center_sym.append(c_stats["symmetric_chamfer_like"])
        left_sym.append(l_stats["symmetric_chamfer_like"])
        right_sym.append(r_stats["symmetric_chamfer_like"])
        center_haus.append(c_stats["symmetric_hausdorff_like"])

        entity_rows.append(
            {
                "entity_type": "lane_segment",
                "entity_id": lane_id,
                "match_index": -1,
                "center_sym_chamfer_like": c_stats["symmetric_chamfer_like"],
                "center_sym_hausdorff_like": c_stats["symmetric_hausdorff_like"],
                "left_sym_chamfer_like": l_stats["symmetric_chamfer_like"],
                "right_sym_chamfer_like": r_stats["symmetric_chamfer_like"],
            }
        )

    worst = sorted(entity_rows, key=lambda x: x["center_sym_hausdorff_like"], reverse=True)[:20]
    metrics = {
        "key_metric_symmetric_hausdorff_like_m": summarize(center_haus),
        "centerline_symmetric_chamfer_like_m": summarize(center_sym),
        "centerline_symmetric_hausdorff_like_m": summarize(center_haus),
        "left_boundary_symmetric_chamfer_like_m": summarize(left_sym),
        "right_boundary_symmetric_chamfer_like_m": summarize(right_sym),
        "pass_rate": {
            "centerline_symmetric_hausdorff_lt_0p2m": (
                float(np.mean(np.asarray(center_haus) < 0.2)) if center_haus else 0.0
            )
        },
        "worst_k_centerline": worst,
    }
    return metrics, entity_rows, {}


def match_geometry_only(
    int_items: List[Dict], ref_items: List[Dict], max_match_distance: float
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    if not int_items or not ref_items:
        return [], list(range(len(int_items))), list(range(len(ref_items)))
    used_ref = set()
    matches = []
    first_points_int = np.array([item["points"][0][:2] for item in int_items])
    first_points_ref = np.array([ref["points"][0][:2] for ref in ref_items])
    last_points_int = np.array([item["points"][-1][:2] for item in int_items])
    last_points_ref = np.array([ref["points"][-1][:2] for ref in ref_items])
    start_distances_matrix = np.linalg.norm(
        first_points_int[:, None] - first_points_ref[None, :], axis=2
    )
    start_distances_mask_matrix = start_distances_matrix < 0.3
    end_distances_matrix = np.linalg.norm(
        last_points_int[:, None] - last_points_ref[None, :], axis=2
    )
    end_distances_mask_matrix = end_distances_matrix < 0.3
    valid_mask_matrix = np.logical_and(
        start_distances_mask_matrix, end_distances_mask_matrix)
    for i, item in enumerate(int_items):
        p_i = points3_to_np(item["points"])
        best_j = -1
        best_score = float("inf")
        ref_indices = np.where(valid_mask_matrix[i])[0]
        for j in ref_indices:
            p_ref = points3_to_np(ref_items[j]["points"])
            score = symmetric_distance_stats(p_i, p_ref)["symmetric_chamfer_like"]
            if score < best_score:
                best_j = j
                best_score = score
        if best_j >= 0 and best_score <= max_match_distance:
            used_ref.add(best_j)
            matches.append((i, best_j, best_score))
    unmatched_i = [i for i in range(len(int_items)) if i not in {m[0] for m in matches}]
    unmatched_ref = [j for j in range(len(ref_items)) if j not in used_ref]
    return matches, unmatched_i, unmatched_ref


def compare_line_strings(
    internal: Dict, reference: Dict, max_match_distance: float
) -> Tuple[Dict, List[Dict]]:
    in_lines = internal["line_strings"]
    ref_lines = reference["line_strings"]
    matches, unmatched_i, unmatched_ref = match_geometry_only(
        in_lines, ref_lines, max_match_distance
    )

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
    internal: Dict, reference: Dict, max_match_distance: float
) -> Tuple[Dict, List[Dict]]:
    in_polys = internal["polygons"]
    ref_polys = reference["polygons"]
    matches, unmatched_i, unmatched_ref = match_geometry_only(
        in_polys, ref_polys, max_match_distance
    )

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
) -> None:
    lane_error_map = {int(r["entity_id"]): float(r["center_sym_hausdorff_like"]) for r in lane_rows}
    line_error_map = {
        int(r["match_index"].split(":")[0]): float(r["sym_hausdorff_like"]) for r in line_rows
    }
    poly_error_map = {
        int(r["match_index"].split(":")[0]): float(r["sym_hausdorff_like"]) for r in poly_rows
    }

    lane_haus = list(lane_error_map.values())
    line_haus = list(line_error_map.values())
    poly_haus = list(poly_error_map.values())
    vmax_lane = max(float(np.max(lane_haus)), 1e-6) if lane_haus else 1.0
    vmax_line = max(float(np.max(line_haus)), 1e-6) if line_haus else 1.0
    vmax_poly = max(float(np.max(poly_haus)), 1e-6) if poly_haus else 1.0
    cmap = plt.cm.viridis
    norm_lane = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax_lane)
    norm_line = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax_line)
    norm_poly = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax_poly)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel 1: Fused overlay (reference first, then internal)
    for lane in reference["lane_segments"]:
        c = points3_to_np(lane["centerline"])
        if len(c):
            ax1.plot(c[:, 0], c[:, 1], color="#2d5016", alpha=0.7, linewidth=1.5)
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
        if len(c):
            ax1.plot(c[:, 0], c[:, 1], color="blue", alpha=0.55, linewidth=0.9)
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
        if len(c):
            err = lane_error_map.get(lane_id, 0.0)
            ax2.plot(c[:, 0], c[:, 1], color=cmap(norm_lane(err)), alpha=0.9, linewidth=1.2)
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
    return [[float(x), float(y)] for x, y in pts[:, :2]]


def render_html_dashboard(
    internal: Dict,
    reference: Dict,
    lane_rows: List[Dict],
    line_rows: List[Dict],
    poly_rows: List[Dict],
    out_path: Path,
) -> None:
    if Environment is None:
        return
    lane_error_map = {int(r["entity_id"]): float(r["center_sym_hausdorff_like"]) for r in lane_rows}
    line_error_map = {
        int(r["match_index"].split(":")[0]): float(r["sym_hausdorff_like"]) for r in line_rows
    }
    poly_error_map = {
        int(r["match_index"].split(":")[0]): float(r["sym_hausdorff_like"]) for r in poly_rows
    }

    lanes_ref = []
    for lane in reference["lane_segments"]:
        c = points3_to_np(lane["centerline"])
        if len(c):
            lanes_ref.append({"coordinates": _pts_to_coords(c)})
    lanes_int = []
    for lane in internal["lane_segments"]:
        c = points3_to_np(lane["centerline"])
        if len(c):
            lane_id = int(lane["id"])
            lanes_int.append(
                {
                    "coordinates": _pts_to_coords(c),
                    "id": lane_id,
                    "hausdorff": lane_error_map.get(lane_id, 0.0),
                }
            )
    lines_ref = []
    for line_string in reference["line_strings"]:
        s = points3_to_np(line_string["points"])
        if len(s):
            lines_ref.append({"coordinates": _pts_to_coords(s)})
    lines_int = []
    for i, line_string in enumerate(internal["line_strings"]):
        s = points3_to_np(line_string["points"])
        if len(s):
            lines_int.append(
                {
                    "coordinates": _pts_to_coords(s),
                    "index": i,
                    "hausdorff": line_error_map.get(i, 0.0),
                }
            )
    polys_ref = []
    for polygon in reference["polygons"]:
        poly = points3_to_np(polygon["points"])
        if len(poly):
            ring = _pts_to_coords(poly)
            if ring and (ring[0] != ring[-1]):
                ring.append(ring[0])
            polys_ref.append({"coordinates": [ring] if ring else []})
    polys_int = []
    for i, polygon in enumerate(internal["polygons"]):
        poly = points3_to_np(polygon["points"])
        if len(poly):
            ring = _pts_to_coords(poly)
            if ring and (ring[0] != ring[-1]):
                ring.append(ring[0])
            polys_int.append(
                {
                    "coordinates": [ring] if ring else [],
                    "index": i,
                    "hausdorff": poly_error_map.get(i, 0.0),
                }
            )

    lane_haus = list(lane_error_map.values())
    line_haus = list(line_error_map.values())
    poly_haus = list(poly_error_map.values())
    vmax_lane = max(float(np.max(lane_haus)), 1e-6) if lane_haus else 1.0
    vmax_line = max(float(np.max(line_haus)), 1e-6) if line_haus else 1.0
    vmax_poly = max(float(np.max(poly_haus)), 1e-6) if poly_haus else 1.0
    metrics_json = {
        "lane": lane_error_map,
        "line": {str(k): v for k, v in line_error_map.items()},
        "poly": {str(k): v for k, v in poly_error_map.items()},
    }

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
        metrics_json=metrics_json,
    )
    out_path.write_text(html, encoding="utf-8")


def write_entity_csv(
    out_dir: Path, lane_rows: List[Dict], line_rows: List[Dict], poly_rows: List[Dict]
) -> None:
    rows = lane_rows + line_rows + poly_rows
    keys = sorted(set().union(*[r.keys() for r in rows])) if rows else []
    with (out_dir / "entity_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
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
    worst_lanes = sorted(lane_rows, key=lambda x: x["center_sym_hausdorff_like"], reverse=True)[:k]
    for row in worst_lanes:
        lane_id = int(row["entity_id"])
        i_lane = next((x for x in internal["lane_segments"] if int(x["id"]) == lane_id), None)
        ref_lane = ref_by_id.get(lane_id)
        payload = {"metrics": row, "internal_lane": i_lane, "reference_lane": ref_lane}
        with (dbg_dir / f"lane_{lane_id}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    worst_lines = sorted(line_rows, key=lambda x: x["sym_hausdorff_like"], reverse=True)[:k]
    for idx, row in enumerate(worst_lines):
        payload = {
            "metrics": row,
            "internal_line": internal["line_strings"][int(row["match_index"].split(":")[0])],
            "reference_line": reference["line_strings"][int(row["match_index"].split(":")[1])],
        }
        with (dbg_dir / f"line_string_{idx}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    worst_polys = sorted(poly_rows, key=lambda x: x["sym_hausdorff_like"], reverse=True)[:k]
    for idx, row in enumerate(worst_polys):
        i_idx, r_idx = map(int, row["match_index"].split(":"))
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
    parser.add_argument("--skip_html", action="store_true")
    parser.add_argument("--output_prefix", type=str, default="")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Open the interactive HTML overlay in the default browser after completion.",
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
    skip_html: bool,
    output_prefix: str,
    open_web: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[2/3] Computing metrics...")
    internal = load_json(internal_map_path)
    reference = load_json(reference_map_path)

    print("[2.0/3] Computing Lane metrics...")
    lane_metrics, lane_rows, mismatch = compare_lane_segments(internal, reference)
    print("[2.3/3] Computing Line String metrics...")
    line_metrics, line_rows = compare_line_strings(internal, reference, max_match_distance)
    print("[2.6/3] Computing Polygon metrics...")
    poly_metrics, poly_rows = compare_polygons(internal, reference, max_match_distance)

    worst_k_by_hausdorff = []
    for r in lane_rows:
        worst_k_by_hausdorff.append(
            {
                "entity_type": "lane_segment",
                "entity_id": r["entity_id"],
                "match_index": -1,
                "sym_hausdorff_like": r["center_sym_hausdorff_like"],
            }
        )
    for r in line_rows:
        worst_k_by_hausdorff.append(
            {
                "entity_type": "line_string",
                "entity_id": -1,
                "match_index": r["match_index"],
                "sym_hausdorff_like": r["sym_hausdorff_like"],
            }
        )
    for r in poly_rows:
        worst_k_by_hausdorff.append(
            {
                "entity_type": "polygon",
                "entity_id": -1,
                "match_index": r["match_index"],
                "sym_hausdorff_like": r["sym_hausdorff_like"],
            }
        )
    worst_k_by_hausdorff = sorted(
        worst_k_by_hausdorff, key=lambda x: x["sym_hausdorff_like"], reverse=True
    )[:20]

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
    summary_csv_path = resolve_output_path(out_dir, output_prefix, "metrics_summary.csv")
    overlay_png_path = resolve_output_path(out_dir, output_prefix, "overlay_and_error_plots.png")
    html_path = resolve_output_path(out_dir, output_prefix, "interactive_overlay.html")

    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _HAUSDORFF_FIRST = (
        "key_metric_symmetric_hausdorff_like_m",
        "centerline_symmetric_hausdorff_like_m",
        "symmetric_hausdorff_like_m",
    )

    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity", "metric", "count", "mean", "median", "p95", "max"])
        for entity in ("lane_segments", "line_strings", "polygons"):
            entity_metrics = metrics[entity]
            seen = set()
            ordered = []
            for k in _HAUSDORFF_FIRST:
                if k in entity_metrics and k not in seen:
                    ordered.append((k, entity_metrics[k]))
                    seen.add(k)
            for metric_name, metric_value in entity_metrics.items():
                if metric_name not in seen:
                    ordered.append((metric_name, metric_value))
                    seen.add(metric_name)
            for metric_name, metric_value in ordered:
                if isinstance(metric_value, dict) and {
                    "count",
                    "mean",
                    "median",
                    "p95",
                    "max",
                }.issubset(metric_value.keys()):
                    writer.writerow(
                        [
                            entity,
                            metric_name,
                            metric_value["count"],
                            metric_value["mean"],
                            metric_value["median"],
                            metric_value["p95"],
                            metric_value["max"],
                        ]
                    )

    write_entity_csv(out_dir, lane_rows, line_rows, poly_rows)
    write_worst_case_debug(
        internal, reference, lane_rows, line_rows, poly_rows, out_dir, top_k_debug
    )
    make_static_plots(internal, reference, lane_rows, line_rows, poly_rows, overlay_png_path)
    if not skip_html:
        render_html_dashboard(internal, reference, lane_rows, line_rows, poly_rows, html_path)
        if open_web:
            webbrowser.open(f"file://{html_path.resolve()}")

    print("Done.")
    print(f"- metrics: {metrics_json_path}")
    print(f"- overlay: {overlay_png_path}")
    if not skip_html:
        print(f"- interactive: {html_path}")


def main() -> None:
    args = parse_args()
    if args.command == "eval-only":
        evaluate_core(
            internal_map_path=args.internal_map,
            reference_map_path=args.reference_map,
            out_dir=args.out_dir,
            max_match_distance=args.max_match_distance,
            top_k_debug=args.top_k_debug,
            skip_html=args.skip_html,
            output_prefix=args.output_prefix,
            open_web=args.web,
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
        skip_html=args.skip_html,
        output_prefix=args.output_prefix,
        open_web=args.web,
    )


if __name__ == "__main__":
    main()
