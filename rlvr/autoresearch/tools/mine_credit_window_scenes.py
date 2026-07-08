#!/usr/bin/env python3
"""Mine R2LPL credit-window scenes from classified route-lineage failures."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from preference_optimization.model_utils import load_model
from rlvr.autoresearch.tools.reproducer_danger_scorer import (
    build_realized_event_scorer,
    build_reproducer_danger_scorer,
    load_credit_windows,
)
from scenario_generation.danger_event_selection import contiguous_index_runs
from scenario_generation.reproducer_rollout import run_segments_batched
from scenario_generation.route_timeline import RouteTimeline, group_routes, route_prefix

_FRAME_RE = re.compile(r"_(\d+)$")
_STEP_FIELDS = {
    "road_border_crossing": "rb_crossing_step",
    "lane_crossing": "lane_crossing_step",
    "static_collision": "static_collision_step",
    "static_near_miss": "static_collision_step",
    "moving_collision": "moving_collision_step",
    "moving_near_miss": "moving_argmin_t",
    "moving_ttc": "ttc_first_unsafe_step",
    "road_border_near": "rb_crossing_step",
    "expert_disagreement": "expert_disagreement_step",
}


def _frame_index(path: Path) -> int:
    m = _FRAME_RE.search(path.stem)
    if m is None:
        raise ValueError(f"Cannot parse route frame index from {path}")
    return int(m.group(1))


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_scene_list(path: Path) -> list[Path]:
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list of NPZ paths")
    paths = [Path(str(p)) for p in raw]
    if not paths:
        raise ValueError(f"{path} is empty")
    return paths


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _validate_credit_config(
    path: Path,
    observed_labels: set[str],
) -> dict[str, dict[str, int | float]]:
    cfg = load_credit_windows(path)
    if cfg is None:
        raise ValueError(f"Credit-window config {path} did not load")
    missing = sorted(label for label in observed_labels if label != "clean" and label not in cfg)
    if missing:
        raise ValueError(f"Credit-window config {path} is missing labels: {missing}")
    return cfg


def _route_files(npz_root: Path) -> dict[str, list[Path]]:
    paths = sorted(npz_root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No route NPZ files under {npz_root}")
    return group_routes(paths)


def _route_files_from_scene_list(scene_list: Path) -> dict[str, list[Path]]:
    return group_routes(_load_scene_list(scene_list))


def _select_event_windows(
    windows: list[dict[str, Any]],
    routes: dict[str, list[Path]],
    source_gap_steps: int = 1,
    *,
    anchor_horizon_steps: int = 40,
    max_rollout_steps: int = 80,
) -> list[dict[str, Any]]:
    if source_gap_steps < 1:
        raise ValueError(f"source_gap_steps must be >= 1, got {source_gap_steps}")
    if anchor_horizon_steps < 1:
        raise ValueError(f"anchor_horizon_steps must be >= 1, got {anchor_horizon_steps}")
    if max_rollout_steps < 1:
        raise ValueError(f"max_rollout_steps must be >= 1, got {max_rollout_steps}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for w in windows:
        grouped.setdefault((str(w["route_key"]), str(w["label"])), []).append(w)
    out: list[dict[str, Any]] = []
    for (route_key, _label), group in grouped.items():
        by_source = {int(w["source_index"]): w for w in group}
        idx_to_frame = {_frame_index(p): i for i, p in enumerate(routes[route_key])}
        frame_by_idx = {i: frame for frame, i in idx_to_frame.items()}
        for run in contiguous_index_runs(by_source, max_gap=source_gap_steps):
            rows = [by_source[idx] for idx in run]
            anchor = min(
                rows,
                key=lambda w: (
                    abs(int(w["violation_step"]) - int(anchor_horizon_steps)),
                    int(w["violation_step"]),
                    int(w["source_index"]),
                ),
            )
            end_index = min(
                int(anchor["source_index"]) + int(max_rollout_steps) - 1,
                len(routes[route_key]) - 1,
            )
            event = dict(anchor)
            event["event_source_start_index"] = int(run[0])
            event["event_source_end_index"] = int(run[-1])
            event["event_member_count"] = len(rows)
            event["start_index"] = int(anchor["source_index"])
            event["start_frame"] = int(anchor["frame_index"])
            event["end_index"] = int(end_index)
            event["end_frame"] = int(frame_by_idx[end_index])
            event["anchor_horizon_steps"] = int(anchor_horizon_steps)
            event["max_rollout_steps"] = int(max_rollout_steps)
            out.append(event)
    return sorted(out, key=lambda w: (str(w["route_key"]), int(w["source_index"]), str(w["label"])))


def _resolve_row(
    row: dict[str, Any],
    routes: dict[str, list[Path]],
    credit: dict[str, int],
    allowed_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    scene = Path(str(row["scene_path"]))
    key = route_prefix(scene)
    if key not in routes:
        raise ValueError(
            f"Scene path is not route-lineage or route is absent from route pool: {scene}"
        )
    frame = _frame_index(scene)
    frame_to_pos = {_frame_index(p): i for i, p in enumerate(routes[key])}
    route_frames = set(frame_to_pos)
    if frame not in route_frames:
        raise ValueError(f"Scene frame {frame} for route {key} is absent from route pool")
    out = []
    for label in row.get("labels", []):
        if label == "clean":
            continue
        if allowed_labels is not None and label not in allowed_labels:
            continue
        step_field = _STEP_FIELDS.get(label)
        step = row.get(step_field) if step_field else None
        if step is None:
            # Risk labels may not have a dedicated crossing step; mine around the source frame.
            step = 0
        offense = int(frame + int(step))
        if offense not in frame_to_pos:
            raise ValueError(
                f"Offense frame {offense} for {scene} label {label} is absent from route pool"
            )
        spec = credit[label]
        width = int(spec["width_frames"])
        gap = int(spec["gap_frames"])
        start_frame = max(min(route_frames), offense - gap - width)
        while start_frame not in frame_to_pos and start_frame < offense:
            start_frame += 1
        out.append(
            {
                "route_key": key,
                "source_scene_path": str(scene),
                "label": label,
                "frame_index": frame,
                "source_index": frame_to_pos[frame],
                "violation_step": int(step),
                "offense_frame": offense,
                "offense_index": frame_to_pos[offense],
                "credit_width": width,
                "credit_gap": gap,
                "credit_width_s": float(spec["width_s"]),
                "credit_gap_s": float(spec["gap_s"]),
                "start_frame": start_frame,
                "start_index": frame_to_pos[start_frame],
                "end_frame": offense,
                "end_index": frame_to_pos[offense],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classified_scenes_jsonl", type=Path, required=True)
    parser.add_argument("--credit_window_config", type=Path, required=True)
    parser.add_argument("--route_npz_root", type=Path, default=None)
    parser.add_argument(
        "--route_scene_list",
        type=Path,
        default=None,
        help="JSON list of route-lineage NPZs to use when a route root is not provided",
    )
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--out_jsonl", type=Path, required=True)
    parser.add_argument("--out_events_json", type=Path, default=None)
    parser.add_argument("--sidecar_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--anchor_horizon_steps",
        type=int,
        default=40,
        help="pick the source scene whose violation ETA is closest to this many steps",
    )
    parser.add_argument(
        "--max_rollout_steps",
        type=int,
        default=80,
        help="simulate closed loop for at most this many route steps from the chosen anchor",
    )
    parser.add_argument(
        "--rollout_length",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--neighbor_history_mode", default="sim", choices=["recorded", "interpolate", "sim"]
    )
    parser.add_argument(
        "--timeline_progress_mode",
        default="clock",
        choices=["pose", "clock"],
        help="how reproduced frames advance during rollout; R2LPL defaults to fixed clock time",
    )
    parser.add_argument(
        "--tracker_mode",
        default="mpc",
        choices=["perfect", "mpc"],
        help="ego tracker used during closed-loop reproduction",
    )
    parser.add_argument(
        "--goal_reach_m",
        type=float,
        default=0.0,
        help="Goal radius for credit-window rollout. Default 0 disables early goal termination.",
    )
    parser.add_argument("--gpu_transform", action="store_true")
    parser.add_argument(
        "--labels",
        default="",
        help="comma-separated labels to mine; empty mines all classified labels",
    )
    parser.add_argument(
        "--verify_reproduced_issue",
        action="store_true",
        help="save windows only for dangerous labels that are reproduced during the rollout",
    )
    parser.add_argument("--reward_config", type=Path, default=None)
    parser.add_argument("--threshold_config", type=Path, default=None)
    parser.add_argument("--danger_decluster_steps", type=int, default=10)
    parser.add_argument(
        "--classified_decluster_steps",
        type=int,
        default=10,
        help="decluster adjacent classified frames before reproducer simulation",
    )
    parser.add_argument("--enable_conflict_detector", action="store_true")
    args = parser.parse_args()
    if (args.route_npz_root is None) == (args.route_scene_list is None):
        raise ValueError("exactly one of --route_npz_root or --route_scene_list is required")
    if args.rollout_length is not None:
        args.max_rollout_steps = int(args.rollout_length)
    if args.anchor_horizon_steps < 1:
        raise ValueError("--anchor_horizon_steps must be >= 1")
    if args.max_rollout_steps < 1:
        raise ValueError("--max_rollout_steps must be >= 1")

    rows = _load_jsonl(args.classified_scenes_jsonl)
    allowed_labels = {label.strip() for label in args.labels.split(",") if label.strip()} or None
    observed = {
        label
        for row in rows
        for label in row.get("labels", [])
        if allowed_labels is None or label in allowed_labels
    }
    credit = _validate_credit_config(args.credit_window_config, observed)
    routes = (
        _route_files(args.route_npz_root)
        if args.route_npz_root is not None
        else _route_files_from_scene_list(args.route_scene_list)
    )
    windows: list[dict[str, Any]] = []
    for row in rows:
        windows.extend(_resolve_row(row, routes, credit, allowed_labels))
    windows = _select_event_windows(
        windows,
        routes,
        source_gap_steps=args.classified_decluster_steps,
        anchor_horizon_steps=args.anchor_horizon_steps,
        max_rollout_steps=args.max_rollout_steps,
    )
    if not windows:
        raise ValueError("No non-clean credit windows resolved from classified scenes")
    if args.out_events_json is not None:
        args.out_events_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_events_json.write_text(json.dumps(windows, indent=2))

    timelines = {key: RouteTimeline(paths, args.sidecar_dir) for key, paths in routes.items()}
    work_units = [
        (timelines[w["route_key"]], int(w["start_index"]), int(w["end_index"]) + 1) for w in windows
    ]
    model, model_args = load_model(args.model_path, args.device)
    model.eval()
    danger_scorer = None
    realized_event_scorer = None
    danger_credit_windows = None
    if args.verify_reproduced_issue:
        if args.reward_config is None or args.threshold_config is None:
            raise ValueError(
                "--verify_reproduced_issue requires --reward_config and --threshold_config"
            )
        if args.out_dir.exists():
            for stale in args.out_dir.glob("*_event_*"):
                if stale.is_dir():
                    shutil.rmtree(stale)
        realized_event_scorer = build_realized_event_scorer(
            reward_config=args.reward_config,
            threshold_config=args.threshold_config,
            device=args.device,
            allowed_labels=allowed_labels,
        )
        danger_credit_windows = load_credit_windows(args.credit_window_config)
    else:
        danger_scorer = (
            build_reproducer_danger_scorer(
                reward_config=args.reward_config,
                threshold_config=args.threshold_config,
                device=args.device,
                enable_conflict_detector=bool(args.enable_conflict_detector),
                allowed_labels=allowed_labels,
            )
            if args.reward_config is not None and args.threshold_config is not None
            else None
        )
    run_segments_batched(
        model,
        model_args,
        work_units,
        device=args.device,
        batch_size=args.batch_size,
        goal_reach_m=args.goal_reach_m,
        route_keys=[w["route_key"] for w in windows],
        gpu_transform=args.gpu_transform,
        neighbor_history_mode=args.neighbor_history_mode,
        tracker_mode=args.tracker_mode,
        timeline_progress_mode=args.timeline_progress_mode,
        credit_save_dir=None if args.verify_reproduced_issue else args.out_dir,
        credit_windows=None if args.verify_reproduced_issue else windows,
        verify_credit_windows=windows if args.verify_reproduced_issue else None,
        danger_save_dir=args.out_dir if args.verify_reproduced_issue else None,
        danger_scorer=danger_scorer,
        realized_event_scorer=realized_event_scorer,
        danger_credit_windows=danger_credit_windows,
        danger_decluster_steps=args.danger_decluster_steps,
    )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        n_rows = 0
        if args.verify_reproduced_issue:
            for saved_dir in sorted(args.out_dir.glob("*_event_*")):
                manifest = _load_json(saved_dir / "manifest.json")
                saved_frames = [int(v) for v in manifest.get("scene_frame_ids_saved", [])]
                realized_label = str(
                    manifest.get("realized_label") or manifest.get("credit_label") or "unknown"
                )
                for scene_idx, scene_path in enumerate(sorted(saved_dir.glob("credit*.npz"))):
                    row = {
                        "scene_path": str(scene_path),
                        "window_dir": str(saved_dir),
                        "event_key": str(saved_dir),
                        "credit_scene_stem": scene_path.stem,
                        "label": realized_label,
                        "labels": [realized_label],
                        "source_label": manifest.get("source_label"),
                        "source_anchor_frame": manifest.get("source_anchor_frame"),
                        "source_anchor_index": manifest.get("source_anchor_index"),
                        "source_event_start_frame": manifest.get("source_event_start_frame"),
                        "source_event_end_frame": manifest.get("source_event_end_frame"),
                        "source_event_member_count": manifest.get("source_event_member_count"),
                        "source_offense_frame": manifest.get("source_offense_frame"),
                        "realized_label": manifest.get("realized_label", realized_label),
                        "realized_step": manifest.get("realized_step"),
                        "realized_frame": manifest.get("realized_frame"),
                        "scene_frame_id": saved_frames[scene_idx]
                        if scene_idx < len(saved_frames)
                        else None,
                        "variant_kind": "reproduced_event_window",
                    }
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                    n_rows += 1
        else:
            for w in windows:
                saved_dir = args.out_dir / (
                    f"{w['route_key']}_{w['start_frame']}_{w['end_frame']}_credit_{w['label']}"
                )
                saved = sorted(saved_dir.glob("credit*.npz"))
                if not saved:
                    raise FileNotFoundError(f"No credit scenes were saved under {saved_dir}")
                for scene_path in saved:
                    row = dict(w)
                    row["scene_path"] = str(scene_path)
                    row["window_dir"] = str(saved_dir)
                    row["event_key"] = str(saved_dir)
                    row["credit_scene_stem"] = scene_path.stem
                    row["variant_kind"] = "event_window"
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                    n_rows += 1
        if n_rows == 0:
            raise ValueError("No credit scenes survived reproduced-issue verification")
    print(f"Wrote credit-window rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()
