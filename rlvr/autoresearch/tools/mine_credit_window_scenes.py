#!/usr/bin/env python3
"""Mine R2LPL credit-window scenes from classified route-lineage failures."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from preference_optimization.model_utils import load_model
from rlvr.autoresearch.tools.reproducer_danger_scorer import (
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _validate_credit_config(path: Path, observed_labels: set[str]) -> dict[str, int]:
    cfg = {str(k): int(v) for k, v in _load_json(path).items() if not str(k).startswith("_")}
    missing = sorted(label for label in observed_labels if label != "clean" and label not in cfg)
    if missing:
        raise ValueError(f"Credit-window config {path} is missing labels: {missing}")
    return cfg


def _route_files(npz_root: Path) -> dict[str, list[Path]]:
    paths = sorted(npz_root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No route NPZ files under {npz_root}")
    return group_routes(paths)


def _collapse_event_windows(
    windows: list[dict[str, Any]], decluster_steps: int
) -> list[dict[str, Any]]:
    if decluster_steps < 1:
        raise ValueError(
            f"classified_decluster_steps must be >= 1 for event grouping, got {decluster_steps}"
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for w in windows:
        grouped.setdefault((str(w["route_key"]), str(w["label"])), []).append(w)
    out: list[dict[str, Any]] = []
    for group in grouped.values():
        by_step = {int(w["offense_index"]): w for w in group}
        for run in contiguous_index_runs(by_step, max_gap=decluster_steps - 1):
            first = by_step[run[0]]
            event = dict(first)
            event["event_offense_start_index"] = int(run[0])
            event["event_offense_end_index"] = int(run[-1])
            event["event_span_steps"] = int(run[-1] - run[0] + 1)
            out.append(event)
    return sorted(
        out, key=lambda w: (str(w["route_key"]), int(w["offense_index"]), str(w["label"]))
    )


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
        width = int(credit[label])
        start_frame = max(min(route_frames), offense - width)
        while start_frame not in frame_to_pos and start_frame < offense:
            start_frame += 1
        out.append(
            {
                "route_key": key,
                "source_scene_path": str(scene),
                "label": label,
                "frame_index": frame,
                "offense_frame": offense,
                "offense_index": frame_to_pos[offense],
                "credit_width": width,
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
    parser.add_argument("--route_npz_root", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--out_jsonl", type=Path, required=True)
    parser.add_argument("--sidecar_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--neighbor_history_mode", default="sim", choices=["recorded", "interpolate", "sim"]
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

    rows = _load_jsonl(args.classified_scenes_jsonl)
    allowed_labels = {label.strip() for label in args.labels.split(",") if label.strip()} or None
    observed = {
        label
        for row in rows
        for label in row.get("labels", [])
        if allowed_labels is None or label in allowed_labels
    }
    credit = _validate_credit_config(args.credit_window_config, observed)
    routes = _route_files(args.route_npz_root)
    windows: list[dict[str, Any]] = []
    for row in rows:
        windows.extend(_resolve_row(row, routes, credit, allowed_labels))
    windows = _collapse_event_windows(windows, args.classified_decluster_steps)
    if not windows:
        raise ValueError("No non-clean credit windows resolved from classified scenes")

    timelines = {key: RouteTimeline(paths, args.sidecar_dir) for key, paths in routes.items()}
    work_units = [
        (timelines[w["route_key"]], int(w["start_index"]), int(w["end_index"]) + 1) for w in windows
    ]
    model, model_args = load_model(args.model_path, args.device)
    model.eval()
    danger_scorer = None
    danger_credit_windows = None
    if args.verify_reproduced_issue:
        if args.reward_config is None or args.threshold_config is None:
            raise ValueError(
                "--verify_reproduced_issue requires --reward_config and --threshold_config"
            )
        danger_scorer = build_reproducer_danger_scorer(
            reward_config=args.reward_config,
            threshold_config=args.threshold_config,
            device=args.device,
            enable_conflict_detector=bool(args.enable_conflict_detector),
            allowed_labels=allowed_labels,
        )
        danger_credit_windows = load_credit_windows(args.credit_window_config)
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
        credit_save_dir=None if args.verify_reproduced_issue else args.out_dir,
        credit_windows=None if args.verify_reproduced_issue else windows,
        verify_credit_windows=windows if args.verify_reproduced_issue else None,
        danger_save_dir=args.out_dir if args.verify_reproduced_issue else None,
        danger_scorer=danger_scorer,
        danger_credit_windows=danger_credit_windows,
        danger_decluster_steps=args.danger_decluster_steps,
    )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        n_rows = 0
        if args.verify_reproduced_issue:
            for saved_dir in sorted(args.out_dir.glob("*_danger_*")):
                label = saved_dir.name.rsplit("_danger_", 1)[-1]
                for scene_path in sorted(saved_dir.glob("credit*.npz")):
                    row = {
                        "scene_path": str(scene_path),
                        "window_dir": str(saved_dir),
                        "credit_scene_stem": scene_path.stem,
                        "label": label,
                        "labels": [label],
                        "variant_kind": "reproduced_credit",
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
                    row["credit_scene_stem"] = scene_path.stem
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                    n_rows += 1
        if n_rows == 0:
            raise ValueError("No credit scenes survived reproduced-issue verification")
    print(f"Wrote credit-window rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()
