"""Build ``ClosedLoopScenarioInput`` from a rollout trace and its window directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from scenario_generation.route_timeline import RouteTimeline
from scenario_generation.scenario_metrics.base import ClosedLoopScenarioInput

SIM_DT_S = 0.1


def read_rollout(path: Path) -> tuple[list[dict], dict]:
    """Per-step rows and the terminated event of a ``*.rollout.jsonl`` trace."""
    steps, terminated = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            event = row.get("event")
            if event is None:
                steps.append(row)
            elif event == "terminated":
                terminated = row
    return steps, terminated


def load_input(
    rollout_path: Path,
    window_dir: Path,
    *,
    anchor_index: int = 0,
    dt: float = SIM_DT_S,
) -> ClosedLoopScenarioInput:
    """Assemble one materialized window's input (a directory with ``scenario.json``).

    ``anchor_index`` picks among the window's anchors.
    """
    window_dir = Path(window_dir)
    meta = json.loads((window_dir / "scenario.json").read_text())
    return load_input_from_frames(
        rollout_path,
        sorted(window_dir.glob("*.npz")),
        window_dir,
        label=str(meta["eval_label"]),
        anchor_frame=int(meta["anchors"][anchor_index]["frame_offset"]),
        dt=dt,
    )


def load_input_from_frames(
    rollout_path: Path,
    npz_paths: list[Path],
    sidecar_dir: Path,
    *,
    label: str,
    anchor_frame: int,
    dt: float = SIM_DT_S,
    span_frames: tuple[int, int] | None = None,
) -> ClosedLoopScenarioInput:
    """Assemble one window's input from its recorded frames (e.g. a registry segment).

    ``span_frames`` is the anchor's event span (see ``ClosedLoopScenarioInput``).

    The recorded frames are read the same way the rollout read them (``RouteTimeline``
    over the same NPZ + sidecars), so recorded index ``i`` here is the cursor's
    ``rec_idx``.
    """
    window_dir = Path(sidecar_dir)
    return load_input_from_timeline(
        rollout_path,
        RouteTimeline(list(npz_paths), sidecar_dir=window_dir),
        window_dir=window_dir,
        label=label,
        anchor_frame=anchor_frame,
        dt=dt,
        span_frames=span_frames,
    )


def load_input_from_timeline(
    rollout_path: Path,
    timeline: RouteTimeline,
    *,
    window_dir: Path,
    label: str,
    anchor_frame: int,
    dt: float = SIM_DT_S,
    span_frames: tuple[int, int] | None = None,
    load_frame: Callable[[int], dict[str, np.ndarray]] | None = None,
) -> ClosedLoopScenarioInput:
    """Assemble one window's input from the timeline the rollout replayed.

    ``timeline`` must cover exactly the window the trace was run on (its index ``i`` is
    the cursor's ``rec_idx``); any ``RouteTimeline``-like source with ``poses``,
    ``speeds`` and ``npz`` works. ``load_frame`` replaces ``timeline.npz`` when the
    source's frames need converting to the NPZ field layout the metrics read.
    """
    steps, terminated = read_rollout(Path(rollout_path))
    if not steps:
        raise ValueError(f"rollout has no steps: {rollout_path}")
    ego_xy = np.array([s["ego"] for s in steps], dtype=np.float64)
    ego_yaw = np.array([s["yaw"] for s in steps], dtype=np.float64)
    ego_speed = np.array([s["speed"] for s in steps], dtype=np.float64)
    rec_idx = np.array([s["rec_idx"] for s in steps], dtype=np.int64)
    collision = np.array([bool(s.get("collision", False)) for s in steps])
    clearance = np.array(
        [float("inf") if s.get("clearance_m") is None else float(s["clearance_m"]) for s in steps]
    )
    collision_rear = np.array([bool(s.get("collision_rear", False)) for s in steps])
    red = np.array([bool(s.get("red_light_violation", False)) for s in steps])
    road_border = None
    if any("rb_dist_m" in s for s in steps):
        road_border = np.array(
            [np.nan if s.get("rb_dist_m") is None else float(s["rb_dist_m"]) for s in steps]
        )
    reached = np.flatnonzero(rec_idx >= anchor_frame)

    return ClosedLoopScenarioInput(
        label=label,
        window_dir=str(window_dir),
        dt=dt,
        anchor_frame=anchor_frame,
        anchor_step=int(reached[0]) if len(reached) else None,
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        ego_speed=ego_speed,
        rec_idx=rec_idx,
        collision=collision,
        clearance_m=clearance,
        red_light_violation=red,
        terminated=str(terminated.get("reason", "unknown")),
        rec_xy=timeline.poses[:, :2].copy(),
        rec_yaw=timeline.poses[:, 2].copy(),
        rec_speed=timeline.speeds.copy(),
        load_frame=timeline.npz if load_frame is None else load_frame,
        span_frames=None if span_frames is None else (int(span_frames[0]), int(span_frames[1])),
        road_border_m=road_border,
        collision_rear=collision_rear,
    )
