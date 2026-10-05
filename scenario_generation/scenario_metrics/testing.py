"""Synthetic ``ClosedLoopScenarioInput`` builders for scenario-metric tests."""

from __future__ import annotations

from typing import Callable

import numpy as np

from scenario_generation.scenario_metrics.base import ClosedLoopScenarioInput


def straight_path(
    n: int, speed: float, dt: float = 0.1, heading: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` poses at constant ``speed`` along ``heading`` from the origin."""
    s = np.arange(n) * speed * dt
    xy = np.stack([s * np.cos(heading), s * np.sin(heading)], axis=1)
    return xy, np.full(n, heading)


def speed_profile_path(
    speeds: np.ndarray, dt: float = 0.1, heading: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Poses along ``heading`` integrating a per-step speed profile (first pose at 0)."""
    s = np.concatenate([[0.0], np.cumsum(np.asarray(speeds, dtype=np.float64)[:-1] * dt)])
    xy = np.stack([s * np.cos(heading), s * np.sin(heading)], axis=1)
    return xy, np.full(len(s), heading)


def make_input(
    *,
    label: str,
    ego_xy: np.ndarray,
    ego_yaw: np.ndarray,
    rec_xy: np.ndarray,
    rec_yaw: np.ndarray,
    anchor_frame: int,
    rec_idx: np.ndarray | None = None,
    ego_speed: np.ndarray | None = None,
    rec_speed: np.ndarray | None = None,
    collision: np.ndarray | None = None,
    clearance_m: np.ndarray | None = None,
    red_light_violation: np.ndarray | None = None,
    terminated: str = "goal",
    frames: dict[int, dict[str, np.ndarray]] | Callable[[int], dict[str, np.ndarray]] | None = None,
    dt: float = 0.1,
    span_frames: tuple[int, int] | None = None,
    collision_rear: np.ndarray | None = None,
) -> ClosedLoopScenarioInput:
    """Build an input with sensible defaults.

    ``rec_idx`` defaults to the cursor tracking the recording one-for-one (clipped to
    the last frame), so ``anchor_step == anchor_frame`` unless given otherwise.
    ``frames`` supplies ``load_frame``; missing frames raise ``KeyError``.
    """
    k, n = len(ego_xy), len(rec_xy)
    if rec_idx is None:
        rec_idx = np.minimum(np.arange(k), n - 1)
    rec_idx = np.asarray(rec_idx, dtype=np.int64)

    def speed_of(xy: np.ndarray) -> np.ndarray:
        v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
        return np.concatenate([v, v[-1:]]) if len(v) else np.zeros(len(xy))

    if callable(frames):
        load_frame = frames
    else:
        table = frames or {}

        def load_frame(i: int) -> dict[str, np.ndarray]:
            return table[i]

    reached = np.flatnonzero(rec_idx >= anchor_frame)
    return ClosedLoopScenarioInput(
        label=label,
        window_dir="<synthetic>",
        dt=dt,
        anchor_frame=anchor_frame,
        anchor_step=int(reached[0]) if len(reached) else None,
        ego_xy=np.asarray(ego_xy, dtype=np.float64),
        ego_yaw=np.asarray(ego_yaw, dtype=np.float64),
        ego_speed=speed_of(np.asarray(ego_xy, dtype=np.float64))
        if ego_speed is None
        else np.asarray(ego_speed),
        rec_idx=rec_idx,
        collision=np.zeros(k, dtype=bool)
        if collision is None
        else np.asarray(collision, dtype=bool),
        clearance_m=np.full(k, np.inf)
        if clearance_m is None
        else np.asarray(clearance_m, dtype=np.float64),
        red_light_violation=np.zeros(k, dtype=bool)
        if red_light_violation is None
        else np.asarray(red_light_violation, dtype=bool),
        terminated=terminated,
        rec_xy=np.asarray(rec_xy, dtype=np.float64),
        rec_yaw=np.asarray(rec_yaw, dtype=np.float64),
        rec_speed=speed_of(np.asarray(rec_xy, dtype=np.float64))
        if rec_speed is None
        else np.asarray(rec_speed),
        load_frame=load_frame,
        span_frames=span_frames,
        collision_rear=None if collision_rear is None else np.asarray(collision_rear, dtype=bool),
    )
