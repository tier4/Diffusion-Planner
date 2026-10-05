"""Shared input/result types for post-hoc closed-loop scenario metrics.

A scenario metric scores one closed-loop window against the recorded human drive it
was cut from, around the window's *anchor* -- the open-loop scene the window was built
for (``scenario.json`` next to the window's frames). It runs after the rollout, from
the per-step ``*.rollout.jsonl`` trace and the window's recorded frames, so it never
touches the simulation hot path.

Two clocks meet here. The live ego is stepped every ``dt`` seconds (index ``k``); the
recorded scene is replayed through a cursor (index ``rec_idx``, one recorded frame per
0.1 s). They drift apart as soon as the model drives slower or faster than the human,
so "after the anchor" is defined on the cursor: ``anchor_step`` is the first sim step
whose replayed frame is at or past the anchor frame -- the moment the live ego is shown
the scene the label describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ClosedLoopScenarioInput:
    """Everything a scenario metric may read, for one closed-loop window.

    World frame is the map frame of the recording. Arrays indexed by ``k`` have one row
    per executed sim step; arrays indexed by ``i`` have one row per recorded frame of
    the window.
    """

    label: str
    window_dir: str
    dt: float

    # Anchor, in recorded-frame index ``i`` (window-local). Several anchors of one label
    # can share a window; the metric is scored for ``anchor_frame``.
    anchor_frame: int
    # First sim step whose replayed frame is at/past ``anchor_frame``; None if the live
    # ego never got that far (a metric then usually reports ``passed=None``).
    anchor_step: int | None

    # Live (closed-loop) ego, index k.
    ego_xy: np.ndarray  # (K, 2)
    ego_yaw: np.ndarray  # (K,)
    ego_speed: np.ndarray  # (K,) m/s, as logged by the rollout
    rec_idx: np.ndarray  # (K,) int, recorded frame the cursor replayed at step k
    collision: np.ndarray  # (K,) bool, ego OBB overlaps a neighbor
    clearance_m: np.ndarray  # (K,) nearest-neighbor OBB distance (inf when none)
    red_light_violation: np.ndarray  # (K,) bool
    terminated: str  # rollout termination reason ("goal", "max_steps", ...)

    # Recorded (human) ego over the window, index i.
    rec_xy: np.ndarray  # (N, 2)
    rec_yaw: np.ndarray  # (N,)
    rec_speed: np.ndarray  # (N,) m/s from pose deltas

    # Lazy loader for the recorded model-input arrays of frame i (ego-centric at the
    # *recorded* pose ``rec_xy[i], rec_yaw[i]``; see ``to_world``). Keys follow the
    # training NPZ: ``lanes``, ``route_lanes``, ``neighbor_agents_past``, ``ego_shape``...
    load_frame: Callable[[int], dict[str, np.ndarray]] = field(repr=False)

    # Optional, for backward compatibility (default: absent).
    # The anchor's event span as recorded frames ``(first, last)``, inclusive and
    # window-local (the registry's ``span_frame_start`` / ``span_frame_stop - 1``); metrics
    # that know spans score over it instead of a fixed horizon after the anchor.
    span_frames: tuple[int, int] | None = None
    # (K,) ego-to-road-border distance per step (the rollout's ``rb_dist_m``; nan where
    # the frame had no border), or None when the trace does not log it.
    road_border_m: np.ndarray | None = None
    # (K,) bool, the collision touches the ego box's rear edge (the replayed agent ran
    # into the ego; a subset of ``collision``), or None when the trace does not log it.
    collision_rear: np.ndarray | None = None

    @property
    def n_steps(self) -> int:
        return len(self.ego_xy)

    @property
    def n_frames(self) -> int:
        return len(self.rec_xy)

    def to_world(self, xy: np.ndarray, frame: int) -> np.ndarray:
        """Map ego-centric points of recorded frame ``frame`` into the world frame."""
        c, s = np.cos(self.rec_yaw[frame]), np.sin(self.rec_yaw[frame])
        xy = np.asarray(xy, dtype=np.float64)
        return np.stack(
            [
                self.rec_xy[frame, 0] + xy[..., 0] * c - xy[..., 1] * s,
                self.rec_xy[frame, 1] + xy[..., 0] * s + xy[..., 1] * c,
            ],
            axis=-1,
        )


@dataclass
class ScenarioResult:
    """One metric's verdict on one window.

    ``passed`` is None when the metric does not apply (e.g. the anchor was never
    reached); ``reason`` then says why, so "not scored" never reads as a pass.
    ``values`` are the numbers the verdict was made from; keep them flat floats so
    they aggregate.
    """

    metric: str
    passed: bool | None
    values: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "passed": self.passed,
            "values": {k: float(v) for k, v in self.values.items()},
            "details": self.details,
            "reason": self.reason,
        }


def path_arclength(path_xy: np.ndarray) -> np.ndarray:
    """Cumulative arc length (m) along a polyline, starting at 0."""
    seg = np.linalg.norm(np.diff(np.asarray(path_xy, dtype=np.float64), axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def project_onto_path(points_xy: np.ndarray, path_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arc length along ``path_xy`` and signed lateral offset (left +) of each point.

    Each point is projected onto its nearest polyline segment (clamped to the
    segment), so a point beyond either end reads the end's arc length; its lateral
    offset is the perpendicular component w.r.t. that segment. Degenerate
    (zero-length) segments are skipped; a path with none left projects everything to
    arc length 0 with the plain distance as the offset.
    """
    path = np.asarray(path_xy, dtype=np.float64)
    pts = np.atleast_2d(np.asarray(points_xy, dtype=np.float64))
    a, b = path[:-1], path[1:]
    d = b - a
    seg_len2 = np.einsum("ij,ij->i", d, d)
    keep = seg_len2 > 1e-12
    if not keep.any():
        return np.zeros(len(pts)), np.linalg.norm(pts - path[0], axis=1)
    a, d, seg_len2 = a[keep], d[keep], seg_len2[keep]
    s0 = path_arclength(path)[:-1][keep]
    rel = pts[:, None, :] - a[None, :, :]  # (P, S, 2)
    t = np.clip(np.einsum("psj,sj->ps", rel, d) / seg_len2, 0.0, 1.0)
    foot = a[None] + t[..., None] * d[None]
    dist = np.linalg.norm(pts[:, None, :] - foot, axis=2)
    j = np.argmin(dist, axis=1)
    rows = np.arange(len(pts))
    seg_len = np.sqrt(seg_len2[j])
    arc = s0[j] + t[rows, j] * seg_len
    # Perpendicular component w.r.t. the nearest segment's direction: a point past an end
    # of the path keeps its sideways offset instead of reading its longitudinal overshoot.
    cross = d[j, 0] * rel[rows, j, 1] - d[j, 1] * rel[rows, j, 0]
    lateral = cross / seg_len
    return arc, lateral


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi
