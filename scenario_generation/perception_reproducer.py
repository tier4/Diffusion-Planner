"""PerceptionReproducer: autoware-faithful cursor over a RouteTimeline.

Replicates Autoware ``planning_debug_tools/perception_reproducer``: the recorded
perception is replayed **keyed on the live ego pose**, not wall-clock. Each step:

1. If the ego has moved more than ``search_radius`` since the queue was last
   built, rebuild a queue of recorded frames whose ego world xy is within
   ``search_radius`` of the live ego, **ordered chronologically** (by frame
   index = recorded time), excluding a **cool-down** set of recently-used frames
   (TTL ``cool_down_sec``). Empty neighborhood -> fall back to the nearest frame.
2. Otherwise keep consuming the existing queue, so the recorded scene plays
   forward in time even while the ego is stopped (the red-light case: cross
   traffic / pedestrians keep moving).
3. A **speed-gap guard** repeats the previous frame when the recorded ego was
   much faster here than the live ego (avoids teleporting objects forward).

The cursor snaps to whole 10 Hz frames (both the log and the sim are 10 Hz). It
only decides *which recorded frame* to reproduce; transforming that frame's
neighbors/map onto the live ego, optional per-track interpolation, and scoring
happen in the rollout.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from scenario_generation.perf_timer import Timers
from scenario_generation.route_timeline import RouteTimeline

# Autoware defaults (planning_debug_tools/perception_reproducer).
DEFAULT_SEARCH_RADIUS_M = 1.5  # -r ; 0 => always publish the single nearest frame
DEFAULT_COOL_DOWN_SEC = 80.0  # -c ; must exceed the ego's max stopping time
# Speed-gap guard: repeat the last frame instead of teleporting objects when the
# recorded ego was much faster here than the live ego.
_SPEED_GAP_RATIO = 2.0
_SPEED_GAP_MIN_REC = 3.0  # m/s


class PerceptionReproducer:
    def __init__(
        self,
        timeline: RouteTimeline,
        search_radius: float = DEFAULT_SEARCH_RADIUS_M,
        cool_down_sec: float = DEFAULT_COOL_DOWN_SEC,
        timers: Timers | None = None,
    ) -> None:
        self.tl = timeline
        self.search_radius = float(search_radius)
        self._base_search_radius = float(
            search_radius
        )  # nominal radius to restore after unsticking
        self.cool_down_sec = float(cool_down_sec)
        self.timers = timers or Timers()
        self.reset()

    def set_search_radius(self, radius: float) -> None:
        """Change the neighborhood radius and force a queue rebuild on the next ``step``.

        Used by the rollout's unstick escalation: temporarily widening the radius lets the
        cursor reach recorded frames further ahead (where a phantom lead/blocker has cleared)
        so the model can proceed on its own, then restore the nominal radius once it moves.
        ``reset`` does NOT touch the radius, so a widened radius persists across a teleport
        unless explicitly restored.
        """
        radius = float(radius)
        if radius == self.search_radius:
            return
        self.search_radius = radius
        self._queue.clear()
        self._last_seq_pos = None  # force a neighborhood rebuild next step at the new radius

    def widen(self, mult: float) -> None:
        """Widen the search radius to ``mult`` x the nominal (base) radius."""
        self.set_search_radius(self._base_search_radius * float(mult))

    def restore_radius(self) -> None:
        """Restore the nominal (base) search radius (no-op if already there)."""
        self.set_search_radius(self._base_search_radius)

    def reset(self, start_idx: int = 0) -> None:
        self._queue: deque[int] = deque()
        self._cool_down: deque[tuple[int, float]] = deque()  # (frame_idx, sim_time_used)
        self._last_seq_pos: np.ndarray | None = None
        self._last_idx: int = start_idx
        self.max_idx_reached: int = start_idx

    def step(self, sim_xy: np.ndarray, sim_speed: float, sim_time: float) -> int:
        """Return the recorded-frame index to reproduce at this sim tick.

        Args:
            sim_xy: (2,) live ego world position.
            sim_speed: live ego speed (m/s) — for the speed-gap guard.
            sim_time: elapsed sim time (s) — drives the cool-down TTL.
        """
        with self.timers("cursor_step"):
            sim_xy = np.asarray(sim_xy, dtype=np.float64)[:2]

            moved = (
                np.inf
                if self._last_seq_pos is None
                else float(np.linalg.norm(sim_xy - self._last_seq_pos))
            )
            if self.search_radius <= 0.0:
                # Degenerate mode: always the single nearest recorded frame.
                idx = self.tl.nearest(sim_xy)
                self._last_idx = idx
                self.max_idx_reached = max(self.max_idx_reached, idx)
                return idx

            if moved > self.search_radius or not self._queue:
                self._last_seq_pos = sim_xy.copy()
                nearby = list(self.tl.query_radius(sim_xy, self.search_radius))
                if not nearby:
                    nearby = [self.tl.nearest(sim_xy)]
                # Expire cool-down entries past their TTL.
                while self._cool_down and (sim_time - self._cool_down[0][1]) > self.cool_down_sec:
                    self._cool_down.popleft()
                cooling = {i for i, _ in self._cool_down}
                # Chronological order == ascending frame index (frame_indices is sorted).
                self._queue = deque(sorted(i for i in nearby if i not in cooling))

            repeat = len(self._queue) == 0
            if not repeat:
                front = self._queue[0]
                rec_dist = float(np.linalg.norm(sim_xy - self.tl.poses[front, :2]))
                rec_speed = float(self.tl.speeds[front])
                repeat = (
                    rec_speed > sim_speed * _SPEED_GAP_RATIO
                    and rec_speed > _SPEED_GAP_MIN_REC
                    and rec_dist > self.search_radius
                )

            if repeat:
                idx = self._last_idx
            else:
                idx = self._queue.popleft()
                self._last_idx = idx
                self._cool_down.append((idx, sim_time))

            self.max_idx_reached = max(self.max_idx_reached, idx)
            return idx
