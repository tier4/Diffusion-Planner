"""PerceptionReproducer: autoware-faithful cursor over a RouteTimeline.

Replicates Autoware ``planning_debug_tools/perception_reproducer``: the recorded
perception is replayed **keyed on the live ego pose**, not wall-clock. Each step:

1. If the ego has moved more than ``search_radius`` since the queue was last
   built, rebuild a queue of recorded frames whose ego world xy is within
   ``search_radius`` of the live ego, **ordered chronologically** (by frame
   index = recorded time), excluding a **cool-down** set of recently-used frames
   (TTL ``cool_down_sec``). Frames earlier than the furthest frame already reached
   are never re-entered, so a closed-loop ego that drifts near an old part of the
   route cannot rewind the replay. Empty neighborhood -> fall back to the nearest
   non-rewinding frame.
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

# Heading gate threshold (rad): fixed; only the on/off switch is configurable.
# A candidate recorded frame is only eligible if its recorded ego yaw is within
# +/- this of the live ego yaw. Stops the xy-only KDTree from grabbing a future
# opposite-heading frame (U-turn / self-crossing route) that would re-center the
# scene rotated ~180 deg. 90 deg is lenient enough for normal curves/junctions.
YAW_GATE_RAD = np.pi / 2.0


def _wrap_to_pi(a: float) -> float:
    """Wrap an angle (rad) to (-pi, pi]."""
    return (float(a) + np.pi) % (2.0 * np.pi) - np.pi


class PerceptionReproducer:
    def __init__(
        self,
        timeline: RouteTimeline,
        search_radius: float = DEFAULT_SEARCH_RADIUS_M,
        cool_down_sec: float = DEFAULT_COOL_DOWN_SEC,
        timers: Timers | None = None,
        yaw_gate: bool = True,
    ) -> None:
        self.tl = timeline
        self.search_radius = float(search_radius)
        self._base_search_radius = float(
            search_radius
        )  # nominal radius to restore after unsticking
        self.cool_down_sec = float(cool_down_sec)
        # On/off only; threshold is the fixed ``YAW_GATE_RAD`` (default ON).
        self.yaw_gate = bool(yaw_gate)
        self.timers = timers or Timers()
        # Cumulative route-level telemetry: how long the cursor spent advancing vs
        # republishing. Persist across ``reset`` (only ``__init__`` clears them) so a
        # whole-route summary isn't wiped on a teleport.
        self.normal_steps: int = 0
        self.repeat_steps: int = 0
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
        # --- Observable per-episode run state ---------------------------------------
        # The cursor exposes only its own two intrinsic states (the escalation actions
        # expand/teleport are owned by the rollout, not the frame picker):
        #   "normal" : consuming the queue, perception advancing with the ego
        #   "repeat" : queue drained / speed-gap guard -> republishing the last frame
        # Both are persistent duration states set by ``step`` (step-based, fixed 10 Hz;
        # seconds are ``steps * DT`` for the caller). Reset per-episode; the cumulative
        # normal/repeat totals live in __init__.
        self.last_was_repeat: bool = False
        self.state: str = "normal"
        # consecutive ticks the base state (normal/repeat) has held (reset on switch)
        self.state_run_steps: int = 0

    def step(
        self,
        sim_xy: np.ndarray,
        sim_speed: float,
        sim_time: float,
        sim_yaw: float | None = None,
    ) -> int:
        """Return the recorded-frame index to reproduce at this sim tick.

        Args:
            sim_xy: (2,) live ego world position.
            sim_speed: live ego speed (m/s) — for the speed-gap guard.
            sim_time: elapsed sim time (s) — drives the cool-down TTL.
            sim_yaw: live ego heading (rad). Used by the heading gate when ``yaw_gate`` is
                on (default). Production pose-mode always passes this; omit only in tests
                that don't care about yaw (gate is then a no-op for that step).
        """
        with self.timers("cursor_step"):
            sim_xy = np.asarray(sim_xy, dtype=np.float64)[:2]

            moved = (
                np.inf
                if self._last_seq_pos is None
                else float(np.linalg.norm(sim_xy - self._last_seq_pos))
            )
            if self.search_radius <= 0.0:
                # Degenerate mode: nearest recorded frame, but still never rewind
                # once this cursor has reached a later recorded frame. First pick of an
                # index is ``normal``; republishing the same index (ego not moving) is
                # ``repeat`` so unstick can escalate.
                idx = max(self.tl.nearest(sim_xy), self.max_idx_reached)
                repeat = self.state_run_steps > 0 and idx == self._last_idx
                self._last_idx = idx
                self.max_idx_reached = max(self.max_idx_reached, idx)
                self._update_base_state(repeat=repeat)
                return idx

            if moved > self.search_radius or not self._queue:
                self._last_seq_pos = sim_xy.copy()
                nearby = list(self.tl.query_radius(sim_xy, self.search_radius))
                if not nearby:
                    nearest = self.tl.nearest(sim_xy)
                    nearby = [nearest if nearest >= self.max_idx_reached else self.max_idx_reached]
                # Expire cool-down entries past their TTL.
                while self._cool_down and (sim_time - self._cool_down[0][1]) > self.cool_down_sec:
                    self._cool_down.popleft()
                cooling = {i for i, _ in self._cool_down}
                # Chronological order == ascending frame index (frame_indices is sorted).
                cand = [i for i in nearby if i >= self.max_idx_reached and i not in cooling]
                if self.yaw_gate and sim_yaw is not None and cand:
                    # Heading gate (default ON): drop candidates whose recorded heading is >
                    # YAW_GATE_RAD off the live ego (future U-turn / self-crossing frames).
                    # If EVERY forward candidate is wrong-heading, leave the queue empty ->
                    # ``repeat`` (hold the last good, correct-heading frame) rather than fall
                    # back to a wrong-heading pick.
                    cand = [
                        i
                        for i in cand
                        if abs(_wrap_to_pi(self.tl.poses[i, 2] - sim_yaw)) <= YAW_GATE_RAD
                    ]
                self._queue = deque(sorted(cand))

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
            self._update_base_state(repeat)
            return idx

    def _update_base_state(self, repeat: bool) -> None:
        """Refresh the cursor's observable state each ``step``: "repeat" when republishing
        the last frame, else "normal". Tracks how long the current state has held
        (state_run_steps) and the cumulative steps in each (for the route summary). The
        escalation actions (widen radius / teleport) are owned and counted by the rollout.
        """
        state = "repeat" if repeat else "normal"
        # run-length: keep counting if the state is unchanged, else restart at 1
        self.state_run_steps = self.state_run_steps + 1 if state == self.state else 1
        self.state = state
        self.last_was_repeat = repeat
        if repeat:
            self.repeat_steps += 1
        else:
            self.normal_steps += 1
