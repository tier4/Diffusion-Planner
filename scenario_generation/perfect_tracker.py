"""Perfect trajectory tracker: the vehicle drives exactly what the model plans.

Every step the vehicle lands on the first point of the predicted trajectory, heading along the path.
No dynamics, no limits. The MPC tracker (physically limited, follows with some error) lives in
``mpc_tracker.py``.
"""

from __future__ import annotations

import math

import numpy as np

# Distance along the reference over which the heading is read. Below it, point spacing at low speed
# (centimetres) makes the direction noisy, so the current heading is kept.
MIN_HEADING_DISTANCE_M = 0.5


def place_on_trajectory(
    current_pose: np.ndarray, ref_xy: np.ndarray, dt: float
) -> tuple[np.ndarray, float]:
    """Exact perfect tracking: the vehicle ends the step on the reference's first point.

    The heading is the direction the reference runs from that point, read over at least
    ``MIN_HEADING_DISTANCE_M``. It is deliberately not the model's heading output: the vehicle's
    reference point is the rear axle, which moves in the direction the vehicle points, so a vehicle
    that follows the positions exactly points along the path. When the reference is shorter than
    that distance the current heading is kept.

    Args:
        current_pose: (3,) [x, y, yaw] in world frame.
        ref_xy: (N, 2) reference positions in world frame, first point one ``dt`` ahead.
        dt: timestep (seconds).

    Returns:
        new_pose: (3,) [x, y, yaw].
        speed: distance moved this step / dt.
    """
    x, y, yaw = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
    if len(ref_xy) < 1:
        return np.array([x, y, yaw], dtype=np.float64), 0.0
    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    tx, ty = float(ref_xy[0, 0]), float(ref_xy[0, 1])
    far = np.flatnonzero(np.linalg.norm(ref_xy - ref_xy[0], axis=1) >= MIN_HEADING_DISTANCE_M)
    heading = math.atan2(ref_xy[far[0], 1] - ty, ref_xy[far[0], 0] - tx) if len(far) else yaw
    speed = math.hypot(tx - x, ty - y) / dt
    return np.array([tx, ty, heading], dtype=np.float64), speed


class PerfectTracker:
    """Perfect trajectory tracking: every step the vehicle lands exactly on the reference's first
    point, heading along the path (see :func:`place_on_trajectory`). No dynamics, no limits.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        # Parallel to MPCTracker.last_*. No steering model: last_steering stays 0.0 and
        # last_yaw_rate is the heading change per step.
        self.last_accel: float = 0.0
        self.last_yaw_rate: float = 0.0
        self.last_steering: float = 0.0
        self._prev_speed: float = 0.0

    def track(
        self,
        x0: np.ndarray,
        ref_world: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Advance one step along the reference trajectory.

        Args:
            x0: (4,) [x, y, yaw, v] current state in world frame.
            ref_world: (N, 2+) reference [x, y, ...] in world frame; only positions are used.

        Returns:
            new_pos: (3,) [x, y, yaw] after one dt step.
            new_speed: scalar speed after one dt step.
        """
        ref_world = np.asarray(ref_world)
        ref_xy = ref_world[:, :2] if len(ref_world) else np.zeros((0, 2))
        new_pos, speed = place_on_trajectory(np.asarray(x0)[:3], ref_xy, self.dt)
        dh = (float(new_pos[2]) - float(x0[2]) + math.pi) % (2 * math.pi) - math.pi
        self.last_yaw_rate = dh / self.dt
        v_prev = float(x0[3]) if len(x0) > 3 else self._prev_speed
        self.last_accel = (speed - v_prev) / self.dt
        self._prev_speed = speed
        self.last_steering = 0.0
        return new_pos.astype(np.float32), speed

    def reset(self):
        """No-op (no internal state to clear)."""
        pass
