import math

import numpy as np

from scenario_generation.reproducer_rollout import _goal_xy_from_npz_goal


class _TinyTimeline:
    def __init__(self):
        self.poses = np.array(
            [
                [10.0, 20.0, math.pi / 2.0],
                [11.0, 20.0, math.pi / 2.0],
                [12.0, 20.0, math.pi / 2.0],
            ],
            dtype=np.float64,
        )

    def npz(self, _idx):
        return {"goal_pose": np.array([4.0, 1.5, 1.0], dtype=np.float32)}


def test_route_goal_recovers_world_xy_from_npz_goal_pose():
    goal = _goal_xy_from_npz_goal(_TinyTimeline(), idx=0, fallback_idx=2)

    np.testing.assert_allclose(goal, np.array([8.5, 24.0]), atol=1e-6)


def test_route_goal_falls_back_to_segment_endpoint_when_npz_goal_is_zero():
    class ZeroGoalTimeline(_TinyTimeline):
        def npz(self, _idx):
            return {"goal_pose": np.zeros(4, dtype=np.float32)}

    goal = _goal_xy_from_npz_goal(ZeroGoalTimeline(), idx=0, fallback_idx=2)

    np.testing.assert_allclose(goal, np.array([12.0, 20.0]), atol=1e-6)
