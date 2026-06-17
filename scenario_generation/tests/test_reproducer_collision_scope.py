"""score_step must count collisions with MOVING neighbors AND rear-end hits.

The avoidance reward's ``compute_static_collision_penalty`` deliberately scores
only *stopped* neighbors and filters out rear-end contacts (the ego being struck
from behind). For mining we want the opposite: a raw oriented-bounding-box overlap
check against EVERY neighbor, with no stopped-only / ego-speed / direction gate, so
that a model that drifts into a moving car or gets rear-ended is flagged.
"""

import numpy as np

from scenario_generation.reproducer_rollout import score_step

EGO_SHAPE = np.array([4.76, 7.24, 2.29], dtype=np.float32)  # wheelbase, length, width


def _neighbors(rows: list[list[float]]) -> np.ndarray:
    """Build a (N, 11) live-ego-frame neighbor array [x,y,cos,sin,vx,vy,w,l,oh3]."""
    a = np.zeros((len(rows), 11), dtype=np.float32)
    for i, r in enumerate(rows):
        a[i, :8] = r
        a[i, 8] = 1.0  # vehicle one-hot
    return a


def test_rear_end_moving_neighbor_is_a_collision():
    """Ego moving forward, a moving neighbor overlaps it from BEHIND (x < 0)."""
    rear = _neighbors([[-3.0, 0.0, 1.0, 0.0, 5.0, 0.0, 2.0, 5.0]])  # vx=5 => moving
    clr, collision, m = score_step(rear, EGO_SHAPE, ego_speed=8.0, device="cpu")
    assert m == 1
    assert collision is True, f"rear-end moving overlap not flagged (clr={clr:.3f})"
    assert clr < 0.5


def test_moving_neighbor_ahead_overlap_is_a_collision():
    ahead = _neighbors([[4.0, 0.0, 1.0, 0.0, 6.0, 0.0, 2.0, 4.0]])  # moving, overlaps
    clr, collision, _ = score_step(ahead, EGO_SHAPE, ego_speed=8.0, device="cpu")
    assert collision is True, f"moving overlap not flagged (clr={clr:.3f})"


def test_far_apart_is_not_a_collision():
    far = _neighbors([[30.0, 12.0, 1.0, 0.0, 8.0, 0.0, 2.0, 4.0]])
    clr, collision, _ = score_step(far, EGO_SHAPE, ego_speed=8.0, device="cpu")
    assert collision is False
    assert clr > 1.0


def test_collision_counted_even_when_ego_stopped():
    """No ego-speed gate: an overlap counts regardless of ego speed."""
    overlap = _neighbors([[4.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 4.0]])
    _, collision, _ = score_step(overlap, EGO_SHAPE, ego_speed=0.0, device="cpu")
    assert collision is True


def test_no_valid_neighbors_returns_inf():
    clr, collision, m = score_step(np.zeros((3, 11), np.float32), EGO_SHAPE, 5.0, "cpu")
    assert m == 0 and collision is False and clr == float("inf")
