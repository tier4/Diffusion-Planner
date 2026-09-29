import numpy as np
import pytest

from scenario_generation.scenario_metrics import registry
from scenario_generation.scenario_metrics.base import (
    ScenarioResult,
    path_arclength,
    project_onto_path,
)
from scenario_generation.scenario_metrics.testing import make_input, straight_path


def test_project_onto_path_arc_and_signed_lateral():
    path = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    arc, lat = project_onto_path(
        np.array([[5.0, 1.0], [5.0, -2.0], [11.0, 5.0], [-3.0, 0.0]]), path
    )
    np.testing.assert_allclose(arc, [5.0, 5.0, 15.0, 0.0])
    # (-3, 0) lies on the path's backward extension: no sideways offset.
    np.testing.assert_allclose(lat, [1.0, -2.0, -1.0, 0.0])


def test_project_onto_path_skips_degenerate_segments():
    path = np.array([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
    arc, lat = project_onto_path(np.array([[2.0, 1.0]]), path)
    np.testing.assert_allclose(arc, [2.0])
    np.testing.assert_allclose(lat, [1.0])
    np.testing.assert_allclose(path_arclength(path), [0.0, 0.0, 4.0])


def test_anchor_step_follows_the_cursor_not_the_clock():
    xy, yaw = straight_path(50, 5.0)
    # Cursor lingers on frame 3 for ten steps: the anchor (frame 10) is reached later.
    rec_idx = np.minimum(np.concatenate([np.arange(4), np.full(10, 3), np.arange(4, 40)]), 49)
    inp = make_input(
        label="x", ego_xy=xy, ego_yaw=yaw, rec_xy=xy, rec_yaw=yaw, anchor_frame=10, rec_idx=rec_idx
    )
    assert inp.anchor_step == 20
    never = make_input(
        label="x", ego_xy=xy[:5], ego_yaw=yaw[:5], rec_xy=xy, rec_yaw=yaw, anchor_frame=10
    )
    assert never.anchor_step is None


def test_to_world_uses_the_recorded_pose():
    xy = np.array([[10.0, 5.0], [11.0, 5.0]])
    yaw = np.array([np.pi / 2, np.pi / 2])
    inp = make_input(label="x", ego_xy=xy, ego_yaw=yaw, rec_xy=xy, rec_yaw=yaw, anchor_frame=0)
    np.testing.assert_allclose(inp.to_world(np.array([[1.0, 0.0]]), 0), [[10.0, 6.0]], atol=1e-12)


def test_unknown_label_is_not_applicable_and_duplicate_registration_raises():
    xy, yaw = straight_path(5, 1.0)
    result = registry.score(
        make_input(
            label="no_such_label", ego_xy=xy, ego_yaw=yaw, rec_xy=xy, rec_yaw=yaw, anchor_frame=0
        )
    )
    assert isinstance(result, ScenarioResult) and result.passed is None

    registry.register("__dup_test__")(lambda inp: None)
    try:
        with pytest.raises(ValueError):
            registry.register("__dup_test__")(lambda inp: None)
    finally:
        registry.METRICS.pop("__dup_test__")
