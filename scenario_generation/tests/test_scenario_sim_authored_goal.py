"""Tests for the goal a scenario_sim rollout plans towards.

Pure Python: the builder is stubbed, so nothing here needs lanelet2 or a map on disk.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from scenario_generation.scenario_sim_route import _goal_lane_position, resolve_route

_XOSC = """<?xml version="1.0"?>
<OpenSCENARIO>
  <Storyboard>
    <Init>
      <Actions>
        <Private entityRef="ego">
          <PrivateAction>
            <RoutingAction>
              <AcquirePositionAction>
                <Position>
                  <LanePosition roadId="" laneId="{lane}" s="{s}" offset="{offset}"/>
                </Position>
              </AcquirePositionAction>
            </RoutingAction>
          </PrivateAction>
        </Private>
        <Private entityRef="npc">
          <PrivateAction>
            <RoutingAction>
              <AcquirePositionAction>
                <Position>
                  <LanePosition roadId="" laneId="999" s="7.0" offset="0"/>
                </Position>
              </AcquirePositionAction>
            </RoutingAction>
          </PrivateAction>
        </Private>
      </Actions>
    </Init>
  </Storyboard>
</OpenSCENARIO>
"""

_XOSC_NO_GOAL = """<?xml version="1.0"?>
<OpenSCENARIO>
  <Storyboard>
    <Init>
      <Actions>
        <Private entityRef="ego">
          <PrivateAction/>
        </Private>
      </Actions>
    </Init>
  </Storyboard>
</OpenSCENARIO>
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "scenario.xosc"
    p.write_text(text)
    return p


class _StubBuilder:
    """The builder surface ``resolve_route`` uses, with the map's answers set by the test."""

    def __init__(self, *, known_ids=(20,), route=(10, 20), lane_pose=(5.0, 6.0, 0.25)):
        self._known = set(known_ids)
        self._route = list(route)
        self._lane_pose = lane_pose
        self.find_route_calls = 0

    def snap_to_nearest_ll(self, ego_xy, heading_rad=None):
        return 10

    def has_lanelet_id(self, ll_id):
        return ll_id in self._known

    def route_between(self, start_ll, goal_ll):
        return list(self._route)

    def lane_position_pose(self, ll_id, s, offset=0.0):
        return None if self._lane_pose is None else np.array(self._lane_pose, dtype=np.float32)

    def find_route(self, start_ll, min_len_m, deterministic=False):
        self.find_route_calls += 1
        return [10, 11, 12]

    def _route_goal(self, route_ll_ids):
        return np.array([99.0, 99.0, 0.0], dtype=np.float32)


def test_goal_lane_position_keeps_s_and_offset(tmp_path: Path) -> None:
    osc = _write(tmp_path, _XOSC.format(lane=20, s="12.5", offset="-1.25"))
    assert _goal_lane_position(osc, "ego") == (20, 12.5, -1.25)


def test_goal_lane_position_defaults_the_optional_attributes(tmp_path: Path) -> None:
    osc = _write(
        tmp_path,
        '<?xml version="1.0"?><OpenSCENARIO><Storyboard><Init><Actions>'
        '<Private entityRef="ego"><PrivateAction><RoutingAction><AcquirePositionAction>'
        '<Position><LanePosition laneId="7"/></Position>'
        "</AcquirePositionAction></RoutingAction></PrivateAction></Private>"
        "</Actions></Init></Storyboard></OpenSCENARIO>",
    )
    assert _goal_lane_position(osc, "ego") == (7, 0.0, 0.0)


def test_goal_lane_position_none_when_the_scenario_authors_no_goal(tmp_path: Path) -> None:
    assert _goal_lane_position(_write(tmp_path, _XOSC_NO_GOAL), "ego") is None


def test_goal_lane_position_rejects_a_non_finite_attribute(tmp_path: Path) -> None:
    """``float`` parses "nan" and "inf" without complaint; a goal must not."""
    for bad in ("nan", "inf", "-inf"):
        osc = _write(tmp_path, _XOSC.format(lane=20, s=bad, offset="0"))
        assert _goal_lane_position(osc, "ego") is None
        osc = _write(tmp_path, _XOSC.format(lane=20, s="1.0", offset=bad))
        assert _goal_lane_position(osc, "ego") is None


def test_resolve_route_plans_towards_the_authored_lane_position(tmp_path: Path) -> None:
    osc = _write(tmp_path, _XOSC.format(lane=20, s="12.5", offset="0"))
    builder = _StubBuilder()
    route, goal = resolve_route(builder, np.zeros(2), 0.0, osc, "ego")
    assert route == [10, 20]
    np.testing.assert_allclose(goal, [5.0, 6.0, 0.25])
    assert builder.find_route_calls == 0


def test_resolve_route_falls_back_to_find_route_when_the_goal_is_off_the_map(
    tmp_path: Path,
) -> None:
    osc = _write(tmp_path, _XOSC.format(lane=20, s="12.5", offset="0"))
    builder = _StubBuilder(known_ids=())
    route, goal = resolve_route(builder, np.zeros(2), 0.0, osc, "ego")
    assert route == [10, 11, 12]
    assert builder.find_route_calls == 1
    np.testing.assert_allclose(goal, [99.0, 99.0, 0.0])


def test_lane_position_pose_interpolates_arc_length_and_offsets_to_the_left() -> None:
    from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder

    class _Entry:
        raw_centerline = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        cum_arc_lengths = np.array([0.0, 10.0, 20.0])

    builder = object.__new__(LaneletSceneBuilder)
    builder._cache = {5: _Entry()}

    # s names a point along the lanelet, not its end.
    pose = builder.lane_position_pose(5, 12.5)
    np.testing.assert_allclose(pose[:2], [12.5, 0.0], atol=1e-5)
    assert math.isclose(float(pose[2]), 0.0, abs_tol=1e-6)

    # Lanelet2 offsets are positive to the left of the direction of travel.
    left = builder.lane_position_pose(5, 12.5, 2.0)
    np.testing.assert_allclose(left[:2], [12.5, 2.0], atol=1e-5)

    assert builder.lane_position_pose(404, 1.0) is None
