"""Resolve the ego route a scenario declares.

Interim: the route is computed in Python via ``LaneletSceneBuilder`` (shortestPath /
find_route) rather than by the mission planner, so it is not guaranteed to match the route
the production stack would resolve for the same scenario.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder


def _ego_action_scopes(root: ET.Element, ego_name: str):
    """The elements whose private actions belong to the ego: its ``Init`` block and every
    ``ManeuverGroup`` listing it as an actor."""
    for private in root.iter("Private"):
        if private.get("entityRef") == ego_name:
            yield private
    for group in root.iter("ManeuverGroup"):
        if any(ref.get("entityRef") == ego_name for ref in group.iter("EntityRef")):
            yield group


def _goal_lane_position(osc_path: str | Path, ego_name: str) -> tuple[int, float, float] | None:
    """``(lanelet id, s, offset)`` from the ego's own ``AcquirePositionAction`` LanePosition,
    if the scenario authors one. SSv2 lane ids are lanelet ids.

    ``s`` is carried, not dropped: the scenarios judge arrival against this very point, so a
    goal taken as the lanelet's end instead aims the planner past what is being measured.
    """
    root = ET.parse(str(osc_path)).getroot()
    for scope in _ego_action_scopes(root, ego_name):
        for routing in scope.iter("AcquirePositionAction"):
            lp = routing.find(".//LanePosition")
            if lp is not None and "laneId" in lp.attrib:
                try:
                    ll_id = int(lp.attrib["laneId"])
                    arc = float(lp.attrib.get("s", 0.0))
                    offset = float(lp.attrib.get("offset", 0.0))
                except ValueError:
                    return None
                # ``float`` accepts "nan" and "inf", which would reach the planner as a goal
                # rather than as the malformed attribute they are.
                if not (math.isfinite(arc) and math.isfinite(offset)):
                    return None
                return ll_id, arc, offset
    return None


def resolve_route(
    builder: LaneletSceneBuilder,
    ego_xy: np.ndarray,
    ego_heading: float,
    osc_path: str | Path,
    ego_name: str,
    *,
    min_len_m: float = 120.0,
) -> tuple[list[int], np.ndarray]:
    """Ordered lanelet ids for the ego route, and the goal pose to plan towards.

    Snaps the sim's actual start pose to a lanelet, then takes the shortest path to the
    scenario's goal when one is authored and reachable, else a forward route of at least
    ``min_len_m``. That fallback is resolved deterministically: a branch picked at random
    would make route_lanes, progress and the road-border geometry differ between runs of the
    same scenario, which is not something an evaluation may leave to chance.

    The goal pose is the authored ``LanePosition`` when there is one, so the planner aims at
    the point the scenario judges arrival against. It falls back to the route's last centreline
    point only when the scenario authors no goal, or the map does not carry that lanelet.
    """
    start_ll = builder.snap_to_nearest_ll(ego_xy, heading_rad=ego_heading)
    if start_ll is None:
        raise RuntimeError(f"Could not snap ego start pose {ego_xy} to any lanelet")
    authored = _goal_lane_position(osc_path, ego_name)
    if authored is not None and builder.has_lanelet_id(authored[0]):
        goal_ll, s, offset = authored
        route = builder.route_between(start_ll, goal_ll)
        if route:
            # ``has_lanelet_id`` established the lanelet is cached, which is the only thing
            # ``lane_position_pose`` needs to answer.
            return route, builder.lane_position_pose(goal_ll, s, offset)
        print(
            f"  [scenario_sim][WARN] goal lanelet {goal_ll} unreachable from {start_ll}; "
            "falling back to find_route"
        )
    route = builder.find_route(start_ll, min_len_m, deterministic=True)
    return route, builder._route_goal(route)
