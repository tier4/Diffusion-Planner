"""Pretty-print the contents of a saved ``Route`` pickle.

Usage::

    python -m scenario_generation.tools.inspect_route path/to/my_route.pkl

Prints:

* map path the route was authored against
* start / goal world poses and snapped lanelet ids
* waypoints (ordered) with snapped lanelet ids
* resolved lanelet path length and first/last 5 lanelet ids

Does not require the map or a model — pure pickle inspection.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from scenario_generation.route import Route


def _fmt_pose(pose) -> str:
    x, y, h = pose
    return f"({x:.2f}, {y:.2f}, {math.degrees(float(h)):.1f}°)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path, help="Route pickle file")
    args = parser.parse_args()

    route = Route.load(args.route)

    print(f"Route from {args.route}")
    print(f"  map_path:          {route.map_path}")
    print(f"  start_pose:        {_fmt_pose(route.start_pose)}  "
          f"lanelet_id={route.start_lanelet_id}")
    print(f"  goal_pose:         {_fmt_pose(route.goal_pose)}  "
          f"lanelet_id={route.goal_lanelet_id}")
    print(f"  waypoints:         {route.num_waypoints()}")
    for i, (wp, wl) in enumerate(zip(route.waypoint_poses, route.waypoint_lanelet_ids), 1):
        print(f"    #{i}: {_fmt_pose(wp)}  lanelet_id={wl}")
    if route.is_resolved():
        ids = route.route_lanelet_ids
        head = ids[:5]
        tail = ids[-5:] if len(ids) > 5 else []
        print(f"  route_lanelet_ids: {len(ids)} lanelets")
        print(f"    first 5: {head}")
        if tail and tail != head:
            print(f"    last 5:  {tail}")
    else:
        print("  route_lanelet_ids: *unresolved* "
              "(replay will fall back to find_route)")


if __name__ == "__main__":
    main()
