"""Traffic-light source Protocol for closed-loop replay.

This module exists purely to reserve the interface for a future session that
wires real traffic-light state into ``replay.py``. Today ``replay.py`` ignores
the source; the map-data lane tensor's 5-dim traffic-light one-hot block at
channel indices ``[8:13]`` (see :class:`scenario_generation.scene_context.MapData`
docstring) stays at its default "none" value.

When a real source is added:

* Read each lanelet's ``trafficLights()`` regulatory elements on the map.
* Run a cycle state-machine (green/amber/red) per group_id.
* Per step, for every lanelet in ``scene.map_data``, write the corresponding
  one-hot into ``scene.map_data.lanes[ll_idx, :, 8:13]`` — same value for all
  20 points of that lanelet.

The writes propagate automatically to both the ego and NPC-as-ego inferences
because they read from the same shared ``MapData``.
"""

from __future__ import annotations

from typing import Protocol


# Channel indices inside the 5-dim traffic-light one-hot block at lane dims
# [8:13] per ``scene_context.MapData``.
TL_GREEN = 0
TL_YELLOW = 1
TL_RED = 2
TL_WHITE = 3
TL_NONE = 4


class TrafficLightSource(Protocol):
    """A callable-shaped object supplying the traffic-light colour for a given
    lanelet at a given simulation time.

    Implementations must be deterministic given their internal state so that
    replays are reproducible with a fixed seed.
    """

    def color_for_lanelet(self, lanelet_id: int, t_sec: float) -> int:
        """Return the active traffic-light channel index for ``lanelet_id``.

        Args:
            lanelet_id: Target lanelet, as stored in ``MapData``'s index-to-id
                mapping.
            t_sec: Simulation time in seconds since replay start.

        Returns:
            One of ``TL_GREEN``, ``TL_YELLOW``, ``TL_RED``, ``TL_WHITE``,
            ``TL_NONE``.
        """
        ...


class AllNoneTLSource:
    """Default source that returns ``TL_NONE`` for every lanelet at every time.

    Matches the pre-existing behaviour of the scenario_generation pipeline
    (lane tensors built by ``_route_to_33dim`` default the traffic block to
    "none"). Use this as the replay default until a real source is wired in.
    """

    def color_for_lanelet(self, lanelet_id: int, t_sec: float) -> int:
        return TL_NONE
