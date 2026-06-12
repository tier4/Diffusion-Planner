"""Traffic-light controller for closed-loop replay.

Discovers traffic lights from the lanelet2 map via ``lanelet.trafficLights()``
regulatory elements and runs independent per-group state machines (green →
amber → red → green) for TL groups along the ego's forward route.

Follows the same pattern as ``route_traffic_light_publisher.py``:
  1. Walk the ego route from the closest lanelet forward (not-yet-passed).
  2. For each lanelet with a TL, lazily initialise a per-group-id state
     machine with randomised timing.
  3. Every tick, advance each active group's state machine and write the
     5-dim one-hot into ``scene.map_data.lanes[:, :, 8:13]``.

Perpendicular groups (cross-traffic at the same intersection) are discovered
by proximity and given the opposite phase of the route-direction group.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from scenario_generation.scene_context import MapData, SceneContext

# Channel indices inside the 5-dim traffic-light one-hot block at lane dims
# [8:13] per ``scene_context.MapData``.
TL_GREEN = 0
TL_YELLOW = 1
TL_RED = 2
TL_WHITE = 3
TL_NONE = 4

# Hex colours for the visualisation overlay.
TL_HEX = {
    TL_GREEN: "#22bb22",
    TL_YELLOW: "#ddaa00",
    TL_RED: "#dd2222",
}


class TrafficLightSource(Protocol):
    """A callable-shaped object supplying the traffic-light colour for a given
    lanelet at a given simulation time."""

    def color_for_lanelet(self, lanelet_id: int, t_sec: float) -> int: ...


class AllNoneTLSource:
    """Default source that returns ``TL_NONE`` for every lanelet at every time."""

    def color_for_lanelet(self, lanelet_id: int, t_sec: float) -> int:
        return TL_NONE


# ── Cycle timing defaults (from route_traffic_light_publisher.py) ───────────

AMBER_DURATION_RANGE = (3.0, 5.0)
GREEN_RED_DURATION_RANGE = (10.0, 30.0)
# Perpendicular lights turn GREEN this many seconds after the route light
# turns RED (reverse-sync delay from route_traffic_light_publisher.py:636).
PERP_GREEN_AFTER_DELAY = 2.0


@dataclass
class _GroupState:
    """Per-TL-group independent state machine (mirrors route_traffic_light_publisher.py)."""

    color: int
    last_change_time: float
    duration: float


def _opposite_color(color: int, time_in_color: float) -> int:
    """Return the perpendicular colour using the reverse-sync pattern from
    ``route_traffic_light_publisher.py:_update_crosswalk_light_state``.

    When route is GREEN → perp is RED.
    When route is AMBER → perp is RED.
    When route is RED → perp stays RED for ``PERP_GREEN_AFTER_DELAY``
    seconds, then turns GREEN.
    """
    if color == TL_GREEN:
        return TL_RED
    if color == TL_YELLOW:
        return TL_RED
    # Route is RED.
    if time_in_color < PERP_GREEN_AFTER_DELAY:
        return TL_RED
    return TL_GREEN


class TrafficLightController:
    """Manages traffic light state for closed-loop replay.

    Only controls TL groups along the ego's route (forward from ego) and
    their perpendicular counterparts. Each group runs its own independent
    state machine, lazily initialised the first time it appears.
    """

    def __init__(
        self,
        builder,  # LaneletSceneBuilder
        route_lanelet_ids: list[int],
        seed: int | None = None,
    ) -> None:
        self._rng = random.Random(seed)

        # Full map: {lanelet_id → TL reg_element_id (group_id)}.
        self._ll_to_group: dict[int, int] = builder.get_traffic_light_groups()

        # group_id → set of lanelet_ids
        self._group_to_lls: dict[int, set[int]] = {}
        for ll_id, gid in self._ll_to_group.items():
            self._group_to_lls.setdefault(gid, set()).add(ll_id)

        # group_id → geometric center (for proximity)
        self._group_centers: dict[int, np.ndarray] = {}
        for gid, ll_ids in self._group_to_lls.items():
            pts = []
            for ll_id in ll_ids:
                if ll_id in builder._cache:
                    pts.append(builder._cache[ll_id].raw_centerline.mean(axis=0))
            if pts:
                self._group_centers[gid] = np.mean(pts, axis=0)

        # Build signal groups from shared light_bulbs. Regulatory elements
        # that reference the same physical bulbs ARE the same signal and
        # MUST show the same colour. This replaces the broken heading-
        # based parallel/perpendicular heuristic.
        bulb_map = builder.get_traffic_light_bulb_groups()
        # bulbs_key → set of reg_element_ids that share those bulbs
        self._signal_groups: dict[frozenset, set[int]] = {}
        for reg_id, bulbs in bulb_map.items():
            if not bulbs:
                continue
            # Collect ALL existing groups that share any bulb with this
            # reg (transitive: A∩C and B∩C means A, B, C all merge).
            overlapping_bulbs = bulbs
            overlapping_regs: set[int] = {reg_id}
            keys_to_remove: list[frozenset] = []
            for existing_bulbs, existing_regs in self._signal_groups.items():
                if overlapping_bulbs & existing_bulbs:
                    overlapping_bulbs = overlapping_bulbs | existing_bulbs
                    overlapping_regs |= existing_regs
                    keys_to_remove.append(existing_bulbs)
            for k in keys_to_remove:
                del self._signal_groups[k]
            self._signal_groups[overlapping_bulbs] = overlapping_regs

        # reg_element_id → canonical signal_id (use min reg_id in the group)
        self._group_to_signal: dict[int, int] = {}
        for _bulbs, reg_ids in self._signal_groups.items():
            canonical = min(reg_ids)
            for rid in reg_ids:
                self._group_to_signal[rid] = canonical

        # Route group discovery: ordered list of unique TL group_ids on the
        # ego's route, preserving encounter order.
        self._route_group_ids: list[int] = []
        seen: set[int] = set()
        for ll_id in route_lanelet_ids:
            gid = self._ll_to_group.get(ll_id)
            if gid is not None and gid not in seen:
                self._route_group_ids.append(gid)
                seen.add(gid)

        # Route signal IDs (canonical IDs for the physical lights on the route)
        self._route_signal_ids: set[int] = set()
        for rgid in self._route_group_ids:
            sig = self._group_to_signal.get(rgid, rgid)
            self._route_signal_ids.add(sig)

        # For each route signal, find nearby signals that are DIFFERENT
        # physical lights → those are perpendicular (cross-traffic).
        # Same-signal groups automatically get the same colour via
        # _group_to_signal lookup.
        proximity_m = 50.0
        self._perp_signals: dict[int, set[int]] = {}  # route_signal → {perp signals}
        for rgid in self._route_group_ids:
            rc = self._group_centers.get(rgid)
            if rc is None:
                continue
            route_sig = self._group_to_signal.get(rgid, rgid)
            perp_sigs: set[int] = set()
            for ogid, oc in self._group_centers.items():
                if ogid == rgid:
                    continue
                other_sig = self._group_to_signal.get(ogid, ogid)
                if other_sig == route_sig or other_sig in self._route_signal_ids:
                    continue
                if float(np.linalg.norm(rc - oc)) < proximity_m:
                    perp_sigs.add(other_sig)
            if perp_sigs:
                self._perp_signals.setdefault(route_sig, set()).update(perp_sigs)

        # Inverse: perp_signal → route_signal
        self._perp_to_route_signal: dict[int, int] = {}
        for rsig, perps in self._perp_signals.items():
            for psig in perps:
                self._perp_to_route_signal.setdefault(psig, rsig)

        # Per-group state machines. Lazily initialised in _ensure_state().
        self._group_states: dict[int, _GroupState] = {}

        # Forward-route tracking: index into route_lanelet_ids of the closest
        # lanelet to ego. Updated each tick via _update_forward_index().
        self._route_ll_ids = list(route_lanelet_ids)
        self._route_ll_centers: list[np.ndarray] = []
        for ll_id in self._route_ll_ids:
            if ll_id in builder._cache:
                self._route_ll_centers.append(builder._cache[ll_id].raw_centerline.mean(axis=0))
            else:
                self._route_ll_centers.append(np.zeros(2, dtype=np.float32))
        self._forward_idx: int = 0

        n_total = len(self._ll_to_group)
        n_route = len(self._route_group_ids)
        n_signals = len(self._signal_groups)
        n_route_sigs = len(self._route_signal_ids)
        n_perp_sigs = sum(len(v) for v in self._perp_signals.values())
        print(
            f"  [TrafficLightController] {n_total} lanelets with TLs, "
            f"{n_signals} physical signals, {n_route} groups on route "
            f"({n_route_sigs} signals), {n_perp_sigs} perpendicular signals"
        )

    # ── State machine helpers ─────────────────────────────────────────────

    def _ensure_state(self, gid: int, t: float) -> _GroupState:
        """Lazily initialise a group's state machine (random initial colour)."""
        if gid not in self._group_states:
            initial = self._rng.choice([TL_GREEN, TL_RED])
            self._group_states[gid] = _GroupState(
                color=initial,
                last_change_time=t,
                duration=self._duration_for(initial),
            )
        return self._group_states[gid]

    def _duration_for(self, color: int) -> float:
        if color == TL_YELLOW:
            return self._rng.uniform(*AMBER_DURATION_RANGE)
        return self._rng.uniform(*GREEN_RED_DURATION_RANGE)

    def _advance_group(self, gid: int, t: float) -> None:
        """Advance one group's independent state machine (green→amber→red→green).

        Mirrors ``route_traffic_light_publisher.py:_update_traffic_light_state``.
        """
        state = self._ensure_state(gid, t)
        if t - state.last_change_time < state.duration:
            return
        if state.color == TL_GREEN:
            nxt = TL_YELLOW
        elif state.color == TL_YELLOW:
            nxt = TL_RED
        else:
            nxt = TL_GREEN
        state.color = nxt
        state.last_change_time = t
        state.duration = self._duration_for(nxt)

    # ── Forward-route tracking ────────────────────────────────────────────

    def _update_forward_index(self, ego_xy: np.ndarray) -> None:
        """Move ``_forward_idx`` to the closest route lanelet to ego."""
        if not self._route_ll_centers:
            return
        best_d = float("inf")
        best_i = self._forward_idx
        # Only search forward from current index (no going backwards).
        for i in range(self._forward_idx, len(self._route_ll_centers)):
            d = float(np.linalg.norm(self._route_ll_centers[i] - ego_xy))
            if d < best_d:
                best_d = d
                best_i = i
            elif d > best_d + 50.0:
                break  # past the closest, stop searching
        self._forward_idx = best_i

    def _forward_route_groups(self) -> list[int]:
        """Return route group_ids from the ego's current position forward."""
        seen: set[int] = set()
        result: list[int] = []
        for i in range(self._forward_idx, len(self._route_ll_ids)):
            gid = self._ll_to_group.get(self._route_ll_ids[i])
            if gid is not None and gid not in seen:
                result.append(gid)
                seen.add(gid)
        return result

    # ── Per-step update ───────────────────────────────────────────────────

    def tick(
        self,
        scene: SceneContext,
        sim_time_s: float,
        map_data_ll_ids: list[int],
        ego_xy: np.ndarray | None = None,
    ) -> None:
        """Advance TL states and write into ``scene.map_data.lanes[:, :, 8:13]``.

        Args:
            scene: Current scene context (``map_data.lanes`` is mutated).
            sim_time_s: Simulation time in seconds since replay start.
            map_data_ll_ids: Ordered list of lanelet IDs corresponding to
                rows of ``scene.map_data.lanes``.
            ego_xy: Current ego XY position for forward-route tracking.
        """
        if ego_xy is not None:
            self._update_forward_index(ego_xy)

        # Advance state machines for forward route groups only.
        # Perpendicular groups derive their colour from the route group
        # via _opposite_color() — they have no independent state machine.
        for rgid in self._forward_route_groups():
            self._advance_group(rgid, sim_time_s)

        self._write_to_lanes(scene.map_data, map_data_ll_ids, sim_time_s)

    def color_for_group(self, gid: int, t: float) -> int:
        """Public accessor for the current colour of a signal group."""
        return self._color_for_group(gid, t)

    def _color_for_group(self, gid: int, t: float) -> int:
        """Return the current colour for a group.

        Three cases:
        1. Route group with its own state machine → return state.color
        2. Same-signal group (shares light_bulbs with a route group) →
           same colour as the route group
        3. Perpendicular signal (different physical light, nearby) →
           opposite colour with safety delay
        """
        # Case 1: route group with active state machine
        state = self._group_states.get(gid)
        if state is not None:
            return state.color

        # Resolve to canonical signal ID
        sig = self._group_to_signal.get(gid, gid)

        # Case 2: same physical signal as a route group
        if sig in self._route_signal_ids:
            # Find the route group that shares this signal and use its state
            for rgid in self._route_group_ids:
                if self._group_to_signal.get(rgid, rgid) == sig:
                    rs = self._group_states.get(rgid)
                    if rs is not None:
                        return rs.color
                    break

        # Case 3: perpendicular signal
        parent_rsig = self._perp_to_route_signal.get(sig)
        if parent_rsig is not None:
            # Find the route group driving this parent signal
            for rgid in self._route_group_ids:
                if self._group_to_signal.get(rgid, rgid) == parent_rsig:
                    rs = self._group_states.get(rgid)
                    if rs is not None:
                        time_in_color = t - rs.last_change_time
                        return _opposite_color(rs.color, time_in_color)
                    break

        return TL_NONE

    def _write_to_lanes(
        self,
        map_data: MapData,
        map_data_ll_ids: list[int],
        t: float,
    ) -> None:
        """Write TL one-hot into ``map_data.lanes[:, :, 8:13]``.

        Matches the C++ encoding at ``lane_segments.cpp:267-292``:
        - Lanelet has no TL regulatory element → NO_TRAFFIC_LIGHT (ch 4)
        - Lanelet has TL but we have no state for it → WHITE (ch 3)
        - Lanelet has TL and we know the color → GREEN/YELLOW/RED
        """
        lanes = map_data.lanes  # (N, 20, 33)
        for row_idx, ll_id in enumerate(map_data_ll_ids):
            if row_idx >= lanes.shape[0]:
                break
            gid = self._ll_to_group.get(ll_id)
            if gid is None:
                # No TL regulatory element on this lanelet.
                lanes[row_idx, :, 8:13] = 0.0
                lanes[row_idx, :, 12] = 1.0  # NO_TRAFFIC_LIGHT
                continue
            color = self._color_for_group(gid, t)
            if color == TL_NONE:
                # TL exists but we don't control this group (not on route
                # or perpendicular). C++ encodes this as WHITE ("TL exists
                # but perception has no data").
                lanes[row_idx, :, 8:13] = 0.0
                lanes[row_idx, :, 8 + TL_WHITE] = 1.0
            else:
                lanes[row_idx, :, 8:13] = 0.0
                lanes[row_idx, :, 8 + color] = 1.0

    def write_to_route_lanes(
        self,
        route_lanes: np.ndarray,
        route_ll_ids: list[int],
        t: float,
    ) -> None:
        """Write TL one-hot into a route_lanes array ``[:, :, 8:13]``.

        Same logic as ``_write_to_lanes`` but for the per-agent route tensor
        (shape ``(max_segments, 20, 33)``). Call this after every
        ``_route_to_33dim`` build so the model sees TL state in route context.
        """
        for row_idx, ll_id in enumerate(route_ll_ids):
            if row_idx >= route_lanes.shape[0]:
                break
            gid = self._ll_to_group.get(ll_id)
            if gid is None:
                continue  # no TL regulatory element, keep NO_TRAFFIC_LIGHT
            color = self._color_for_group(gid, t)
            if color == TL_NONE:
                # TL exists but uncontrolled → WHITE (matches C++)
                route_lanes[row_idx, :, 8:13] = 0.0
                route_lanes[row_idx, :, 8 + TL_WHITE] = 1.0
            else:
                route_lanes[row_idx, :, 8:13] = 0.0
                route_lanes[row_idx, :, 8 + color] = 1.0

    # ── Query API (for visualisation) ─────────────────────────────────────

    def get_lanelet_color(self, ll_id: int, t: float = 0.0) -> str | None:
        """Return hex colour string for a TL-affected lanelet, or None."""
        gid = self._ll_to_group.get(ll_id)
        if gid is None:
            return None
        color = self._color_for_group(gid, t)
        return TL_HEX.get(color)

    def get_group_for_lanelet(self, ll_id: int) -> int | None:
        """Return TL group_id for a lanelet, or None if no TL."""
        return self._ll_to_group.get(ll_id)
