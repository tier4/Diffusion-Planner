"""Core engine: Lanelet2 map -> SceneContext generation.

Loads a Lanelet2 map, builds a routing graph, and generates synthetic driving
scenes with ego + N neighbors placed on lanes with feasible routes and history.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from scenario_generation.scene_context import Agent, AgentType, MapData, SceneContext

# lanelet2 requires ROS/Autoware Python paths
_ROS_FALLBACK_PATHS = ["/opt/ros/humble/lib/python3.10/site-packages"]
_AUTOWARE_DIR = os.environ.get("AUTOWARE_INSTALL", os.path.expanduser("~/autoware/install"))
_ROS_FALLBACK_PATHS.append(
    f"{_AUTOWARE_DIR}/autoware_lanelet2_extension_python/local/lib/python3.10/dist-packages"
)
for _p in _ROS_FALLBACK_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

POINTS_PER_LANELET = 20

# Polygon / LineString tensor constants — matched to the C++ Autoware planner
# (``autoware_diffusion_planner/include/autoware/diffusion_planner/dimensions.hpp``
# + ``conversion/lanelet.hpp``).
POINTS_PER_POLYGON = 40
POINTS_PER_LINE_STRING = 20
POLYGON_TYPE_INTERSECTION_AREA = 0
POLYGON_TYPE_NUM = 1
LINE_STRING_TYPE_STOP_LINE = 0
LINE_STRING_TYPE_ROAD_BORDER = 1
LINE_STRING_TYPE_NUM = 2


# ── LineType enum (local copy to avoid ROS runtime dep) ──────────────────────

class LineType(IntEnum):
    CROSSWALK = 0
    CURBSTONE = 1
    GUARD_RAIL = 2
    LINE_THICK = 3
    LINE_THIN = 4
    PEDESTRIAN_MARKING = 5
    ROAD_BORDER = 6
    ROAD_SHOULDER = 7
    VIRTUAL = 8
    ZEBRA_MARKING = 9
    NUM = 10

    @classmethod
    def from_str(cls, type_str: str) -> LineType:
        return _LINE_TYPE_MAP.get(type_str, LineType.VIRTUAL)


_LINE_TYPE_MAP = {
    "crosswalk": LineType.CROSSWALK,
    "curbstone": LineType.CURBSTONE,
    "guard_rail": LineType.GUARD_RAIL,
    "line_thick": LineType.LINE_THICK,
    "line_thin": LineType.LINE_THIN,
    "pedestrian_marking": LineType.PEDESTRIAN_MARKING,
    "road_border": LineType.ROAD_BORDER,
    "road_shoulder": LineType.ROAD_SHOULDER,
    "virtual": LineType.VIRTUAL,
    "zebra_marking": LineType.ZEBRA_MARKING,
}

KPH_TO_MPS = 1000.0 / 3600.0


# ── Placement result ─────────────────────────────────────────────────────────

@dataclass
class AgentPlacement:
    lanelet_id: int
    position_xy: np.ndarray
    heading: float
    speed: float
    length: float
    width: float
    wheelbase: float
    is_ego: bool = False


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _resample_linestring_to_tiles(
    pts: np.ndarray, num_points: int, max_step_m: float,
) -> list[np.ndarray]:
    """Mirror the C++ ``resample_line_string`` at
    ``autoware_diffusion_planner/src/conversion/lanelet.cpp:109``.

    A single polyline is split into N segments so that no sampled-point
    step exceeds ``max_step_m``. Each segment becomes a ``(num_points, 2)``
    tile interpolated uniformly along its arc length.

    For a 500 m border with ``num_points=20`` and ``max_step_m=5.0``, the
    unsplit step would be ``500 / 19 ≈ 26`` m — six times over. So we
    split into ``ceil(26 / 5) = 6`` tiles of ~83 m each, each internally
    spaced at ~4.4 m per sample. The output then carries six separate
    tile entries in the line_strings tensor, each passing the AABB filter
    independently, giving the model the same spatial resolution it saw
    during training.
    """
    if len(pts) < 2 or num_points < 2:
        return [pts.astype(np.float32)]

    # Cumulative arc length along the polyline.
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(arc[-1])
    if total < 1e-6:
        tile = np.tile(pts[:1].astype(np.float32), (num_points, 1))
        return [tile]

    step_m = total / (num_points - 1)
    safe_max = max(max_step_m, 1e-6)
    n_tiles = max(1, int(np.ceil(step_m / safe_max)))
    tile_len = total / n_tiles

    # Pre-build a linear interpolator over arc length.
    def _point_at(s: float) -> np.ndarray:
        s = max(0.0, min(total, s))
        idx = int(np.searchsorted(arc, s) - 1)
        idx = max(0, min(idx, len(arc) - 2))
        seg = arc[idx + 1] - arc[idx]
        t = 0.0 if seg < 1e-9 else (s - arc[idx]) / seg
        return pts[idx] + t * (pts[idx + 1] - pts[idx])

    tiles: list[np.ndarray] = []
    for i in range(n_tiles):
        s_start = i * tile_len
        inner_step = tile_len / (num_points - 1)
        tile = np.zeros((num_points, 2), dtype=np.float32)
        for j in range(num_points):
            tile[j] = _point_at(s_start + j * inner_step)
        tiles.append(tile)
    return tiles


def _interpolate_lane(waypoints: np.ndarray, num_points: int = POINTS_PER_LANELET) -> np.ndarray:
    """Arc-length interpolation of a polyline to exactly num_points.

    Ported from lanelet_converter.py _interpolate_lane().
    """
    assert len(waypoints) >= 2
    distances = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        distances[i] = distances[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])

    total_length = distances[-1]
    if total_length < 1e-9:
        return np.tile(waypoints[0], (num_points, 1))

    step = total_length / (num_points - 1)
    result = [waypoints[0]]
    seg_idx = 0

    for i in range(1, num_points - 1):
        target = i * step
        while seg_idx + 1 < len(distances) and distances[seg_idx + 1] < target:
            seg_idx += 1
        if seg_idx >= len(distances) - 1:
            seg_idx = len(distances) - 2

        seg_start = distances[seg_idx]
        seg_end = distances[seg_idx + 1]
        safe_len = max(seg_end - seg_start, 1e-6)
        t = max(0.0, min(1.0, (target - seg_start) / safe_len))
        result.append(waypoints[seg_idx] + t * (waypoints[seg_idx + 1] - waypoints[seg_idx]))

    result.append(waypoints[-1])
    return np.array(result, dtype=np.float32)


def _one_hot(class_idx: int, num_classes: int = LineType.NUM.value) -> np.ndarray:
    arr = np.zeros(num_classes, dtype=np.float32)
    if 0 <= class_idx < num_classes:
        arr[class_idx] = 1.0
    return arr


def _obb_corners(x: float, y: float, heading: float, length: float, width: float) -> np.ndarray:
    """Return (4, 2) corners of an oriented bounding box centered on rear axle."""
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    rear_overhang = (length - length * 0.65) / 2.0
    # Longitudinal: from -rear_overhang to length - rear_overhang
    dx_lo, dx_hi = -rear_overhang, length - rear_overhang
    dy_lo, dy_hi = -width / 2.0, width / 2.0
    local_corners = np.array([
        [dx_lo, dy_lo], [dx_hi, dy_lo], [dx_hi, dy_hi], [dx_lo, dy_hi],
    ])
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return (local_corners @ rot.T) + np.array([x, y])


def _project_onto_axis(corners: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    proj = corners @ axis
    return float(proj.min()), float(proj.max())


def _obb_collides(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    """SAT collision test between two OBBs given as (4, 2) corners."""
    for corners in (corners_a, corners_b):
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-9:
                continue
            axis /= norm
            min_a, max_a = _project_onto_axis(corners_a, axis)
            min_b, max_b = _project_onto_axis(corners_b, axis)
            if max_a < min_b or max_b < min_a:
                return False
    return True


def _polyline_arc_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)))


def _heading_at_point(centerline_2d: np.ndarray, idx: int) -> float:
    """Compute heading at a point index from centerline direction."""
    if idx < len(centerline_2d) - 1:
        dx, dy = centerline_2d[idx + 1] - centerline_2d[idx]
    else:
        dx, dy = centerline_2d[idx] - centerline_2d[idx - 1]
    return float(math.atan2(dy, dx))


# ── Cached lanelet data ──────────────────────────────────────────────────────

@dataclass
class _CachedLanelet:
    ll_id: int
    raw_centerline: np.ndarray      # (N, 2) original points
    raw_left: np.ndarray            # (N, 2)
    raw_right: np.ndarray           # (N, 2)
    interp_centerline: np.ndarray   # (20, 2)
    interp_left: np.ndarray         # (20, 2)
    interp_right: np.ndarray        # (20, 2)
    left_line_type: LineType
    right_line_type: LineType
    speed_limit_mps: float
    has_speed_limit: bool
    subtype: str
    arc_length: float = 0.0
    cum_arc_lengths: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))


# ── Main builder class ───────────────────────────────────────────────────────

class LaneletSceneBuilder:
    """Loads a Lanelet2 map and generates synthetic SceneContext objects."""

    def __init__(self, lanelet_path: str):
        import lanelet2
        from autoware_lanelet2_extension_python.projection import MGRSProjector

        projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
        self._lanelet_map = lanelet2.io.load(str(lanelet_path), projection)

        # Japanese Autoware maps (e.g. Shinagawa-Odaiba) tag main driving
        # lanes as ``subtype=road_shoulder`` — but the stock Germany/Vehicle
        # traffic rules treat that subtype as non-drivable for vehicles,
        # disconnecting them from the routing graph (0 successors / 0
        # predecessors). Rewrite the subtype in-memory to ``road`` so the
        # routing graph includes them. Geometry and all other attributes
        # stay untouched.
        n_promoted = 0
        for ll in self._lanelet_map.laneletLayer:
            if ("subtype" in ll.attributes
                    and ll.attributes["subtype"] == "road_shoulder"):
                ll.attributes["subtype"] = "road"
                n_promoted += 1
        if n_promoted:
            print(f"LaneletSceneBuilder: promoted {n_promoted} "
                  f"road_shoulder→road lanelets for routing")

        traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )
        self._routing_graph = lanelet2.routing.RoutingGraph(self._lanelet_map, traffic_rules)

        # Cache per-lanelet geometry
        self._cache: dict[int, _CachedLanelet] = {}
        self._ll_by_id = {}  # lanelet2 objects by id
        # Ordered lanelet IDs from the most recent _build_map_data call.
        # The TrafficLightController reads this to map row indices in
        # map_data.lanes back to lanelet IDs.
        self._last_map_data_ids: list[int] = []
        self._vehicle_subtypes = {"road", "highway", "road_shoulder"}
        # Subset of ``_vehicle_subtypes`` that actually has routing
        # connections under Germany/Vehicle traffic rules. Used when snapping
        # a user click to a lanelet the routing graph can plan through —
        # ``road_shoulder`` is a vehicle-drivable geometry but has 0
        # successors / predecessors so snapping a start or goal to it makes
        # routing return None.
        self._routable_subtypes = {"road", "highway"}

        for ll in self._lanelet_map.laneletLayer:
            subtype = ll.attributes["subtype"] if "subtype" in ll.attributes else ""
            raw_lb = np.array([(p.x, p.y) for p in ll.leftBound], dtype=np.float32)
            raw_rb = np.array([(p.x, p.y) for p in ll.rightBound], dtype=np.float32)

            if len(raw_lb) < 2 or len(raw_rb) < 2:
                continue

            # Compute centerline as midpoint of arc-length-resampled bounds.
            # The lanelet2 Python binding's ll.centerline produces broken
            # geometry for U-turn lanelets with asymmetric left/right bounds
            # (it shortcuts instead of following the loop). The C++
            # centerline3d() handles this correctly; resampling + midpoint
            # replicates that behaviour.
            n_cl = max(len(raw_lb), len(raw_rb))
            raw_cl = ((_interpolate_lane(raw_lb, n_cl)
                       + _interpolate_lane(raw_rb, n_cl)) * 0.5).astype(np.float32)

            self._ll_by_id[ll.id] = ll

            lt_left = LineType.from_str(
                ll.leftBound.attributes["type"] if "type" in ll.leftBound.attributes else ""
            )
            lt_right = LineType.from_str(
                ll.rightBound.attributes["type"] if "type" in ll.rightBound.attributes else ""
            )

            speed_str = ll.attributes["speed_limit"] if "speed_limit" in ll.attributes else ""
            if speed_str:
                speed_mps = float(speed_str) * KPH_TO_MPS
                has_sl = True
            else:
                speed_mps = 0.0
                has_sl = False

            # Interpolate to 20 points for 33-dim conversion
            interp_cl = _interpolate_lane(raw_cl, POINTS_PER_LANELET)
            interp_lb = _interpolate_lane(raw_lb, POINTS_PER_LANELET)
            interp_rb = _interpolate_lane(raw_rb, POINTS_PER_LANELET)

            diffs = np.linalg.norm(np.diff(raw_cl[:, :2], axis=0), axis=1)
            cum_arc = np.concatenate([[0.0], np.cumsum(diffs)]).astype(np.float64)

            self._cache[ll.id] = _CachedLanelet(
                ll_id=ll.id,
                raw_centerline=raw_cl,
                raw_left=raw_lb,
                raw_right=raw_rb,
                interp_centerline=interp_cl,
                interp_left=interp_lb,
                interp_right=interp_rb,
                left_line_type=lt_left,
                right_line_type=lt_right,
                speed_limit_mps=speed_mps,
                has_speed_limit=has_sl,
                subtype=subtype,
                arc_length=float(cum_arc[-1]),
                cum_arc_lengths=cum_arc,
            )

        # ── Polygon + LineString cache (matches the C++ Autoware planner's
        #    convert_to_internal_lanelet_map at ../diffusion_planner/src/
        #    conversion/lanelet.cpp:294-317). Polygons are filtered to
        #    ``intersection_area`` only; LineStrings to ``stop_line`` and
        #    ``road_border`` (both populate channels 2/3 of the line_strings
        #    tensor as one-hot LINE_STRING_TYPE_STOP_LINE=0 / _ROAD_BORDER=1).
        self._polygons_cache: list[tuple[np.ndarray, int]] = []  # [(points (N,2), type)]
        self._line_strings_cache: list[tuple[np.ndarray, int]] = []  # same shape

        for poly in self._lanelet_map.polygonLayer:
            ptype = poly.attributes["type"] if "type" in poly.attributes else ""
            if ptype != "intersection_area":
                continue
            pts = np.array([(p.x, p.y) for p in poly], dtype=np.float32)
            if len(pts) < 2:
                continue
            interp = _interpolate_lane(pts, POINTS_PER_POLYGON)
            self._polygons_cache.append((interp, POLYGON_TYPE_INTERSECTION_AREA))

        # Matches C++ resample_line_string (max_step_m = 5 m). A long border
        # (e.g. 500 m) gets split into multiple 20-point tiles so no tile
        # step exceeds 5 m. Without this, a 500 m border becomes a single
        # 20-point coarse tile (26 m per step) — way below the model's
        # training-time resolution and a likely cause of poor performance.
        line_string_max_step_m = 5.0
        for ls in self._lanelet_map.lineStringLayer:
            lstype = ls.attributes["type"] if "type" in ls.attributes else ""
            if lstype not in ("stop_line", "road_border"):
                continue
            pts = np.array([(p.x, p.y) for p in ls], dtype=np.float32)
            if len(pts) < 2:
                continue
            type_idx = (
                LINE_STRING_TYPE_STOP_LINE if lstype == "stop_line"
                else LINE_STRING_TYPE_ROAD_BORDER
            )
            tiles = _resample_linestring_to_tiles(
                pts, POINTS_PER_LINE_STRING, line_string_max_step_m,
            )
            for tile in tiles:
                self._line_strings_cache.append((tile, type_idx))

        print(f"LaneletSceneBuilder: cached {len(self._polygons_cache)} "
              f"intersection polygons, {len(self._line_strings_cache)} line "
              f"strings (stop_line + road_border)")

        # Pre-compute vectorised anchor arrays for fast spatial queries
        # (used by closest_lanelets). Rows align with ``self._vehicle_ll_ids``.
        self._vehicle_ll_ids: np.ndarray = np.array([
            ll_id for ll_id, c in self._cache.items()
            if c.subtype in self._vehicle_subtypes
        ], dtype=np.int64)
        n_v = len(self._vehicle_ll_ids)
        self._vehicle_centers = np.zeros((n_v, 2), dtype=np.float32)
        self._vehicle_firsts = np.zeros((n_v, 2), dtype=np.float32)
        self._vehicle_lasts = np.zeros((n_v, 2), dtype=np.float32)
        self._vehicle_backs = np.zeros((n_v, 2), dtype=np.float32)  # -2 index
        for i, ll_id in enumerate(self._vehicle_ll_ids):
            cl = self._cache[int(ll_id)].raw_centerline
            self._vehicle_centers[i] = cl.mean(axis=0)
            self._vehicle_firsts[i] = cl[0]
            self._vehicle_lasts[i] = cl[-1]
            self._vehicle_backs[i] = cl[-2] if len(cl) >= 2 else cl[-1]

        print(f"LaneletSceneBuilder: cached {len(self._cache)} lanelets "
              f"({n_v} drivable), routing graph built")

    # ── 33-dim conversion ────────────────────────────────────────────────

    def lanelet_ids(self) -> list[int]:
        """Public accessor for cached lanelet ids (sorted, deterministic)."""
        return sorted(self._cache.keys())

    def has_lanelet_id(self, ll_id: int) -> bool:
        """True iff the lanelet with this id was loaded into the cache."""
        return int(ll_id) in self._cache

    def raw_centerline(self, ll_id: int) -> np.ndarray:
        """Public accessor for a lanelet's raw centerline (N, 2 or 3) array.

        Returns a copy so callers can't accidentally mutate the cache.
        Raises KeyError if the id is not loaded.
        """
        return self._cache[int(ll_id)].raw_centerline.copy()

    def lanelet_to_33dim(self, ll_id: int) -> tuple[np.ndarray, float, bool]:
        """Convert a lanelet to (20, 33) tensor + speed limit info.

        Works in world frame (no ego transform).
        """
        c = self._cache[ll_id]
        centerline = c.interp_centerline.copy()         # (20, 2)
        left_bound = c.interp_left.copy()               # (20, 2)
        right_bound = c.interp_right.copy()              # (20, 2)

        # Direction vectors
        diff = np.zeros_like(centerline)
        diff[:-1] = centerline[1:] - centerline[:-1]
        if len(centerline) > 1:
            diff[-1] = diff[-2]

        # Boundary offsets relative to centerline
        left_offset = left_bound - centerline
        right_offset = right_bound - centerline

        # Traffic light: default no-light for generated scenarios
        traffic = np.zeros((POINTS_PER_LANELET, 5), dtype=np.float32)
        traffic[:, 4] = 1.0

        # Line type one-hot (tiled to all 20 points)
        lt_left = np.tile(_one_hot(c.left_line_type.value), (POINTS_PER_LANELET, 1))
        lt_right = np.tile(_one_hot(c.right_line_type.value), (POINTS_PER_LANELET, 1))

        segment = np.concatenate([
            centerline,    # [0:2]
            diff,          # [2:4]
            left_offset,   # [4:6]
            right_offset,  # [6:8]
            traffic,       # [8:13]
            lt_left,       # [13:23]
            lt_right,      # [23:33]
        ], axis=1)

        assert segment.shape == (POINTS_PER_LANELET, 33), f"Bad shape: {segment.shape}"
        return segment, c.speed_limit_mps, c.has_speed_limit

    # ── Traffic light discovery ─────────────────────────────────────────

    def get_traffic_light_groups(self) -> dict[int, int]:
        """Return ``{lanelet_id: traffic_light_group_id}`` for all lanelets
        with ``trafficLights()`` regulatory elements.

        The group_id is the ``.id`` of the first traffic-light regulatory
        element on each lanelet (mirrors the C++ planner's per-LaneSegment
        traffic_light_id and ``route_traffic_light_publisher.py:172``).
        """
        result: dict[int, int] = {}
        for ll_id, ll in self._ll_by_id.items():
            try:
                tl_list = ll.trafficLights()
            except Exception:
                continue
            if tl_list:
                result[ll_id] = tl_list[0].id
        return result

    def get_traffic_light_bulb_groups(self) -> dict[int, frozenset]:
        """Return ``{reg_element_id: frozenset(light_bulb_linestring_ids)}``.

        Two regulatory elements that share the same ``light_bulbs``
        LineStrings are controlled by the same physical traffic light and
        MUST show the same colour.
        """
        result: dict[int, frozenset] = {}
        seen_regs: set[int] = set()
        for ll in self._ll_by_id.values():
            try:
                tl_list = ll.trafficLights()
            except Exception:
                continue
            for tl_reg in tl_list:
                if tl_reg.id in seen_regs:
                    continue
                seen_regs.add(tl_reg.id)
                bulb_ids: set[int] = set()
                try:
                    params = tl_reg.parameters
                    if "light_bulbs" in params:
                        for bulb_ls in params["light_bulbs"]:
                            bulb_ids.add(bulb_ls.id)
                except Exception:
                    pass
                result[tl_reg.id] = frozenset(bulb_ids)
        return result

    # ── Spatial query ────────────────────────────────────────────────────

    def lanelets_in_rect(
        self, xmin: float, ymin: float, xmax: float, ymax: float,
    ) -> list[int]:
        """Return lanelet IDs that overlap the rectangle (even partially).

        Uses AABB intersection: includes lanelets whose centerline bounding box
        overlaps the selection rectangle. This catches lanelets that cross the
        rectangle boundary even if no individual point falls inside.

        Only includes vehicle-drivable subtypes.
        """
        result = []
        for ll_id, c in self._cache.items():
            if c.subtype not in self._vehicle_subtypes:
                continue
            cl = c.raw_centerline
            # AABB overlap test: lanelet bbox vs selection rect
            if (cl[:, 0].max() >= xmin and cl[:, 0].min() <= xmax and
                    cl[:, 1].max() >= ymin and cl[:, 1].min() <= ymax):
                result.append(ll_id)
        return result

    def closest_lanelets(
        self, xy: np.ndarray, max_n: int, mask_range: float = 100.0,
    ) -> list[int]:
        """Return up to ``max_n`` drivable lanelet IDs around ``xy``, closest first.

        Mirrors the lane-filter strategy used by the Autoware/Diffusion-Planner
        ROS node (see ``diffusion_planner_ros/lanelet2_utils/lanelet_converter.py``):

        1. **AABB pre-filter**: a lanelet passes when any of its center,
           first centerline point, or last centerline point falls inside the
           ``±mask_range`` square around ``xy``. The ROS node uses 100 m.
        2. **Sort by min-endpoint-distance** to ``xy`` (the score used by
           ``key_func`` in ``create_lane_tensor``: the smaller of the
           distances from ``xy`` to the first and second-to-last centerline
           points).
        3. **Cap at ``max_n``**.

        ``max_n`` should typically be ≤ ``tensor_converter._NUM_LANES``
        (currently 140). The ROS node passes 70; training NPZs show ~65
        non-zero slots per scene, so 70–140 is the sweet spot.

        Args:
            xy: 2-element world-frame query point ``[x, y]``.
            max_n: Maximum number of lanelets to return.
            mask_range: Half-side of the square AABB pre-filter in metres.

        Returns:
            Lanelet IDs sorted closest-first. Length ≤ ``max_n``.
        """
        pos = np.asarray(xy, dtype=np.float32)[:2]
        if len(self._vehicle_ll_ids) == 0:
            return []

        # Vectorised AABB: accept when any of (center, first, last) is inside.
        def _inside(points: np.ndarray) -> np.ndarray:
            dx = np.abs(points[:, 0] - pos[0])
            dy = np.abs(points[:, 1] - pos[1])
            return (dx < mask_range) & (dy < mask_range)

        mask = (
            _inside(self._vehicle_centers)
            | _inside(self._vehicle_firsts)
            | _inside(self._vehicle_lasts)
        )
        if not mask.any():
            return []

        # Score by min-distance of first / second-to-last endpoint to xy
        # (mirrors the ROS node's key_func in create_lane_tensor).
        kept_ids = self._vehicle_ll_ids[mask]
        d_first = np.sum((self._vehicle_firsts[mask] - pos) ** 2, axis=1)
        d_back = np.sum((self._vehicle_backs[mask] - pos) ** 2, axis=1)
        score = np.minimum(d_first, d_back)
        if len(kept_ids) > max_n:
            top_idx = np.argpartition(score, max_n - 1)[:max_n]
            top_idx = top_idx[np.argsort(score[top_idx])]
        else:
            top_idx = np.argsort(score)
        return [int(kept_ids[i]) for i in top_idx]

    def lanelets_near_point(
        self, xy: np.ndarray, radius: float,
    ) -> list[int]:
        """Return drivable lanelet IDs whose centerline passes within ``radius``
        metres of ``xy``.

        Used by the NPC spawn manager to pick candidate lanelets near the ego.
        First filters by an AABB of side ``2*radius`` (re-uses
        :meth:`lanelets_in_rect`) then refines with exact centerline-point
        distance so diagonal-corner false positives are rejected.
        """
        x, y = float(xy[0]), float(xy[1])
        candidates = self.lanelets_in_rect(
            x - radius, y - radius, x + radius, y + radius,
        )
        radius_sq = radius * radius
        result = []
        for ll_id in candidates:
            cl = self._cache[ll_id].raw_centerline
            dx = cl[:, 0] - x
            dy = cl[:, 1] - y
            if np.min(dx * dx + dy * dy) <= radius_sq:
                result.append(ll_id)
        return result

    def snap_to_nearest_ll(
        self,
        xy: np.ndarray,
        candidate_ids: list[int] | None = None,
        routable_only: bool = True,
        reachable_from: int | None = None,
        reachable_range_m: float = 10000.0,
        heading_rad: float | None = None,
        heading_weight_m_per_deg: float = 0.5,
    ) -> int | None:
        """Snap a world-frame ``(x, y)`` position to the nearest drivable lanelet.

        Scoring: when ``heading_rad`` is provided, the candidate score is
        ``distance_m + heading_weight_m_per_deg * |delta_heading_deg|``. At
        the default weight (0.5), a 180° mismatch adds 90 m to the distance —
        so a slightly-farther lane pointing the right way wins over a
        slightly-closer lane pointing the wrong way. This is critical for
        dense maps (e.g. Shinagawa) where parallel opposite-direction roads
        often lie within a few metres of each other.

        Args:
            xy: 2-element array-like ``[x, y]`` in MGRS world frame.
            candidate_ids: Optional restriction to a subset of lanelets. When
                ``None``, all cached drivable lanelets are considered.
            routable_only: When ``True`` (default), only lanelets with subtype
                in ``self._routable_subtypes`` (``road`` / ``highway``) are
                considered. This avoids snapping start/goal clicks onto
                ``road_shoulder`` lanelets — those are drivable geometrically
                but have no routing connections under Germany/Vehicle rules,
                so ``route_between`` would return ``None``.
            reachable_from: When set to a lanelet id, only candidates in
                ``RoutingGraph.reachableSet(that_lanelet, reachable_range_m)``
                are considered. Use this for snapping a goal / waypoint so
                the click doesn't land on a geometrically-close but
                topologically-disconnected sub-network.
            reachable_range_m: Distance horizon for the reachable-set
                computation (ignored when ``reachable_from`` is None).
            heading_rad: Desired travel direction at the query point. When
                provided, candidates with lane direction far from this are
                penalised (see scoring formula above). ``None`` falls back
                to pure nearest-distance snapping.
            heading_weight_m_per_deg: Penalty weight applied to the heading
                delta. 0.5 m/deg is a good default; raise it to snap more
                aggressively along the click direction.

        Returns:
            The lanelet id with the best combined score, or ``None`` when no
            candidate passes the filters.
        """
        pos = np.asarray(xy, dtype=np.float32)[:2]
        allowed_subtypes = (
            self._routable_subtypes if routable_only else self._vehicle_subtypes
        )
        if candidate_ids is None:
            candidate_ids = [
                ll_id for ll_id, c in self._cache.items()
                if c.subtype in allowed_subtypes
            ]
        else:
            candidate_ids = [
                ll_id for ll_id in candidate_ids
                if ll_id in self._cache
                and self._cache[ll_id].subtype in allowed_subtypes
            ]

        if reachable_from is not None and reachable_from in self._ll_by_id:
            source_ll = self._ll_by_id[reachable_from]
            reachable_ids = {
                ll.id for ll in self._routing_graph.reachableSet(
                    source_ll, reachable_range_m,
                )
            }
            candidate_ids = [ll for ll in candidate_ids if ll in reachable_ids]

        best_ll_id = None
        best_score = float("inf")
        for ll_id in candidate_ids:
            cl = self._cache[ll_id].raw_centerline
            diff = cl - pos
            dist2 = diff[:, 0] ** 2 + diff[:, 1] ** 2
            idx = int(np.argmin(dist2))
            dist = float(np.sqrt(dist2[idx]))

            score = dist
            if heading_rad is not None and len(cl) >= 2:
                # Lane direction at the closest centerline point.
                if idx < len(cl) - 1:
                    dxdy = cl[idx + 1] - cl[idx]
                else:
                    dxdy = cl[idx] - cl[idx - 1]
                lane_hdg = math.atan2(float(dxdy[1]), float(dxdy[0]))
                delta = lane_hdg - heading_rad
                # Wrap into [-pi, pi] and take absolute value (degrees).
                delta = math.atan2(math.sin(delta), math.cos(delta))
                delta_deg = abs(math.degrees(delta))
                score += heading_weight_m_per_deg * delta_deg

            if score < best_score:
                best_score = score
                best_ll_id = ll_id
        return best_ll_id

    def is_lanelet_straight(
        self, ll_id: int, curvature_threshold: float = 0.3,
    ) -> bool:
        """Check if a lanelet is straight enough for NPC spawning.

        Iterates consecutive centerline segments and returns False when the
        maximum heading change between any two adjacent segments exceeds
        ``curvature_threshold`` radians (default 0.3 rad ≈ 17°).

        Ported from the Autoware planning-simulator ``stopped_vehicle_utils.is_lanelet_straight`` helper.
        NPC history synthesis via :meth:`generate_history` can produce unnatural
        trajectories on sharp curves, so the spawn manager rejects non-straight
        candidates.
        """
        if ll_id not in self._cache:
            return False
        cl = self._cache[ll_id].raw_centerline
        if len(cl) < 3:
            return True  # too few points to measure curvature

        diffs = np.diff(cl, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        valid = seg_lens > 1e-6
        if valid.sum() < 2:
            return True
        angles = np.arctan2(diffs[valid, 1], diffs[valid, 0])

        delta = np.diff(angles)
        # Wrap each delta into [-pi, pi].
        delta = np.arctan2(np.sin(delta), np.cos(delta))
        return bool(np.max(np.abs(delta)) <= curvature_threshold)

    # ── Routing ──────────────────────────────────────────────────────────

    def route_between(
        self, start_ll_id: int, goal_ll_id: int,
    ) -> list[int] | None:
        """Shortest path between two lanelet IDs using the lanelet2 routing graph.

        Returns a list of lanelet IDs ``[start_ll_id, ..., goal_ll_id]`` or
        ``None`` when start and goal are in disconnected components / reverse
        directions. Wraps :class:`lanelet2.routing.RoutingGraph.shortestPath`.
        """
        if start_ll_id not in self._ll_by_id or goal_ll_id not in self._ll_by_id:
            return None
        start_ll = self._ll_by_id[start_ll_id]
        goal_ll = self._ll_by_id[goal_ll_id]
        path = self._routing_graph.shortestPath(start_ll, goal_ll)
        if path is None:
            return None
        return [ll.id for ll in path]

    def route_with_waypoints(
        self,
        start_ll_id: int,
        via_ll_ids: list[int],
        goal_ll_id: int,
    ) -> list[int] | None:
        """Shortest path forced through ``via_ll_ids`` in the given order.

        Uses :class:`lanelet2.routing.RoutingGraph.shortestPathWithVia`. When
        ``via_ll_ids`` is empty, delegates to :meth:`route_between`.

        Returns the resolved lanelet id sequence or ``None`` when any
        consecutive pair in ``[start, *via, goal]`` is unreachable.
        """
        if not via_ll_ids:
            return self.route_between(start_ll_id, goal_ll_id)
        if start_ll_id not in self._ll_by_id or goal_ll_id not in self._ll_by_id:
            return None
        for vid in via_ll_ids:
            if vid not in self._ll_by_id:
                return None
        start_ll = self._ll_by_id[start_ll_id]
        goal_ll = self._ll_by_id[goal_ll_id]
        via_lls = [self._ll_by_id[i] for i in via_ll_ids]
        path = self._routing_graph.shortestPathWithVia(start_ll, via_lls, goal_ll)
        if path is None:
            return None
        return [ll.id for ll in path]

    def find_route(self, start_ll_id: int, min_length_m: float = 120.0) -> list[int]:
        """Find a forward route from start_lanelet of at least min_length_m.

        Follows one randomly selected successor at each step until the
        accumulated arc length reaches min_length_m or a dead end.
        """
        if start_ll_id not in self._ll_by_id:
            return [start_ll_id]

        start_ll = self._ll_by_id[start_ll_id]
        route_ids = [start_ll_id]
        total_len = self._cache[start_ll_id].arc_length
        current_ll = start_ll

        max_steps = 100
        for _ in range(max_steps):
            if total_len >= min_length_m:
                break
            following = self._routing_graph.following(current_ll)
            if not following:
                break
            # Random branch selection for variety
            next_ll = random.choice(list(following))
            if next_ll.id not in self._cache:
                break
            route_ids.append(next_ll.id)
            total_len += self._cache[next_ll.id].arc_length
            current_ll = next_ll

        return route_ids

    # ── History generation ───────────────────────────────────────────────

    def generate_history(
        self,
        position_xy: np.ndarray,
        heading: float,
        speed: float,
        lanelet_id: int,
        n_steps: int = 31,
        dt: float = 0.1,
    ) -> tuple[np.ndarray, set[int]]:
        """Generate past trajectory by tracing backward along the centerline.

        Returns ((n_steps, 3) [x, y, heading_rad], set of traversed lanelet IDs).
        Index -1 is current timestep.
        """
        history = np.zeros((n_steps, 3), dtype=np.float32)
        history[-1] = [position_xy[0], position_xy[1], heading]

        if speed < 0.1 or lanelet_id not in self._cache:
            history[:, 0] = position_xy[0]
            history[:, 1] = position_xy[1]
            history[:, 2] = heading
            return history, {lanelet_id}

        # Build dense backward polyline ordered [current_pos, ..., far_behind]
        bw_pts, traversed_ids = self._build_backward_polyline(lanelet_id, position_xy, heading, n_steps, speed, dt)

        diffs = np.linalg.norm(np.diff(bw_pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(diffs)])

        # Sample history positions at evenly-spaced backward distances
        seg_idx = 0
        for step in range(n_steps - 2, -1, -1):
            backward_dist = (n_steps - 1 - step) * speed * dt
            # Walk forward in arc lengths to find the right segment
            while seg_idx + 1 < len(arc) and arc[seg_idx + 1] < backward_dist:
                seg_idx += 1
            if seg_idx >= len(bw_pts) - 1:
                # Ran out of polyline: hold last valid position
                pos = bw_pts[-1]
                if len(bw_pts) >= 2:
                    fwd = bw_pts[-2] - bw_pts[-1]
                    h = math.atan2(fwd[1], fwd[0])
                else:
                    h = heading
            else:
                seg_len = arc[seg_idx + 1] - arc[seg_idx]
                safe_len = max(seg_len, 1e-6)
                t = (backward_dist - arc[seg_idx]) / safe_len
                t = max(0.0, min(1.0, t))
                pos = bw_pts[seg_idx] + t * (bw_pts[seg_idx + 1] - bw_pts[seg_idx])
                # Heading = forward driving direction (opposite of backward direction)
                fwd = bw_pts[seg_idx] - bw_pts[seg_idx + 1]
                h = math.atan2(fwd[1], fwd[0])

            # Small lateral noise
            lateral = np.array([-math.sin(h), math.cos(h)])
            pos = pos + lateral * np.random.normal(0, 0.05)
            history[step] = [pos[0], pos[1], h]

        return history, traversed_ids

    def _build_backward_polyline(
        self, start_ll_id: int, start_pos: np.ndarray, heading: float,
        n_steps: int, speed: float, dt: float,
    ) -> tuple[np.ndarray, set[int]]:
        """Build a polyline going backward from start position.

        Projects start_pos onto the centerline arc to find the exact parameter,
        then only includes points that are strictly behind the agent along the
        driving direction.

        Returns (polyline ordered [current_pos, ..., far_behind], set of traversed lanelet IDs).
        """
        needed_dist = speed * dt * (n_steps + 5)
        c = self._cache[start_ll_id]
        cl = c.raw_centerline

        arc = c.cum_arc_lengths

        # Project start_pos onto the centerline: find the segment and parameter
        best_param = 0.0
        best_proj = cl[0].copy()
        best_dist = float("inf")
        for i in range(len(cl) - 1):
            seg = cl[i + 1] - cl[i]
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-9:
                continue
            seg_dir = seg / seg_len
            t = float(np.dot(start_pos[:2] - cl[i], seg_dir))
            t = max(0.0, min(seg_len, t))
            proj = cl[i] + seg_dir * t
            d = float(np.linalg.norm(start_pos[:2] - proj))
            if d < best_dist:
                best_dist = d
                best_proj = proj
                best_param = arc[i] + t

        # Start from the snapped projection point on the centerline
        points = [best_proj.copy()]
        total_dist = 0.0
        for i in range(len(cl) - 1, -1, -1):
            if arc[i] < best_param - 0.1:
                d = float(np.linalg.norm(cl[i] - points[-1]))
                if d > 0.01:
                    total_dist += d
                    points.append(cl[i].copy())

        # Trace through predecessors
        current_ll_id = start_ll_id
        visited = {start_ll_id}
        for _ in range(50):
            if total_dist >= needed_dist:
                break
            if current_ll_id not in self._ll_by_id:
                break
            current_ll = self._ll_by_id[current_ll_id]
            prev_lls = self._routing_graph.previous(current_ll)
            if not prev_lls:
                break
            prev_ll = random.choice(list(prev_lls))
            if prev_ll.id in visited or prev_ll.id not in self._cache:
                break
            visited.add(prev_ll.id)
            prev_cl = self._cache[prev_ll.id].raw_centerline
            for pt in prev_cl[::-1]:
                d = float(np.linalg.norm(pt - points[-1]))
                if d > 0.01:
                    total_dist += d
                    points.append(pt.copy())
            current_ll_id = prev_ll.id

        if len(points) < 2:
            behind = start_pos[:2] - np.array([math.cos(heading), math.sin(heading)]) * 0.5
            points = [start_pos[:2].copy(), behind]

        return np.array(points, dtype=np.float32), visited

    # ── Agent placement ──────────────────────────────────────────────────

    def place_agents(
        self,
        lanelet_ids: list[int],
        n_neighbors: int,
        min_separation_m: float = 8.0,
        min_speed: float = 3.0,
        max_speed: float = 12.0,
        ego_pose: tuple[float, float, float] | None = None,
    ) -> list[AgentPlacement]:
        """Place ego + n_neighbors on lanes, collision-free.

        Args:
            ego_pose: Optional (x, y, heading_rad) for manual ego placement.
                Snaps to the nearest lanelet centerline.
        """
        candidates = [
            ll_id for ll_id in lanelet_ids
            if ll_id in self._cache and self._cache[ll_id].arc_length > 5.0
        ]
        if not candidates:
            candidates = [ll_id for ll_id in lanelet_ids if ll_id in self._cache]
        if not candidates:
            raise ValueError("No valid lanelets in the selected rectangle")

        with_successors = [
            ll_id for ll_id in candidates
            if ll_id in self._ll_by_id and
            len(list(self._routing_graph.following(self._ll_by_id[ll_id]))) > 0
        ]
        preferred = with_successors if with_successors else candidates

        placements: list[AgentPlacement] = []
        placed_corners: list[np.ndarray] = []

        # Ego placement: manual pose or random
        if ego_pose is not None:
            ego_placement = self._place_ego_at_pose(ego_pose, lanelet_ids, min_speed, max_speed)
        else:
            ego_placement = self._try_place_one(
                preferred, placed_corners, min_separation_m,
                min_speed, max_speed, is_ego=True,
            )
        if ego_placement is not None:
            corners = _obb_corners(
                ego_placement.position_xy[0], ego_placement.position_xy[1],
                ego_placement.heading, ego_placement.length, ego_placement.width,
            )
            placed_corners.append(corners)
            placements.append(ego_placement)

        # Neighbor placements
        for _ in range(n_neighbors):
            placement = self._try_place_one(
                candidates, placed_corners, min_separation_m,
                min_speed, max_speed, is_ego=False,
            )
            if placement is not None:
                corners = _obb_corners(
                    placement.position_xy[0], placement.position_xy[1],
                    placement.heading, placement.length, placement.width,
                )
                placed_corners.append(corners)
                placements.append(placement)

        return placements

    def _place_ego_at_pose(
        self,
        ego_pose: tuple[float, float, float],
        lanelet_ids: list[int],
        min_speed: float,
        max_speed: float,
    ) -> AgentPlacement | None:
        """Place ego at a user-specified position, snapped to nearest lanelet."""
        ex, ey, eheading = ego_pose
        pos = np.array([ex, ey], dtype=np.float32)

        best_ll_id = self.snap_to_nearest_ll(pos, candidate_ids=lanelet_ids)
        if best_ll_id is None:
            return None

        # Snap to centerline and use the lane heading at that point
        cl = self._cache[best_ll_id].raw_centerline
        dists = np.linalg.norm(cl - pos, axis=1)
        closest_idx = int(np.argmin(dists))
        snapped_pos = cl[closest_idx].copy()
        lane_heading = _heading_at_point(cl, closest_idx)

        # Use user heading if provided, otherwise lane heading
        heading = eheading if abs(eheading) > 0.01 else lane_heading

        speed = random.uniform(min_speed, max_speed)
        length = random.uniform(4.0, 5.0)
        width = random.uniform(1.7, 2.0)

        return AgentPlacement(
            lanelet_id=best_ll_id,
            position_xy=snapped_pos.astype(np.float32),
            heading=heading,
            speed=speed,
            length=length,
            width=width,
            wheelbase=length * 0.65,
            is_ego=True,
        )

    def _try_place_one(
        self,
        candidate_ids: list[int],
        existing_corners: list[np.ndarray],
        min_sep: float,
        min_speed: float,
        max_speed: float,
        is_ego: bool,
        max_retries: int = 50,
    ) -> AgentPlacement | None:
        """Try to place one agent without collision."""
        for _ in range(max_retries):
            ll_id = random.choice(candidate_ids)
            c = self._cache[ll_id]
            cl = c.raw_centerline

            arc_lengths = c.cum_arc_lengths
            total = arc_lengths[-1]
            if total < 1.0:
                continue

            # Sample away from endpoints
            margin = min(3.0, total * 0.1)
            target_arc = random.uniform(margin, total - margin)

            # Find the segment
            seg_idx = int(np.searchsorted(arc_lengths, target_arc)) - 1
            seg_idx = max(0, min(seg_idx, len(cl) - 2))
            seg_len = arc_lengths[seg_idx + 1] - arc_lengths[seg_idx]
            if seg_len < 1e-6:
                continue
            t = (target_arc - arc_lengths[seg_idx]) / seg_len
            pos = cl[seg_idx] + t * (cl[seg_idx + 1] - cl[seg_idx])

            heading = _heading_at_point(cl, seg_idx)
            speed = random.uniform(min_speed, max_speed)

            # Vehicle dimensions
            length = random.uniform(4.0, 5.0)
            width = random.uniform(1.7, 2.0)
            wheelbase = length * 0.65

            # Collision check
            corners = _obb_corners(pos[0], pos[1], heading, length, width)
            collides = False
            for existing in existing_corners:
                if _obb_collides(corners, existing):
                    collides = True
                    break
                # Also check center-to-center distance as fast reject
                ex_center = existing.mean(axis=0)
                if np.linalg.norm(pos - ex_center) < min_sep:
                    collides = True
                    break

            if not collides:
                return AgentPlacement(
                    lanelet_id=ll_id,
                    position_xy=pos.astype(np.float32),
                    heading=heading,
                    speed=speed,
                    length=length,
                    width=width,
                    wheelbase=wheelbase,
                    is_ego=is_ego,
                )

        return None

    # ── Scene assembly ───────────────────────────────────────────────────

    def build_scene_context(
        self,
        rect: tuple[float, float, float, float] | None = None,
        n_neighbors: int = 5,
        min_separation_m: float = 8.0,
        min_speed: float = 3.0,
        max_speed: float = 12.0,
        route_length_m: float = 120.0,
        ego_pose: tuple[float, float, float] | None = None,
        lanelet_ids: list[int] | None = None,
    ) -> SceneContext:
        """Generate a complete SceneContext from a rectangle or lanelet ID list.

        Provide either ``rect`` (extracts lanelets via AABB) or ``lanelet_ids``
        (pre-saved set from the GUI). Routes and history that extend beyond the
        initial set are retroactively added to the map data.
        """
        if lanelet_ids is not None:
            ll_ids = [lid for lid in lanelet_ids if lid in self._cache]
        elif rect is not None:
            xmin, ymin, xmax, ymax = rect
            ll_ids = self.lanelets_in_rect(xmin, ymin, xmax, ymax)
        else:
            raise ValueError("Provide either rect or lanelet_ids")

        if not ll_ids:
            raise ValueError("No valid lanelets in the selection")

        placements = self.place_agents(
            ll_ids, n_neighbors, min_separation_m, min_speed, max_speed,
            ego_pose=ego_pose,
        )
        if not placements:
            raise ValueError("Could not place any agents in the selected area")

        if not any(p.is_ego for p in placements):
            raise ValueError("Ego placement failed")

        # Build agents, collecting all traversed lanelet IDs for the map
        all_lanelet_ids = set(ll_ids)
        agents: list[Agent] = []
        nb_idx = 0
        for p in placements:
            if p.is_ego:
                agent_id = "ego"
            else:
                agent_id = f"neighbor_{nb_idx}"
                nb_idx += 1

            # Route (extends beyond selection rect)
            route_ll_ids = self.find_route(p.lanelet_id, route_length_m)
            all_lanelet_ids.update(route_ll_ids)

            # History (traces backward through predecessors)
            history, history_ll_ids = self.generate_history(
                p.position_xy, p.heading, p.speed, p.lanelet_id,
            )
            all_lanelet_ids.update(history_ll_ids)

            # Past velocities from history
            velocities = np.zeros((31, 2), dtype=np.float32)
            for t in range(1, 31):
                velocities[t] = (history[t, :2] - history[t - 1, :2]) / 0.1
            velocities[0] = velocities[1]

            # Goal pose: end of route
            goal = self._route_goal(route_ll_ids)

            # Route lanes as 33-dim
            route_lanes, route_sl, route_hsl = self._route_to_33dim(route_ll_ids)

            agent = Agent(
                id=agent_id,
                agent_type=AgentType.VEHICLE,
                length=p.length,
                width=p.width,
                wheelbase=p.wheelbase,
                past_trajectory=history,
                past_velocities=velocities,
                acceleration=np.zeros(2, dtype=np.float32),
                steering_angle=0.0,
                yaw_rate=0.0,
                goal_pose=goal,
                route_lanes=route_lanes,
                route_speed_limit=route_sl,
                route_has_speed_limit=route_hsl,
            )
            agents.append(agent)

        # Map data: all lanelets from selection + routes + history
        map_data = self._build_map_data(list(all_lanelet_ids))

        return SceneContext(
            agents=agents,
            map_data=map_data,
            ego_agent_id="ego",
            dt=0.1,
        )

    def _route_goal(self, route_ll_ids: list[int]) -> np.ndarray:
        """Get goal pose from the last point of the route's last lanelet."""
        last_id = route_ll_ids[-1]
        if last_id in self._cache:
            cl = self._cache[last_id].raw_centerline
            pos = cl[-1]
            heading = _heading_at_point(cl, len(cl) - 1)
            return np.array([pos[0], pos[1], heading], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _route_to_33dim(
        self, route_ll_ids: list[int], max_segments: int = 25,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert route lanelet IDs to 33-dim arrays."""
        segments = []
        speed_limits = []
        has_speed = []

        for ll_id in route_ll_ids[:max_segments]:
            if ll_id not in self._cache:
                continue
            seg, sl, hsl = self.lanelet_to_33dim(ll_id)
            segments.append(seg)
            speed_limits.append(sl)
            has_speed.append(hsl)

        if not segments:
            return (
                np.zeros((max_segments, POINTS_PER_LANELET, 33), dtype=np.float32),
                np.zeros((max_segments, 1), dtype=np.float32),
                np.zeros((max_segments, 1), dtype=bool),
            )

        # Pad to max_segments
        n = len(segments)
        lanes = np.zeros((max_segments, POINTS_PER_LANELET, 33), dtype=np.float32)
        sl_arr = np.zeros((max_segments, 1), dtype=np.float32)
        hsl_arr = np.zeros((max_segments, 1), dtype=bool)

        for j in range(n):
            lanes[j] = segments[j]
            sl_arr[j, 0] = speed_limits[j]
            hsl_arr[j, 0] = has_speed[j]

        return lanes, sl_arr, hsl_arr

    # ── Polygon / LineString tensors (match C++ Autoware) ────────────────

    def build_polygons_tensor(
        self,
        center_xy: np.ndarray,
        max_n: int = 10,
        mask_range: float = 100.0,
    ) -> np.ndarray:
        """Return a ``(max_n, POINTS_PER_POLYGON, 2 + POLYGON_TYPE_NUM)`` array
        of intersection-area polygon points in **world frame** (not ego),
        sorted by min-distance-to-``center_xy``.

        The world-frame coordinates are written to channels [0:2]; channel
        ``2 + type`` is set to 1.0 for the polygon's type. The
        ``MapData.polygons`` field consumed by ``MapTensorCache`` carries the
        world frame; the cache transforms to ego frame at inference time.

        Mirrors C++
        ``LaneSegmentContext::create_line_tensor<Polygon>(...)`` filter +
        sort logic at
        ``autoware_diffusion_planner/src/preprocessing/lane_segments.cpp:347``.
        """
        return self._build_line_or_polygon_tensor(
            self._polygons_cache, center_xy, max_n,
            POINTS_PER_POLYGON, POLYGON_TYPE_NUM, mask_range,
        )

    def road_border_polylines(self) -> list[np.ndarray]:
        """Return raw road-border polylines (world frame, ``(K, 2)`` arrays).

        Filters the internal line-string cache to only ``road_border`` entries
        (``stop_line`` entries are skipped). Use for overlaying road borders
        on visualizations — call sites should not reach into
        ``_line_strings_cache`` directly.
        """
        return [
            pts for pts, type_idx in self._line_strings_cache
            if type_idx == LINE_STRING_TYPE_ROAD_BORDER
        ]

    def build_line_strings_tensor(
        self,
        center_xy: np.ndarray,
        max_n: int = 60,
        mask_range: float = 100.0,
    ) -> np.ndarray:
        """Return a ``(max_n, POINTS_PER_LINE_STRING, 2 + LINE_STRING_TYPE_NUM)``
        array of stop_line + road_border line-string points in **world frame**,
        sorted by min-distance-to-``center_xy``. Channel layout matches
        training NPZs:

            [0]: x (world)
            [1]: y (world)
            [2]: one-hot stop_line
            [3]: one-hot road_border
        """
        return self._build_line_or_polygon_tensor(
            self._line_strings_cache, center_xy, max_n,
            POINTS_PER_LINE_STRING, LINE_STRING_TYPE_NUM, mask_range,
        )

    def _build_line_or_polygon_tensor(
        self,
        cache: list[tuple[np.ndarray, int]],
        center_xy: np.ndarray,
        max_n: int,
        num_points: int,
        num_types: int,
        mask_range: float,
    ) -> np.ndarray:
        """Shared implementation for ``build_polygons_tensor`` and
        ``build_line_strings_tensor``. AABB pre-filter + min-distance sort +
        top-N truncate, matching the C++ ``create_line_tensor`` template."""
        cx, cy = float(center_xy[0]), float(center_xy[1])
        x_min, x_max = cx - mask_range, cx + mask_range
        y_min, y_max = cy - mask_range, cy + mask_range

        scored: list[tuple[float, np.ndarray, int]] = []
        for pts, type_idx in cache:
            inside = (
                ((pts[:, 0] > x_min) & (pts[:, 0] < x_max)
                 & (pts[:, 1] > y_min) & (pts[:, 1] < y_max)).any()
            )
            if not inside:
                continue
            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy
            min_d = float(np.sqrt(dx * dx + dy * dy).min())
            scored.append((min_d, pts, type_idx))
        scored.sort(key=lambda t: t[0])

        out = np.zeros((max_n, num_points, 2 + num_types), dtype=np.float32)
        for i, (_, pts, type_idx) in enumerate(scored[:max_n]):
            n = min(len(pts), num_points)
            out[i, :n, 0] = pts[:n, 0]
            out[i, :n, 1] = pts[:n, 1]
            out[i, :n, 2 + type_idx] = 1.0
        return out

    # ── Route segment selection (match C++ Autoware) ─────────────────────

    def select_route_segment_indices(
        self,
        route_lanelet_ids: list[int],
        center_xy: np.ndarray,
        max_segments: int = 25,
        mask_range: float = 100.0,
    ) -> list[int]:
        """Return the forward-near-ego subset of a saved route, in the order
        the ego will encounter them.

        Algorithm matches the C++ Autoware planner
        (``LaneSegmentContext::select_route_segment_indices`` at
        ``autoware_diffusion_planner/src/preprocessing/lane_segments.cpp:64``):

        1. Find the route lanelet *closest* to ``center_xy`` by min
           centerline-point distance — that's the ego's current "spot" on
           the route.
        2. Starting from that index, walk **forward** through the saved
           route. For each next lanelet, check if any centerline point is
           inside the AABB ``±mask_range`` around ``center_xy``.

           - Lanelets outside the AABB *before* ego enters the valid region
             are skipped (route hasn't started near ego yet).
           - The first lanelet inside the AABB marks "entered valid region".
           - Once entered, the first lanelet *outside* the AABB triggers a
             break (we've gone past the relevant portion).
        3. Cap at ``max_segments`` (default 25 = ``_NUM_ROUTE``).

        This is what makes ``route_lanes`` a sliding window of forward
        context that matches training distribution (median 4 non-zero
        slots) instead of a frozen-at-init full route.
        """
        if not route_lanelet_ids:
            return []
        cx, cy = float(center_xy[0]), float(center_xy[1])

        # Step 1: closest route lanelet to ego.
        closest_idx = 0
        closest_dist = float("inf")
        for i, ll_id in enumerate(route_lanelet_ids):
            if ll_id not in self._cache:
                continue
            cl = self._cache[ll_id].raw_centerline
            dx = cl[:, 0] - cx
            dy = cl[:, 1] - cy
            d = float(np.sqrt(dx * dx + dy * dy).min())
            if d < closest_dist:
                closest_dist = d
                closest_idx = i

        # Step 2: walk forward from closest_idx with AABB gating.
        selected: list[int] = []
        has_entered = False
        for i in range(closest_idx, len(route_lanelet_ids)):
            ll_id = route_lanelet_ids[i]
            if ll_id not in self._cache:
                continue
            cl = self._cache[ll_id].raw_centerline
            inside = (
                ((cl[:, 0] > cx - mask_range) & (cl[:, 0] < cx + mask_range)
                 & (cl[:, 1] > cy - mask_range) & (cl[:, 1] < cy + mask_range)).any()
            )
            if not inside:
                if has_entered:
                    break
                continue
            has_entered = True
            selected.append(ll_id)
            if len(selected) >= max_segments:
                break
        return selected

    def _build_map_data(
        self,
        ll_ids: list[int],
        center_xy: np.ndarray | None = None,
    ) -> MapData:
        """Build MapData from lanelet IDs.

        Includes ALL lanelets in the selection (no cap). At inference time the
        tensor converter selects the N closest lanelets per ego agent.

        When ``center_xy`` is provided, ``polygons`` and ``line_strings`` are
        populated with the closest 10 / 60 elements (intersection areas /
        stop_line + road_border) within ±100 m of the center, sorted by
        distance — matching the C++ Autoware planner's preprocessing. This
        gives the model the road-border + intersection context it was
        trained with. Without ``center_xy`` (legacy callers), they remain
        zero-filled.
        """
        segments = []
        speed_limits = []
        has_speed = []
        used_ids: list[int] = []

        for ll_id in ll_ids:
            if ll_id not in self._cache:
                continue
            seg, sl, hsl = self.lanelet_to_33dim(ll_id)
            segments.append(seg)
            speed_limits.append(sl)
            has_speed.append(hsl)
            used_ids.append(ll_id)

        # Store the ordered lanelet IDs for external consumers (e.g.
        # TrafficLightController) that need to map row indices back to
        # lanelet IDs.
        self._last_map_data_ids = used_ids

        n = len(segments)
        if n == 0:
            n = 1  # at least one slot to avoid zero-size arrays
        lanes = np.zeros((n, POINTS_PER_LANELET, 33), dtype=np.float32)
        sl_arr = np.zeros((n, 1), dtype=np.float32)
        hsl_arr = np.zeros((n, 1), dtype=bool)

        for j in range(len(segments)):
            lanes[j] = segments[j]
            sl_arr[j, 0] = speed_limits[j]
            hsl_arr[j, 0] = has_speed[j]

        if center_xy is not None:
            polygons = self.build_polygons_tensor(center_xy)
            line_strings = self.build_line_strings_tensor(center_xy)
        else:
            polygons = np.zeros((10, POINTS_PER_POLYGON, 2 + POLYGON_TYPE_NUM), dtype=np.float32)
            line_strings = np.zeros((60, POINTS_PER_LINE_STRING, 2 + LINE_STRING_TYPE_NUM), dtype=np.float32)

        return MapData(
            lanes=lanes,
            lanes_speed_limit=sl_arr,
            lanes_has_speed_limit=hsl_arr,
            polygons=polygons,
            line_strings=line_strings,
            static_objects=np.zeros((5, 10), dtype=np.float32),
        )
