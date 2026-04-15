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

        traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )
        self._routing_graph = lanelet2.routing.RoutingGraph(self._lanelet_map, traffic_rules)

        # Cache per-lanelet geometry
        self._cache: dict[int, _CachedLanelet] = {}
        self._ll_by_id = {}  # lanelet2 objects by id
        self._vehicle_subtypes = {"road", "highway", "road_shoulder"}

        for ll in self._lanelet_map.laneletLayer:
            subtype = ll.attributes["subtype"] if "subtype" in ll.attributes else ""
            raw_cl = np.array([(p.x, p.y) for p in ll.centerline], dtype=np.float32)
            raw_lb = np.array([(p.x, p.y) for p in ll.leftBound], dtype=np.float32)
            raw_rb = np.array([(p.x, p.y) for p in ll.rightBound], dtype=np.float32)

            if len(raw_cl) < 2:
                continue

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

        print(f"LaneletSceneBuilder: cached {len(self._cache)} lanelets, "
              f"routing graph built")

    # ── 33-dim conversion ────────────────────────────────────────────────

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

    # ── Routing ──────────────────────────────────────────────────────────

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
        for i in range(len(cl) - 1, -1, -1):
            if arc[i] < best_param - 0.1:
                if np.linalg.norm(cl[i] - points[-1]) > 0.01:
                    points.append(cl[i].copy())

        total_dist = sum(
            np.linalg.norm(points[j] - points[j - 1]) for j in range(1, len(points))
        )

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
            # Predecessor exit connects to current entrance: add from exit to entrance
            for pt in prev_cl[::-1]:
                if np.linalg.norm(pt - points[-1]) > 0.01:
                    points.append(pt.copy())
            total_dist += self._cache[prev_ll.id].arc_length
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

        # Find nearest lanelet centerline
        best_ll_id = None
        best_dist = float("inf")
        for ll_id in lanelet_ids:
            if ll_id not in self._cache:
                continue
            cl = self._cache[ll_id].raw_centerline
            dists = np.linalg.norm(cl - pos, axis=1)
            d = float(dists.min())
            if d < best_dist:
                best_dist = d
                best_ll_id = ll_id

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

    def _build_map_data(self, ll_ids: list[int]) -> MapData:
        """Build MapData from lanelet IDs.

        Includes ALL lanelets in the selection (no cap). At inference time the
        tensor converter selects the N closest lanelets per ego agent.
        """
        segments = []
        speed_limits = []
        has_speed = []

        for ll_id in ll_ids:
            if ll_id not in self._cache:
                continue
            seg, sl, hsl = self.lanelet_to_33dim(ll_id)
            segments.append(seg)
            speed_limits.append(sl)
            has_speed.append(hsl)

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

        return MapData(
            lanes=lanes,
            lanes_speed_limit=sl_arr,
            lanes_has_speed_limit=hsl_arr,
            polygons=np.zeros((10, 40, 3), dtype=np.float32),
            line_strings=np.zeros((60, 20, 4), dtype=np.float32),
            static_objects=np.zeros((5, 10), dtype=np.float32),
        )
