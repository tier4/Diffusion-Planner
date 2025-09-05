from __future__ import annotations

import lanelet2
import numpy as np
import torch
from autoware_lanelet2_extension_python.projection import MGRSProjector
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from shapely import LineString

from .constant import MAP_TYPE_MAPPING, T4_LANE, T4_ROADEDGE, T4_ROADLINE
from .map import MapType
from .polylines_base import BoundaryType
from .static_map import (
    AWMLStaticMap,
    BoundarySegment,
    LaneSegment,
    Polyline,
)
from .uuid import uuid


def _interpolate_points(line, num_point):
    line = LineString(line)
    new_line = np.concatenate(
        [line.interpolate(d).coords._coords for d in np.linspace(0, line.length, num_point)]
    )
    return new_line


# cspell: ignore MGRS


def _get_attribute(attribute_map, key: str, default: str) -> str:
    """Return attribute value from AttributeMap with default fallback.

    Args:
    ----
        attribute_map: AttributeMap object.
        key (str): Attribute key to retrieve.
        default (str): Default value if key is not found.

    Returns:
    -------
        str: Attribute value or default if key is not found.

    """
    if key in attribute_map:
        return attribute_map[key]
    else:
        return default


def _get_boundary_type(linestring: lanelet2.core.LineString3d) -> BoundaryType:
    """Return the `BoundaryType` from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): LineString instance.

    Returns:
    -------
        BoundaryType: BoundaryType instance.

    """
    line_type = _get_attribute(linestring.attributes, "type", "")
    line_subtype = _get_attribute(linestring.attributes, "subtype", "")
    if line_type == "virtual" and line_subtype == "":
        return MapType.UNKNOWN
    elif line_type in T4_ROADEDGE:
        return MAP_TYPE_MAPPING[line_type]
    elif line_subtype in T4_ROADLINE:
        return MAP_TYPE_MAPPING[line_subtype]
    else:
        return MapType.UNKNOWN


def _get_boundary_segment(linestring: lanelet2.core.LineString3d) -> BoundarySegment:
    """Return the `BoundarySegment` from linestring.

    Args:
    ----
        linestring (lanelet2.core.LineString3d): LineString instance.

    Returns:
    -------
        BoundarySegment: BoundarySegment instance.

    """
    boundary_type = _get_boundary_type(linestring)
    waypoints = _interpolate_lane(np.array([(line.x, line.y, line.z) for line in linestring]))
    polyline = Polyline(polyline_type=boundary_type, waypoints=waypoints)
    return BoundarySegment(linestring.id, polyline)


def _get_speed_limit_mph(lanelet: lanelet2.core.Lanelet) -> float | None:
    """Return the lane speed limit in miles per hour (mph).

    Args:
    ----
        lanelet (lanelet2.core.Lanelet): Lanelet instance.

    Returns:
    -------
        float | None: If the lane has the speed limit return float, otherwise None.

    """
    kph2mph = 0.621371
    speed_limit_str = _get_attribute(lanelet.attributes, "speed_limit", "")
    if speed_limit_str:
        return float(speed_limit_str) * kph2mph
    else:
        return None


def _interpolate_lane(waypoints: NDArray):
    # Compute cumulative distances (arc length)
    distances = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        distances[i] = distances[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])

    # Generate new arc lengths with fixed spacing (0.5 meters)
    new_distances = np.arange(0, distances[-1], 0.5)
    new_distances = np.append(new_distances, distances[-1])  # Ensure last point is included

    # Interpolate x, y, z separately
    interp_x = interp1d(distances, waypoints[:, 0], kind="linear")
    interp_y = interp1d(distances, waypoints[:, 1], kind="linear")
    interp_z = interp1d(distances, waypoints[:, 2], kind="linear")

    # Compute new waypoints
    new_waypoints = np.vstack(
        (interp_x(new_distances), interp_y(new_distances), interp_z(new_distances))
    ).T

    # Ensure the first and last points remain unchanged
    # Ensure the first waypoint is exactly the same without duplication
    if not np.allclose(new_waypoints[0], waypoints[0]):
        new_waypoints = np.vstack((waypoints[0], new_waypoints))

    # Ensure the last waypoint is exactly the same without duplication
    if not np.allclose(new_waypoints[-1], waypoints[-1]):
        new_waypoints = np.vstack((new_waypoints, waypoints[-1]))
    return new_waypoints


def _identify_current_light_status(turn_direction: int, traffic_light_elements: list) -> int:
    """
    Identify the current traffic light status based on turn direction and traffic light elements.
    ref: https://github.com/tier4/lanelet2_python_api_for_autoware/blob/rosless_lanelet2/interaction_with_cache_json.ipynb

    Args:
        turn_direction: Integer representing the turn direction (0=straight, 1=left, 2=right)
        traffic_light_elements: List of dictionaries containing traffic light information
                               (color, shape, status, confidence)

    Returns:
        int: The color of the relevant traffic light (0=UNKNOWN, 1=RED, 2=AMBER, 3=GREEN, 4=WHITE)
    """
    # Filter out ineffective elements (color == 0)
    effective_elements = [element for element in traffic_light_elements if element.color != 0]

    # If no effective elements, return UNKNOWN (0)
    if not effective_elements:
        return 0

    # If only one effective element, return its color
    if len(effective_elements) == 1:
        return effective_elements[0].color

    # For multiple elements, find the one that matches the turn direction
    # Map turn direction to corresponding arrow shape
    direction_to_shape_map = {
        0: 4,  # straight -> UP_ARROW
        1: 2,  # left -> LEFT_ARROW
        2: 3,  # right -> RIGHT_ARROW
    }

    target_shape = direction_to_shape_map.get(turn_direction, 0)

    # First priority: Find elements with exactly matching direction
    matching_elements = [element for element in effective_elements if element.shape == target_shape]
    if matching_elements:
        # If multiple matching elements, take the one with highest confidence
        return max(matching_elements, key=lambda x: x.confidence).color

    # Second priority: Find circle elements
    circle_elements = [element for element in effective_elements if element.shape == 1]  # CIRCLE
    if circle_elements:
        # If multiple circle elements, take the one with highest confidence
        return max(circle_elements, key=lambda x: x.confidence).color

    # If no matching direction or circle, return the element with highest confidence
    return max(effective_elements, key=lambda x: x.confidence).color


def convert_lanelet(filename: str) -> AWMLStaticMap:
    """Convert lanelet (.osm) to map info.

    Note:
    ----
        Currently, following subtypes are skipped:
            walkway

    Args:
    ----
        filename (str): Path to osm file.

    Returns:
    -------
        AWMLStaticMap: Static map data.

    """
    projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    lanelet_map = lanelet2.io.load(filename, projection)

    lane_segments: dict[int, LaneSegment] = {}
    taken_boundary_ids: list[int] = []
    for lanelet in lanelet_map.laneletLayer:
        lanelet_subtype = _get_attribute(lanelet.attributes, "subtype", "")

        # NOTE: skip walkway because it contains stop_line as boundary
        if lanelet_subtype in T4_LANE:
            # lane
            lane_type = MAP_TYPE_MAPPING[lanelet_subtype]
            lane_waypoints = _interpolate_lane(
                np.array([(line.x, line.y, line.z) for line in lanelet.centerline])
            )
            lane_polyline = Polyline(polyline_type=lane_type, waypoints=lane_waypoints)
            speed_limit_mph = _get_speed_limit_mph(lanelet)

            # road line or road edge
            left_linestring = lanelet.leftBound
            right_linestring = lanelet.rightBound
            left_boundary = _get_boundary_segment(left_linestring)
            right_boundary = _get_boundary_segment(right_linestring)
            taken_boundary_ids.extend((left_linestring.id, right_linestring.id))

            turn_direction_str = _get_attribute(lanelet.attributes, "turn_direction", "unknown")
            turn_direction_int = {
                "unknown": -1,
                "straight": 0,
                "left": 1,
                "right": 2,
            }[turn_direction_str]

            lane_segments[lanelet.id] = LaneSegment(
                id=lanelet.id,
                polyline=lane_polyline,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                speed_limit_mph=speed_limit_mph,
                traffic_lights=lanelet.trafficLights(),
                turn_direction=turn_direction_int,
            )

    boundary_segments: dict[int, BoundarySegment] = {}
    for linestring in lanelet_map.lineStringLayer:
        type_name: str = _get_attribute(linestring.attributes, "type", "")
        if (
            type_name in T4_ROADEDGE or type_name in T4_ROADLINE
        ) and linestring.id not in taken_boundary_ids:
            boundary_segments[linestring.id] = _get_boundary_segment(linestring)

    # generate uuid from map filepath
    map_id = uuid(filename, digit=16)
    map = AWMLStaticMap(map_id, lane_segments=lane_segments)
    map = _fix_point_num(map)
    return map


def _fix_point_num(map: AWMLStaticMap):
    for segment_id, segment in map.lane_segments.items():
        centerlines = segment.polyline.waypoints
        left_boundary = segment.left_boundary.polyline.waypoints
        right_boundary = segment.right_boundary.polyline.waypoints

        # Fix the number of points to 20
        segment.polyline.waypoints = _interpolate_points(centerlines, 20)
        segment.left_boundary.polyline.waypoints = _interpolate_points(left_boundary, 20)
        segment.right_boundary.polyline.waypoints = _interpolate_points(right_boundary, 20)

        # To crop the map faster, we need to set the center of the segment
        segment.center = np.mean(centerlines[:, 0:2], axis=0)
    return map


def process_segment(
    segment,
    inv_transform_matrix_4x4,
    center_x,
    center_y,
    mask_range,
    traffic_light_recognition,
):
    centerlines = segment.polyline.waypoints
    left_boundary = segment.left_boundary.polyline.waypoints
    right_boundary = segment.right_boundary.polyline.waypoints

    def judge_inside(x, y):
        return (
            (x > center_x - mask_range)
            & (x < center_x + mask_range)
            & (y > center_y - mask_range)
            & (y < center_y + mask_range)
        )

    inside_center = judge_inside(segment.center[0], segment.center[1])
    inside_first = judge_inside(centerlines[0, 0], centerlines[0, 1])
    inside_last = judge_inside(centerlines[-1, 0], centerlines[-1, 1])
    if (not inside_center) and (not inside_first) and (not inside_last):
        return None

    # Convert to base_link
    centerlines_4xN = np.vstack((centerlines.T, np.ones(centerlines.shape[0])))
    centerlines_ego = inv_transform_matrix_4x4 @ centerlines_4xN
    centerlines = centerlines_ego[:3, :].T
    left_boundaries_4xN = np.vstack((left_boundary.T, np.ones(left_boundary.shape[0])))
    left_boundaries_ego = inv_transform_matrix_4x4 @ left_boundaries_4xN
    left_boundary = left_boundaries_ego[:3, :].T
    right_boundaries_4xN = np.vstack((right_boundary.T, np.ones(right_boundary.shape[0])))
    right_boundaries_ego = inv_transform_matrix_4x4 @ right_boundaries_4xN
    right_boundary = right_boundaries_ego[:3, :].T

    left_boundary -= centerlines
    right_boundary -= centerlines

    diff_centerlines = centerlines[1:] - centerlines[:-1]
    diff_centerlines = np.insert(diff_centerlines, diff_centerlines.shape[0], 0, axis=0)

    traffic_light = [0, 0, 0, 0, 0]  # (green, yellow, red, unknown, no traffic light)
    if len(segment.traffic_lights) == 0:
        traffic_light = [0, 0, 0, 0, 1]  # no traffic light
    else:
        if len(segment.traffic_lights) > 1:
            print(
                f"Warning: more than one traffic light in segment {segment.id}, using the first one."
            )
        traffic_light_id = segment.traffic_lights[0].id
        if traffic_light_id in traffic_light_recognition:
            elements = traffic_light_recognition[traffic_light_id]
            traffic_light_color = _identify_current_light_status(segment.turn_direction, elements)
            # https://github.com/autowarefoundation/autoware_msgs/blob/main/autoware_perception_msgs/msg/TrafficLightElement.msg
            if traffic_light_color == 0:  # UNKNOWN
                traffic_light[3] = 1
            elif traffic_light_color == 1:  # RED
                traffic_light[2] = 1
            elif traffic_light_color == 2:  # AMBER
                traffic_light[1] = 1
            elif traffic_light_color == 3:  # GREEN
                traffic_light[0] = 1
            elif traffic_light_color == 4:  # WHITE
                traffic_light[3] = 1
            else:
                assert False, f"Unexpected traffic light color: {traffic_light_color}"
        else:
            traffic_light[3] = 1
    traffic_light = np.tile(traffic_light, (centerlines.shape[0], 1))

    line_data = np.concatenate(
        (
            centerlines[:, 0:2],  # xy
            diff_centerlines[:, 0:2],  # xy
            left_boundary[:, 0:2],  # xy
            right_boundary[:, 0:2],  # xy
            traffic_light,
        ),
        axis=1,
    )

    # convert from miles per hour to meters per second
    speed_limit_mps = segment.speed_limit_mph * 0.44704

    return line_data, speed_limit_mps


def create_lane_tensor(
    lane_segments: list,
    map2bl_mat4x4: NDArray,
    center_x: float,
    center_y: float,
    mask_range: float,
    traffic_light_recognition: dict,
    num_segments: int,
    dev: torch.device,
    do_sort: bool,
) -> list[np.ndarray]:
    result_list = []
    for segment in lane_segments:
        curr_data = process_segment(
            segment,
            map2bl_mat4x4,
            center_x,
            center_y,
            mask_range,
            traffic_light_recognition,
        )
        if curr_data is None:
            continue
        result_list.append(curr_data)

    # sort by distance from the first and last point
    def key_func(item):
        line_data, speed_limit_mps = item
        return min(
            np.linalg.norm(line_data[0, :2]),
            np.linalg.norm(line_data[-1, :2]),
        )

    if do_sort:
        result_list = sorted(result_list, key=key_func)

    result_list = result_list[0:num_segments]

    lanes_tensor = torch.zeros((1, num_segments, 20, 13), dtype=torch.float32, device=dev)
    lanes_speed_limit = torch.zeros((1, num_segments, 1), dtype=torch.float32, device=dev)
    lanes_has_speed_limit = torch.zeros((1, num_segments, 1), dtype=torch.bool, device=dev)

    for i, result_list in enumerate(result_list):
        line_data, speed_limit = result_list
        lanes_tensor[0, i] = torch.from_numpy(line_data).cuda()
        assert speed_limit is not None
        lanes_speed_limit[0, i] = speed_limit
        lanes_has_speed_limit[0, i] = speed_limit is not None

    return lanes_tensor, lanes_speed_limit, lanes_has_speed_limit
