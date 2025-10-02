from __future__ import annotations

import lanelet2
import numpy as np
import torch
from autoware_lanelet2_extension_python.projection import MGRSProjector
from diffusion_planner.dimensions import (
    POINTS_PER_LANELET,
    POINTS_PER_LINE_STRING,
    POINTS_PER_POLYGON,
)
from numpy.typing import NDArray

from .lanelet_map import (
    Lanelet,
    LaneletMap,
    LineString,
    LineType,
    Polygon,
)

# cspell: ignore MGRS


def _get_attribute(attribute_map, key: str, default: str) -> str:
    if key in attribute_map:
        return attribute_map[key]
    else:
        return default


def _interpolate_lane(waypoints: NDArray, num_points: int):
    assert len(waypoints) >= 2, "At least two waypoints are required"

    # Compute cumulative distances (arc length)
    distances = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        diff = waypoints[i] - waypoints[i - 1]
        norm = np.sqrt(np.sum(diff**2))
        distances[i] = distances[i - 1] + norm

    total_length = distances[-1]

    # Generate target arc lengths
    result = []

    # Always include the first point
    result.append(waypoints[0])

    step = total_length / (num_points - 1)
    seg_idx = 0

    for i in range(1, num_points - 1):
        target = i * step

        # Find the correct segment containing the target arc length
        while seg_idx + 1 < len(distances) and distances[seg_idx + 1] < target:
            seg_idx += 1

        # Ensure we don't go past the last segment
        if seg_idx >= len(distances) - 1:
            seg_idx = len(distances) - 2

        # Interpolate between waypoints[seg_idx] and waypoints[seg_idx + 1]
        seg_start = distances[seg_idx]
        seg_end = distances[seg_idx + 1]
        seg_length = seg_end - seg_start

        # Calculate interpolation parameter, handling zero-length segments
        safe_seg_length = max(seg_length, 1e-6)
        t = (target - seg_start) / safe_seg_length
        # Clamp t to [0, 1] to ensure we don't extrapolate
        t = max(0.0, min(1.0, t))

        # Linear interpolation
        interpolated_point = waypoints[seg_idx] + t * (waypoints[seg_idx + 1] - waypoints[seg_idx])
        result.append(interpolated_point)

    # Always include the last point
    result.append(waypoints[-1])

    new_waypoints = np.array(result)
    assert new_waypoints.shape[0] == num_points, (
        f"Unexpected number of waypoints: {new_waypoints.shape[0]}"
    )
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


def convert_lanelet(filename: str) -> LaneletMap:
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
        LaneletMap: Map data.

    """
    projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    lanelet_map = lanelet2.io.load(filename, projection)

    lanelets: dict[int, Lanelet] = {}
    for lanelet in lanelet_map.laneletLayer:
        lanelet_subtype = _get_attribute(lanelet.attributes, "subtype", "")
        if lanelet_subtype not in ("road", "highway", "road_shoulder", "bicycle_lane"):
            continue

        centerline = _interpolate_lane(
            np.array([(line.x, line.y, line.z) for line in lanelet.centerline]),
            POINTS_PER_LANELET,
        )
        left_boundary = _interpolate_lane(
            np.array([(line.x, line.y, line.z) for line in lanelet.leftBound]),
            POINTS_PER_LANELET,
        )
        right_boundary = _interpolate_lane(
            np.array([(line.x, line.y, line.z) for line in lanelet.rightBound]),
            POINTS_PER_LANELET,
        )

        turn_direction_str = _get_attribute(lanelet.attributes, "turn_direction", "unknown")
        turn_direction_int = {
            "unknown": -1,
            "straight": 0,
            "left": 1,
            "right": 2,
        }[turn_direction_str]

        line_type_left = _get_attribute(lanelet.leftBound.attributes, "type", "")
        line_type_left = LineType.from_str(line_type_left)
        line_type_right = _get_attribute(lanelet.rightBound.attributes, "type", "")
        line_type_right = LineType.from_str(line_type_right)

        kph2mph = 0.621371
        speed_limit_str = _get_attribute(lanelet.attributes, "speed_limit", "")
        speed_limit = float(speed_limit_str) * kph2mph if speed_limit_str else None

        lanelets[lanelet.id] = Lanelet(
            id=lanelet.id,
            centerline=centerline,
            left_boundary=left_boundary,
            left_line_type=line_type_left,
            right_boundary=right_boundary,
            right_line_type=line_type_right,
            speed_limit_mph=speed_limit,
            traffic_lights=lanelet.trafficLights(),
            turn_direction=turn_direction_int,
            center=np.mean(centerline[:, 0:2], axis=0),
        )

    polygons: dict[int, Polygon] = {}
    for polygon in lanelet_map.polygonLayer:
        polygon_type = _get_attribute(polygon.attributes, "type", "")
        polygon_subtype = _get_attribute(polygon.attributes, "subtype", "")
        if polygon_type not in ("intersection_area",):
            continue
        polygon_points = np.array([(point.x, point.y, point.z) for point in polygon])
        polygon_points = interpolate_func(polygon_points, POINTS_PER_POLYGON)
        polygons[polygon.id] = Polygon(
            id=polygon.id,
            polyline=polygon_points,
            type=polygon_type,
            subtype=polygon_subtype,
        )

    line_strings: dict[int, LineString] = {}
    for line_string in lanelet_map.lineStringLayer:
        line_string_type = _get_attribute(line_string.attributes, "type", "")
        line_string_subtype = _get_attribute(line_string.attributes, "subtype", "")
        if line_string_type not in ("stop_line",):
            continue
        line_string_points = np.array([(point.x, point.y, point.z) for point in line_string])
        line_string_points = interpolate_func(line_string_points, POINTS_PER_LINE_STRING)
        line_strings[line_string.id] = LineString(
            id=line_string.id,
            polyline=line_string_points,
            type=line_string_type,
            subtype=line_string_subtype,
        )

    print(f"{len(lanelets)} lanelets are loaded.")
    print(f"{len(polygons)} polygons are loaded.")
    print(f"{len(line_strings)} line strings are loaded.")

    map = LaneletMap(lanelets=lanelets, polygons=polygons, line_strings=line_strings)
    return map


def one_hot_encode(class_index: int, num_class: int) -> NDArray:
    assert 0 <= class_index < num_class, "class_index out of range"
    one_hot = np.zeros(num_class, dtype=np.float32)
    one_hot[class_index] = 1.0
    return one_hot


def judge_inside(center_x, center_y, x, y):
    mask_range = 100.0
    return (
        (x > center_x - mask_range)
        & (x < center_x + mask_range)
        & (y > center_y - mask_range)
        & (y < center_y + mask_range)
    )


def transform(inv_transform_matrix_4x4, points):
    points_4xN = np.vstack((points.T, np.ones(points.shape[0])))
    points_ego = inv_transform_matrix_4x4 @ points_4xN
    points = points_ego[:3, :].T
    return points


def process_lanelet(
    lanelet,
    inv_transform_matrix_4x4,
    center_x,
    center_y,
    traffic_light_recognition,
):
    centerline = lanelet.centerline
    left_boundary = lanelet.left_boundary
    right_boundary = lanelet.right_boundary

    inside_center = judge_inside(center_x, center_y, lanelet.center[0], lanelet.center[1])
    inside_first = judge_inside(center_x, center_y, centerline[0, 0], centerline[0, 1])
    inside_last = judge_inside(center_x, center_y, centerline[-1, 0], centerline[-1, 1])
    if (not inside_center) and (not inside_first) and (not inside_last):
        return None

    # Convert to base_link
    # print(inv_transform_matrix_4x4)
    # print("before")
    # for i in range(centerline.shape[0]):
    #     print(f"{centerline[i][0]:.6f}, {centerline[i][1]:.6f}, {centerline[i][2]:.6f}")
    centerline = transform(inv_transform_matrix_4x4, centerline)
    # print("after")
    # for i in range(centerline.shape[0]):
    #     print(f"{centerline[i][0]:.6f}, {centerline[i][1]:.6f}, {centerline[i][2]:.6f}")
    # exit(1)
    left_boundary = transform(inv_transform_matrix_4x4, left_boundary)
    right_boundary = transform(inv_transform_matrix_4x4, right_boundary)

    left_boundary -= centerline
    right_boundary -= centerline

    diff_centerline = centerline[1:] - centerline[:-1]
    diff_centerline = np.insert(diff_centerline, diff_centerline.shape[0], 0, axis=0)

    traffic_light = [0, 0, 0, 0, 0]  # (green, yellow, red, unknown, no traffic light)
    if len(lanelet.traffic_lights) == 0:
        traffic_light = [0, 0, 0, 0, 1]  # no traffic light
    else:
        if len(lanelet.traffic_lights) > 1:
            print(
                f"Warning: more than one traffic light in lanelet {lanelet.id}, using the first one."
            )
        traffic_light_id = lanelet.traffic_lights[0].id
        if traffic_light_id in traffic_light_recognition:
            elements = traffic_light_recognition[traffic_light_id]
            traffic_light_color = _identify_current_light_status(lanelet.turn_direction, elements)
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
    traffic_light = np.tile(traffic_light, (centerline.shape[0], 1))

    left_line_type = lanelet.left_line_type
    right_line_type = lanelet.right_line_type
    left_line_type_onehot = one_hot_encode(left_line_type.value, LineType.NUM.value)
    right_line_type_onehot = one_hot_encode(right_line_type.value, LineType.NUM.value)
    left_line_type_onehot = np.tile(left_line_type_onehot, (centerline.shape[0], 1))
    right_line_type_onehot = np.tile(right_line_type_onehot, (centerline.shape[0], 1))

    line_data = np.concatenate(
        (
            centerline[:, 0:2],  # xy
            diff_centerline[:, 0:2],  # xy
            left_boundary[:, 0:2],  # xy
            right_boundary[:, 0:2],  # xy
            traffic_light,
            left_line_type_onehot,
            right_line_type_onehot,
        ),
        axis=1,
    )
    assert line_data.shape == (POINTS_PER_LANELET, Lanelet.TENSOR_DIM), (
        f"Unexpected shape: {line_data.shape}"
    )

    # convert from miles per hour to meters per second
    speed_limit_mps = lanelet.speed_limit_mph * 0.44704

    return line_data, speed_limit_mps


def create_lane_tensor(
    lanelets: dict[int, Lanelet],
    map2bl_mat4x4: NDArray,
    center_x: float,
    center_y: float,
    traffic_light_recognition: dict,
    num_segments: int,
    dev: torch.device,
    do_sort: bool,
) -> list[np.ndarray]:
    result_list = []
    for lanelet in lanelets:
        curr_data = process_lanelet(
            lanelet,
            map2bl_mat4x4,
            center_x,
            center_y,
            traffic_light_recognition,
        )
        if curr_data is None:
            continue
        result_list.append(curr_data)

    # sort by distance from the first and last point
    def key_func(item):
        line_data, speed_limit_mps = item
        back_index = -2  # -1 is sometimes the same as next first point, so use -2
        return min(
            np.linalg.norm(line_data[0, :2]),
            np.linalg.norm(line_data[back_index, :2]),
        )

    if do_sort:
        result_list = sorted(result_list, key=key_func)

    result_list = result_list[0:num_segments]

    lanes_tensor = torch.zeros(
        (1, num_segments, POINTS_PER_LANELET, Lanelet.TENSOR_DIM),
        dtype=torch.float32,
        device=dev,
    )
    lanes_speed_limit = torch.zeros((1, num_segments, 1), dtype=torch.float32, device=dev)
    lanes_has_speed_limit = torch.zeros((1, num_segments, 1), dtype=torch.bool, device=dev)

    for i, result_list in enumerate(result_list):
        line_data, speed_limit = result_list
        lanes_tensor[0, i] = torch.from_numpy(line_data).cuda()
        assert speed_limit is not None
        lanes_speed_limit[0, i] = speed_limit
        lanes_has_speed_limit[0, i] = speed_limit is not None

    return lanes_tensor, lanes_speed_limit, lanes_has_speed_limit


def create_line_tensor(
    polygons: dict[int, Polygon],
    map2bl_mat4x4: NDArray,
    center_x: float,
    center_y: float,
    num_elements: int,
    num_points: int,
    dev: torch.device,
) -> list[np.ndarray]:
    result_list = []
    for polygon in polygons:
        polyline = polygon.polyline
        inside_at_least_one = False
        for point in polyline:
            if judge_inside(center_x, center_y, point[0], point[1]):
                inside_at_least_one = True
                break
        if not inside_at_least_one:
            continue
        polyline = transform(map2bl_mat4x4, polyline)
        result_list.append(polyline[:, 0:2])

    # sort by distance from the first and last point
    def key_func(item):
        return np.linalg.norm(item[:, 0:2], axis=1).min()

    result_list = sorted(result_list, key=key_func)
    result_list = result_list[0:num_elements]
    result_list += [np.zeros((num_points, 2), dtype=np.float32)] * (num_elements - len(result_list))
    result_list = np.array(result_list, dtype=np.float32)
    tensor_data = (
        torch.from_numpy(result_list)
        .reshape((1, num_elements, num_points, 2))
        .to(torch.float32)
        .to(dev)
    )
    return tensor_data
