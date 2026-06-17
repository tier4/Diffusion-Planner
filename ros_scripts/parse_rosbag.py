import argparse
import json
import logging
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rosbag2_py
import torch
import yaml
from autoware_perception_msgs.msg import (
    TrackedObjects,
    TrafficLightGroupArray,
)
from autoware_planning_msgs.msg import LaneletRoute
from autoware_vehicle_msgs.msg import TurnIndicatorsReport
from diffusion_planner.dimensions import *
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import (
    LINE_STRING_TYPE_MAP,
    LINE_STRING_TYPE_NUM,
    POLYGON_TYPE_MAP,
    POLYGON_TYPE_NUM,
    convert_lanelet,
    create_lane_tensor,
    create_line_tensor,
)
from diffusion_planner_ros.utils import (
    convert_tracked_objects_to_tensor,
    create_current_ego_state,
    filter_route_lanelets,
    get_nearest_msg,
    get_transform_matrix,
    parse_timestamp,
    parse_traffic_light_recognition,
    pose_to_mat4x4,
    rot3x3_to_heading_cos_sin,
    tracking_one_step,
)
from geometry_msgs.msg import AccelWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
from tqdm import tqdm

"""
This script makes npz files from a rosbag.
[Contents of a npz file]
version                     int32   ()
ego_agent_past              float32 (INPUT_T + 1, 3)
ego_current_state           float32 (10,)
ego_agent_future            float32 (OUTPUT_T, 3)
neighbor_agents_past        float32 (MAX_NUM_NEIGHBORS, INPUT_T + 1, 11)
neighbor_agents_future      float32 (MAX_NUM_NEIGHBORS, OUTPUT_T, 3)
static_objects              float32 (5, 10)
lanes                       float32 (NUM_SEGMENTS_IN_LANE, POINTS_PER_LANELET, SEGMENT_POINT_DIM)
lanes_speed_limit           float32 (NUM_SEGMENTS_IN_LANE, 1)
lanes_has_speed_limit       bool    (NUM_SEGMENTS_IN_LANE, 1)
route_lanes                 float32 (NUM_SEGMENTS_IN_ROUTE, POINTS_PER_LANELET, SEGMENT_POINT_DIM)
route_lanes_speed_limit     float32 (NUM_SEGMENTS_IN_ROUTE, 1)
route_lanes_has_speed_limit bool    (NUM_SEGMENTS_IN_ROUTE, 1)
turn_indicators             int32   (INPUT_T + 1)
"""
PAST_TIME_STEPS = INPUT_T + 1

# Synthetic clock period for frame assembly (10 Hz), matching the C++ build_sequences.
CLOCK_PERIOD_NS = 100_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("rosbag_path", type=Path)
    parser.add_argument("vector_map_path", type=Path)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--min_frames", type=int, default=1700)
    parser.add_argument("--search_nearest_route", type=int, default=1)
    return parser.parse_args()


@dataclass
class FrameData:
    timestamp: int
    route: LaneletRoute
    tracked_objects: TrackedObjects
    kinematic_state: Odometry
    acceleration: AccelWithCovarianceStamped
    traffic_signals: TrafficLightGroupArray
    turn_indicator: TurnIndicatorsReport


@dataclass
class SequenceData:
    """This class means one sequence of data.
    It contains exactly one route msg and multiple other msgs.
    A rosbag may contain multiple sequences.
    """

    data_list: list[FrameData]
    route: LaneletRoute


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, ratio: float) -> np.ndarray:
    """Shortest-arc quaternion slerp ([x, y, z, w]), matching tf2::slerp."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        res = q0 + ratio * (q1 - q0)
        return res / np.linalg.norm(res)
    theta0 = np.arccos(dot)
    sin0 = np.sin(theta0)
    s0 = np.sin((1.0 - ratio) * theta0) / sin0
    s1 = np.sin(ratio * theta0) / sin0
    return s0 * q0 + s1 * q1


def _interpolated_pose_mat4x4(pose0, pose1, ratio: float) -> np.ndarray:
    """Linear position + slerp orientation, matching calc_interpolated_pose(..., false)."""
    ratio = min(max(ratio, 0.0), 1.0)
    p0 = np.array([pose0.position.x, pose0.position.y, pose0.position.z])
    p1 = np.array([pose1.position.x, pose1.position.y, pose1.position.z])
    q0 = np.array(
        [pose0.orientation.x, pose0.orientation.y, pose0.orientation.z, pose0.orientation.w]
    )
    q1 = np.array(
        [pose1.orientation.x, pose1.orientation.y, pose1.orientation.z, pose1.orientation.w]
    )
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_quat(_slerp_quat(q0, q1, ratio)).as_matrix()
    mat[:3, 3] = p0 + ratio * (p1 - p0)
    return mat


def create_ego_sequence_interp(
    data_list, start_idx, num_timesteps, map2bl_matrix_4x4, reference_time_sec
):
    """Time-based ego sequence matching C++ create_ego_sequence + create_ego_agent_past.

    Collects odom from start_idx until its header stamp reaches reference_time, then samples
    num_timesteps poses at PREDICTION_TIME_STEP_S (0.1 s) spacing ending at reference_time,
    linearly interpolating by header stamp. Returns (num_timesteps, 3) of [x, y, yaw] in ego
    frame, or None if the data does not cover reference_time.
    """
    odoms = []
    for j in range(max(0, start_idx), len(data_list)):
        odoms.append(data_list[j].kinematic_state)
        if _stamp_to_sec(data_list[j].kinematic_state.header.stamp) >= reference_time_sec:
            break
    if len(odoms) == 0 or _stamp_to_sec(odoms[-1].header.stamp) < reference_time_sec:
        return None

    dt = 0.1  # PREDICTION_TIME_STEP_S
    first_sec = _stamp_to_sec(odoms[0].header.stamp)
    last_sec = _stamp_to_sec(odoms[-1].header.stamp)

    out = np.zeros((num_timesteps, 3), dtype=np.float64)
    search_start = 0
    for t in range(num_timesteps):
        # t=0 is the oldest, t=num_timesteps-1 is the reference time.
        target_sec = reference_time_sec - (num_timesteps - 1 - t) * dt
        if target_sec <= first_sec:
            pose_mat = pose_to_mat4x4(odoms[0].pose.pose)
        elif target_sec >= last_sec:
            pose_mat = pose_to_mat4x4(odoms[-1].pose.pose)
        else:
            while search_start + 1 < len(odoms):
                t_next = _stamp_to_sec(odoms[search_start + 1].header.stamp)
                if target_sec <= t_next:
                    break
                search_start += 1
            t0 = _stamp_to_sec(odoms[search_start].header.stamp)
            t1 = _stamp_to_sec(odoms[search_start + 1].header.stamp)
            ratio = (target_sec - t0) / (t1 - t0) if t1 > t0 else 0.0
            pose_mat = _interpolated_pose_mat4x4(
                odoms[search_start].pose.pose, odoms[search_start + 1].pose.pose, ratio
            )
        pose_ego = map2bl_matrix_4x4 @ pose_mat
        out[t, 0] = pose_ego[0, 3]
        out[t, 1] = pose_ego[1, 3]
        out[t, 2] = np.arctan2(pose_ego[1, 0], pose_ego[0, 0])
    return out


def create_ego_sequence(data_list, i, OUTPUT_T, map2bl_matrix_4x4):
    ego_future_x = []
    ego_future_y = []
    ego_future_yaw = []
    for j in range(OUTPUT_T):
        index = min(i + j + 1, len(data_list) - 1)
        x = data_list[index].kinematic_state.pose.pose.position.x
        y = data_list[index].kinematic_state.pose.pose.position.y
        z = data_list[index].kinematic_state.pose.pose.position.z
        qx = data_list[index].kinematic_state.pose.pose.orientation.x
        qy = data_list[index].kinematic_state.pose.pose.orientation.y
        qz = data_list[index].kinematic_state.pose.pose.orientation.z
        qw = data_list[index].kinematic_state.pose.pose.orientation.w
        rot = Rotation.from_quat([qx, qy, qz, qw])
        pose_in_map = np.eye(4)
        pose_in_map[:3, :3] = rot.as_matrix()
        pose_in_map[0, 3] = x
        pose_in_map[1, 3] = y
        pose_in_map[2, 3] = z
        pose_in_bl = map2bl_matrix_4x4 @ pose_in_map
        ego_future_x.append(pose_in_bl[0, 3])
        ego_future_y.append(pose_in_bl[1, 3])
        rot = Rotation.from_matrix(pose_in_bl[:3, :3])
        yaw = rot.as_euler("xyz")[2]
        ego_future_yaw.append(yaw)

    ego_future_np = np.concatenate(
        [
            np.array(ego_future_x).reshape(-1, 1),
            np.array(ego_future_y).reshape(-1, 1),
            np.array(ego_future_yaw).reshape(-1, 1),
        ],
        axis=1,
    )
    return ego_future_np


# Neighbor-agent input builders live in the venv-safe `reproducer_inputs` module so both this
# converter and the closed-loop Perception Reproducer share the exact same (parity-matched) code.
from reproducer_inputs import build_neighbor_future, build_neighbor_past  # noqa: E402


def create_neighbor_future(
    tracked_objs: dict,
    map2bl_matrix_4x4: np.ndarray,
    max_num_objects: int,
    max_timesteps: int,
) -> torch.Tensor:
    """
    This function is similar to utils.convert_tracked_objects_to_tensor, but there are some discrepancies.
    """
    neighbor = torch.zeros((1, max_num_objects, max_timesteps, 11))

    # [Discrepancy1] for future tensor, sort is not needed

    for i, (object_id_bytes, tracked_obj) in enumerate(tracked_objs.items()):
        if i >= max_num_objects:
            break
        label_in_model = tracked_obj.class_label
        # [Discrepancy2] for future tensor, write values between 0 to len(tracked_obj.kinematics_list) - 1
        for j in range(min(max_timesteps, len(tracked_obj.kinematics_list))):
            # [Discrepancy3] for future tensor, the order is forward
            kinematics = tracked_obj.kinematics_list[j]
            shape = tracked_obj.shape_list[j]
            pose_in_map_4x4 = pose_to_mat4x4(kinematics.pose_with_covariance.pose)
            pose_in_bl_4x4 = map2bl_matrix_4x4 @ pose_in_map_4x4
            cos, sin = rot3x3_to_heading_cos_sin(pose_in_bl_4x4[0:3, 0:3])
            twist_in_local = np.array(
                [
                    kinematics.twist_with_covariance.twist.linear.x,
                    kinematics.twist_with_covariance.twist.linear.y,
                    kinematics.twist_with_covariance.twist.linear.z,
                ]
            )
            twist_in_map = pose_in_map_4x4[0:3, 0:3] @ twist_in_local
            twist_in_bl = map2bl_matrix_4x4[0:3, 0:3] @ twist_in_map
            neighbor[0, i, j, 0] = pose_in_bl_4x4[0, 3]  # x
            neighbor[0, i, j, 1] = pose_in_bl_4x4[1, 3]  # y
            neighbor[0, i, j, 2] = cos  # heading cos
            neighbor[0, i, j, 3] = sin  # heading sin
            neighbor[0, i, j, 4] = twist_in_bl[0]  # velocity x
            neighbor[0, i, j, 5] = twist_in_bl[1]  # velocity y
            # I don't know why but sometimes the length and width from autoware are 0
            neighbor[0, i, j, 6] = max(shape.dimensions.y, 1.0)  # width
            neighbor[0, i, j, 7] = max(shape.dimensions.x, 1.0)  # length
            neighbor[0, i, j, 8] = label_in_model == 0  # vehicle
            neighbor[0, i, j, 9] = label_in_model == 1  # pedestrian
            neighbor[0, i, j, 10] = label_in_model == 2  # bicycle
    return neighbor


def tracking_past_and_future(data_list, i, map2bl_matrix_4x4):
    # tracking for past (including current frame)
    tracking_past = {}
    for frame_data in data_list[i - PAST_TIME_STEPS + 1 : i + 1]:
        tracking_past = tracking_one_step(frame_data.tracked_objects, tracking_past)

    # sort tracking_past by distance to the ego
    def sort_key(item):
        _, tracked_obj = item
        last_kinematics = tracked_obj.kinematics_list[-1]
        pose_in_map = pose_to_mat4x4(last_kinematics.pose_with_covariance.pose)
        pose_in_bl = map2bl_matrix_4x4 @ pose_in_map
        return np.linalg.norm(pose_in_bl[0:2, 3])

    tracking_past = dict(sorted(tracking_past.items(), key=sort_key))

    # tracking for future (for ground truth)
    tracking_future = deepcopy(tracking_past)
    # reset lost_time and list
    for key in tracking_future.keys():
        tracking_future[key].lost_time = 1
        tracking_future[key].shape_list = tracking_future[key].shape_list[-1:]
        tracking_future[key].kinematics_list = tracking_future[key].kinematics_list[-1:]
    # tracking
    for frame_data in data_list[i : i + OUTPUT_T]:
        tracking_future = tracking_one_step(
            frame_data.tracked_objects,
            tracking_future,
            lost_time_limit=100000,  # to avoid lost
        )
        # filter tracking_future by tracking_past
        # (remove the objects that are not in tracking_past)
        tracking_future = {
            key: value for key, value in tracking_future.items() if key in tracking_past
        }
    # erase losing from tracking_future
    for key, val in tracking_future.items():
        total = len(val.kinematics_list)
        lost_time = val.lost_time
        valid_t = total - lost_time
        val.shape_list = val.shape_list[0:valid_t]
        val.kinematics_list = val.kinematics_list[0:valid_t]

    assert len(tracking_past.keys()) == len(tracking_future.keys())
    return tracking_past, tracking_future


def get_relative_goal_pose(goal_pose, map2bl_matrix_4x4):
    """Get the relative goal pose in the base link frame."""
    pose_in_map = pose_to_mat4x4(goal_pose)
    pose_in_bl = map2bl_matrix_4x4 @ pose_in_map
    x = pose_in_bl[0, 3]
    y = pose_in_bl[1, 3]
    yaw = np.arctan2(pose_in_bl[1, 0], pose_in_bl[0, 0])
    return np.array([x, y, yaw], dtype=np.float32)


def build_sequences_from_rosbag(
    rosbag_path: Path,
    limit: int = -1,
    search_nearest_route: int = 1,
    logger: logging.Logger | None = None,
) -> list:
    """Read a rosbag and assemble fixed-rate (10 Hz) per-route sequences of FrameData.

    This is the C++ build_sequences port (synthetic clock + zero-order hold on rosbag time +
    persistent traffic-light TTL map). Returns a list of SequenceData, each frame carrying the
    ego kinematic_state, acceleration, tracked_objects, a traffic recognition snapshot
    (group_id -> elements) and the route. Shared by the npz converter (`main`) and the
    closed-loop Perception Reproducer.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # parse rosbag
    serialization_format = "cdr"
    metadata_yaml_path = rosbag_path / "metadata.yaml"
    metadata_yaml = yaml.safe_load(metadata_yaml_path.read_text(encoding="utf-8"))
    storage_id = metadata_yaml["rosbag2_bagfile_information"]["storage_identifier"]
    storage_options = rosbag2_py.StorageOptions(uri=str(rosbag_path), storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    target_topic_list = [
        "/localization/kinematic_state",
        "/localization/acceleration",
        "/perception/object_recognition/tracking/objects",
        "/perception/traffic_light_recognition/traffic_signals",
        "/planning/mission_planning/route",
        "/vehicle/status/turn_indicators_status",
    ]

    storage_filter = rosbag2_py.StorageFilter(topics=target_topic_list)
    reader.set_filter(storage_filter)

    # Read all messages, keeping each message's rosbag (recording) time `t`.
    # The C++ build_sequences keys every topic off the rosbag time, NOT the header stamp.
    topic_name_to_timed = {topic: [] for topic in target_topic_list}
    parse_num = 0
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic not in target_topic_list:
            continue
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        topic_name_to_timed[topic].append((t, msg))
        parse_num += 1
        if limit > 0 and parse_num >= limit:
            break

    for key, value in topic_name_to_timed.items():
        logger.info(f"{key}: {len(value)} msgs")

    kinematic_states = topic_name_to_timed["/localization/kinematic_state"]
    accelerations = topic_name_to_timed["/localization/acceleration"]
    tracked_objects_msgs = topic_name_to_timed["/perception/object_recognition/tracking/objects"]
    turn_indicators = topic_name_to_timed["/vehicle/status/turn_indicators_status"]
    traffic_signals = topic_name_to_timed["/perception/traffic_light_recognition/traffic_signals"]
    route_entries = topic_name_to_timed["/planning/mission_planning/route"]

    if len(route_entries) == 0:
        logger.info("No route messages; nothing to build.")
        return []

    required_topics = [kinematic_states, accelerations, tracked_objects_msgs, turn_indicators]
    if any(len(timed) == 0 for timed in required_topics):
        logger.info("A required topic has no messages; nothing to build.")
        return []

    # Merge consecutive routes that share a start_pose (FreeSpacePlanner sometimes only
    # changes goal_pose, so such routes belong to one sequence). route_to_group maps a
    # route index -> its merged-sequence index.
    sequence_data_list = []
    route_to_group = [0] * len(route_entries)
    for j in range(len(route_entries)):
        if j > 0 and route_entries[j][1].start_pose == route_entries[j - 1][1].start_pose:
            route_to_group[j] = len(sequence_data_list) - 1
        else:
            route_to_group[j] = len(sequence_data_list)
            sequence_data_list.append(SequenceData([], route_entries[j][1]))

    # Build per-route sequences of fixed-rate (10 Hz) frames. Pure assembly: every tick
    # produces exactly one frame carrying the latest message at-or-before the tick for each
    # topic (zero-order hold). Skip decisions happen later in the per-frame loop.
    earliest_route = min(t for t, _ in route_entries)
    clock_start = max(
        kinematic_states[0][0],
        accelerations[0][0],
        tracked_objects_msgs[0][0],
        turn_indicators[0][0],
        earliest_route,
    )
    # End once any required topic is exhausted; beyond its last message we could only carry
    # stale data forward. Traffic is drop-tolerated and does not gate the clock.
    clock_end = min(
        kinematic_states[-1][0],
        accelerations[-1][0],
        tracked_objects_msgs[-1][0],
        turn_indicators[-1][0],
    )
    logger.info(f"clock_start={clock_start} clock_end={clock_end} period_ns={CLOCK_PERIOD_NS}")

    def advance_cursor(timed, cursor, tick):
        # Advance to the largest index whose rosbag_time is <= tick.
        while cursor + 1 < len(timed) and timed[cursor + 1][0] <= tick:
            cursor += 1
        return cursor

    kin_cursor = accel_cursor = tracked_cursor = turn_ind_cursor = traffic_high_cursor = -1
    traffic_low_cursor = 0
    # Persistent traffic-light state (group_id -> (stamp_ns, elements)), maintained at 10 Hz
    # with a 5 s TTL exactly like the C++ build_sequences / process_traffic_signals.
    traffic_map = {}
    traffic_ttl_ns = int(5.0 * 1e9)
    tick = clock_start
    while tick <= clock_end:
        kin_cursor = advance_cursor(kinematic_states, kin_cursor, tick)
        accel_cursor = advance_cursor(accelerations, accel_cursor, tick)
        tracked_cursor = advance_cursor(tracked_objects_msgs, tracked_cursor, tick)
        turn_ind_cursor = advance_cursor(turn_indicators, turn_ind_cursor, tick)
        traffic_high_cursor = advance_cursor(traffic_signals, traffic_high_cursor, tick)

        kinematic = kinematic_states[kin_cursor][1]
        accel = accelerations[accel_cursor][1]
        tracked = tracked_objects_msgs[tracked_cursor][1]
        turn_ind = turn_indicators[turn_ind_cursor][1]

        # Fold the traffic msgs that arrived since the previous tick into the persistent map
        # (latest stamp per light group), then expire entries older than the TTL.
        for k in range(traffic_low_cursor, traffic_high_cursor + 1):
            msg = traffic_signals[k][1]
            msg_stamp = parse_timestamp(msg.stamp)
            for signal in msg.traffic_light_groups:
                gid = signal.traffic_light_group_id
                if gid not in traffic_map or msg_stamp > traffic_map[gid][0]:
                    traffic_map[gid] = (msg_stamp, signal.elements)
        traffic_low_cursor = traffic_high_cursor + 1
        for gid in [g for g, (stamp, _) in traffic_map.items() if tick - stamp > traffic_ttl_ns]:
            del traffic_map[gid]
        # Snapshot the current recognition state (group_id -> elements) for this frame.
        traffic = {gid: elements for gid, (_stamp, elements) in traffic_map.items()}

        # Resolve the route for this tick (latest route with rosbag_time <= tick).
        max_route_index = 0
        if search_nearest_route:
            best_route_time = None
            for j in range(len(route_entries)):
                route_time = route_entries[j][0]
                if route_time <= tick and (
                    best_route_time is None or route_time >= best_route_time
                ):
                    best_route_time = route_time
                    max_route_index = j

        group = route_to_group[max_route_index]
        sequence_data_list[group].data_list.append(
            FrameData(
                timestamp=tick,
                route=sequence_data_list[group].route,
                tracked_objects=tracked,
                kinematic_state=kinematic,
                acceleration=accel,
                traffic_signals=traffic,
                turn_indicator=turn_ind,
            )
        )
        tick += CLOCK_PERIOD_NS

    # Frames are pushed in tick order; keep each sequence ascending by timestamp.
    for seq in sequence_data_list:
        seq.data_list.sort(key=lambda fd: fd.timestamp)

    return sequence_data_list


def main(
    rosbag_path: Path,
    vector_map_path: Path,
    save_dir: Path,
    step: int,
    limit: int,
    min_frames: int,
    search_nearest_route: bool,
):
    log_dir = save_dir.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{rosbag_path.stem}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    vector_map = convert_lanelet(str(vector_map_path))

    sequence_data_list = build_sequences_from_rosbag(
        rosbag_path, limit=limit, search_nearest_route=search_nearest_route, logger=logger
    )
    if not sequence_data_list:
        return

    map_name = rosbag_path.stem
    sequence_num = len(sequence_data_list)
    logger.info(f"Total {sequence_num} sequences")

    for seq_id, seq in enumerate(sequence_data_list):
        logger.info(f"Processing sequence {seq_id + 1}/{sequence_num}")

        data_list = seq.data_list
        n = len(data_list)
        logger.info(f"Total {n} frames")

        if n == 0:
            continue
        # Match the C++ converter: use the sequence's last frame ego pose as the goal pose
        # (FreeSpacePlanner only updates goal_pose, so the route's own goal is unreliable).
        goal_pose = data_list[-1].kinematic_state.pose.pose

        # if less than min_frames (default 3 min), skip this sequence
        if n < min_frames:
            logger.info(
                f"Skip this sequence because the number of frames {n} is less than {min_frames} frames"
            )
            continue

        # list[FrameData] -> npz
        progress = tqdm(total=(n - PAST_TIME_STEPS - OUTPUT_T) // step)
        stopping_count = 0
        for i in range(PAST_TIME_STEPS, n, step):
            progress.update(1)
            token = f"{seq_id:08d}{i:08d}"

            bl2map_matrix_4x4, map2bl_matrix_4x4 = get_transform_matrix(
                data_list[i].kinematic_state
            )

            # traffic_signals already holds the persistent {group_id: elements} snapshot.
            traffic_light_recognition = data_list[i].traffic_signals

            # lanes
            lanes_tensor, lanes_speed_limit, lanes_has_speed_limit = create_lane_tensor(
                vector_map.lanelets.values(),
                map2bl_mat4x4=map2bl_matrix_4x4,
                center_x=data_list[i].kinematic_state.pose.pose.position.x,
                center_y=data_list[i].kinematic_state.pose.pose.position.y,
                traffic_light_recognition=traffic_light_recognition,
                num_segments=NUM_SEGMENTS_IN_LANE,
                dev="cpu",
                do_sort=True,
            )

            # routes
            route_lanelets = [
                vector_map.lanelets[segment.preferred_primitive.id]
                for segment in data_list[i].route.segments
            ]
            route_lanelets = filter_route_lanelets(route_lanelets, data_list[i].kinematic_state)
            route_tensor, route_speed_limit, route_has_speed_limit = create_lane_tensor(
                route_lanelets,
                map2bl_mat4x4=map2bl_matrix_4x4,
                center_x=data_list[i].kinematic_state.pose.pose.position.x,
                center_y=data_list[i].kinematic_state.pose.pose.position.y,
                traffic_light_recognition=traffic_light_recognition,
                num_segments=NUM_SEGMENTS_IN_ROUTE,
                dev="cpu",
                do_sort=False,
            )
            goal_pose_bl = get_relative_goal_pose(goal_pose, map2bl_matrix_4x4)

            # ego (time-based interpolation, matching the C++ converter)
            past_reference_sec = _stamp_to_sec(data_list[i].kinematic_state.header.stamp)
            ego_past_np = create_ego_sequence_interp(
                data_list,
                i - PAST_TIME_STEPS + 1,
                PAST_TIME_STEPS,
                map2bl_matrix_4x4,
                past_reference_sec,
            )
            if ego_past_np is None:
                logger.info(f"Failed to create ego past at frame {i}")
                break
            ego_tensor = create_current_ego_state(
                data_list[i].kinematic_state, data_list[i].acceleration, wheel_base=2.75
            ).squeeze(0)
            future_reference_sec = past_reference_sec + OUTPUT_T * 0.1
            ego_future_np = create_ego_sequence_interp(
                data_list, i + 1, OUTPUT_T, map2bl_matrix_4x4, future_reference_sec
            )
            if ego_future_np is None:
                logger.info(f"Reached end of sequence at frame {i}")
                break

            # (1)自車が止まっている
            # (2)目の前のlanelet segmentが赤信号である
            # (3)GTのTrajectoryが進むように出ている
            # このようなデータはスキップする

            # まず停止判定
            is_stop = data_list[i].kinematic_state.twist.twist.linear.x < 0.1
            if is_stop:
                stopping_count += 1
            else:
                stopping_count = 0

            # goal付近で1秒以上止まっていたら終了
            distance_to_goal_pose = np.linalg.norm(
                np.array([ego_future_np[-1, 0], ego_future_np[-1, 1]])
                - np.array([goal_pose_bl[0], goal_pose_bl[1]])
            )
            if stopping_count >= 10 and distance_to_goal_pose < 5.0:
                logger.info(
                    f"finish at {i} because stopping_count={stopping_count} and distance_to_goal_pose={distance_to_goal_pose:.2f}"
                )
                break

            is_red_light = route_tensor[:, 1, 0, 8 + 2].item()  # next segment
            sum_mileage = 0.0
            for j in range(OUTPUT_T - 1):
                sum_mileage += np.linalg.norm(ego_future_np[j, :2] - ego_future_np[j + 1, :2])
            is_future_forward = sum_mileage > 0.1
            if is_stop and is_red_light and is_future_forward:
                logger.info(
                    f"Skip this frame {i} because it is stop at red light and future trajectory is forward"
                )
                continue

            # neighbor (ported from the C++ AgentData / process_neighbor_agents_and_future)
            neighbor_past_np, neighbor_agent_ids = build_neighbor_past(
                data_list,
                i,
                map2bl_matrix_4x4,
                max_num_objects=MAX_NUM_NEIGHBORS,
                time_length=PAST_TIME_STEPS,
            )
            neighbor_past_tensor = torch.from_numpy(neighbor_past_np)
            neighbor_future_tensor = torch.from_numpy(
                build_neighbor_future(
                    data_list,
                    i,
                    map2bl_matrix_4x4,
                    neighbor_agent_ids,
                    max_num_objects=MAX_NUM_NEIGHBORS,
                    out_t=OUTPUT_T,
                )
            )

            # polygon
            polygon_tensor = create_line_tensor(
                vector_map.polygons.values(),
                map2bl_matrix_4x4,
                center_x=data_list[i].kinematic_state.pose.pose.position.x,
                center_y=data_list[i].kinematic_state.pose.pose.position.y,
                num_elements=NUM_POLYGONS,
                num_points=POINTS_PER_POLYGON,
                dev="cpu",
                type_map=POLYGON_TYPE_MAP,
                num_types=POLYGON_TYPE_NUM,
            )

            # line_string
            line_string_tensor = create_line_tensor(
                vector_map.line_strings.values(),
                map2bl_matrix_4x4,
                center_x=data_list[i].kinematic_state.pose.pose.position.x,
                center_y=data_list[i].kinematic_state.pose.pose.position.y,
                num_elements=NUM_LINE_STRINGS,
                num_points=POINTS_PER_LINE_STRING,
                dev="cpu",
                type_map=LINE_STRING_TYPE_MAP,
                num_types=LINE_STRING_TYPE_NUM,
            )

            # turn_indicators
            turn_indicators = np.array(
                [
                    data_list[max(0, i - INPUT_T + j)].turn_indicator.report
                    for j in range(INPUT_T + 1)
                ],
                dtype=np.int32,
            )

            curr_data = {
                "version": 2,
                "ego_agent_past": ego_past_np,
                "ego_current_state": ego_tensor.numpy(),
                "ego_agent_future": ego_future_np,
                "neighbor_agents_past": neighbor_past_tensor.numpy(),
                "neighbor_agents_future": neighbor_future_tensor.numpy(),
                "static_objects": np.zeros((5, 10), dtype=np.float32),
                # (wheel_base, length, width); matches the C++ converter's ego_shape option
                "ego_shape": np.array([2.75, 4.34, 1.70], dtype=np.float32),
                "lanes": lanes_tensor.squeeze(0).numpy(),
                "lanes_speed_limit": lanes_speed_limit.squeeze(0).numpy(),
                "lanes_has_speed_limit": lanes_has_speed_limit.squeeze(0).numpy(),
                "route_lanes": route_tensor.squeeze(0).numpy(),
                "route_lanes_speed_limit": route_speed_limit.squeeze(0).numpy(),
                "route_lanes_has_speed_limit": route_has_speed_limit.squeeze(0).numpy(),
                "turn_indicators": turn_indicators,
                "goal_pose": goal_pose_bl,
                "polygons": polygon_tensor.squeeze(0).numpy(),
                "line_strings": line_string_tensor.squeeze(0).numpy(),
            }
            # save the data
            save_dir.mkdir(parents=True, exist_ok=True)
            output_file = f"{save_dir}/{map_name}_{token}.npz"
            np.savez(output_file, **curr_data)

            # save other info
            pose_dict = {
                "timestamp": data_list[i].timestamp,
                "x": data_list[i].kinematic_state.pose.pose.position.x,
                "y": data_list[i].kinematic_state.pose.pose.position.y,
                "z": data_list[i].kinematic_state.pose.pose.position.z,
                "qx": data_list[i].kinematic_state.pose.pose.orientation.x,
                "qy": data_list[i].kinematic_state.pose.pose.orientation.y,
                "qz": data_list[i].kinematic_state.pose.pose.orientation.z,
                "qw": data_list[i].kinematic_state.pose.pose.orientation.w,
            }
            with open(f"{save_dir}/{map_name}_{token}.json", "w") as f:
                json.dump(pose_dict, f)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
