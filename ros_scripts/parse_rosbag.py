import argparse
import json
import logging
from collections import defaultdict
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
from geometry_msgs.msg import AccelWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from diffusion_planner_ros.lanelet2_utils.lanelet_converter import (
    convert_lanelet,
    create_lane_tensor,
)
from diffusion_planner_ros.utils import (
    convert_tracked_objects_to_tensor,
    create_current_ego_state,
    filter_target_segments,
    get_nearest_msg,
    get_transform_matrix,
    parse_timestamp,
    parse_traffic_light_recognition,
    pose_to_mat4x4,
    rot3x3_to_heading_cos_sin,
    tracking_one_step,
)

"""
This script makes npz files from a rosbag.
[Contents of a npz file]
map_name                    <U26    ()
token                       <U16    ()
ego_agent_past              float32 (21, 3)
ego_current_state           float32 (10,)
ego_agent_future            float32 (80, 3)
neighbor_agents_past        float32 (32, 21, 11)
neighbor_agents_future      float32 (32, 80, 3)
static_objects              float32 (5, 10)
lanes                       float32 (70, 20, 21)
lanes_speed_limit           float32 (70, 1)
lanes_has_speed_limit       bool    (70, 1)
route_lanes                 float32 (25, 20, 21)
route_lanes_speed_limit     float32 (25, 1)
route_lanes_has_speed_limit bool    (25, 1)
turn_indicator              int32   (1)
"""
PAST_TIME_STEPS = 21
FUTURE_TIME_STEPS = 80
NEIGHBOR_NUM = 32
STATIC_NUM = 5
LANE_NUM = 70
LANE_LEN = 20
ROUTE_NUM = 25
ROUTE_LEN = 20


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


def create_ego_sequence(data_list, i, future_time_steps, map2bl_matrix_4x4):
    ego_future_x = []
    ego_future_y = []
    ego_future_yaw = []
    for j in range(future_time_steps):
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
            neighbor[0, i, j, 7] = max(shape.dimensions.x, 1.0)  # lendth
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
    for frame_data in data_list[i : i + FUTURE_TIME_STEPS]:
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
    # erase losting from tracking_future
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
    yaw = Rotation.from_matrix(pose_in_bl[0:3, 0:3]).as_euler("xyz")[2]
    return np.array([x, y, yaw], dtype=np.float32)


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

    topic_name_to_data = defaultdict(list)
    parse_num = 0
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        if topic in target_topic_list:
            topic_name_to_data[topic].append(msg)
            parse_num += 1
            if limit > 0 and parse_num >= limit:
                break

    for key, value in topic_name_to_data.items():
        logger.info(f"{key}: {len(value)} msgs")

    route_msgs = topic_name_to_data["/planning/mission_planning/route"]
    sequence_data_list = [SequenceData([], route_msg) for route_msg in route_msgs]

    # convert to FrameData
    # The base topic is "/perception/object_recognition/tracking/objects" (10Hz)
    n = len(topic_name_to_data["/perception/object_recognition/tracking/objects"])
    logger.info(f"{n=}")
    progress_bar = tqdm(total=n)
    for i in range(n):
        tracking = topic_name_to_data["/perception/object_recognition/tracking/objects"][i]
        timestamp = parse_timestamp(tracking.header.stamp)
        latest_msgs = {
            "/localization/kinematic_state": None,
            "/localization/acceleration": None,
            "/perception/traffic_light_recognition/traffic_signals": None,
            "/vehicle/status/turn_indicators_status": None,
        }

        ok = True
        for key in latest_msgs.keys():
            curr_msg, curr_index = get_nearest_msg(topic_name_to_data[key], tracking.header.stamp)
            if curr_msg is None:
                logger.info(f"Cannot find {key} msg")
                ok = False
                break
            topic_name_to_data[key] = topic_name_to_data[key][curr_index:]
            latest_msgs[key] = curr_msg
            msg_stamp = curr_msg.header.stamp if hasattr(curr_msg, "header") else curr_msg.stamp
            msg_stamp_int = parse_timestamp(msg_stamp)
            diff = abs(timestamp - msg_stamp_int)
            if diff > int(0.2 * 1e9):
                logger.info(f"Over 200 msec: {key} {len(topic_name_to_data[key])=}, {diff=:,}")
                ok = False

        # check kinematic_state
        if ok:
            kinematic_state = latest_msgs["/localization/kinematic_state"]
            covariance = kinematic_state.pose.covariance
            covariance_xx = covariance[0]
            covariance_yy = covariance[7]
            if covariance_xx > 1e-1 or covariance_yy > 1e-1:
                logger.info(f"Invalid kinematic_state {covariance_xx=:.5f}, {covariance_yy=:.5f}")
                ok = False

        # check route
        if search_nearest_route:
            # find the latest route msg
            max_route_index = -1
            max_route_timestamp = 0
            for j in range(len(route_msgs)):
                route_msg = route_msgs[j]
                route_stamp = parse_timestamp(route_msg.header.stamp)
                if max_route_timestamp <= route_stamp <= timestamp:
                    max_route_timestamp = route_stamp
                    max_route_index = j
            if max_route_index == -1:
                logger.info(f"Cannot find route msg at {i}")
                continue
        else:
            # use the first route msg
            max_route_index = 0

        sequence = sequence_data_list[max_route_index]

        if not ok:
            if len(sequence.data_list) == 0:
                # At the beginning of recording, some msgs may be missing
                # Skip this frame
                logger.info(f"Skip this frame {i=}/{n=}")
                continue
            else:
                # If the msg is missing in the middle of recording, we can use the msgs to this point
                logger.info(f"Finish at this frame {i=}/{n=}")
                break

        sequence.data_list.append(
            FrameData(
                timestamp=timestamp,
                route=sequence.route,
                tracked_objects=tracking,
                kinematic_state=latest_msgs["/localization/kinematic_state"],
                acceleration=latest_msgs["/localization/acceleration"],
                traffic_signals=latest_msgs[
                    "/perception/traffic_light_recognition/traffic_signals"
                ],
                turn_indicator=latest_msgs["/vehicle/status/turn_indicators_status"],
            )
        )
        progress_bar.update(1)

    # FreeSpacePlannerの影響で、最後の方でgoal_poseだけ微妙に変わることがある
    # そういうものは結合する
    for i in range(len(sequence_data_list) - 2, -1, -1):
        route_msg_l = sequence_data_list[i].route
        route_msg_r = sequence_data_list[i + 1].route
        if route_msg_l.start_pose != route_msg_r.start_pose:
            logger.info(
                f"Route start pose mismatch: {route_msg_l.start_pose} != {route_msg_r.start_pose}"
            )
            continue
        logger.info(f"Concatenate sequence {i} and {i + 1}")
        logger.info(f"Before {len(sequence_data_list[i].data_list)=} frames")
        sequence_data_list[i].data_list.extend(sequence_data_list[i + 1].data_list)
        logger.info(f"After {len(sequence_data_list[i].data_list)=} frames")
        sequence_data_list.pop(i + 1)

    map_name = rosbag_path.stem
    sequence_num = len(sequence_data_list)
    logger.info(f"Total {sequence_num} sequences")

    for seq_id, seq in enumerate(sequence_data_list):
        logger.info(f"Processing sequence {seq_id + 1}/{sequence_num}")

        data_list = seq.data_list
        n = len(data_list)
        logger.info(f"Total {n} frames")

        # if less than min_frames (default 3 min), skip this sequence
        if n < min_frames:
            logger.info(
                f"Skip this sequence because the number of frames {n} is less than {min_frames} frames"
            )
            continue

        # list[FrameData] -> npz
        progress = tqdm(total=(n - PAST_TIME_STEPS - FUTURE_TIME_STEPS) // step)
        stopping_count = 0
        for i in range(PAST_TIME_STEPS, n, step):
            progress.update(1)
            token = f"{seq_id:08d}{i:08d}"

            bl2map_matrix_4x4, map2bl_matrix_4x4 = get_transform_matrix(
                data_list[i].kinematic_state
            )

            traffic_light_recognition = parse_traffic_light_recognition(
                data_list[i].traffic_signals
            )

            # lanes
            lanes_tensor, lanes_speed_limit, lanes_has_speed_limit = create_lane_tensor(
                vector_map.lane_segments.values(),
                map2bl_mat4x4=map2bl_matrix_4x4,
                center_x=data_list[i].kinematic_state.pose.pose.position.x,
                center_y=data_list[i].kinematic_state.pose.pose.position.y,
                mask_range=100,
                traffic_light_recognition=traffic_light_recognition,
                num_segments=70,
                dev="cpu",
                do_sort=True,
            )

            # routes
            target_segments = [
                vector_map.lane_segments[segment.preferred_primitive.id]
                for segment in data_list[i].route.segments
            ]
            target_segments = filter_target_segments(target_segments, data_list[i].kinematic_state)
            route_tensor, route_speed_limit, route_has_speed_limit = create_lane_tensor(
                target_segments,
                map2bl_mat4x4=map2bl_matrix_4x4,
                center_x=data_list[i].kinematic_state.pose.pose.position.x,
                center_y=data_list[i].kinematic_state.pose.pose.position.y,
                mask_range=100,
                traffic_light_recognition=traffic_light_recognition,
                num_segments=25,
                dev="cpu",
                do_sort=False,
            )
            goal_pose = data_list[i].route.goal_pose
            goal_pose_bl = get_relative_goal_pose(goal_pose, map2bl_matrix_4x4)

            # ego
            ego_past_np = create_ego_sequence(
                data_list, i - PAST_TIME_STEPS + 1, PAST_TIME_STEPS, map2bl_matrix_4x4
            )
            ego_tensor = create_current_ego_state(
                data_list[i].kinematic_state, data_list[i].acceleration, wheel_base=2.79
            ).squeeze(0)
            ego_future_np = create_ego_sequence(data_list, i, FUTURE_TIME_STEPS, map2bl_matrix_4x4)

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

            is_red_light = route_tensor[:, 1, 0, -3].item()  # next segment
            sum_mileage = 0.0
            for j in range(FUTURE_TIME_STEPS - 1):
                sum_mileage += np.linalg.norm(ego_future_np[j, :2] - ego_future_np[j + 1, :2])
            is_future_forward = sum_mileage > 0.1
            if is_stop and is_red_light and is_future_forward:
                logger.info(
                    f"Skip this frame {i} because it is stop at red light and future trajectory is forward"
                )
                continue

            # neighbor
            tracking_past, tracking_future = tracking_past_and_future(
                data_list, i, map2bl_matrix_4x4
            )
            neighbor_past_tensor = convert_tracked_objects_to_tensor(
                tracked_objs=tracking_past,
                map2bl_matrix_4x4=map2bl_matrix_4x4,
                max_num_objects=NEIGHBOR_NUM,
                max_timesteps=PAST_TIME_STEPS,
            ).squeeze(0)
            neighbor_future_tensor = create_neighbor_future(
                tracked_objs=tracking_future,
                map2bl_matrix_4x4=map2bl_matrix_4x4,
                max_num_objects=NEIGHBOR_NUM,
                max_timesteps=FUTURE_TIME_STEPS,
            ).squeeze(0)
            # (32, 80, 11) -> (32, 80, 3)
            neighbor_future_tensor = neighbor_future_tensor[:, :, :4]
            # fixed cos(2) sin(3) -> heading
            neighbor_future_tensor[:, :, 2] = np.arctan2(
                neighbor_future_tensor[:, :, 3], neighbor_future_tensor[:, :, 2]
            )
            neighbor_future_tensor = neighbor_future_tensor[:, :, :3]

            curr_data = {
                "map_name": map_name,
                "token": token,
                "ego_agent_past": ego_past_np,
                "ego_current_state": ego_tensor.numpy(),
                "ego_agent_future": ego_future_np,
                "neighbor_agents_past": neighbor_past_tensor.numpy(),
                "neighbor_agents_future": neighbor_future_tensor.numpy(),
                "static_objects": np.zeros((STATIC_NUM, 10), dtype=np.float32),
                "lanes": lanes_tensor.squeeze(0).numpy(),
                "lanes_speed_limit": lanes_speed_limit.squeeze(0).numpy(),
                "lanes_has_speed_limit": lanes_has_speed_limit.squeeze(0).numpy(),
                "route_lanes": route_tensor.squeeze(0).numpy(),
                "route_lanes_speed_limit": route_speed_limit.squeeze(0).numpy(),
                "route_lanes_has_speed_limit": route_has_speed_limit.squeeze(0).numpy(),
                "turn_indicator": data_list[i].turn_indicator.report,
                "goal_pose": goal_pose_bl,
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
