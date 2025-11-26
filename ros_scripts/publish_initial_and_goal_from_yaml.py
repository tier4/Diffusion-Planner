import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import lanelet2
import numpy as np
import rclpy
import yaml
from autoware_internal_debug_msgs.msg import ProcessingTimeTree
from autoware_lanelet2_extension_python.projection import MGRSProjector
from autoware_perception_msgs.msg import (
    TrafficLightElement,
    TrafficLightGroup,
    TrafficLightGroupArray,
)
from geometry_msgs.msg import PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tier4_simulation_msgs.msg import DummyObject
from unique_identifier_msgs.msg import UUID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=Path)
    return parser.parse_args()


class TrafficLightStateMachine:
    """信号の状態を管理するクラス"""

    # RED = 1, YELLOW = 2, GREEN = 3
    RED = 1
    YELLOW = 2
    GREEN = 3
    VELOCITY_THRESHOLD = 0.1  # m/s, 静止とみなす閾値

    def __init__(self):
        self.traffic_light_status = self.YELLOW
        self.is_moving = False
        self.stopped_time = 0.0
        self.green_time = 0.0
        self.yellow_time = 0.0
        self.last_callback_time = None

    def update_traffic_light_color(self, vx: float, dt: float) -> int:
        """
        速度と時間差分に基づいて信号の色を決定する

        Args:
            vx: 車両の速度 (m/s)
            dt: 前回からの時間差分 (秒)

        Returns:
            信号の色 (RED=1, YELLOW=2, GREEN=3)
        """
        if self.traffic_light_status == self.RED:
            if abs(vx) > self.VELOCITY_THRESHOLD:
                self.is_moving = True
                self.stopped_time = 0.0
            else:
                # 静止している
                if self.is_moving:
                    # 動いている状態から静止状態への遷移
                    self.is_moving = False
                    self.stopped_time = 0.0

                self.stopped_time += dt
                # 静止していくらか経ったら緑
                if self.stopped_time >= 3.0:
                    self.traffic_light_status = self.GREEN
                    self.stopped_time = 0.0
                    self.green_time = 0.0
                    self.yellow_time = 0.0
        elif self.traffic_light_status == self.GREEN:
            self.green_time += dt
            # 緑から黄色への遷移
            if self.green_time >= 20.0:
                self.traffic_light_status = self.YELLOW
                self.stopped_time = 0.0
                self.green_time = 0.0
                self.yellow_time = 0.0
        elif self.traffic_light_status == self.YELLOW:
            self.yellow_time += dt
            # 黄色から赤への遷移
            if self.yellow_time >= 5.0:
                self.traffic_light_status = self.RED
                self.stopped_time = 0.0
                self.green_time = 0.0
                self.yellow_time = 0.0

        return self.traffic_light_status


def _get_attribute(attribute_map, key: str, default: str) -> str:
    if key in attribute_map:
        return attribute_map[key]
    else:
        return default


if __name__ == "__main__":
    args = parse_args()
    yaml_path = args.yaml_path

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    print(f"{data=}")

    rclpy.init(args=sys.argv)
    node = Node("publish_initial_and_goal_from_yaml")
    node.get_logger().info(f'Node "{node.get_name()}" has been started.')
    pub_initialpose = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
    pub_goal = node.create_publisher(PoseStamped, "/planning/mission_planning/goal", 10)
    pub_checkpoint = node.create_publisher(PoseStamped, "/planning/mission_planning/checkpoint", 10)
    pub_object = node.create_publisher(
        DummyObject, "/simulation/dummy_perception_publisher/object_info", 10
    )
    pub_traffic_light = node.create_publisher(
        TrafficLightGroupArray, "/perception/traffic_light_recognition/traffic_signals", 10
    )

    # 信号の状態機械を初期化
    traffic_light_sm = TrafficLightStateMachine()

    osm_path = Path(
        "/media/shintarosakoda/sandisk4t/data/nas_copy/tieriv_dataset/driving_dataset/map/2025-07-29/23-34-26_new/lanelet2_map.osm"
    )
    projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    lanelet_map = lanelet2.io.load(str(osm_path), projection)
    traffic_light_group_ids = set()
    for regulatory in lanelet_map.regulatoryElementLayer:
        reg_subtype = _get_attribute(regulatory.attributes, "subtype", "")
        if reg_subtype == "traffic_light":
            traffic_light_group_ids.add(regulatory.id)

    traffic_light_group_id_list = sorted(traffic_light_group_ids)

    def callback_kinematic_state(msg: Odometry):
        global traffic_light_group_id_list
        stamp = msg.header.stamp
        stamp.sec += 1

        # 速度を取得
        vx = msg.twist.twist.linear.x

        # 時間差分を計算
        current_time = time.time()
        if traffic_light_sm.last_callback_time is None:
            traffic_light_sm.last_callback_time = current_time
            dt = 0.0
        else:
            dt = current_time - traffic_light_sm.last_callback_time
            traffic_light_sm.last_callback_time = current_time

        # 信号の色を更新
        traffic_light_sm.update_traffic_light_color(vx, dt)

        pub_traffic_light.publish(
            TrafficLightGroupArray(
                stamp=stamp,
                traffic_light_groups=[
                    TrafficLightGroup(
                        traffic_light_group_id=id,
                        elements=[
                            TrafficLightElement(
                                color=traffic_light_sm.traffic_light_status,
                                shape=1,  # CIRCLE
                                status=2,  # SOLID_ON
                                confidence=1.0,
                            )
                        ],
                        predictions=[],
                    )
                    for id in traffic_light_group_id_list
                ],
            )
        )

    sub_kinematic_state = node.create_subscription(
        Odometry, "/localization/kinematic_state", callback_kinematic_state, 10
    )
    node.get_logger().info("Publishers created.")

    time_list = defaultdict(list)

    def callback_processing_time_tree(msg: ProcessingTimeTree):
        nodes = msg.nodes
        for node in nodes:
            time_list[node.name].append(float(node.processing_time))

        for key, value in time_list.items():
            mean = np.mean(value)
            std = np.std(value)
            print(f"{key}\t {mean:.1f} ± {std:.1f} msec")

    sub_processing_time_tree = node.create_subscription(
        ProcessingTimeTree,
        "/planning/trajectory_generator/diffusion_planner_node/debug/processing_time_detail_ms",
        callback_processing_time_tree,
        10,
    )

    initialpose = PoseWithCovarianceStamped()
    initialpose.header.frame_id = "map"
    initialpose.header.stamp = node.get_clock().now().to_msg()
    initialpose.pose.pose.position.x = data["initialpose"]["pose"]["pose"]["position"]["x"]
    initialpose.pose.pose.position.y = data["initialpose"]["pose"]["pose"]["position"]["y"]
    initialpose.pose.pose.position.z = data["initialpose"]["pose"]["pose"]["position"]["z"]
    initialpose.pose.pose.orientation.x = data["initialpose"]["pose"]["pose"]["orientation"]["x"]
    initialpose.pose.pose.orientation.y = data["initialpose"]["pose"]["pose"]["orientation"]["y"]
    initialpose.pose.pose.orientation.z = data["initialpose"]["pose"]["pose"]["orientation"]["z"]
    initialpose.pose.pose.orientation.w = data["initialpose"]["pose"]["pose"]["orientation"]["w"]
    for i in range(36):
        initialpose.pose.covariance[i] = data["initialpose"]["pose"]["covariance"][i]
    pub_initialpose.publish(initialpose)
    node.get_logger().info(f"Published initial pose: {initialpose}")
    time.sleep(3)

    goal_pose = PoseStamped()
    goal_pose.header.frame_id = "map"
    goal_pose.header.stamp = node.get_clock().now().to_msg()
    goal_pose.pose.position.x = data["goal"]["pose"]["pose"]["position"]["x"]
    goal_pose.pose.position.y = data["goal"]["pose"]["pose"]["position"]["y"]
    goal_pose.pose.position.z = data["goal"]["pose"]["pose"]["position"]["z"]
    goal_pose.pose.orientation.x = data["goal"]["pose"]["pose"]["orientation"]["x"]
    goal_pose.pose.orientation.y = data["goal"]["pose"]["pose"]["orientation"]["y"]
    goal_pose.pose.orientation.z = data["goal"]["pose"]["pose"]["orientation"]["z"]
    goal_pose.pose.orientation.w = data["goal"]["pose"]["pose"]["orientation"]["w"]
    pub_goal.publish(goal_pose)
    node.get_logger().info(f"Published goal pose: {goal_pose}")

    if "checkpoint" in data:
        time.sleep(1)
        checkpoint = PoseStamped()
        checkpoint.header.frame_id = "map"
        checkpoint.header.stamp = node.get_clock().now().to_msg()
        checkpoint.pose.position.x = data["checkpoint"]["pose"]["position"]["x"]
        checkpoint.pose.position.y = data["checkpoint"]["pose"]["position"]["y"]
        checkpoint.pose.position.z = data["checkpoint"]["pose"]["position"]["z"]
        checkpoint.pose.orientation.x = data["checkpoint"]["pose"]["orientation"]["x"]
        checkpoint.pose.orientation.y = data["checkpoint"]["pose"]["orientation"]["y"]
        checkpoint.pose.orientation.z = data["checkpoint"]["pose"]["orientation"]["z"]
        checkpoint.pose.orientation.w = data["checkpoint"]["pose"]["orientation"]["w"]
        pub_checkpoint.publish(checkpoint)
        node.get_logger().info(f"Published checkpoint pose: {checkpoint}")

    def get_pose(name: str):
        pose = PoseWithCovariance()
        pose.pose.position.x = data[name]["pose"]["position"]["x"]
        pose.pose.position.y = data[name]["pose"]["position"]["y"]
        pose.pose.position.z = data[name]["pose"]["position"]["z"]
        pose.pose.orientation.x = data[name]["pose"]["orientation"]["x"]
        pose.pose.orientation.y = data[name]["pose"]["orientation"]["y"]
        pose.pose.orientation.z = data[name]["pose"]["orientation"]["z"]
        pose.pose.orientation.w = data[name]["pose"]["orientation"]["w"]
        pose.covariance[0] = 0.0008999999845400453
        pose.covariance[7] = 0.0008999999845400453
        pose.covariance[14] = 0.0008999999845400453
        pose.covariance[35] = 0.007615434937179089
        return pose

    if "pedestrian" in data:
        time.sleep(1)
        pedestrian = DummyObject()
        pedestrian.header.frame_id = "map"
        pedestrian.header.stamp = node.get_clock().now().to_msg()
        pedestrian.id = UUID()
        for i in range(16):
            pedestrian.id.uuid[i] = i
        pedestrian.initial_state.pose_covariance = get_pose("pedestrian")
        pedestrian.classification.label = 7
        pedestrian.classification.probability = 1.0
        pedestrian.shape.type = 1  # BOX
        pedestrian.shape.footprint.points = []
        pedestrian.shape.dimensions.x = 0.6
        pedestrian.shape.dimensions.y = 0.6
        pedestrian.shape.dimensions.z = 2.0
        pedestrian.max_velocity = +33.29999923706055
        pedestrian.min_velocity = -33.29999923706055
        pedestrian.action = 0
        pub_object.publish(pedestrian)
        node.get_logger().info(f"Published pedestrian pose: {pedestrian}")

    if "bus_avoidance" in data:
        time.sleep(1)
        bus = DummyObject()
        bus.header.frame_id = "map"
        bus.header.stamp = node.get_clock().now().to_msg()
        bus.id = UUID()
        for i in range(16):
            bus.id.uuid[i] = i
        bus.initial_state.pose_covariance = get_pose("bus_avoidance")
        bus.classification.label = 3
        bus.classification.probability = 1.0
        bus.shape.type = 0  # BOX
        bus.shape.footprint.points = []
        bus.shape.dimensions.x = 10.5
        bus.shape.dimensions.y = 2.5
        bus.shape.dimensions.z = 3.5
        bus.max_velocity = +33.29999923706055
        bus.min_velocity = -33.29999923706055
        bus.action = 0
        pub_object.publish(bus)
        node.get_logger().info(f"Published bus pose: {bus}")

    # start spin
    rclpy.spin(node)
