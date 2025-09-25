import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rclpy
import yaml
from autoware_internal_debug_msgs.msg import ProcessingTimeTree
from autoware_perception_msgs.msg import (
    TrafficLightElement,
    TrafficLightGroup,
    TrafficLightGroupArray,
)
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tier4_simulation_msgs.msg import DummyObject
from unique_identifier_msgs.msg import UUID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=Path)
    return parser.parse_args()


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

    # RED = 1, YELLOW = 2, GREEN = 3
    RED = 1
    YELLOW = 2
    GREEN = 3
    traffic_light_status = GREEN
    counter = 0

    def callback_kinematic_state(msg: Odometry):
        stamp = msg.header.stamp
        stamp.sec += 1
        global counter, traffic_light_status
        counter += 1
        # 各色で遷移時間を変える
        if traffic_light_status == GREEN:
            if counter % 400 == 0:
                counter = 0
                traffic_light_status = YELLOW
        elif traffic_light_status == YELLOW:
            if counter % 300 == 0:
                counter = 0
                traffic_light_status = RED
        elif traffic_light_status == RED:
            if counter % 1400 == 0:
                counter = 0
                traffic_light_status = GREEN

        traffic_light_group_id_list = [
            10221,
            10222,
            10249,
            10250,
            10267,
            10270,
            10276,
            10307,
            10309,
            10323,
            10324,
            10583,
            10584,
            179518,
            179596,
            179970,
            190343,
            190426,
            2118985,
            2119347,
            2119369,
            2119402,
            2119414,
            2119540,
        ]
        pub_traffic_light.publish(
            TrafficLightGroupArray(
                stamp=stamp,
                traffic_light_groups=[
                    TrafficLightGroup(
                        traffic_light_group_id=id,
                        elements=[
                            TrafficLightElement(
                                color=traffic_light_status,
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

    if "pedestrian" in data:
        time.sleep(1)
        pedestrian = DummyObject()
        pedestrian.header.frame_id = "map"
        pedestrian.header.stamp = node.get_clock().now().to_msg()
        pedestrian.id = UUID()
        for i in range(16):
            pedestrian.id.uuid[i] = i
        pose = pedestrian.initial_state.pose_covariance.pose
        pose.position.x = data["pedestrian"]["pose"]["position"]["x"]
        pose.position.y = data["pedestrian"]["pose"]["position"]["y"]
        pose.position.z = data["pedestrian"]["pose"]["position"]["z"]
        pose.orientation.x = data["pedestrian"]["pose"]["orientation"]["x"]
        pose.orientation.y = data["pedestrian"]["pose"]["orientation"]["y"]
        pose.orientation.z = data["pedestrian"]["pose"]["orientation"]["z"]
        pose.orientation.w = data["pedestrian"]["pose"]["orientation"]["w"]
        cov = pedestrian.initial_state.pose_covariance.covariance
        cov[0] = 0.0008999999845400453
        cov[7] = 0.0008999999845400453
        cov[14] = 0.0008999999845400453
        cov[35] = 0.007615434937179089
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

    if "bus" in data and False:
        time.sleep(1)
        bus = DummyObject()
        bus.header.frame_id = "map"
        bus.header.stamp = node.get_clock().now().to_msg()
        bus.id = UUID()
        for i in range(16):
            bus.id.uuid[i] = i
        pose = bus.initial_state.pose_covariance.pose
        pose.position.x = data["bus"]["pose"]["position"]["x"]
        pose.position.y = data["bus"]["pose"]["position"]["y"]
        pose.position.z = data["bus"]["pose"]["position"]["z"]
        pose.orientation.x = data["bus"]["pose"]["orientation"]["x"]
        pose.orientation.y = data["bus"]["pose"]["orientation"]["y"]
        pose.orientation.z = data["bus"]["pose"]["orientation"]["z"]
        pose.orientation.w = data["bus"]["pose"]["orientation"]["w"]
        cov = bus.initial_state.pose_covariance.covariance
        cov[0] = 0.0008999999845400453
        cov[7] = 0.0008999999845400453
        cov[14] = 0.0008999999845400453
        cov[35] = 0.007615434937179089
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
