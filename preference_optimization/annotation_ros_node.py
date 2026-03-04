"""ROS2 annotation server node for Lichtblick panels.

Direct cutover transport: ROS2 topics via ros2_foxglove_bridge.
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import rclpy
import torch
import numpy as np
from autoware_perception_msgs.msg import ObjectClassification, Shape, TrackedObject, TrackedObjectKinematics, TrackedObjects
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, TransformStamped, Vector3
from matplotlib.backends.backend_agg import FigureCanvasAgg
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from unique_identifier_msgs.msg import UUID
from visualization_msgs.msg import Marker, MarkerArray

from preference_optimization.annotation_gui import PreferenceAnnotator
from preference_optimization.model_utils import load_model


class AnnotationRosNode(Node):
    def __init__(
        self,
        model_path: Path,
        npz_list: Path,
        target_count: int | None,
        device: str,
    ) -> None:
        super().__init__("annotation_ros_node")

        torch_device = torch.device(device)
        policy_model, model_args = load_model(model_path, torch_device)
        policy_model.eval()

        with open(npz_list, "r") as f:
            npz_paths = json.load(f)
        if target_count is None:
            target_count = len(npz_paths)

        self.annotator = PreferenceAnnotator(policy_model, model_args, npz_paths, target_count)
        self.params = {
            "noise_scale": 2.5,
            "fde_threshold": 2.0,
            "ade_threshold": 1.0,
            "max_retries": 50,
            "zoom_level": 5,
            "time_step": 40,
            "gt_similarity_mode": True,
            "enable_initial_pruning": True,
            "initial_pos_threshold": 0.055,
            "initial_yaw_threshold_deg": 0.55,
            "enable_guidance": False,
            "use_collision": True,
            "use_route_following": False,
            "use_lane_keeping": False,
            "use_centerline_following": False,
            "guidance_scale": 0.5,
        }
        self.training_status = {
            "phase": "annotation",
            "message": "Ready for annotation",
            "epoch": 0,
            "total_epochs": 0,
            "batch": 0,
            "total_batches": 0,
            "metrics": {},
        }

        latched_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )


        cmd_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.state_pub = self.create_publisher(String, "/annotation/state", latched_qos)
        self.cmd_sub = self.create_subscription(String, "/annotation/cmd", self._on_cmd, cmd_qos)
        self.det_pub = self.create_publisher(Trajectory, "/annotation/data/trajectory/deterministic", latched_qos)
        self.stoch_pub = self.create_publisher(Trajectory, "/annotation/data/trajectory/stochastic", latched_qos)
        self.gt_pub = self.create_publisher(Trajectory, "/annotation/data/trajectory/ground_truth", latched_qos)
        self.ego_hist_pub = self.create_publisher(Trajectory, "/annotation/data/trajectory/ego_history", latched_qos)
        self.gt_snippet_pub = self.create_publisher(Trajectory, "/annotation/data/trajectory/gt_snippet", latched_qos)
        self.map_marker_pub = self.create_publisher(MarkerArray, "/annotation/data/map_markers", latched_qos)
        self.footprint_pub = self.create_publisher(MarkerArray, "/annotation/data/footprints", latched_qos)
        self.tracked_objects_pub = self.create_publisher(TrackedObjects, "/annotation/data/tracked_objects", latched_qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_timer = self.create_timer(0.1, self._publish_dynamic_streams_tick)
        self.static_timer = self.create_timer(0.1, self._publish_static_streams_tick)
        self.trajectory_timer = self.create_timer(0.1, self._publish_trajectory_streams_tick)

        self.last_payload: dict[str, Any] = {}
        self._cached_det_np: Any = None
        self._cached_stoch_np: Any = None
        self._static_markers_dirty = True
        self._cached_map_markers: list[Marker] = []
        self._pending_marker_clear = False
        self._last_sample_index = -1
        self._load_sample()
    
    def wait_for_annotation_complete(self, poll_interval_sec: float = 0.5) -> list[dict]:
        while not self.annotator.annotation_complete:
            time.sleep(poll_interval_sec)
        return list(self.annotator.preferences)

    def reset_annotation_round(self, target_count: int | None = None) -> None:
        if target_count is None:
            target_count = len(self.annotator.npz_paths)
        self.annotator = PreferenceAnnotator(
            self.annotator.policy_model,
            self.annotator.model_args,
            self.annotator.original_npz_paths,
            target_count,
        )
        self.params["time_step"] = 40
        self.training_status = {
            "phase": "annotation",
            "message": "Ready for annotation",
            "epoch": 0,
            "total_epochs": 0,
            "batch": 0,
            "total_batches": 0,
            "metrics": {},
        }
        self._load_sample()

    def update_training_status(
        self,
        *,
        phase: str,
        message: str,
        epoch: int = 0,
        total_epochs: int = 0,
        batch: int = 0,
        total_batches: int = 0,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.training_status = {
            "phase": phase,
            "message": message,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "batch": batch,
            "total_batches": total_batches,
            "metrics": metrics or {},
        }
        self._publish_current()

    def _fig_to_base64(self, fig) -> str | None:
        if fig is None:
            return None
        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _quat_from_heading(heading: float) -> tuple[float, float, float, float]:
        return (0.0, 0.0, math.sin(heading / 2.0), math.cos(heading / 2.0))

    def _trajectory_to_msg(self, trajectory: Any) -> Trajectory:
        msg = Trajectory()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = "map"
        if trajectory is None:
            return msg
        traj_np = torch.tensor(trajectory).cpu().numpy()
        for i, row in enumerate(traj_np):
            point = TrajectoryPoint()
            x, y, cos_h, sin_h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            heading = math.atan2(sin_h, cos_h)
            qx, qy, qz, qw = self._quat_from_heading(heading)
            point.pose.position.x = x
            point.pose.position.y = y
            point.pose.position.z = 0.0
            point.pose.orientation.x = qx
            point.pose.orientation.y = qy
            point.pose.orientation.z = qz
            point.pose.orientation.w = qw
            # Duration.nanosec must be in [0, 2^32-1], so split total time into sec+nsec.
            sec = i // 10
            nsec = (i % 10) * 100_000_000
            point.time_from_start = Duration(sec=sec, nanosec=nsec)
            msg.points.append(point)
        return msg

    def _trajectory_from_rows(self, rows: Any) -> Trajectory:
        msg = Trajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        if rows is None:
            return msg
        rows_np = torch.as_tensor(rows).cpu().numpy()
        for i, row in enumerate(rows_np):
            point = TrajectoryPoint()
            x = float(row[0]) if len(row) > 0 else 0.0
            y = float(row[1]) if len(row) > 1 else 0.0
            if len(row) > 3:
                heading = math.atan2(float(row[3]), float(row[2]))
            elif len(row) > 2:
                heading = float(row[2])
            else:
                heading = 0.0
            qx, qy, qz, qw = self._quat_from_heading(heading)
            point.pose.position.x = x
            point.pose.position.y = y
            point.pose.position.z = 0.0
            point.pose.orientation.x = qx
            point.pose.orientation.y = qy
            point.pose.orientation.z = qz
            point.pose.orientation.w = qw
            sec = i // 10
            nsec = (i % 10) * 100_000_000
            point.time_from_start = Duration(sec=sec, nanosec=nsec)
            msg.points.append(point)
        return msg

    def _gt_to_msg(self) -> Trajectory:
        msg = Trajectory()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = "map"
        if self.annotator.current_data is None or "ego_agent_future" not in self.annotator.current_data:
            return msg
        gt_np = self.annotator.current_data["ego_agent_future"][0].cpu().numpy()
        for i, row in enumerate(gt_np):
            point = TrajectoryPoint()
            x, y, heading = float(row[0]), float(row[1]), float(row[2])
            qx, qy, qz, qw = self._quat_from_heading(heading)
            point.pose.position.x = x
            point.pose.position.y = y
            point.pose.position.z = 0.0
            point.pose.orientation.x = qx
            point.pose.orientation.y = qy
            point.pose.orientation.z = qz
            point.pose.orientation.w = qw
            # Duration.nanosec must be in [0, 2^32-1], so split total time into sec+nsec.
            sec = i // 10
            nsec = (i % 10) * 100_000_000
            point.time_from_start = Duration(sec=sec, nanosec=nsec)
            msg.points.append(point)
        return msg

    def _ego_history_to_msg(self) -> Trajectory:
        if self.annotator.current_data is None or "ego_agent_past" not in self.annotator.current_data:
            return Trajectory()
        return self._trajectory_from_rows(self.annotator.current_data["ego_agent_past"][0])

    def _gt_snippet_to_msg(self) -> Trajectory:
        if self.annotator.current_data is None or "ego_agent_future" not in self.annotator.current_data:
            return Trajectory()
        gt = self.annotator.current_data["ego_agent_future"][0]
        idx = max(0, min(int(self.params["time_step"]), int(gt.shape[0]) - 1))
        return self._trajectory_from_rows(gt[: idx + 1])

    def _publish_state(self, payload: dict[str, Any]) -> None:
        state_msg = String()
        state_msg.data = json.dumps(payload)
        self.state_pub.publish(state_msg)

    def _rebuild_map_marker_cache(self) -> None:
        now = self.get_clock().now().to_msg()
        markers: list[Marker] = []
        map_z = -0.01

        def _add_polyline(
            marker_id: int, pts: Any, ns: str, r: float, g: float, b: float, width: float = 0.25, z: float = map_z
        ) -> int:
            arr = torch.as_tensor(pts).cpu().numpy()
            if arr.ndim > 2:
                for sub in arr:
                    marker_id = _add_polyline(marker_id, sub, ns, r, g, b, width, z)
                return marker_id
            if arr.ndim < 2:
                return marker_id
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = "map"
            marker.ns = ns
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = width
            marker.lifetime = Duration(sec=2, nanosec=0)
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 0.9
            for row in arr:
                flat = row.reshape(-1)
                if flat.size < 2:
                    continue
                marker.points.append(Point(x=float(flat[0]), y=float(flat[1]), z=z))
            if len(marker.points) < 2:
                return marker_id
            markers.append(marker)
            return marker_id + 1

        marker_id = 0
        data = self.annotator.current_data or {}

        if "lanes" in data:
            lanes_np = torch.as_tensor(data["lanes"]).cpu().numpy()
            if lanes_np.ndim >= 4:
                lane_batch = lanes_np[0]
                for lane in lane_batch:
                    if lane.shape[-1] < 8:
                        continue
                    left = np.stack((lane[:, 0] + lane[:, 4], lane[:, 1] + lane[:, 5]), axis=1)
                    right = np.stack((lane[:, 0] + lane[:, 6], lane[:, 1] + lane[:, 7]), axis=1)
                    marker_id = _add_polyline(marker_id, left, "lane_left", 0.25, 0.75, 1.0, width=0.16)
                    marker_id = _add_polyline(marker_id, right, "lane_right", 0.25, 0.75, 1.0, width=0.16)

        if "route_lanes" in data:
            route_np = torch.as_tensor(data["route_lanes"]).cpu().numpy()
            if route_np.ndim >= 4:
                for route in route_np[0]:
                    marker_id = _add_polyline(marker_id, route[:, :2], "route_lanes", 0.55, 0.55, 0.0, width=0.28)
            else:
                marker_id = _add_polyline(marker_id, route_np, "route_lanes", 0.55, 0.55, 0.0, width=0.28)

        if "line_strings" in data:
            line_np = torch.as_tensor(data["line_strings"]).cpu().numpy()
            marker_id = _add_polyline(marker_id, line_np, "line_strings", 1.0, 0.2, 0.2, width=0.15)

        if "polygons" in data:
            poly_np = torch.as_tensor(data["polygons"]).cpu().numpy()
            polygons = poly_np[0] if poly_np.ndim >= 4 else poly_np
            for poly in polygons:
                if poly.ndim < 2 or poly.shape[0] < 3:
                    continue
                marker = Marker()
                marker.header.stamp = now
                marker.header.frame_id = "map"
                marker.ns = "polygons"
                marker.id = marker_id
                marker.type = Marker.TRIANGLE_LIST
                marker.action = Marker.ADD
                marker.lifetime = Duration(sec=2, nanosec=0)
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5
                marker.color.a = 0.35
                for i in range(1, poly.shape[0] - 1):
                    p0 = poly[0].reshape(-1)
                    p1 = poly[i].reshape(-1)
                    p2 = poly[i + 1].reshape(-1)
                    if p0.size < 2 or p1.size < 2 or p2.size < 2:
                        continue
                    marker.points.append(Point(x=float(p0[0]), y=float(p0[1]), z=map_z))
                    marker.points.append(Point(x=float(p1[0]), y=float(p1[1]), z=map_z))
                    marker.points.append(Point(x=float(p2[0]), y=float(p2[1]), z=map_z))
                if marker.points:
                    markers.append(marker)
                    marker_id += 1

        if "static_objects" in data:
            static_np = torch.as_tensor(data["static_objects"]).cpu().numpy()
            objects = static_np[0] if static_np.ndim >= 3 else static_np
            for obj in objects:
                flat = obj.reshape(-1)
                if flat.size < 4 or np.sum(np.abs(flat[:4])) < 1e-6:
                    continue
                x, y, cos_h, sin_h = float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
                heading = math.atan2(sin_h, cos_h)
                qx, qy, qz, qw = self._quat_from_heading(heading)
                width = float(flat[4]) if flat.size > 4 else 1.0
                length = float(flat[5]) if flat.size > 5 else 1.0
                marker = Marker()
                marker.header.stamp = now
                marker.header.frame_id = "map"
                marker.ns = "static_objects"
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.lifetime = Duration(sec=2, nanosec=0)
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = 0.4
                marker.pose.orientation.x = qx
                marker.pose.orientation.y = qy
                marker.pose.orientation.z = qz
                marker.pose.orientation.w = qw
                marker.scale = Vector3(x=max(length, 0.1), y=max(width, 0.1), z=0.8)
                marker.color.r = 0.7
                marker.color.g = 0.45
                marker.color.b = 0.2
                marker.color.a = 0.45
                markers.append(marker)
                marker_id += 1

        for key, color in (
            ("map_polylines", (0.6, 0.6, 0.6)),
        ):
            if key not in data:
                continue
            tensor = torch.as_tensor(data[key]).cpu().numpy()
            marker_id = _add_polyline(marker_id, tensor, key, *color)

        goal_key = "goal_pose" if "goal_pose" in data else ("goal" if "goal" in data else None)
        if goal_key is not None:
            goal_np = torch.as_tensor(data[goal_key]).cpu().numpy().reshape(-1)
            if goal_np.size >= 2:
                goal = Marker()
                goal.header.stamp = now
                goal.header.frame_id = "map"
                goal.ns = "goal"
                goal.id = marker_id
                goal.type = Marker.SPHERE
                goal.action = Marker.ADD
                goal.lifetime = Duration(sec=2, nanosec=0)
                goal.pose.position.x = float(goal_np[0])
                goal.pose.position.y = float(goal_np[1])
                goal.pose.position.z = 0.05
                goal.scale = Vector3(x=0.8, y=0.8, z=0.8)
                goal.color.r = 1.0
                goal.color.g = 0.2
                goal.color.b = 0.2
                goal.color.a = 0.9
                markers.append(goal)
        self._cached_map_markers = markers

    def _clear_map_markers(self) -> None:
        clear_msg = MarkerArray()
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.action = Marker.DELETEALL
        clear_msg.markers.append(marker)
        self.map_marker_pub.publish(clear_msg)

    def _publish_map_markers_tick(self) -> None:
        if self._static_markers_dirty:
            self._rebuild_map_marker_cache()
            self._static_markers_dirty = False
        if self._pending_marker_clear:
            self._clear_map_markers()
            self._pending_marker_clear = False
        if not self._cached_map_markers:
            return
        msg = MarkerArray()
        now = self.get_clock().now().to_msg()
        for template in self._cached_map_markers:
            marker = copy.deepcopy(template)
            marker.header.stamp = now
            marker.lifetime = Duration(sec=2, nanosec=0)
            msg.markers.append(marker)
        self.map_marker_pub.publish(msg)

    def _publish_tracked_objects(self) -> None:
        msg = TrackedObjects()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        data = self.annotator.current_data or {}
        neighbors = data.get("neighbor_agents_past")
        if neighbors is None:
            self.tracked_objects_pub.publish(msg)
            return
        neighbors_np = torch.as_tensor(neighbors).cpu().numpy()
        if neighbors_np.ndim < 3:
            self.tracked_objects_pub.publish(msg)
            return
        agents = neighbors_np[0] if neighbors_np.ndim == 4 else neighbors_np
        ego_xy: tuple[float, float] | None = None
        if "ego_current_state" in data:
            ego_state = torch.as_tensor(data["ego_current_state"]).cpu().numpy().reshape(-1)
            if ego_state.size >= 2:
                ego_xy = (float(ego_state[0]), float(ego_state[1]))
        for agent_row in agents:
            if agent_row.shape[0] == 0:
                continue
            row = agent_row[-1]
            if ego_xy is not None and len(row) > 1:
                dx = float(row[0]) - ego_xy[0]
                dy = float(row[1]) - ego_xy[1]
                # Do not publish ego as a tracked object.
                if dx * dx + dy * dy < 1.0:
                    continue
            tracked = TrackedObject()
            obj_id = UUID()
            obj_id.uuid = list(uuid.uuid4().bytes)
            tracked.object_id = obj_id
            cls = ObjectClassification()
            cls.label = ObjectClassification.CAR
            cls.probability = 0.5
            tracked.classification = [cls]
            kinematics = TrackedObjectKinematics()
            kinematics.pose_with_covariance.pose.position.x = float(row[0]) if len(row) > 0 else 0.0
            kinematics.pose_with_covariance.pose.position.y = float(row[1]) if len(row) > 1 else 0.0
            heading = math.atan2(float(row[3]), float(row[2])) if len(row) > 3 else (float(row[2]) if len(row) > 2 else 0.0)
            qx, qy, qz, qw = self._quat_from_heading(heading)
            kinematics.pose_with_covariance.pose.orientation.x = qx
            kinematics.pose_with_covariance.pose.orientation.y = qy
            kinematics.pose_with_covariance.pose.orientation.z = qz
            kinematics.pose_with_covariance.pose.orientation.w = qw
            tracked.kinematics = kinematics
            shape = Shape()
            shape.type = Shape.BOUNDING_BOX
            shape.dimensions = Vector3(x=4.5, y=1.8, z=1.6)
            tracked.shape = shape
            msg.objects.append(tracked)
        self.tracked_objects_pub.publish(msg)

    def _publish_static_streams_tick(self) -> None:
        self._publish_map_markers_tick()
        self._publish_tracked_objects()

    def _publish_trajectory_streams_tick(self) -> None:
        self.det_pub.publish(self._trajectory_to_msg(self.annotator.trajectory_1))
        self.stoch_pub.publish(self._trajectory_to_msg(self.annotator.trajectory_2))
        self.gt_pub.publish(self._gt_to_msg())
        self.ego_hist_pub.publish(self._ego_history_to_msg())
        self.gt_snippet_pub.publish(self._gt_snippet_to_msg())

    def _marker_for_base_link(self, frame_id: str, marker_id: int, color: tuple[float, float, float]) -> Marker:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = frame_id
        marker.ns = "ego_footprint"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.lifetime = Duration(sec=2, nanosec=0)
        # Keep vehicle body slightly above map/path to avoid z-fighting.
        marker.pose.position.z = 0.77
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=4.8, y=2.0, z=1.5)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        return marker

    def _ego_current_marker(self) -> Marker | None:
        data = self.annotator.current_data or {}
        if "ego_current_state" not in data:
            return None
        ego_state = torch.as_tensor(data["ego_current_state"]).cpu().numpy().reshape(-1)
        if ego_state.size < 4:
            return None
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "ego_current"
        marker.id = 9000
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.lifetime = Duration(sec=2, nanosec=0)
        marker.pose.position.x = float(ego_state[0])
        marker.pose.position.y = float(ego_state[1])
        marker.pose.position.z = 0.6
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=1.2, y=1.2, z=1.2)
        marker.color.r = 0.1
        marker.color.g = 0.45
        marker.color.b = 1.0
        marker.color.a = 1.0
        return marker

    def _publish_footprint_markers(self) -> None:
        msg = MarkerArray()
        ego_marker = self._ego_current_marker()
        if ego_marker is not None:
            msg.markers.append(ego_marker)
        msg.markers.append(self._marker_for_base_link("deterministic_base_link", 0, (0.1, 0.9, 0.1)))
        msg.markers.append(self._marker_for_base_link("stochastic_base_link", 1, (1.0, 0.6, 0.0)))
        self.footprint_pub.publish(msg)

    def _trajectory_pose_at_index(self, trajectory: Any, idx: int) -> tuple[float, float, float]:
        if trajectory is None:
            return (0.0, 0.0, 0.0)
        arr = torch.as_tensor(trajectory).cpu().numpy()
        if arr.shape[0] == 0:
            return (0.0, 0.0, 0.0)
        i = max(0, min(idx, arr.shape[0] - 1))
        row = arr[i]
        x = float(row[0]) if len(row) > 0 else 0.0
        y = float(row[1]) if len(row) > 1 else 0.0
        heading = math.atan2(float(row[3]), float(row[2])) if len(row) > 3 else (float(row[2]) if len(row) > 2 else 0.0)
        return (x, y, heading)

    def _broadcast_tf(self, child_frame_id: str, x: float, y: float, heading: float) -> None:
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "map"
        tf.child_frame_id = child_frame_id
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = 0.0
        qx, qy, qz, qw = self._quat_from_heading(heading)
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

    def _publish_tf_links(self) -> None:
        idx = int(self.params["time_step"])
        det = self._trajectory_pose_at_index(self._cached_det_np, idx)
        stoch = self._trajectory_pose_at_index(self._cached_stoch_np, idx)
        self._broadcast_tf("deterministic_base_link", *det)
        self._broadcast_tf("stochastic_base_link", *stoch)

    def _publish_dynamic_streams_tick(self) -> None:
        self._publish_tf_links()
        self._publish_footprint_markers()

    def _state_payload(self, plots, metric_text, progress_text, metrics_text, sidebar_status, history_display) -> dict[str, Any]:
        if plots is None:
            plot_payload = self.last_payload.get(
                "plots",
                {"trajectory": None, "velocity": None, "lateral": None},
            )
        else:
            plot_payload = {
                "trajectory": self._fig_to_base64(plots[0]),
                "velocity": self._fig_to_base64(plots[1]),
                "lateral": self._fig_to_base64(plots[2]),
            }
        payload = {
            "texts": {
                "metric": metric_text or "",
                "progress": progress_text or "",
                "metrics": metrics_text or "",
                "metrics_full_table": metrics_text or "",
                "metrics_ade_fde_table": metrics_text or "",
                "sidebar": sidebar_status or "",
                "history": history_display or "",
            },
            "plots": plot_payload,
            "params": self.params,
            "status": {
                "current_index": self.annotator.current_index,
                "total_samples": len(self.annotator.npz_paths),
                "total_preferences": len(self.annotator.preferences),
                "target_count": self.annotator.target_count,
                "annotation_complete": self.annotator.annotation_complete,
                "current_filter": self.annotator.current_filter,
                "auto_skip_labeled": self.annotator.auto_skip_labeled,
                "current_jump_size": self.annotator.current_jump_size,
                "is_pruned": getattr(self.annotator, "is_pruned", False),
                "initial_displacement": getattr(self.annotator, "initial_displacement", 0.0),
                "initial_yaw_diff": getattr(self.annotator, "initial_yaw_diff", 0.0),
                "gt_available": getattr(self.annotator, "gt_available", False),
            },
            "training": self.training_status,
        }
        self.last_payload = payload
        return payload

    def _publish_all(self, payload: dict[str, Any]) -> None:
        self._publish_state(payload)
        self._cached_det_np = self.annotator.trajectory_1
        self._cached_stoch_np = self.annotator.trajectory_2
        self._publish_trajectory_streams_tick()
        self._publish_static_streams_tick()
        self._publish_dynamic_streams_tick()

    def _load_sample(self) -> None:
        kw = self._get_annotator_kwargs()
        result = self.annotator.load_sample(
            kw["noise_scale"],
            kw["fde_threshold"],
            kw["ade_threshold"],
            kw["max_retries"],
            kw["zoom_level"],
            kw["gt_similarity_mode"],
            enable_initial_pruning=kw["enable_initial_pruning"],
            initial_pos_threshold=kw["initial_pos_threshold"],
            initial_yaw_threshold_deg=kw["initial_yaw_threshold_deg"],
            enable_guidance=kw["enable_guidance"],
            use_collision=kw["use_collision"],
            use_route_following=kw["use_route_following"],
            use_lane_keeping=kw["use_lane_keeping"],
            use_centerline_following=kw["use_centerline_following"],
            guidance_scale=kw["guidance_scale"],
            time_step=kw["time_step"],
        )
        self._pending_marker_clear = True
        self._static_markers_dirty = True
        self._refresh(result)

    def _refresh(self, result_tuple: tuple[Any, ...]) -> None:
        if self._last_sample_index >= 0 and self.annotator.current_index != self._last_sample_index:
            self._pending_marker_clear = True
            self._static_markers_dirty = True
        self._last_sample_index = self.annotator.current_index
        plots = self.annotator.update_time_display(self.params["time_step"], self.params["zoom_level"])
        metric_text, progress_text, metrics_text, sidebar_status, history_display = result_tuple[3:8]
        payload = self._state_payload(plots, metric_text, progress_text, metrics_text, sidebar_status, history_display)
        self._publish_all(payload)

    def _publish_current(self) -> None:
        payload = self._state_payload(
            None,
            self.last_payload.get("texts", {}).get("metric", ""),
            self.last_payload.get("texts", {}).get("progress", ""),
            self.last_payload.get("texts", {}).get("metrics", ""),
            self.last_payload.get("texts", {}).get("sidebar", ""),
            self.last_payload.get("texts", {}).get("history", ""),
        )
        self._publish_all(payload)

    def _publish_current_with_refreshed_plots(self) -> None:
        plots = self.annotator.update_time_display(self.params["time_step"], self.params["zoom_level"])
        payload = self._state_payload(
            plots,
            self.last_payload.get("texts", {}).get("metric", ""),
            self.last_payload.get("texts", {}).get("progress", ""),
            self.last_payload.get("texts", {}).get("metrics", ""),
            self.last_payload.get("texts", {}).get("sidebar", ""),
            self.last_payload.get("texts", {}).get("history", ""),
        )
        self._publish_all(payload)

    def _get_annotator_kwargs(self) -> dict[str, Any]:
        """Return full param dict for annotator method calls."""
        return {
            "noise_scale": self.params["noise_scale"],
            "fde_threshold": self.params["fde_threshold"],
            "ade_threshold": self.params["ade_threshold"],
            "max_retries": self.params["max_retries"],
            "zoom_level": self.params["zoom_level"],
            "gt_similarity_mode": self.params["gt_similarity_mode"],
            "enable_initial_pruning": self.params["enable_initial_pruning"],
            "initial_pos_threshold": self.params["initial_pos_threshold"],
            "initial_yaw_threshold_deg": self.params["initial_yaw_threshold_deg"],
            "enable_guidance": self.params["enable_guidance"],
            "use_collision": self.params["use_collision"],
            "use_route_following": self.params["use_route_following"],
            "use_lane_keeping": self.params["use_lane_keeping"],
            "use_centerline_following": self.params["use_centerline_following"],
            "guidance_scale": self.params["guidance_scale"],
            "time_step": self.params["time_step"],
        }

    def _publish_time_step_update(self) -> None:
        # Lightweight update path: keep existing plots and trajectories, only update time-step state + TF/footprint.
        payload = self._state_payload(
            None,
            self.last_payload.get("texts", {}).get("metric", ""),
            self.last_payload.get("texts", {}).get("progress", ""),
            self.last_payload.get("texts", {}).get("metrics", ""),
            self.last_payload.get("texts", {}).get("sidebar", ""),
            self.last_payload.get("texts", {}).get("history", ""),
        )
        self._publish_state(payload)
        self._publish_dynamic_streams_tick()

    def _on_cmd(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            action = data.get("type", "")
            payload = data.get("payload", {})
            if action == "get_state":
                self._publish_current()
            elif action == "set_params":
                for key, value in payload.items():
                    if key in self.params:
                        self.params[key] = value
                self._publish_current()
            elif action == "load_sample":
                self._load_sample()
            elif action == "regenerate":
                kw = self._get_annotator_kwargs()
                self._refresh(
                    self.annotator.regenerate(
                        kw["noise_scale"],
                        kw["fde_threshold"],
                        kw["ade_threshold"],
                        kw["max_retries"],
                        kw["zoom_level"],
                        kw["gt_similarity_mode"],
                        enable_initial_pruning=kw["enable_initial_pruning"],
                        initial_pos_threshold=kw["initial_pos_threshold"],
                        initial_yaw_threshold_deg=kw["initial_yaw_threshold_deg"],
                        enable_guidance=kw["enable_guidance"],
                        use_collision=kw["use_collision"],
                        use_route_following=kw["use_route_following"],
                        use_lane_keeping=kw["use_lane_keeping"],
                        use_centerline_following=kw["use_centerline_following"],
                        guidance_scale=kw["guidance_scale"],
                        time_step=kw["time_step"],
                    )
                )
            elif action == "select_winner":
                winner = payload.get("winner", "trajectory_2")
                if winner == "orange":
                    winner = "trajectory_2"
                elif winner == "green":
                    winner = "trajectory_1"
                kw = self._get_annotator_kwargs()
                self._refresh(
                    self.annotator.select_winner(
                        winner,
                        kw["noise_scale"],
                        kw["fde_threshold"],
                        kw["ade_threshold"],
                        kw["max_retries"],
                        kw["zoom_level"],
                        kw["gt_similarity_mode"],
                        enable_initial_pruning=kw["enable_initial_pruning"],
                        initial_pos_threshold=kw["initial_pos_threshold"],
                        initial_yaw_threshold_deg=kw["initial_yaw_threshold_deg"],
                        enable_guidance=kw["enable_guidance"],
                        use_collision=kw["use_collision"],
                        use_route_following=kw["use_route_following"],
                        use_lane_keeping=kw["use_lane_keeping"],
                        use_centerline_following=kw["use_centerline_following"],
                        guidance_scale=kw["guidance_scale"],
                        time_step=kw["time_step"],
                    )
                )
            elif action == "select_gt_as_winner":
                kw = self._get_annotator_kwargs()
                self._refresh(
                    self.annotator.select_gt_as_winner(
                        kw["noise_scale"],
                        kw["fde_threshold"],
                        kw["ade_threshold"],
                        kw["max_retries"],
                        kw["zoom_level"],
                        kw["gt_similarity_mode"],
                        enable_initial_pruning=kw["enable_initial_pruning"],
                        initial_pos_threshold=kw["initial_pos_threshold"],
                        initial_yaw_threshold_deg=kw["initial_yaw_threshold_deg"],
                        enable_guidance=kw["enable_guidance"],
                        use_collision=kw["use_collision"],
                        use_route_following=kw["use_route_following"],
                        use_lane_keeping=kw["use_lane_keeping"],
                        use_centerline_following=kw["use_centerline_following"],
                        guidance_scale=kw["guidance_scale"],
                        time_step=kw["time_step"],
                    )
                )
            elif action == "jump":
                delta = int(payload.get("delta", 0))
                self.annotator.update_jump_size(delta)
                kw = self._get_annotator_kwargs()
                self._refresh(
                    self.annotator.jump(
                        delta,
                        kw["noise_scale"],
                        kw["fde_threshold"],
                        kw["ade_threshold"],
                        kw["max_retries"],
                        kw["zoom_level"],
                        kw["gt_similarity_mode"],
                        enable_initial_pruning=kw["enable_initial_pruning"],
                        initial_pos_threshold=kw["initial_pos_threshold"],
                        initial_yaw_threshold_deg=kw["initial_yaw_threshold_deg"],
                        enable_guidance=kw["enable_guidance"],
                        use_collision=kw["use_collision"],
                        use_route_following=kw["use_route_following"],
                        use_lane_keeping=kw["use_lane_keeping"],
                        use_centerline_following=kw["use_centerline_following"],
                        guidance_scale=kw["guidance_scale"],
                        time_step=kw["time_step"],
                    )
                )
            elif action == "jump_to_index":
                kw = self._get_annotator_kwargs()
                self._refresh(
                    self.annotator.jump_to_index(
                        int(payload.get("target_index", 1)),
                        kw["noise_scale"],
                        kw["fde_threshold"],
                        kw["ade_threshold"],
                        kw["max_retries"],
                        kw["zoom_level"],
                        kw["gt_similarity_mode"],
                        enable_initial_pruning=kw["enable_initial_pruning"],
                        initial_pos_threshold=kw["initial_pos_threshold"],
                        initial_yaw_threshold_deg=kw["initial_yaw_threshold_deg"],
                        enable_guidance=kw["enable_guidance"],
                        use_collision=kw["use_collision"],
                        use_route_following=kw["use_route_following"],
                        use_lane_keeping=kw["use_lane_keeping"],
                        use_centerline_following=kw["use_centerline_following"],
                        guidance_scale=kw["guidance_scale"],
                        time_step=kw["time_step"],
                    )
                )
            elif action == "jump_to_next_unlabeled":
                kw = self._get_annotator_kwargs()
                self._refresh(
                    self.annotator.jump_to_next_unlabeled(
                        kw["noise_scale"],
                        kw["fde_threshold"],
                        kw["ade_threshold"],
                        kw["max_retries"],
                        kw["zoom_level"],
                        kw["gt_similarity_mode"],
                        enable_initial_pruning=kw["enable_initial_pruning"],
                        initial_pos_threshold=kw["initial_pos_threshold"],
                        initial_yaw_threshold_deg=kw["initial_yaw_threshold_deg"],
                        enable_guidance=kw["enable_guidance"],
                        use_collision=kw["use_collision"],
                        use_route_following=kw["use_route_following"],
                        use_lane_keeping=kw["use_lane_keeping"],
                        use_centerline_following=kw["use_centerline_following"],
                        guidance_scale=kw["guidance_scale"],
                        time_step=kw["time_step"],
                    )
                )
            elif action == "toggle_filter":
                kw = self._get_annotator_kwargs()
                self._refresh(
                    self.annotator.toggle_filter(
                        payload.get("filter_mode", "All"),
                        kw["noise_scale"],
                        kw["fde_threshold"],
                        kw["ade_threshold"],
                        kw["max_retries"],
                        kw["zoom_level"],
                        kw["gt_similarity_mode"],
                        enable_initial_pruning=kw["enable_initial_pruning"],
                        initial_pos_threshold=kw["initial_pos_threshold"],
                        initial_yaw_threshold_deg=kw["initial_yaw_threshold_deg"],
                        enable_guidance=kw["enable_guidance"],
                        use_collision=kw["use_collision"],
                        use_route_following=kw["use_route_following"],
                        use_lane_keeping=kw["use_lane_keeping"],
                        use_centerline_following=kw["use_centerline_following"],
                        guidance_scale=kw["guidance_scale"],
                        time_step=kw["time_step"],
                    )
                )
            elif action == "set_auto_skip":
                self.annotator.auto_skip_labeled = bool(payload.get("enabled", False))
                self._publish_current()
            elif action == "update_time":
                self.params["time_step"] = int(payload.get("time_step", self.params["time_step"]))
                self._publish_time_step_update()
            elif action == "update_zoom":
                self.params["zoom_level"] = int(payload.get("zoom_level", self.params["zoom_level"]))
                self._publish_current_with_refreshed_plots()
            elif action == "launch_training":
                self._refresh(self.annotator.launch_training())
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed handling command: {exc}")

class AnnotationRosServer:
    """Background runner wrapper to keep train_dpo integration compact."""

    def __init__(self, model_path: Path, npz_list: Path, target_count: int | None, device: str):
        self.model_path = model_path
        self.npz_list = npz_list
        self.target_count = target_count
        self.device = device
        self.node: AnnotationRosNode | None = None
        self.executor = None
        self.thread: threading.Thread | None = None

    def start_background(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        if not rclpy.ok():
            rclpy.init()
        self.node = AnnotationRosNode(
            model_path=self.model_path,
            npz_list=self.npz_list,
            target_count=self.target_count,
            device=self.device,
        )
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)

        def _spin() -> None:
            self.executor.spin()

        self.thread = threading.Thread(target=_spin, daemon=True)
        self.thread.start()

    def stop_background(self) -> None:
        if self.executor is not None:
            self.executor.shutdown()
        if self.node is not None:
            self.node.destroy_node()
        if self.thread is not None:
            self.thread.join(timeout=5)
        if rclpy.ok():
            rclpy.shutdown()

    def wait_for_annotation_complete(self, poll_interval_sec: float = 0.5) -> list[dict]:
        if self.node is None:
            return []
        return self.node.wait_for_annotation_complete(poll_interval_sec=poll_interval_sec)

    def reset_annotation_round(self, target_count: int | None = None) -> None:
        if self.node is None:
            return
        self.node.reset_annotation_round(target_count=target_count)

    def update_training_status(self, **kwargs) -> None:
        if self.node is None:
            return
        self.node.update_training_status(**kwargs)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS2 annotation server node.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--npz-list", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = AnnotationRosNode(
        model_path=args.model_path,
        npz_list=args.npz_list,
        target_count=args.target_count,
        device=args.device,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

