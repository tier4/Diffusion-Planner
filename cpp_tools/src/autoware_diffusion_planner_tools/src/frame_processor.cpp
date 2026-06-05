// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "frame_processor.hpp"

#include "io.hpp"

#include <autoware/diffusion_planner/constants.hpp>
#include <autoware/diffusion_planner/conversion/agent.hpp>
#include <autoware/diffusion_planner/dimensions.hpp>
#include <autoware/diffusion_planner/preprocessing/preprocessing_utils.hpp>
#include <autoware/diffusion_planner/preprocessing/traffic_signals.hpp>
#include <autoware/diffusion_planner/utils/utils.hpp>
#include <autoware_utils/geometry/geometry.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace autoware::diffusion_planner;
using namespace autoware_perception_msgs::msg;
using namespace geometry_msgs::msg;

std::optional<std::vector<float>> create_ego_sequence(
  const std::vector<FrameData> & data_list, const int64_t start_idx, const size_t num_timesteps,
  const Eigen::Matrix4d & map2bl_matrix, const rclcpp::Time & reference_time,
  const bool use_interpolation)
{
  std::deque<nav_msgs::msg::Odometry> odom_deque;

  if (use_interpolation) {
    for (size_t j = static_cast<size_t>(std::max(int64_t(0), start_idx)); j < data_list.size();
         ++j) {
      odom_deque.push_back(data_list[j].kinematic_state);
      if (rclcpp::Time(data_list[j].kinematic_state.header.stamp) >= reference_time) {
        break;
      }
    }

    if (odom_deque.empty() || rclcpp::Time(odom_deque.back().header.stamp) < reference_time) {
      return std::nullopt;
    }

    return preprocess::create_ego_agent_past(
      odom_deque, num_timesteps, map2bl_matrix, reference_time);
  } else {
    for (size_t j = 0; j < num_timesteps; ++j) {
      const int64_t index =
        std::min(start_idx + static_cast<int64_t>(j), static_cast<int64_t>(data_list.size()) - 1);
      if (index < 0) {
        return std::nullopt;
      }
      odom_deque.push_back(data_list[index].kinematic_state);
    }

    if (odom_deque.empty()) {
      return std::nullopt;
    }

    return preprocess::create_ego_agent_past(odom_deque, num_timesteps, map2bl_matrix);
  }
}

std::pair<std::vector<float>, std::vector<float>> process_neighbor_agents_and_future(
  const std::vector<FrameData> & data_list, const int64_t current_idx,
  const Eigen::Matrix4d & map2bl_matrix)
{
  const int64_t start_idx =
    std::max(static_cast<int64_t>(0), current_idx - INPUT_T_WITH_CURRENT + 1);
  const bool ignore_unknown_agents = true;
  AgentData agent_data_past;
  for (int64_t t = 0; t < INPUT_T_WITH_CURRENT; ++t) {
    const int64_t frame_idx = start_idx + t;
    if (frame_idx >= static_cast<int64_t>(data_list.size())) {
      break;
    }
    agent_data_past.update_histories(data_list[frame_idx].tracked_objects, ignore_unknown_agents);
  }
  const auto transformed_histories =
    agent_data_past.transformed_and_trimmed_histories(map2bl_matrix, MAX_NUM_NEIGHBORS);
  const std::vector<float> neighbor_past =
    flatten_histories_to_vector(transformed_histories, MAX_NUM_NEIGHBORS, INPUT_T_WITH_CURRENT);

  const std::vector<AgentHistory> agent_histories = transformed_histories;
  std::unordered_map<std::string, AgentHistory> id_to_history;
  for (size_t i = 0; i < agent_histories.size(); ++i) {
    const auto object_id = agent_histories[i].get_latest_state().object_id;
    id_to_history.emplace(object_id, AgentHistory(OUTPUT_T));
    id_to_history.at(object_id).update(
      agent_histories[i].get_latest_state().original_info,
      agent_histories[i].get_latest_state().timestamp);
  }

  std::vector<float> neighbor_future(MAX_NUM_NEIGHBORS * OUTPUT_T * NEIGHBOR_FUTURE_DIM, 0.0f);
  for (int64_t agent_idx = 0; agent_idx < static_cast<int64_t>(agent_histories.size());
       ++agent_idx) {
    const std::string & agent_id_str = agent_histories[agent_idx].get_latest_state().object_id;
    AgentHistory & future_history = id_to_history.at(agent_id_str);
    for (int64_t t = 1; t <= OUTPUT_T; ++t) {
      const int64_t future_frame_idx = current_idx + t;
      if (future_frame_idx >= static_cast<int64_t>(data_list.size())) {
        break;
      }
      const auto & future_objects = data_list[future_frame_idx].tracked_objects.objects;
      bool found = false;
      for (const auto & obj : future_objects) {
        const std::string obj_id = autoware_utils_uuid::to_hex_string(obj.object_id);
        if (obj_id == agent_id_str) {
          future_history.update(obj, data_list[future_frame_idx].kinematic_state.header.stamp);
          found = true;
          break;
        }
      }
      if (!found) {
        break;
      }
    }
    future_history.apply_transform(map2bl_matrix);

    const std::vector<float> arr = future_history.as_array();
    for (int64_t t = 0; t < OUTPUT_T; ++t) {
      const int64_t base_idx = agent_idx * OUTPUT_T * NEIGHBOR_FUTURE_DIM + t * NEIGHBOR_FUTURE_DIM;
      for (int64_t d = 0; d < NEIGHBOR_FUTURE_DIM; ++d) {
        if (t * AGENT_STATE_DIM + d >= arr.size()) {
          break;
        }
        neighbor_future[base_idx + d] = arr[t * AGENT_STATE_DIM + d];
      }
    }
  }

  return std::make_pair(neighbor_past, neighbor_future);
}

void process_sequence(
  SequenceData & seq, const int64_t seq_id,
  const preprocess::LaneSegmentContext & lane_segment_context, const int64_t step,
  const bool use_interpolation, const int64_t convert_red, const int64_t convert_yellow,
  const float ego_wheel_base, const std::vector<float> & ego_shape,
  const std::string & save_dir, const std::string & rosbag_dir_name)
{
  const int64_t n = static_cast<int64_t>(seq.data_list.size());

  // Use the last frame's pose as goal
  seq.route.goal_pose = seq.data_list.back().kinematic_state.pose.pose;

  int64_t stopping_count = 0;
  for (int64_t i = INPUT_T_WITH_CURRENT; i < n; i += step) {
    const std::string token = create_token(seq_id, i);

    const Eigen::Matrix4d bl2map =
      utils::pose_to_matrix4d(seq.data_list[i].kinematic_state.pose.pose);
    const Eigen::Matrix4d map2bl = utils::inverse(bl2map);

    // Ego past
    const rclcpp::Time past_reference_time(seq.data_list[i].kinematic_state.header.stamp);
    const auto ego_past_opt = create_ego_sequence(
      seq.data_list, i - INPUT_T_WITH_CURRENT + 1, INPUT_T_WITH_CURRENT, map2bl,
      past_reference_time, use_interpolation);
    if (!ego_past_opt) {
      std::cout << "Failed to create ego past at frame " << i << std::endl;
      break;
    }
    const std::vector<float> & ego_past = ego_past_opt.value();

    // Ego future
    const rclcpp::Time future_reference_time =
      past_reference_time +
      rclcpp::Duration::from_seconds(OUTPUT_T * constants::PREDICTION_TIME_STEP_S);
    const auto ego_future_opt = create_ego_sequence(
      seq.data_list, i + 1, OUTPUT_T, map2bl, future_reference_time, use_interpolation);
    if (!ego_future_opt) {
      std::cout << "Reached end of sequence at frame " << i << "/" << n << std::endl;
      break;
    }
    const std::vector<float> & ego_future = ego_future_opt.value();

    // Ego current state
    const std::vector<float> ego_current = preprocess::create_ego_current_state(
      seq.data_list[i].kinematic_state, seq.data_list[i].acceleration, ego_wheel_base);

    // Neighbor agents (past and future)
    const auto [neighbor_past, neighbor_future] =
      process_neighbor_agents_and_future(seq.data_list, i, map2bl);

    // Ego position for lane queries
    const Point & ego_pos = seq.data_list[i].kinematic_state.pose.pose.position;
    const double center_x = ego_pos.x;
    const double center_y = ego_pos.y;
    const double center_z = ego_pos.z;

    // Traffic signals
    std::map<lanelet::Id, preprocess::TrafficSignalStamped> traffic_light_id_map;
    const rclcpp::Time current_time(seq.data_list[i].tracked_objects.header.stamp);
    std::vector<autoware_perception_msgs::msg::TrafficLightGroupArray::ConstSharedPtr> msg_vec;
    for (const auto & traffic_signal_msg : seq.data_list[i].traffic_signals) {
      msg_vec.push_back(
        std::make_shared<autoware_perception_msgs::msg::TrafficLightGroupArray>(
          traffic_signal_msg));
    }
    preprocess::process_traffic_signals(msg_vec, traffic_light_id_map, current_time, 5.0);

    // Lane features
    const std::vector<int64_t> lane_segment_indices =
      lane_segment_context.select_lane_segment_indices(
        map2bl, center_x, center_y, NUM_SEGMENTS_IN_LANE);
    const auto [lanes, lanes_speed_limit] = lane_segment_context.create_tensor_data_from_indices(
      map2bl, traffic_light_id_map, lane_segment_indices, NUM_SEGMENTS_IN_LANE);

    std::vector<bool> lanes_has_speed_limit(lanes_speed_limit.size());
    for (size_t idx = 0; idx < lanes_speed_limit.size(); ++idx) {
      lanes_has_speed_limit[idx] =
        (lanes_speed_limit[idx] > std::numeric_limits<float>::epsilon());
    }

    // Route lane features
    const std::vector<int64_t> segment_indices =
      lane_segment_context.select_route_segment_indices(
        seq.route, center_x, center_y, center_z, NUM_SEGMENTS_IN_ROUTE);
    const auto [route_lanes, route_lanes_speed_limit] =
      lane_segment_context.create_tensor_data_from_indices(
        map2bl, traffic_light_id_map, segment_indices, NUM_SEGMENTS_IN_ROUTE);

    std::vector<bool> route_lanes_has_speed_limit(route_lanes_speed_limit.size());
    for (size_t idx = 0; idx < route_lanes_speed_limit.size(); ++idx) {
      route_lanes_has_speed_limit[idx] =
        (route_lanes_speed_limit[idx] > std::numeric_limits<float>::epsilon());
    }

    const std::vector<float> polygons =
      lane_segment_context.create_polygon_tensor(map2bl, center_x, center_y);
    const std::vector<float> line_strings =
      lane_segment_context.create_line_string_tensor(map2bl, center_x, center_y);

    // Goal pose in base_link frame
    const geometry_msgs::msg::Pose & goal_pose = seq.route.goal_pose;
    const Eigen::Matrix4d goal_pose_in_map = utils::pose_to_matrix4d(goal_pose);
    const Eigen::Matrix4d goal_pose_in_bl = map2bl * goal_pose_in_map;
    const float goal_x = goal_pose_in_bl(0, 3);
    const float goal_y = goal_pose_in_bl(1, 3);
    const float yaw = std::atan2(goal_pose_in_bl(1, 0), goal_pose_in_bl(0, 0));
    const std::vector<float> goal_pose_vec = {goal_x, goal_y, std::cos(yaw), std::sin(yaw)};

    // Stop detection
    const bool is_stop = seq.data_list[i].kinematic_state.twist.twist.linear.x < 0.1;
    if (is_stop) {
      stopping_count++;
    } else {
      stopping_count = 0;
    }

    // Check if ego is stopped near goal
    const float ego_future_last_x = ego_future[(OUTPUT_T - 1) * 4 + 0];
    const float ego_future_last_y = ego_future[(OUTPUT_T - 1) * 4 + 1];
    const float distance_to_goal_pose = std::sqrt(
      (ego_future_last_x - goal_x) * (ego_future_last_x - goal_x) +
      (ego_future_last_y - goal_y) * (ego_future_last_y - goal_y));

    if (stopping_count > INPUT_T && distance_to_goal_pose < 5.0) {
      std::cout << "finish at " << i << " because stopping_count=" << stopping_count
                << " and distance_to_goal_pose=" << distance_to_goal_pose << std::endl;
      break;
    }

    // Red/yellow light check (second route segment, first point)
    const int64_t segment_idx = 1;
    const int64_t point_idx = 0;
    const int64_t red_light_index = segment_idx * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM +
                                    point_idx * SEGMENT_POINT_DIM + TRAFFIC_LIGHT_RED;
    const bool is_red_light = route_lanes[red_light_index] > 0.5 && !convert_red;
    const int64_t yellow_light_index = segment_idx * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM +
                                       point_idx * SEGMENT_POINT_DIM + TRAFFIC_LIGHT_YELLOW;
    const bool is_yellow_light = route_lanes[yellow_light_index] > 0.5 && !convert_yellow;
    const bool is_red_or_yellow = is_red_light || is_yellow_light;

    float sum_mileage = 0.0;
    for (int64_t j = 0; j < OUTPUT_T - 1; ++j) {
      const float dx = ego_future[(j + 1) * 4 + 0] - ego_future[j * 4 + 0];
      const float dy = ego_future[(j + 1) * 4 + 1] - ego_future[j * 4 + 1];
      sum_mileage += std::sqrt(dx * dx + dy * dy);
    }
    const bool is_future_forward = sum_mileage > 1.0;

    const std::vector<float> static_objects(STATIC_OBJECTS_SHAPE[1] * STATIC_OBJECTS_SHAPE[2], 0.0f);

    std::vector<int32_t> turn_indicators(INPUT_T_WITH_CURRENT);
    for (int64_t t = 0; t < INPUT_T_WITH_CURRENT; ++t) {
      turn_indicators[t] =
        seq.data_list[std::max(int64_t(0), i - INPUT_T_WITH_CURRENT + 1 + t)].turn_indicator.report;
    }

    if (is_stop && is_red_or_yellow && is_future_forward) {
      std::cout << "Skip this frame " << i
                << " because it is stop at red or yellow light and future trajectory is forward"
                << std::endl;
      save_frame_json(
        save_dir, rosbag_dir_name, token, seq.data_list[i].kinematic_state,
        seq.data_list[i].timestamp, SkippingInfo::red_or_yellow_light());
      continue;
    }
    if (stopping_count > (INPUT_T + 5) && is_red_or_yellow) {
      std::cout << "Skip this frame " << i << " because stopping_count=" << stopping_count
                << " and red or yellow light" << std::endl;
      save_frame_json(
        save_dir, rosbag_dir_name, token, seq.data_list[i].kinematic_state,
        seq.data_list[i].timestamp, SkippingInfo::vehicle_stopped());
      continue;
    }

    save_frame_data(
      save_dir, rosbag_dir_name, token, ego_past, ego_current, ego_future, neighbor_past,
      neighbor_future, static_objects, lanes, lanes_speed_limit, lanes_has_speed_limit,
      route_lanes, route_lanes_speed_limit, route_lanes_has_speed_limit, polygons, line_strings,
      goal_pose_vec, turn_indicators, ego_shape);
    save_frame_json(
      save_dir, rosbag_dir_name, token, seq.data_list[i].kinematic_state,
      seq.data_list[i].timestamp, SkippingInfo::accepted());

    if (i % 100 == 0) {
      std::cout << "Processed frame " << i << "/" << n << std::endl;
    }
  }
}
