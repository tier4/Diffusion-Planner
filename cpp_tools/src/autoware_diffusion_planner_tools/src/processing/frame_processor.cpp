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

#include "processing/frame_processor.hpp"

#include "io/frame_writer.hpp"
#include "processing/ego_sequence.hpp"
#include "processing/frame_filters.hpp"
#include "processing/neighbor_processor.hpp"
#include "types/skipping_info.hpp"
#include "utils/timestamp_utils.hpp"

#include <Eigen/Core>
#include <autoware/diffusion_planner/constants.hpp>
#include <autoware/diffusion_planner/dimensions.hpp>
#include <autoware/diffusion_planner/preprocessing/preprocessing_utils.hpp>
#include <autoware/diffusion_planner/preprocessing/traffic_signals.hpp>
#include <autoware/diffusion_planner/utils/utils.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

void process_sequence(
  SequenceData & seq, const int64_t seq_id, const ConverterOptions & options,
  const autoware::diffusion_planner::preprocess::LaneSegmentContext & lane_segment_context,
  const timestamp_stats::TimestampStatsMap & timestamp_stats_map)
{
  using autoware::diffusion_planner::INPUT_T;
  using autoware::diffusion_planner::INPUT_T_WITH_CURRENT;
  using autoware::diffusion_planner::NUM_SEGMENTS_IN_LANE;
  using autoware::diffusion_planner::NUM_SEGMENTS_IN_ROUTE;
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POINTS_PER_SEGMENT;
  using autoware::diffusion_planner::SEGMENT_POINT_DIM;
  using autoware::diffusion_planner::STATIC_OBJECTS_SHAPE;
  using autoware::diffusion_planner::TRAFFIC_LIGHT_RED;
  using autoware::diffusion_planner::TRAFFIC_LIGHT_YELLOW;
  namespace constants = autoware::diffusion_planner::constants;
  namespace preprocess = autoware::diffusion_planner::preprocess;
  namespace utils = autoware::diffusion_planner::utils;

  const int64_t n = static_cast<int64_t>(seq.data_list.size());

  std::cout << "Processing sequence " << seq_id + 1 << " with " << n << " frames" << std::endl;

  std::ostringstream seq_id_stream;
  seq_id_stream << "sequence_" << std::setfill('0') << std::setw(8) << seq_id;
  const std::string sequence_id_str = seq_id_stream.str();
  const int64_t start_ts = seq.data_list.empty() ? 0 : seq.data_list.front().timestamp;
  const int64_t end_ts = seq.data_list.empty() ? 0 : seq.data_list.back().timestamp;

  // Calculate the traveled distance
  double traveled_distance = 0.0;
  for (int64_t i = 1; i < n; ++i) {
    const auto & pos1 = seq.data_list[i - 1].kinematic_state.pose.pose.position;
    const auto & pos2 = seq.data_list[i].kinematic_state.pose.pose.position;
    const double dx = pos2.x - pos1.x;
    const double dy = pos2.y - pos1.y;
    traveled_distance += std::sqrt(dx * dx + dy * dy);
  }

  if (n < options.min_frames) {
    std::cout << "Skipping sequence with only " << n << " frames (min: " << options.min_frames
              << ")" << std::endl;
    save_route_json(
      options.save_dir, options.rosbag_dir_name, sequence_id_str, n, traveled_distance, start_ts,
      end_ts, SkippingInfo::insufficient_frames(n, options.min_frames), timestamp_stats_map);
    return;
  }

  std::cout << "Traveled distance: " << traveled_distance << " meters" << std::endl;
  if (traveled_distance < options.min_distance) {
    std::cout << "Skipping sequence with traveled distance " << traveled_distance
              << " meters (min: " << options.min_distance << " meters)" << std::endl;
    save_route_json(
      options.save_dir, options.rosbag_dir_name, sequence_id_str, n, traveled_distance, start_ts,
      end_ts, SkippingInfo::insufficient_distance(traveled_distance, options.min_distance),
      timestamp_stats_map);
    return;
  }

  save_route_json(
    options.save_dir, options.rosbag_dir_name, sequence_id_str, n, traveled_distance, start_ts,
    end_ts, SkippingInfo::accepted(), timestamp_stats_map);

  // Replace the goal pose with the last frame's pose
  seq.route.goal_pose = seq.data_list.back().kinematic_state.pose.pose;

  // Process frames with stopping count tracking
  int64_t stopping_count = 0;
  for (int64_t i = INPUT_T_WITH_CURRENT; i < n; i += options.step) {
    // Create token in canonical format: seq_id(8digits) + "_" + i(8digits)
    const std::string token = create_token(seq_id, i);

    // Get transformation matrix
    const Eigen::Matrix4d bl2map =
      utils::pose_to_matrix4d(seq.data_list[i].kinematic_state.pose.pose);
    const Eigen::Matrix4d map2bl = utils::inverse(bl2map);

    // Create ego sequences
    const rclcpp::Time past_reference_time(seq.data_list[i].kinematic_state.header.stamp);
    const auto ego_past_opt = create_ego_sequence(
      seq.data_list, i - INPUT_T_WITH_CURRENT + 1, INPUT_T_WITH_CURRENT, map2bl,
      past_reference_time, options.use_interpolation);
    if (!ego_past_opt) {
      std::cout << "Failed to create ego past at frame " << i << std::endl;
      break;
    }
    const std::vector<float> & ego_past = ego_past_opt.value();

    const rclcpp::Time future_reference_time =
      past_reference_time +
      rclcpp::Duration::from_seconds(OUTPUT_T * constants::PREDICTION_TIME_STEP_S);
    const auto ego_future_opt = create_ego_sequence(
      seq.data_list, i + 1, OUTPUT_T, map2bl, future_reference_time, options.use_interpolation);
    if (!ego_future_opt) {
      std::cout << "Reached end of sequence at frame " << i << "/" << n << std::endl;
      break;
    }
    const std::vector<float> & ego_future = ego_future_opt.value();

    // Create ego current state
    const std::vector<float> ego_current = preprocess::create_ego_current_state(
      seq.data_list[i].kinematic_state, seq.data_list[i].acceleration, options.ego_wheel_base);

    // Process neighbor agents (both past and future with consistent agent ordering)
    const auto [neighbor_past, neighbor_future] =
      process_neighbor_agents_and_future(seq.data_list, i, map2bl);

    // Process lanes and routes
    const auto & ego_pos = seq.data_list[i].kinematic_state.pose.pose.position;
    const double center_x = ego_pos.x;
    const double center_y = ego_pos.y;
    const double center_z = ego_pos.z;

    // Process traffic signals for this frame using the traffic signals from FrameData
    std::map<lanelet::Id, preprocess::TrafficSignalStamped> traffic_light_id_map;
    const auto current_stamp = seq.data_list[i].tracked_objects.header.stamp;
    const rclcpp::Time current_time(current_stamp);

    std::vector<autoware_perception_msgs::msg::TrafficLightGroupArray::ConstSharedPtr> msg_vec;
    for (const auto & traffic_signal_msg : seq.data_list[i].traffic_signals) {
      msg_vec.push_back(std::make_shared<autoware_perception_msgs::msg::TrafficLightGroupArray>(
        traffic_signal_msg));
    }
    preprocess::process_traffic_signals(msg_vec, traffic_light_id_map, current_time, 5.0);

    // Get lanes data with speed limits
    const std::vector<int64_t> lane_segment_indices =
      lane_segment_context.select_lane_segment_indices(
        map2bl, center_x, center_y, NUM_SEGMENTS_IN_LANE);
    const auto [lanes, lanes_speed_limit] = lane_segment_context.create_tensor_data_from_indices(
      map2bl, traffic_light_id_map, lane_segment_indices, NUM_SEGMENTS_IN_LANE);

    // Create has_speed_limit flags based on speed_limit values
    std::vector<bool> lanes_has_speed_limit(lanes_speed_limit.size());
    for (size_t idx = 0; idx < lanes_speed_limit.size(); ++idx) {
      lanes_has_speed_limit[idx] =
        (lanes_speed_limit[idx] > std::numeric_limits<float>::epsilon());
    }

    // Get route lanes data with speed limits
    const std::vector<int64_t> segment_indices =
      lane_segment_context.select_route_segment_indices(
        seq.route, center_x, center_y, center_z, NUM_SEGMENTS_IN_ROUTE);
    const auto [route_lanes, route_lanes_speed_limit] =
      lane_segment_context.create_tensor_data_from_indices(
        map2bl, traffic_light_id_map, segment_indices, NUM_SEGMENTS_IN_ROUTE);

    // Create route_lanes_has_speed_limit based on speed_limit values
    std::vector<bool> route_lanes_has_speed_limit(route_lanes_speed_limit.size());
    for (size_t idx = 0; idx < route_lanes_speed_limit.size(); ++idx) {
      route_lanes_has_speed_limit[idx] =
        (route_lanes_speed_limit[idx] > std::numeric_limits<float>::epsilon());
    }

    const std::vector<float> polygons =
      lane_segment_context.create_polygon_tensor(map2bl, center_x, center_y);
    const std::vector<float> line_strings =
      lane_segment_context.create_line_string_tensor(map2bl, center_x, center_y);

    // Get goal pose
    const geometry_msgs::msg::Pose & goal_pose = seq.route.goal_pose;
    const Eigen::Matrix4d goal_pose_in_map = utils::pose_to_matrix4d(goal_pose);
    const Eigen::Matrix4d goal_pose_in_bl = map2bl * goal_pose_in_map;
    const float goal_x = goal_pose_in_bl(0, 3);
    const float goal_y = goal_pose_in_bl(1, 3);
    const float yaw = std::atan2(goal_pose_in_bl(1, 0), goal_pose_in_bl(0, 0));
    const std::vector<float> goal_pose_vec = {goal_x, goal_y, std::cos(yaw), std::sin(yaw)};

    // Such data should be skipped.
    // (1)Ego vehicle is stopped
    // (2)The lanelet segment in front is a red light
    // (3)The GT trajectory is moving forward.

    // (1)Ego vehicle is stopped
    const bool is_stop = seq.data_list[i].kinematic_state.twist.twist.linear.x < 0.1;
    if (is_stop) {
      stopping_count++;
    } else {
      stopping_count = 0;
    }

    // if ego vehicle is stopped and close to goal, finish
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

    // Check for red light (next segment)
    // route_tensor[:, 1, 0, -3] corresponds to second segment, first point, red light flag
    const int64_t segment_idx = 1;  // next segment
    const int64_t point_idx = 0;    // first point
    const int64_t red_light_index = segment_idx * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM +
                                    point_idx * SEGMENT_POINT_DIM + TRAFFIC_LIGHT_RED;
    const bool is_red_light = route_lanes[red_light_index] > 0.5 && !options.convert_red;
    const int64_t yellow_light_index = segment_idx * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM +
                                       point_idx * SEGMENT_POINT_DIM + TRAFFIC_LIGHT_YELLOW;
    const bool is_yellow_light = route_lanes[yellow_light_index] > 0.5 && !options.convert_yellow;
    const bool is_red_or_yellow = is_red_light || is_yellow_light;

    float sum_mileage = 0.0;
    for (int64_t j = 0; j < OUTPUT_T - 1; ++j) {
      const float dx = ego_future[(j + 1) * 4 + 0] - ego_future[j * 4 + 0];
      const float dy = ego_future[(j + 1) * 4 + 1] - ego_future[j * 4 + 1];
      sum_mileage += std::sqrt(dx * dx + dy * dy);
    }
    const bool is_future_forward = sum_mileage > 1.0;

    // Create placeholder data for static objects
    const std::vector<float> static_objects(
      STATIC_OBJECTS_SHAPE[1] * STATIC_OBJECTS_SHAPE[2], 0.0f);

    std::vector<int32_t> turn_indicators(INPUT_T_WITH_CURRENT);
    for (int64_t t = 0; t < INPUT_T_WITH_CURRENT; ++t) {
      turn_indicators[t] = seq.data_list[std::max(int64_t(0), i - INPUT_T_WITH_CURRENT + 1 + t)]
                             .turn_indicator.report;
    }

    if (is_stop && is_red_or_yellow && is_future_forward) {
      std::cout << "Skip this frame " << i
                << " because it is stop at red or yellow light and future trajectory is forward"
                << std::endl;
      save_frame_json(
        options.save_dir, options.rosbag_dir_name, token, seq.data_list[i].kinematic_state,
        seq.data_list[i].timestamp, SkippingInfo::red_or_yellow_light());
      continue;
    }
    if (stopping_count > (INPUT_T + 5) && is_red_or_yellow) {
      std::cout << "Skip this frame " << i << " because stopping_count=" << stopping_count
                << " and red or yellow light" << std::endl;
      save_frame_json(
        options.save_dir, options.rosbag_dir_name, token, seq.data_list[i].kinematic_state,
        seq.data_list[i].timestamp, SkippingInfo::vehicle_stopped());
      continue;
    }

    // Collision-free filter (ported from filter_collision_free_npz.py), always applied:
    // drop frames whose GT ego trajectory collides with a static object, neighbor,
    // or road border.
    {
      const frame_filters::CollisionResult collision = frame_filters::check_collision(
        ego_future, options.ego_shape, static_objects, neighbor_future, neighbor_past, line_strings,
        options.static_object_margin, options.neighbor_margin, options.road_border_margin,
        options.collision_time_stride);
      if (collision.collided()) {
        std::cout << "Skip this frame " << i << " due to collision (";
        for (size_t r = 0; r < collision.reasons.size(); ++r) {
          if (r > 0) std::cout << ", ";
          std::cout << collision.reasons[r];
        }
        std::cout << ")" << std::endl;
        save_frame_json(
          options.save_dir, options.rosbag_dir_name, token, seq.data_list[i].kinematic_state,
          seq.data_list[i].timestamp, SkippingInfo::collision(collision.reasons));
        continue;
      }
    }

    save_frame_data(
      options.save_dir, options.rosbag_dir_name, token, ego_past, ego_current, ego_future,
      neighbor_past, neighbor_future, static_objects, lanes, lanes_speed_limit,
      lanes_has_speed_limit, route_lanes, route_lanes_speed_limit, route_lanes_has_speed_limit,
      polygons, line_strings, goal_pose_vec, turn_indicators, options.ego_shape);
    save_frame_json(
      options.save_dir, options.rosbag_dir_name, token, seq.data_list[i].kinematic_state,
      seq.data_list[i].timestamp, SkippingInfo::accepted());

    if (i % 100 == 0) {
      std::cout << "Processed frame " << i << "/" << n << std::endl;
    }
  }
}
