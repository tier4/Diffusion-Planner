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

#include "label_builder.hpp"

#include "autoware/ml_planner/dimensions.hpp"
#include "autoware/ml_planner/preprocessing/items/agent.hpp"
#include "autoware/ml_planner/preprocessing/items/ego_history.hpp"
#include "autoware/ml_planner/preprocessing/items/traffic_signals.hpp"
#include "autoware/ml_planner/preprocessing/items/turn_indicators.hpp"
#include "autoware/ml_planner/utils/utils.hpp"

#include <Eigen/Dense>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <string>
#include <vector>

namespace autoware::ml_planner::data {

namespace {

using autoware_perception_msgs::msg::TrackedObjects;
using nav_msgs::msg::Odometry;

double stamp_sec_of(const std_msgs::msg::Header &header) {
  return rclcpp::Time(header.stamp).seconds();
}

} // namespace

preprocess::TensorMapResult create_label_data_map(
    const rclcpp::Time &frame_time,
    const preprocess::MessageView<Odometry> &ego_msgs,
    const preprocess::MessageView<TrackedObjects> &objects_msgs,
    const preprocess::MessageView<
        autoware_vehicle_msgs::msg::TurnIndicatorsReport> &turn_indicators_msgs,
    const preprocess::MessageView<
        autoware_perception_msgs::msg::TrafficLightGroupArray>
        &traffic_signals_msgs,
    const autoware_planning_msgs::msg::LaneletRoute &route,
    const preprocess::LaneSegmentContext &map_context,
    const std::vector<preprocess::SelectedAgent> &selected_agents,
    const LabelBuilderParams &params) {
  const double frame_sec = frame_time.seconds();
  const double horizon_s =
      static_cast<double>(params.num_future_steps) * params.time_step_s;

  // The frame is only usable as a training sample if the ego odometry covers
  // the full future horizon.
  if (ego_msgs.empty() ||
      stamp_sec_of(ego_msgs.back().header) < frame_sec + horizon_s) {
    return tl::unexpected(
        std::string{"Ego odometry does not cover the future horizon"});
  }

  // Ego frame at frame_time: the newest odometry at or before the frame,
  // identical to the input builder.
  const Odometry *current_odom = nullptr;
  for (const Odometry &msg : ego_msgs) {
    if (stamp_sec_of(msg.header) <= frame_sec) {
      current_odom = &msg;
    } else {
      break;
    }
  }
  if (current_odom == nullptr) {
    return tl::unexpected(
        std::string{"No ego odometry at or before the frame time"});
  }
  const Eigen::Matrix4d ego_to_map_transform =
      utils::pose_to_matrix4d(current_odom->pose.pose);
  const Eigen::Matrix4d map_to_ego_transform =
      utils::inverse(ego_to_map_transform);

  preprocess::TensorMap label_data_map;

  // Ego future: reuse the ego history item with the grid anchored at the end
  // of the horizon, which yields frame_time + [1 .. num_future_steps] * dt.
  label_data_map["ego_agent_future"] = preprocess::create_ego_history(
      ego_msgs, static_cast<size_t>(params.num_future_steps),
      map_to_ego_transform,
      frame_time + rclcpp::Duration::from_seconds(horizon_s));

  // Neighbor futures, ordered like the neighbor_agents_past input.
  label_data_map["neighbor_agents_future"] =
      preprocess::create_neighbor_agent_sequence(
          objects_msgs, selected_agents, frame_time, map_to_ego_transform,
          MAX_NUM_NEIGHBORS, static_cast<size_t>(params.num_future_steps),
          params.time_step_s, preprocess::AgentSequenceDirection::Future,
          params.neighbor_observation_timeout_s);

  // Turn indicator future: zero-order hold on the same future grid.
  label_data_map["turn_indicators_future"] = preprocess::create_turn_indicators(
      turn_indicators_msgs,
      frame_time + rclcpp::Duration::from_seconds(horizon_s),
      params.num_future_steps, params.time_step_s);

  // Traffic light futures: same segment selection (and row order) as the
  // "lanes" / "route_lanes" inputs at frame_time, sampled on the future grid.
  {
    const auto center_x =
        static_cast<float>(current_odom->pose.pose.position.x);
    const auto center_y =
        static_cast<float>(current_odom->pose.pose.position.y);
    const auto center_z =
        static_cast<float>(current_odom->pose.pose.position.z);
    const rclcpp::Time grid_end_time =
        frame_time + rclcpp::Duration::from_seconds(horizon_s);

    const std::vector<int64_t> lane_indices =
        map_context.select_lane_segment_indices(map_to_ego_transform, center_x,
                                                center_y, NUM_SEGMENTS_IN_LANE);
    label_data_map["lane_traffic_light_future"] =
        preprocess::create_traffic_light_past(
            traffic_signals_msgs,
            map_context.get_traffic_light_ids(lane_indices),
            NUM_SEGMENTS_IN_LANE, grid_end_time, params.num_future_steps,
            params.time_step_s, params.traffic_light_timeout_s);

    const std::vector<int64_t> route_indices =
        map_context.select_route_segment_indices(
            route, center_x, center_y, center_z, NUM_SEGMENTS_IN_ROUTE);
    label_data_map["route_traffic_light_future"] =
        preprocess::create_traffic_light_past(
            traffic_signals_msgs,
            map_context.get_traffic_light_ids(route_indices),
            NUM_SEGMENTS_IN_ROUTE, grid_end_time, params.num_future_steps,
            params.time_step_s, params.traffic_light_timeout_s);
  }

  return label_data_map;
}

} // namespace autoware::ml_planner::data
