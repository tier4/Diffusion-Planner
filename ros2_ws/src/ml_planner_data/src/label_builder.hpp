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

#ifndef ML_PLANNER_DATA__SRC__LABEL_BUILDER_HPP_
#define ML_PLANNER_DATA__SRC__LABEL_BUILDER_HPP_

#include "autoware/ml_planner/preprocessing/input_builder.hpp"
#include "autoware/ml_planner/preprocessing/preprocessing_utils.hpp"

#include <rclcpp/time.hpp>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <cstdint>
#include <deque>
#include <vector>

namespace autoware::ml_planner::data {

struct LabelBuilderParams {
  /// Future grid: frame_time + (i + 1) * time_step_s, i = 0 .. num_future_steps
  /// - 1.
  int64_t num_future_steps{80}; // OUTPUT_T
  double time_step_s{0.1};      // PREDICTION_TIME_STEP_S
  /// A neighbor future grid sample is valid only if an observation exists
  /// within this age relative to the grid time.
  double neighbor_observation_timeout_s{0.3};
  /// Traffic light observations older than this relative to a grid time are
  /// treated as no-data (same meaning as in the input builder).
  double traffic_light_timeout_s{0.2};
};

/**
 * @brief Build the training label tensors for one frame (pure function).
 *
 * All message windows must cover [frame_time - HISTORY_WINDOW_S,
 * frame_time + num_future_steps * time_step_s]. All features are expressed in
 * the ego frame at frame_time (identical to the input tensors).
 *
 * Produced keys:
 * - "ego_agent_future"            (num_future_steps, 6)
 *       [x, y, cos_yaw, sin_yaw, velocity, yaw_rate], interpolated
 * - "neighbor_agents_future"      (MAX_NUM_NEIGHBORS, num_future_steps, 4)
 *       [x, y, cos_yaw, sin_yaw], zero-order hold on message stamps.
 *       Rows are ordered identically to the "neighbor_agents_past" input.
 *       Steps without a sufficiently fresh observation stay all-zero, which
 *       consumers must treat as invalid (a valid step always has
 *       cos^2 + sin^2 = 1, so it can never be all-zero).
 * - "turn_indicators_future"      (num_future_steps,)
 *       TurnIndicatorsReport.report, zero-order hold
 * - "lane_traffic_light_future"   (NUM_SEGMENTS_IN_LANE, num_future_steps,
 *                                  TRAFFIC_LIGHT_ONE_HOT_DIM)
 * - "route_traffic_light_future"  (NUM_SEGMENTS_IN_ROUTE, num_future_steps,
 *                                  TRAFFIC_LIGHT_ONE_HOT_DIM)
 *       Traffic light states on the future grid, zero-order hold. Segment
 *       rows are selected at frame_time and ordered identically to the
 *       "lanes" / "route_lanes" inputs.
 *
 * Returns an error if ego odometry does not cover the full future horizon or
 * no ego odometry is available at or before the frame time.
 */
preprocess::TensorMapResult create_label_data_map(
    const rclcpp::Time &frame_time,
    const preprocess::MessageView<nav_msgs::msg::Odometry> &ego_msgs,
    const preprocess::MessageView<autoware_perception_msgs::msg::TrackedObjects>
        &objects_msgs,
    const preprocess::MessageView<
        autoware_vehicle_msgs::msg::TurnIndicatorsReport> &turn_indicators_msgs,
    const preprocess::MessageView<
        autoware_perception_msgs::msg::TrafficLightGroupArray>
        &traffic_signals_msgs,
    const autoware_planning_msgs::msg::LaneletRoute &route,
    const preprocess::LaneSegmentContext &map_context,
    const std::vector<preprocess::SelectedAgent> &selected_agents,
    const LabelBuilderParams &params);

} // namespace autoware::ml_planner::data

#endif // ML_PLANNER_DATA__SRC__LABEL_BUILDER_HPP_
