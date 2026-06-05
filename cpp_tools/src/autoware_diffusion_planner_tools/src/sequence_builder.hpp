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

#pragma once

#include "data_types.hpp"

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <deque>
#include <string>
#include <vector>

// Template must live in the header so all TUs can instantiate it.
template <typename T>
std::vector<T> check_and_update_msg(
  std::deque<T> & msgs, const builtin_interfaces::msg::Time & target_stamp)
{
  const int64_t target_time = parse_timestamp(target_stamp);
  std::vector<T> result;
  int64_t best_index = -1;

  for (int64_t i = 0; i < static_cast<int64_t>(msgs.size()); ++i) {
    const auto & msg = msgs[i];
    builtin_interfaces::msg::Time msg_stamp;
    if constexpr (
      std::is_same_v<T, nav_msgs::msg::Odometry> ||
      std::is_same_v<T, autoware_perception_msgs::msg::TrackedObjects> ||
      std::is_same_v<T, geometry_msgs::msg::AccelWithCovarianceStamped>) {
      msg_stamp = msg.header.stamp;
    } else if constexpr (
      std::is_same_v<T, autoware_vehicle_msgs::msg::TurnIndicatorsReport> ||
      std::is_same_v<T, autoware_perception_msgs::msg::TrafficLightGroupArray>) {
      msg_stamp = msg.stamp;
    }

    const int64_t msg_time = parse_timestamp(msg_stamp);
    const int64_t time_diff = target_time - msg_time;

    if (time_diff < 0) {
      break;
    }

    if (time_diff <= static_cast<int64_t>(2e8)) {  // 200 msec threshold
      result.push_back(msg);
      best_index = i;
    }
  }

  if (best_index >= 0) {
    msgs.erase(msgs.begin(), msgs.begin() + best_index);
  }
  return result;
}

std::vector<SequenceData> build_sequences(
  const std::deque<autoware_perception_msgs::msg::TrackedObjects> & tracked_objects_msgs,
  std::deque<nav_msgs::msg::Odometry> & kinematic_states,
  std::deque<geometry_msgs::msg::AccelWithCovarianceStamped> & accelerations,
  std::deque<autoware_perception_msgs::msg::TrafficLightGroupArray> & traffic_signals,
  std::deque<autoware_vehicle_msgs::msg::TurnIndicatorsReport> & turn_indicators,
  const std::vector<autoware_planning_msgs::msg::LaneletRoute> & route_msgs,
  bool search_nearest_route, const std::string & save_dir,
  const std::string & rosbag_dir_name);

void merge_sequences(std::vector<SequenceData> & sequences);

void sort_sequences(std::vector<SequenceData> & sequences);
