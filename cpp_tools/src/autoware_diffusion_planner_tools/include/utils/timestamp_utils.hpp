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

#ifndef UTILS__TIMESTAMP_UTILS_HPP_
#define UTILS__TIMESTAMP_UTILS_HPP_

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <cstdint>
#include <deque>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

inline std::string create_token(const int64_t seq_id, const int64_t frame_id)
{
  std::ostringstream token_stream;
  token_stream << std::setfill('0') << std::setw(8) << seq_id << "_" << std::setw(8) << frame_id;
  return token_stream.str();
}

inline int64_t parse_timestamp(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}

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

    // Only consider past messages, break if future message is encountered
    if (time_diff < 0) {
      break;
    }

    if (time_diff <= static_cast<int64_t>(2e8)) {  // 200 msec within loop
      result.push_back(msg);                       // collect all within threshold
      best_index = i;
    }
  }

  // Remove processed messages up to the selected index
  if (best_index >= 0) {
    msgs.erase(msgs.begin(), msgs.begin() + best_index);
  }
  return result;
}

#endif  // UTILS__TIMESTAMP_UTILS_HPP_
