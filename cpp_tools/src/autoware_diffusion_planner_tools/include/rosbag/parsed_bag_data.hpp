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

#ifndef ROSBAG__PARSED_BAG_DATA_HPP_
#define ROSBAG__PARSED_BAG_DATA_HPP_

#include "timestamp_stats.hpp"
#include "types/skipping_info.hpp"

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

struct ParsedBagData
{
  std::deque<nav_msgs::msg::Odometry> kinematic_states;
  std::deque<geometry_msgs::msg::AccelWithCovarianceStamped> accelerations;
  std::deque<autoware_perception_msgs::msg::TrackedObjects> tracked_objects_msgs;
  std::deque<autoware_vehicle_msgs::msg::TurnIndicatorsReport> turn_indicators;
  std::vector<autoware_planning_msgs::msg::LaneletRoute> route_msgs;
  std::deque<autoware_perception_msgs::msg::TrafficLightGroupArray> traffic_signals;
  timestamp_stats::TimestampStatsMap timestamp_stats_map;

  explicit ParsedBagData(const std::vector<std::string> & target_topics)
  : timestamp_stats_map(target_topics)
  {
  }
};

ParsedBagData load_rosbag(const std::string & rosbag_path, int64_t limit);

// Returns std::nullopt if no topics are missing; otherwise returns SkippingInfo describing
// which topics are missing.
std::optional<SkippingInfo> check_missing_topics(const ParsedBagData & data);

#endif  // ROSBAG__PARSED_BAG_DATA_HPP_
