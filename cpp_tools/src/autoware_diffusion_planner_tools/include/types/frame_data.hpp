// Copyright 2025 TIER IV, Inc.
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

#ifndef TYPES__FRAME_DATA_HPP_
#define TYPES__FRAME_DATA_HPP_

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <cstdint>
#include <vector>

struct FrameData
{
  int64_t timestamp;
  autoware_perception_msgs::msg::TrackedObjects tracked_objects;
  nav_msgs::msg::Odometry kinematic_state;
  geometry_msgs::msg::AccelWithCovarianceStamped acceleration;
  std::vector<autoware_perception_msgs::msg::TrafficLightGroupArray> traffic_signals;
  autoware_vehicle_msgs::msg::TurnIndicatorsReport turn_indicator;
};

struct SequenceData
{
  std::vector<FrameData> data_list;
  autoware_planning_msgs::msg::LaneletRoute route;
};

#endif  // TYPES__FRAME_DATA_HPP_
