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

#pragma once

#include <autoware/diffusion_planner/constants.hpp>
#include <autoware/diffusion_planner/dimensions.hpp>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <builtin_interfaces/msg/time.hpp>

#include <cstdint>
#include <string>
#include <vector>

// Constants derived from dimensions.hpp
constexpr int64_t NEIGHBOR_PAST_DIM = autoware::diffusion_planner::NEIGHBOR_SHAPE[3];
constexpr int64_t NEIGHBOR_FUTURE_DIM = 4;  // x, y, cos(yaw), sin(yaw)

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

enum class MissingTopicType {
  KinematicState,  // /localization/kinematic_state
  Acceleration,    // /localization/acceleration
  TrackedObjects,  // /perception/object_recognition/tracking/objects
  Route,           // /planning/mission_planning/route
  TurnIndicators,  // /vehicle/status/turn_indicators_status
  TrafficSignals,  // /perception/traffic_light_recognition/traffic_signals
};

enum class IncompleteDataType {
  KinematicState,  // Kinematic state message missing
  Acceleration,    // Acceleration message missing
  TrackedObjects,  // Tracked objects message missing
  TrafficSignals,  // Traffic signals message missing
  TurnIndicators,  // Turn indicators message missing
};

enum class SkippingLabel {
  NotSkipped,  // Frame/route was accepted (no skip)

  // Rosbag-level skipping reasons (missing topics)
  MissingRequiredTopic,  // Required ROS topic is missing from rosbag

  // Frame-level skipping reasons (incomplete data)
  IncompleteData,  // Some messages are missing at the beginning of recording

  // Sequence-level skipping reasons
  InsufficientFrames,    // Sequence has fewer frames than minimum required
  InsufficientDistance,  // Traveled distance of sequence is too short

  // Frame processing skipping reasons
  RedOrYellowLight,  // At red or yellow light with forward future trajectory
  VehicleStopped,    // Ego vehicle is stopped
};

struct SkippingInfo
{
  SkippingLabel label;
  std::string details;
  std::vector<MissingTopicType> missing_topic_types;
  std::vector<IncompleteDataType> incomplete_data_types;

  static SkippingInfo accepted() { return {SkippingLabel::NotSkipped, "Accepted", {}, {}}; }

  static SkippingInfo missing_topics(const std::vector<MissingTopicType> & types)
  {
    static const std::string topic_map[] = {"KinematicState", "Acceleration",   "TrackedObjects",
                                            "Route",          "TurnIndicators", "TrafficSignals"};
    std::string details = "Missing topics: ";
    for (size_t i = 0; i < types.size(); ++i) {
      if (i > 0) details += ", ";
      details += topic_map[static_cast<int>(types[i])];
    }
    return {SkippingLabel::MissingRequiredTopic, details, types, {}};
  }

  static SkippingInfo incomplete_data(const std::vector<IncompleteDataType> & types)
  {
    static const std::string data_map[] = {
      "KinematicState", "Acceleration", "TrackedObjects", "TrafficSignals", "TurnIndicators"};
    std::string details = "Incomplete data: ";
    for (size_t i = 0; i < types.size(); ++i) {
      if (i > 0) details += ", ";
      details += data_map[static_cast<int>(types[i])];
    }
    return {SkippingLabel::IncompleteData, details, {}, types};
  }

  static SkippingInfo insufficient_frames(int64_t actual, int64_t minimum)
  {
    return {
      SkippingLabel::InsufficientFrames,
      "Only " + std::to_string(actual) + " frames (minimum: " + std::to_string(minimum) + ")",
      {},
      {}};
  }

  static SkippingInfo insufficient_distance(double traveled, double minimum)
  {
    return {
      SkippingLabel::InsufficientDistance,
      "Traveled distance " + std::to_string(traveled) + "m (minimum: " + std::to_string(minimum) +
        "m)",
      {},
      {}};
  }

  static SkippingInfo vehicle_stopped()
  {
    return {SkippingLabel::VehicleStopped, "Ego vehicle is stopped", {}, {}};
  }

  static SkippingInfo red_or_yellow_light()
  {
    return {
      SkippingLabel::RedOrYellowLight,
      "At red/yellow light with forward moving future trajectory",
      {},
      {}};
  }
};

inline int64_t parse_timestamp(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}

inline double to_millisecond(const int64_t timestamp_ns)
{
  return static_cast<double>(timestamp_ns) / 1e6;
}
