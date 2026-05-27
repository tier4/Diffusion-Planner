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

#ifndef TYPES__SKIPPING_INFO_HPP_
#define TYPES__SKIPPING_INFO_HPP_

#include <cstdint>
#include <string>
#include <vector>

// Detailed categorization of missing topics
enum class MissingTopicType {
  KinematicState,  // /localization/kinematic_state
  Acceleration,    // /localization/acceleration
  TrackedObjects,  // /perception/object_recognition/tracking/objects
  Route,           // /planning/mission_planning/route
  TurnIndicators,  // /vehicle/status/turn_indicators_status
  TrafficSignals,  // /perception/traffic_light_recognition/traffic_signals
};

// Detailed categorization of incomplete data at frame level
enum class IncompleteDataType {
  KinematicState,  // Kinematic state message missing
  Acceleration,    // Acceleration message missing
  TrackedObjects,  // Tracked objects message missing
  TrafficSignals,  // Traffic signals message missing
  TurnIndicators,  // Turn indicators message missing
};

// Top-level skipping reason with detailed categorization
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

  // Filter skipping reasons (ported from the standalone python filter scripts)
  Collision,  // GT ego trajectory collides with a static object, neighbor, or road border
  OffLane,    // GT ego trajectory is too far from any lane centerline
};

// Structure to hold detailed skipping information
struct SkippingInfo
{
  SkippingLabel label;
  std::string details;  // Human-readable details (e.g., topic name, message type)
  std::vector<MissingTopicType> missing_topic_types;
  std::vector<IncompleteDataType> incomplete_data_types;

  static SkippingInfo accepted() { return {SkippingLabel::NotSkipped, "Accepted", {}, {}}; }

  // Specialized constructors for convenience
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

  // reasons: any of "static_object", "neighbor", "road_border"
  static SkippingInfo collision(const std::vector<std::string> & reasons)
  {
    std::string details = "Collision: ";
    for (size_t i = 0; i < reasons.size(); ++i) {
      if (i > 0) details += ", ";
      details += reasons[i];
    }
    return {SkippingLabel::Collision, details, {}, {}};
  }

  static SkippingInfo off_lane(float mean_distance, float max_distance)
  {
    return {
      SkippingLabel::OffLane,
      "Off lane: mean_dist=" + std::to_string(mean_distance) +
        "m, max_dist=" + std::to_string(max_distance) + "m",
      {},
      {}};
  }
};

#endif  // TYPES__SKIPPING_INFO_HPP_
