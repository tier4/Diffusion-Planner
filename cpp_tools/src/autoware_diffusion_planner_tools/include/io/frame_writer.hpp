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

#ifndef IO__FRAME_WRITER_HPP_
#define IO__FRAME_WRITER_HPP_

#include "nlohmann/json.hpp"
#include "timestamp_stats.hpp"
#include "types/skipping_info.hpp"
#include "types/training_data_binary.hpp"

#include <nav_msgs/msg/odometry.hpp>

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Pure builders — no file I/O, fully unit-testable.
// ---------------------------------------------------------------------------

TrainingDataBinary build_training_data(
  const std::vector<float> & ego_past, const std::vector<float> & ego_current,
  const std::vector<float> & ego_future, const std::vector<float> & neighbor_past,
  const std::vector<float> & neighbor_future, const std::vector<float> & static_objects,
  const std::vector<float> & lanes, const std::vector<float> & lanes_speed_limit,
  const std::vector<bool> & lanes_has_speed_limit, const std::vector<float> & route_lanes,
  const std::vector<float> & route_lanes_speed_limit,
  const std::vector<bool> & route_lanes_has_speed_limit, const std::vector<float> & polygons,
  const std::vector<float> & line_strings, const std::vector<float> & goal_pose,
  const std::vector<int32_t> & turn_indicators, const std::vector<float> & ego_shape);

nlohmann::json build_frame_json(
  const nav_msgs::msg::Odometry & kinematic_state, const int64_t timestamp,
  const SkippingInfo & skipping_info,
  const std::vector<std::string> & neighbor_ids = {});

nlohmann::json build_route_json(
  const int64_t num_frames, const double traveled_distance_m, const int64_t start_timestamp,
  const int64_t end_timestamp, const SkippingInfo & skipping_info,
  const timestamp_stats::TimestampStatsMap & timestamp_stats_map);

// ---------------------------------------------------------------------------
// File-writing wrappers — call the builders above, then persist to disk.
// ---------------------------------------------------------------------------

void save_frame_data(
  const std::string & output_path, const std::string & rosbag_dir_name, const std::string & token,
  const std::vector<float> & ego_past, const std::vector<float> & ego_current,
  const std::vector<float> & ego_future, const std::vector<float> & neighbor_past,
  const std::vector<float> & neighbor_future, const std::vector<float> & static_objects,
  const std::vector<float> & lanes, const std::vector<float> & lanes_speed_limit,
  const std::vector<bool> & lanes_has_speed_limit, const std::vector<float> & route_lanes,
  const std::vector<float> & route_lanes_speed_limit,
  const std::vector<bool> & route_lanes_has_speed_limit, const std::vector<float> & polygons,
  const std::vector<float> & line_strings, const std::vector<float> & goal_pose,
  const std::vector<int32_t> & turn_indicators, const std::vector<float> & ego_shape);

void save_frame_json(
  const std::string & output_path, const std::string & rosbag_dir_name, const std::string & token,
  const nav_msgs::msg::Odometry & kinematic_state, const int64_t timestamp,
  const SkippingInfo & skipping_info, const std::vector<std::string> & neighbor_ids = {});

void save_route_json(
  const std::string & output_path, const std::string & rosbag_dir_name,
  const std::string & identifier, const int64_t num_frames, const double traveled_distance_m,
  const int64_t start_timestamp, const int64_t end_timestamp, const SkippingInfo & skipping_info,
  const timestamp_stats::TimestampStatsMap & timestamp_stats_map);

#endif  // IO__FRAME_WRITER_HPP_
