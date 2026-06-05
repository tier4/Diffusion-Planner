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
#include "timestamp_stats.hpp"

#include <nav_msgs/msg/odometry.hpp>

#include <cstdint>
#include <string>
#include <vector>

std::string create_token(int64_t seq_id, int64_t frame_id);

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
  const nav_msgs::msg::Odometry & kinematic_state, int64_t timestamp,
  const SkippingInfo & skipping_info);

void save_route_json(
  const std::string & output_path, const std::string & rosbag_dir_name,
  const std::string & identifier, int64_t num_frames, double traveled_distance_m,
  int64_t start_timestamp, int64_t end_timestamp, const SkippingInfo & skipping_info,
  const timestamp_stats::TimestampStatsMap & timestamp_stats_map);
