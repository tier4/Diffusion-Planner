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

#include "io/frame_writer.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{
inline double to_millisecond(const int64_t timestamp_ns)
{
  return static_cast<double>(timestamp_ns) / 1e6;
}
}  // namespace

// ---------------------------------------------------------------------------
// Pure builders
// ---------------------------------------------------------------------------

nlohmann::json build_frame_json(
  const nav_msgs::msg::Odometry & kinematic_state, const int64_t timestamp,
  const SkippingInfo & skipping_info, const std::vector<std::string> & neighbor_ids)
{
  std::vector<int> incomplete_types;
  for (const auto & t : skipping_info.incomplete_data_types) {
    incomplete_types.push_back(static_cast<int>(t));
  }

  nlohmann::json j;
  // is_skipped (+ skipping_info.label) is the per-frame "skip_for_training" tag:
  // with --write_skipped_npz the frame is still written for the closed-loop
  // reproducer (gap-free), and training filters on this flag.
  j["is_skipped"] = (skipping_info.label != SkippingLabel::NotSkipped);
  j["timestamp"] = timestamp;
  j["x"] = kinematic_state.pose.pose.position.x;
  j["y"] = kinematic_state.pose.pose.position.y;
  j["z"] = kinematic_state.pose.pose.position.z;
  j["qx"] = kinematic_state.pose.pose.orientation.x;
  j["qy"] = kinematic_state.pose.pose.orientation.y;
  j["qz"] = kinematic_state.pose.pose.orientation.z;
  j["qw"] = kinematic_state.pose.pose.orientation.w;
  j["skipping_info"] = {
    {"label", static_cast<int>(skipping_info.label)},
    {"details", skipping_info.details},
    {"incomplete_data_types", incomplete_types}};
  // Perception track UUIDs aligned 1:1 with the neighbor_past slots (for the
  // reproducer's cross-frame association / interpolation).
  j["neighbor_ids"] = neighbor_ids;
  return j;
}

nlohmann::json build_route_json(
  const int64_t num_frames, const double traveled_distance_m, const int64_t start_timestamp,
  const int64_t end_timestamp, const SkippingInfo & skipping_info,
  const timestamp_stats::TimestampStatsMap & timestamp_stats_map)
{
  std::vector<int> missing_types;
  for (const auto & t : skipping_info.missing_topic_types) {
    missing_types.push_back(static_cast<int>(t));
  }

  nlohmann::json j;
  j["is_skipped"] = (skipping_info.label != SkippingLabel::NotSkipped);
  j["num_frames"] = num_frames;
  j["traveled_distance_m"] = traveled_distance_m;
  j["start_timestamp"] = start_timestamp;
  j["end_timestamp"] = end_timestamp;
  j["skipping_info"] = {
    {"label", static_cast<int>(skipping_info.label)},
    {"details", skipping_info.details},
    {"missing_topic_types", missing_types}};

  nlohmann::json timestamp_stats_json;
  for (const auto & [topic, stats] : timestamp_stats_map.stats_map) {
    nlohmann::json diff_stats_json = {
      {"mean_ms", to_millisecond(stats.diff_mean())},
      {"std_dev_ms", to_millisecond(stats.diff_std_dev())},
      {"min_ms", to_millisecond(stats.diff_min())},
      {"max_ms", to_millisecond(stats.diff_max())}};
    nlohmann::json header_diff_stats_json = {
      {"mean_ms", to_millisecond(stats.header_diff_mean())},
      {"std_dev_ms", to_millisecond(stats.header_diff_std_dev())},
      {"min_ms", to_millisecond(stats.header_diff_min())},
      {"max_ms", to_millisecond(stats.header_diff_max())}};
    nlohmann::json rosbag_diff_stats_json = {
      {"mean_ms", to_millisecond(stats.rosbag_diff_mean())},
      {"std_dev_ms", to_millisecond(stats.rosbag_diff_std_dev())},
      {"min_ms", to_millisecond(stats.rosbag_diff_min())},
      {"max_ms", to_millisecond(stats.rosbag_diff_max())}};
    timestamp_stats_json[topic] = {
      {"monotonic_header", stats.is_monotonic_header()},
      {"monotonic_rosbag", stats.is_monotonic_rosbag()},
      {"diff_stats", diff_stats_json},
      {"header_diff_stats", header_diff_stats_json},
      {"rosbag_diff_stats", rosbag_diff_stats_json}};
  }
  j["timestamp_stats"] = timestamp_stats_json;
  return j;
}

// ---------------------------------------------------------------------------
// File-writing wrappers
// ---------------------------------------------------------------------------

void save_frame_json(
  const std::string & output_path, const std::string & rosbag_dir_name, const std::string & token,
  const nav_msgs::msg::Odometry & kinematic_state, const int64_t timestamp,
  const SkippingInfo & skipping_info, const std::vector<std::string> & neighbor_ids)
{
  namespace fs = std::filesystem;

  fs::create_directories(output_path);

  const nlohmann::json j =
    build_frame_json(kinematic_state, timestamp, skipping_info, neighbor_ids);

  const std::string json_filename = output_path + "/" + rosbag_dir_name + "_" + token + ".json";
  std::ofstream json_file(json_filename);
  if (json_file.is_open()) {
    json_file << std::setw(2) << j << std::endl;
    json_file.close();
  } else {
    std::cerr << "Failed to open JSON file for writing: " << json_filename << std::endl;
  }
}

void save_route_json(
  const std::string & output_path, const std::string & rosbag_dir_name,
  const std::string & identifier, const int64_t num_frames, const double traveled_distance_m,
  const int64_t start_timestamp, const int64_t end_timestamp, const SkippingInfo & skipping_info,
  const timestamp_stats::TimestampStatsMap & timestamp_stats_map)
{
  namespace fs = std::filesystem;

  const std::string routes_dir = output_path + "/routes";
  fs::create_directories(routes_dir);

  const nlohmann::json j = build_route_json(
    num_frames, traveled_distance_m, start_timestamp, end_timestamp, skipping_info,
    timestamp_stats_map);

  const std::string json_filename = routes_dir + "/" + rosbag_dir_name + "_" + identifier + ".json";
  std::ofstream json_file(json_filename);
  if (!json_file.is_open()) {
    std::cerr << "Failed to open route JSON file for writing: " << json_filename << std::endl;
    return;
  }

  json_file << std::setw(2) << j << std::endl;
  if (!json_file) {
    std::cerr << "Failed to write route JSON file: " << json_filename << std::endl;
    return;
  }

  json_file.close();
  if (!json_file) {
    std::cerr << "Failed to close route JSON file: " << json_filename << std::endl;
  }
}
