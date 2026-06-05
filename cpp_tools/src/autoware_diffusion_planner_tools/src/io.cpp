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

#include "io.hpp"

#include "nlohmann/json.hpp"

#include <autoware/diffusion_planner/dimensions.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace autoware::diffusion_planner;

// Training data binary format (fixed-size, version 2)
struct TrainingDataBinary
{
  uint32_t version;

  float ego_agent_past[EGO_HISTORY_SHAPE[1] * EGO_HISTORY_SHAPE[2]];
  float ego_current_state[EGO_CURRENT_STATE_SHAPE[1]];
  float ego_agent_future[OUTPUT_T * EGO_HISTORY_SHAPE[2]];
  float neighbor_agents_past[MAX_NUM_NEIGHBORS * INPUT_T_WITH_CURRENT * NEIGHBOR_PAST_DIM];
  float neighbor_agents_future[MAX_NUM_NEIGHBORS * OUTPUT_T * NEIGHBOR_FUTURE_DIM];
  float static_objects[STATIC_OBJECTS_SHAPE[1] * STATIC_OBJECTS_SHAPE[2]];
  float lanes[NUM_SEGMENTS_IN_LANE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM];
  float lanes_speed_limit[NUM_SEGMENTS_IN_LANE];
  int32_t lanes_has_speed_limit[NUM_SEGMENTS_IN_LANE];
  float route_lanes[NUM_SEGMENTS_IN_ROUTE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM];
  float route_lanes_speed_limit[NUM_SEGMENTS_IN_ROUTE];
  int32_t route_lanes_has_speed_limit[NUM_SEGMENTS_IN_ROUTE];
  float polygons[NUM_POLYGONS * POINTS_PER_POLYGON * (2 + POLYGON_TYPE_NUM)];
  float line_strings[NUM_LINE_STRINGS * POINTS_PER_LINE_STRING * (2 + LINE_STRING_TYPE_NUM)];
  float goal_pose[NEIGHBOR_FUTURE_DIM];
  int32_t turn_indicators[INPUT_T_WITH_CURRENT];
  float ego_shape[EGO_SHAPE_SHAPE[1]];

  TrainingDataBinary() : version(2)
  {
    std::fill(std::begin(ego_agent_past), std::end(ego_agent_past), 0.0f);
    std::fill(std::begin(ego_current_state), std::end(ego_current_state), 0.0f);
    std::fill(std::begin(ego_agent_future), std::end(ego_agent_future), 0.0f);
    std::fill(std::begin(neighbor_agents_past), std::end(neighbor_agents_past), 0.0f);
    std::fill(std::begin(neighbor_agents_future), std::end(neighbor_agents_future), 0.0f);
    std::fill(std::begin(static_objects), std::end(static_objects), 0.0f);
    std::fill(std::begin(lanes), std::end(lanes), 0.0f);
    std::fill(std::begin(lanes_speed_limit), std::end(lanes_speed_limit), 0.0f);
    std::fill(std::begin(lanes_has_speed_limit), std::end(lanes_has_speed_limit), 0);
    std::fill(std::begin(route_lanes), std::end(route_lanes), 0.0f);
    std::fill(std::begin(route_lanes_speed_limit), std::end(route_lanes_speed_limit), 0.0f);
    std::fill(std::begin(route_lanes_has_speed_limit), std::end(route_lanes_has_speed_limit), 0);
    std::fill(std::begin(polygons), std::end(polygons), 0.0f);
    std::fill(std::begin(line_strings), std::end(line_strings), 0.0f);
    std::fill(std::begin(goal_pose), std::end(goal_pose), 0.0f);
    std::fill(std::begin(turn_indicators), std::end(turn_indicators), 0);
    std::fill(std::begin(ego_shape), std::end(ego_shape), 0.0f);
  }
};

std::string create_token(const int64_t seq_id, const int64_t frame_id)
{
  std::ostringstream token_stream;
  token_stream << std::setfill('0') << std::setw(8) << seq_id << "_" << std::setw(8) << frame_id;
  return token_stream.str();
}

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
  const std::vector<int32_t> & turn_indicators, const std::vector<float> & ego_shape)
{
  namespace fs = std::filesystem;

  fs::create_directories(output_path);

  TrainingDataBinary data;
  std::copy(ego_past.begin(), ego_past.end(), data.ego_agent_past);
  std::copy(ego_current.begin(), ego_current.end(), data.ego_current_state);
  std::copy(ego_future.begin(), ego_future.end(), data.ego_agent_future);
  std::copy(neighbor_past.begin(), neighbor_past.end(), data.neighbor_agents_past);
  std::copy(neighbor_future.begin(), neighbor_future.end(), data.neighbor_agents_future);
  std::copy(static_objects.begin(), static_objects.end(), data.static_objects);
  std::copy(lanes.begin(), lanes.end(), data.lanes);
  std::copy(lanes_speed_limit.begin(), lanes_speed_limit.end(), data.lanes_speed_limit);
  std::copy(route_lanes.begin(), route_lanes.end(), data.route_lanes);
  std::copy(
    route_lanes_speed_limit.begin(), route_lanes_speed_limit.end(), data.route_lanes_speed_limit);
  std::copy(polygons.begin(), polygons.end(), data.polygons);
  std::copy(line_strings.begin(), line_strings.end(), data.line_strings);
  std::copy(goal_pose.begin(), goal_pose.end(), data.goal_pose);
  for (size_t i = 0; i < lanes_has_speed_limit.size(); ++i) {
    data.lanes_has_speed_limit[i] = static_cast<int32_t>(lanes_has_speed_limit[i]);
  }
  for (size_t i = 0; i < route_lanes_has_speed_limit.size(); ++i) {
    data.route_lanes_has_speed_limit[i] = static_cast<int32_t>(route_lanes_has_speed_limit[i]);
  }
  std::copy(turn_indicators.begin(), turn_indicators.end(), data.turn_indicators);
  std::copy(ego_shape.begin(), ego_shape.end(), data.ego_shape);

  const std::string binary_filename = output_path + "/" + rosbag_dir_name + "_" + token + ".bin";
  std::ofstream file(binary_filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << binary_filename << std::endl;
    return;
  }
  file.write(reinterpret_cast<const char *>(&data), sizeof(TrainingDataBinary));
  if (file.fail()) {
    std::cerr << "Failed to write data to file: " << binary_filename << std::endl;
  }
  file.close();
}

void save_frame_json(
  const std::string & output_path, const std::string & rosbag_dir_name, const std::string & token,
  const nav_msgs::msg::Odometry & kinematic_state, const int64_t timestamp,
  const SkippingInfo & skipping_info)
{
  namespace fs = std::filesystem;

  fs::create_directories(output_path);

  std::vector<int> incomplete_types;
  for (const auto & t : skipping_info.incomplete_data_types) {
    incomplete_types.push_back(static_cast<int>(t));
  }

  nlohmann::json j;
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
