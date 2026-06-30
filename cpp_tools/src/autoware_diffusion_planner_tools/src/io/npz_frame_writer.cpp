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

#include "io/npz_frame_writer.hpp"

#include "utils/cnpy.hpp"

#include <autoware/diffusion_planner/dimensions.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{

std::vector<float> cos_sin_to_heading(const std::vector<float> & data, size_t num_rows)
{
  const size_t input_cols = 4;
  const size_t output_cols = 3;
  std::vector<float> result(num_rows * output_cols);

  for (size_t i = 0; i < num_rows; ++i) {
    const float x = data[i * input_cols + 0];
    const float y = data[i * input_cols + 1];
    const float cos_val = data[i * input_cols + 2];
    const float sin_val = data[i * input_cols + 3];
    const float heading = std::atan2(sin_val, cos_val);

    result[i * output_cols + 0] = x;
    result[i * output_cols + 1] = y;
    result[i * output_cols + 2] = heading;
  }
  return result;
}

std::vector<float> cos_sin_to_heading_3d(const std::vector<float> & data, size_t dim0, size_t dim1)
{
  const size_t input_cols = 4;
  const size_t output_cols = 3;
  std::vector<float> result(dim0 * dim1 * output_cols);

  for (size_t i = 0; i < dim0; ++i) {
    for (size_t j = 0; j < dim1; ++j) {
      const size_t base_in = (i * dim1 + j) * input_cols;
      const size_t base_out = (i * dim1 + j) * output_cols;

      const float x = data[base_in + 0];
      const float y = data[base_in + 1];
      const float cos_val = data[base_in + 2];
      const float sin_val = data[base_in + 3];
      const float heading = std::atan2(sin_val, cos_val);

      result[base_out + 0] = x;
      result[base_out + 1] = y;
      result[base_out + 2] = heading;
    }
  }
  return result;
}

}  // namespace

void save_frame_data_npz(
  const std::string & output_path, const std::string & rosbag_dir_name, const std::string & token,
  const std::vector<float> & ego_past, const std::vector<float> & ego_current,
  const std::vector<float> & ego_future, const std::vector<float> & neighbor_past,
  const std::vector<float> & neighbor_future, const std::vector<float> & static_objects,
  const std::vector<float> & lanes, const std::vector<float> & lanes_speed_limit,
  const std::vector<uint8_t> & lanes_has_speed_limit, const std::vector<float> & route_lanes,
  const std::vector<float> & route_lanes_speed_limit,
  const std::vector<uint8_t> & route_lanes_has_speed_limit, const std::vector<float> & polygons,
  const std::vector<float> & line_strings, const std::vector<float> & goal_pose,
  const std::vector<int32_t> & turn_indicators, const std::vector<float> & ego_shape)
{
  namespace fs = std::filesystem;
  using autoware::diffusion_planner::INPUT_T_WITH_CURRENT;
  using autoware::diffusion_planner::LINE_STRING_TYPE_NUM;
  using autoware::diffusion_planner::MAX_NUM_NEIGHBORS;
  using autoware::diffusion_planner::NUM_LINE_STRINGS;
  using autoware::diffusion_planner::NUM_POLYGONS;
  using autoware::diffusion_planner::NUM_SEGMENTS_IN_LANE;
  using autoware::diffusion_planner::NUM_SEGMENTS_IN_ROUTE;
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POINTS_PER_LINE_STRING;
  using autoware::diffusion_planner::POINTS_PER_POLYGON;
  using autoware::diffusion_planner::POINTS_PER_SEGMENT;
  using autoware::diffusion_planner::POLYGON_TYPE_NUM;
  using autoware::diffusion_planner::SEGMENT_POINT_DIM;
  using autoware::diffusion_planner::STATIC_OBJECTS_SHAPE;

  fs::create_directories(output_path);

  const std::string npz_filename = output_path + "/" + rosbag_dir_name + "_" + token + ".npz";

  const uint32_t version = 2;
  cnpy::npz_save_compressed(npz_filename, "version", &version, {1}, "w");

  const std::vector<float> ego_past_heading = cos_sin_to_heading(ego_past, INPUT_T_WITH_CURRENT);
  cnpy::npz_save_compressed(
    npz_filename, "ego_agent_past", ego_past_heading.data(), {INPUT_T_WITH_CURRENT, 3}, "a");

  cnpy::npz_save_compressed(
    npz_filename, "ego_current_state", ego_current.data(), {ego_current.size()}, "a");

  const std::vector<float> ego_future_heading = cos_sin_to_heading(ego_future, OUTPUT_T);
  cnpy::npz_save_compressed(
    npz_filename, "ego_agent_future", ego_future_heading.data(), {OUTPUT_T, 3}, "a");

  constexpr int64_t NEIGHBOR_PAST_DIM = 11;
  cnpy::npz_save_compressed(
    npz_filename, "neighbor_agents_past", neighbor_past.data(),
    {MAX_NUM_NEIGHBORS, INPUT_T_WITH_CURRENT, static_cast<size_t>(NEIGHBOR_PAST_DIM)}, "a");

  const std::vector<float> neighbor_future_heading =
    cos_sin_to_heading_3d(neighbor_future, MAX_NUM_NEIGHBORS, OUTPUT_T);
  cnpy::npz_save_compressed(
    npz_filename, "neighbor_agents_future", neighbor_future_heading.data(),
    {MAX_NUM_NEIGHBORS, OUTPUT_T, 3}, "a");

  cnpy::npz_save_compressed(
    npz_filename, "static_objects", static_objects.data(),
    {STATIC_OBJECTS_SHAPE[1], STATIC_OBJECTS_SHAPE[2]}, "a");

  cnpy::npz_save_compressed(
    npz_filename, "lanes", lanes.data(),
    {NUM_SEGMENTS_IN_LANE, POINTS_PER_SEGMENT, SEGMENT_POINT_DIM}, "a");

  cnpy::npz_save_compressed(
    npz_filename, "lanes_speed_limit", lanes_speed_limit.data(), {NUM_SEGMENTS_IN_LANE, 1}, "a");

  cnpy::npz_save_compressed(
    npz_filename, "lanes_has_speed_limit",
    reinterpret_cast<const bool *>(lanes_has_speed_limit.data()), {NUM_SEGMENTS_IN_LANE, 1}, "a");

  cnpy::npz_save_compressed(
    npz_filename, "route_lanes", route_lanes.data(),
    {NUM_SEGMENTS_IN_ROUTE, POINTS_PER_SEGMENT, SEGMENT_POINT_DIM}, "a");

  cnpy::npz_save_compressed(
    npz_filename, "route_lanes_speed_limit", route_lanes_speed_limit.data(),
    {NUM_SEGMENTS_IN_ROUTE, 1}, "a");

  cnpy::npz_save_compressed(
    npz_filename, "route_lanes_has_speed_limit",
    reinterpret_cast<const bool *>(route_lanes_has_speed_limit.data()), {NUM_SEGMENTS_IN_ROUTE, 1},
    "a");

  cnpy::npz_save_compressed(
    npz_filename, "polygons", polygons.data(),
    {NUM_POLYGONS, POINTS_PER_POLYGON, 2 + POLYGON_TYPE_NUM}, "a");

  cnpy::npz_save_compressed(
    npz_filename, "line_strings", line_strings.data(),
    {NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 2 + LINE_STRING_TYPE_NUM}, "a");

  const std::vector<float> goal_pose_heading = cos_sin_to_heading(goal_pose, 1);
  cnpy::npz_save_compressed(npz_filename, "goal_pose", goal_pose_heading.data(), {3}, "a");

  cnpy::npz_save_compressed(
    npz_filename, "turn_indicators", turn_indicators.data(), {turn_indicators.size()}, "a");

  cnpy::npz_save_compressed(npz_filename, "ego_shape", ego_shape.data(), {ego_shape.size()}, "a");
}
