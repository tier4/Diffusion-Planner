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

#include <autoware/diffusion_planner/preprocessing/lane_segments.hpp>

#include <rclcpp/time.hpp>

#include <Eigen/Core>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

std::optional<std::vector<float>> create_ego_sequence(
  const std::vector<FrameData> & data_list, int64_t start_idx, size_t num_timesteps,
  const Eigen::Matrix4d & map2bl_matrix, const rclcpp::Time & reference_time,
  bool use_interpolation);

std::pair<std::vector<float>, std::vector<float>> process_neighbor_agents_and_future(
  const std::vector<FrameData> & data_list, int64_t current_idx,
  const Eigen::Matrix4d & map2bl_matrix);

void process_sequence(
  SequenceData & seq, int64_t seq_id,
  const autoware::diffusion_planner::preprocess::LaneSegmentContext & lane_segment_context,
  int64_t step, bool use_interpolation, int64_t convert_red, int64_t convert_yellow,
  float ego_wheel_base, const std::vector<float> & ego_shape, const std::string & save_dir,
  const std::string & rosbag_dir_name);
