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

#include "processing/ego_sequence.hpp"

#include <autoware/diffusion_planner/preprocessing/preprocessing_utils.hpp>

#include <algorithm>
#include <deque>

std::optional<std::vector<float>> create_ego_sequence(
  const std::vector<FrameData> & data_list, const int64_t start_idx, const size_t num_timesteps,
  const Eigen::Matrix4d & map2bl_matrix, const rclcpp::Time & reference_time,
  const bool use_interpolation)
{
  std::deque<nav_msgs::msg::Odometry> odom_deque;

  if (use_interpolation) {
    // Collect odom messages from start_idx until timestamp >= reference_time
    for (size_t j = static_cast<size_t>(std::max(int64_t(0), start_idx)); j < data_list.size();
         ++j) {
      odom_deque.push_back(data_list[j].kinematic_state);
      if (rclcpp::Time(data_list[j].kinematic_state.header.stamp) >= reference_time) {
        break;
      }
    }

    // Error: data doesn't cover the reference_time
    if (odom_deque.empty() || rclcpp::Time(odom_deque.back().header.stamp) < reference_time) {
      return std::nullopt;
    }

    return autoware::diffusion_planner::preprocess::create_ego_agent_past(
      odom_deque, num_timesteps, map2bl_matrix, reference_time);
  } else {
    // Without interpolation: collect exactly num_timesteps frames by index
    for (size_t j = 0; j < num_timesteps; ++j) {
      const int64_t index =
        std::min(start_idx + static_cast<int64_t>(j), static_cast<int64_t>(data_list.size()) - 1);
      if (index < 0) {
        return std::nullopt;
      }
      odom_deque.push_back(data_list[index].kinematic_state);
    }

    if (odom_deque.empty()) {
      return std::nullopt;
    }

    return autoware::diffusion_planner::preprocess::create_ego_agent_past(
      odom_deque, num_timesteps, map2bl_matrix);
  }
}
