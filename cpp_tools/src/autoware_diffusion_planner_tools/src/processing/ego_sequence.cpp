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
  const bool use_interpolation, const bool allow_hold_past_end)
{
  const int64_t last_idx = static_cast<int64_t>(data_list.size()) - 1;
  if (last_idx < 0 || start_idx < 0) {
    return std::nullopt;
  }

  std::deque<nav_msgs::msg::Odometry> odom_deque;

  if (use_interpolation) {
    // Collect odom messages from start_idx until timestamp >= reference_time. A window that
    // starts one past the last frame still gets the final pose, which is all a fully held
    // window needs.
    for (int64_t j = std::min(start_idx, last_idx); j <= last_idx; ++j) {
      odom_deque.push_back(data_list[j].kinematic_state);
      if (rclcpp::Time(data_list[j].kinematic_state.header.stamp) >= reference_time) {
        break;
      }
    }

    // Data doesn't cover reference_time. create_ego_agent_past holds the final pose for every
    // target time past its last message, which only the callers that verified the sequence ends
    // standing still may accept.
    if (rclcpp::Time(odom_deque.back().header.stamp) < reference_time && !allow_hold_past_end) {
      return std::nullopt;
    }

    return autoware::diffusion_planner::preprocess::create_ego_agent_past(
      odom_deque, num_timesteps, map2bl_matrix, reference_time);
  } else {
    // Without interpolation: collect exactly num_timesteps frames by index, holding the last
    // frame once the window overruns it — the same hold as above, gated the same way.
    if (start_idx + static_cast<int64_t>(num_timesteps) - 1 > last_idx && !allow_hold_past_end) {
      return std::nullopt;
    }

    for (size_t j = 0; j < num_timesteps; ++j) {
      const int64_t index = std::min(start_idx + static_cast<int64_t>(j), last_idx);
      odom_deque.push_back(data_list[index].kinematic_state);
    }

    if (odom_deque.empty()) {
      return std::nullopt;
    }

    return autoware::diffusion_planner::preprocess::create_ego_agent_past(
      odom_deque, num_timesteps, map2bl_matrix);
  }
}
