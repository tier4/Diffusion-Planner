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

#ifndef PROCESSING__EGO_SEQUENCE_HPP_
#define PROCESSING__EGO_SEQUENCE_HPP_

#include "types/frame_data.hpp"

#include <Eigen/Core>
#include <rclcpp/time.hpp>

#include <cstdint>
#include <optional>
#include <vector>

// Build a [num_timesteps, 4] (x, y, cos, sin) ego trajectory in the frame given by
// map2bl_matrix, whose last timestep lands on reference_time.
//
// allow_hold_past_end decides what happens when the requested window reaches past the last frame
// of data_list: false gives up (nullopt), true holds the final pose for the remaining timesteps.
// Holding is only faithful for a sequence that ends standing still, so callers gate it on
// stopped_tail::ends_stopped; pass false for a window that cannot overrun (the past window) to
// keep that intent visible at the call site.
std::optional<std::vector<float>> create_ego_sequence(
  const std::vector<FrameData> & data_list, const int64_t start_idx, const size_t num_timesteps,
  const Eigen::Matrix4d & map2bl_matrix, const rclcpp::Time & reference_time,
  const bool use_interpolation, const bool allow_hold_past_end);

#endif  // PROCESSING__EGO_SEQUENCE_HPP_
