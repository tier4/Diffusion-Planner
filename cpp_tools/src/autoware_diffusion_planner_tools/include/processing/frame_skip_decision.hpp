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

#ifndef PROCESSING__FRAME_SKIP_DECISION_HPP_
#define PROCESSING__FRAME_SKIP_DECISION_HPP_

#include "types/skipping_info.hpp"

#include <cstdint>
#include <vector>

namespace frame_processor
{

// Inputs derived from per-frame state that drive the skip decision.
// All values are plain scalars or pre-computed booleans so the function
// has no side effects and is straightforwardly unit-testable.
struct FrameSkipInputs
{
  int64_t max_msg_age_ns;             // max staleness across required topics at this tick
  double cov_xx;                      // kinematic_state.pose.covariance[0]
  double cov_yy;                      // kinematic_state.pose.covariance[7]
  bool is_stop;                       // linear.x < 0.1
  bool is_red_or_yellow;              // next route segment has red/yellow light
  bool is_future_forward;             // GT future mileage > 1.0 m
  int64_t stopping_count;             // consecutive ticks ego has been stopped
  int64_t no_future_progress_x_step;  // no_future_progress_count * step (scaled ticks)
};

// Filter thresholds forwarded from ConverterOptions.
// Grouping prevents argument-order mistakes at the call site.
struct FrameFilterParams
{
  float static_object_margin;
  float neighbor_margin;
  float road_border_margin;
  int64_t collision_time_stride;
  float offlane_max_score;
  int64_t offlane_time_stride;
};

// Pure skip-reason computation — no I/O, no ROS time, no file system.
// Returns the first matching SkippingInfo in priority order, or accepted().
SkippingInfo decide_frame_skip(
  const FrameSkipInputs & inputs, const std::vector<float> & ego_future,
  const std::vector<float> & ego_shape, const std::vector<float> & static_objects,
  const std::vector<float> & neighbor_future, const std::vector<float> & neighbor_past,
  const std::vector<float> & line_strings, const std::vector<float> & lanes,
  const FrameFilterParams & filter_params);

}  // namespace frame_processor

#endif  // PROCESSING__FRAME_SKIP_DECISION_HPP_
