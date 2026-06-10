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

#include "processing/frame_skip_decision.hpp"

#include "processing/frame_filters.hpp"

#include <autoware/diffusion_planner/dimensions.hpp>

#include <cstdint>
#include <vector>

namespace frame_processor
{

namespace
{
constexpr int64_t kStaleThresholdNs = 500'000'000LL;  // 500 ms
// Skip frames where GT future has not advanced for >=3 s (ego stuck beyond red lights).
// At 10 Hz with step=1 this is 30 ticks; the caller passes no_future_progress_count * step.
constexpr int64_t kStuckThresholdTicks = 30;
}  // namespace

SkippingInfo decide_frame_skip(
  const FrameSkipInputs & inputs,
  const std::vector<float> & ego_future,
  const std::vector<float> & ego_shape,
  const std::vector<float> & static_objects,
  const std::vector<float> & neighbor_future,
  const std::vector<float> & neighbor_past,
  const std::vector<float> & line_strings,
  const std::vector<float> & lanes,
  float static_object_margin,
  float neighbor_margin,
  float road_border_margin,
  int64_t collision_time_stride,
  float offlane_max_score,
  int64_t offlane_time_stride)
{
  using autoware::diffusion_planner::INPUT_T;

  if (inputs.max_msg_age_ns > kStaleThresholdNs) {
    return SkippingInfo::stale_data(inputs.max_msg_age_ns);
  }

  if (inputs.cov_xx > 1e-1 || inputs.cov_yy > 1e-1) {
    return SkippingInfo::invalid_covariance(inputs.cov_xx, inputs.cov_yy);
  }

  if (inputs.is_stop && inputs.is_red_or_yellow && inputs.is_future_forward) {
    return SkippingInfo::red_or_yellow_light();
  }

  if (inputs.stopping_count > (INPUT_T + 5) && inputs.is_red_or_yellow) {
    return SkippingInfo::stopped_at_traffic_light();
  }

  if (inputs.no_future_progress_x_step > kStuckThresholdTicks) {
    const double sustained_s = static_cast<double>(inputs.no_future_progress_x_step) / 10.0;
    return SkippingInfo::no_future_progress(sustained_s);
  }

  if (const frame_filters::CollisionResult collision = frame_filters::check_collision(
        ego_future, ego_shape, static_objects, neighbor_future, neighbor_past, line_strings,
        static_object_margin, neighbor_margin, road_border_margin, collision_time_stride);
      collision.collided()) {
    return SkippingInfo::collision(collision.reasons);
  }

  if (const frame_filters::OffLaneResult offlane =
        frame_filters::compute_offlane_score(ego_future, lanes, offlane_time_stride);
      frame_filters::is_off_lane(offlane, offlane_max_score)) {
    return SkippingInfo::off_lane(offlane.mean_distance, offlane.max_distance);
  }

  return SkippingInfo::accepted();
}

}  // namespace frame_processor
