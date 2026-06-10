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

#include <autoware/diffusion_planner/dimensions.hpp>

#include <gtest/gtest.h>

#include <vector>

using namespace frame_processor;
using autoware::diffusion_planner::INPUT_T;
using autoware::diffusion_planner::MAX_NUM_NEIGHBORS;
using autoware::diffusion_planner::OUTPUT_T;
using autoware::diffusion_planner::POSE_DIM;
using autoware::diffusion_planner::STATIC_OBJECTS_SHAPE;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

FrameSkipInputs make_clear_inputs()
{
  FrameSkipInputs in{};
  in.max_msg_age_ns = 0;
  in.cov_xx = 0.0;
  in.cov_yy = 0.0;
  in.is_stop = false;
  in.is_red_or_yellow = false;
  in.is_future_forward = true;
  in.stopping_count = 0;
  in.no_future_progress_x_step = 0;
  return in;
}

// Vectors sized for a valid call with no objects/lanes (all zeros → no collision/offlane).
struct ZeroVectors
{
  std::vector<float> ego_future;
  std::vector<float> ego_shape;
  std::vector<float> static_objects;
  std::vector<float> neighbor_future;
  std::vector<float> neighbor_past;
  std::vector<float> line_strings;
  std::vector<float> lanes;

  ZeroVectors()
  {
    using namespace autoware::diffusion_planner;
    ego_future.assign(OUTPUT_T * POSE_DIM, 0.0f);
    ego_shape = {2.75f, 4.34f, 1.70f};
    static_objects.assign(STATIC_OBJECTS_SHAPE[1] * STATIC_OBJECTS_SHAPE[2], 0.0f);
    const int64_t past = INPUT_T + 1;
    const int64_t np_dim = 11;
    neighbor_past.assign(MAX_NUM_NEIGHBORS * past * np_dim, 0.0f);
    neighbor_future.assign(MAX_NUM_NEIGHBORS * OUTPUT_T * POSE_DIM, 0.0f);
    const int64_t ls_dim = 2 + LINE_STRING_TYPE_NUM;
    line_strings.assign(NUM_LINE_STRINGS * POINTS_PER_LINE_STRING * ls_dim, 0.0f);
    lanes.assign(NUM_SEGMENTS_IN_LANE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM, 0.0f);
  }
};

SkippingInfo call_decide(const FrameSkipInputs & inputs, const ZeroVectors & vecs)
{
  return decide_frame_skip(
    inputs, vecs.ego_future, vecs.ego_shape, vecs.static_objects, vecs.neighbor_future,
    vecs.neighbor_past, vecs.line_strings, vecs.lanes,
    /*static_object_margin=*/0.0f,
    /*neighbor_margin=*/0.0f,
    /*road_border_margin=*/0.0f,
    /*collision_time_stride=*/5,
    /*offlane_max_score=*/6.0f,
    /*offlane_time_stride=*/1);
}

}  // namespace

// ---------------------------------------------------------------------------
// decide_frame_skip tests
// ---------------------------------------------------------------------------

TEST(DecideFrameSkipTest, AllClearReturnsAccepted)
{
  const ZeroVectors vecs;
  // With no valid centerline points, empty lanes → is_off_lane = true.
  // Provide a non-zero lane point so the score can be evaluated.
  ZeroVectors vecs2 = vecs;
  // Place a centerline point at (1,0) so has_centerline = true, score = distance from (0,0)→1 m.
  vecs2.lanes[0] = 1.0f;

  const FrameSkipInputs inputs = make_clear_inputs();
  const SkippingInfo info = call_decide(inputs, vecs2);
  EXPECT_EQ(info.label, SkippingLabel::NotSkipped);
}

TEST(DecideFrameSkipTest, StaleDataTakesHighestPriority)
{
  ZeroVectors vecs;
  vecs.lanes[0] = 1.0f;

  FrameSkipInputs inputs = make_clear_inputs();
  inputs.max_msg_age_ns = 600'000'000LL;  // > 500 ms threshold

  const SkippingInfo info = call_decide(inputs, vecs);
  EXPECT_EQ(info.label, SkippingLabel::IncompleteData);
  EXPECT_NE(info.details.find("Stale"), std::string::npos);
}

TEST(DecideFrameSkipTest, InvalidCovarianceBeforeOtherChecks)
{
  ZeroVectors vecs;
  vecs.lanes[0] = 1.0f;

  FrameSkipInputs inputs = make_clear_inputs();
  inputs.max_msg_age_ns = 0;  // not stale
  inputs.cov_xx = 0.5;        // > 1e-1 threshold

  const SkippingInfo info = call_decide(inputs, vecs);
  EXPECT_EQ(info.label, SkippingLabel::IncompleteData);
  EXPECT_NE(info.details.find("covariance"), std::string::npos);
}

TEST(DecideFrameSkipTest, RedOrYellowLightSkip)
{
  ZeroVectors vecs;
  vecs.lanes[0] = 1.0f;

  FrameSkipInputs inputs = make_clear_inputs();
  inputs.is_stop = true;
  inputs.is_red_or_yellow = true;
  inputs.is_future_forward = true;

  const SkippingInfo info = call_decide(inputs, vecs);
  EXPECT_EQ(info.label, SkippingLabel::RedOrYellowLight);
}

TEST(DecideFrameSkipTest, StoppedAtTrafficLightSkip)
{
  ZeroVectors vecs;
  vecs.lanes[0] = 1.0f;

  FrameSkipInputs inputs = make_clear_inputs();
  // is_stop && is_future_forward but NOT is_future_forward (so first check doesn't fire)
  inputs.is_stop = true;
  inputs.is_future_forward = false;
  inputs.is_red_or_yellow = true;
  inputs.stopping_count = INPUT_T + 10;  // > INPUT_T + 5

  const SkippingInfo info = call_decide(inputs, vecs);
  EXPECT_EQ(info.label, SkippingLabel::StoppedAtTrafficLight);
}

TEST(DecideFrameSkipTest, NoFutureProgressSkip)
{
  ZeroVectors vecs;
  vecs.lanes[0] = 1.0f;

  FrameSkipInputs inputs = make_clear_inputs();
  inputs.no_future_progress_x_step = 31;  // > kStuckThresholdTicks (30)

  const SkippingInfo info = call_decide(inputs, vecs);
  EXPECT_EQ(info.label, SkippingLabel::NoFutureProgress);
}

TEST(DecideFrameSkipTest, StaleDataWinsOverRedLight)
{
  // When both stale and red light apply, stale takes priority.
  ZeroVectors vecs;
  vecs.lanes[0] = 1.0f;

  FrameSkipInputs inputs = make_clear_inputs();
  inputs.max_msg_age_ns = 600'000'000LL;
  inputs.is_stop = true;
  inputs.is_red_or_yellow = true;
  inputs.is_future_forward = true;

  const SkippingInfo info = call_decide(inputs, vecs);
  EXPECT_EQ(info.label, SkippingLabel::IncompleteData);
}

TEST(DecideFrameSkipTest, OffLaneSkipWhenNoCenterline)
{
  // All-zero lanes → no valid centerline → is_off_lane = true.
  const ZeroVectors vecs;  // lanes all zero
  const FrameSkipInputs inputs = make_clear_inputs();

  const SkippingInfo info = call_decide(inputs, vecs);
  EXPECT_EQ(info.label, SkippingLabel::OffLane);
}
