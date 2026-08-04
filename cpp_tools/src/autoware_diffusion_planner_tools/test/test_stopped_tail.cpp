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
#include "processing/stopped_tail.hpp"

#include <Eigen/Core>
#include <autoware/diffusion_planner/constants.hpp>
#include <autoware/diffusion_planner/dimensions.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace
{

constexpr int64_t kPeriodNs = 100'000'000LL;  // 10 Hz, the rate build_sequences assembles at

// Build a sequence of 10 Hz frames with the given per-frame ego longitudinal speeds. Poses
// advance along +x by speed * period so the frames are self-consistent.
std::vector<FrameData> make_frames(const std::vector<double> & speeds_mps)
{
  constexpr double period_s = static_cast<double>(kPeriodNs) * 1e-9;
  std::vector<FrameData> data_list;
  double x = 0.0;
  for (size_t i = 0; i < speeds_mps.size(); ++i) {
    FrameData frame;
    frame.timestamp = static_cast<int64_t>(i) * kPeriodNs;
    frame.kinematic_state.header.stamp.sec = static_cast<int32_t>(i / 10);
    frame.kinematic_state.header.stamp.nanosec = static_cast<uint32_t>((i % 10) * kPeriodNs);
    frame.kinematic_state.pose.pose.orientation.w = 1.0;
    frame.kinematic_state.pose.pose.position.x = x;
    frame.kinematic_state.twist.twist.linear.x = speeds_mps[i];
    frame.max_msg_age_ns = 0;
    data_list.push_back(frame);
    x += speeds_mps[i] * period_s;
  }
  return data_list;
}

}  // namespace

TEST(EndsStoppedTest, EmptySequenceDoesNotEndStopped)
{
  EXPECT_FALSE(stopped_tail::ends_stopped({}));
}

TEST(EndsStoppedTest, SequenceEndingWhileMovingDoesNotEndStopped)
{
  EXPECT_FALSE(stopped_tail::ends_stopped(make_frames({0.0, 0.0, 3.0})));
}

TEST(EndsStoppedTest, SequenceEndingAtRestEndsStopped)
{
  EXPECT_TRUE(stopped_tail::ends_stopped(make_frames({5.0, 2.0, 0.0})));
}

TEST(EndsStoppedTest, ThresholdIsExclusive)
{
  EXPECT_FALSE(stopped_tail::ends_stopped(make_frames({stopped_tail::kEndStoppedSpeedMps})));
  EXPECT_TRUE(stopped_tail::ends_stopped(make_frames({stopped_tail::kEndStoppedSpeedMps - 1e-3})));
}

TEST(EndsStoppedTest, ReversingEgoDoesNotEndStopped)
{
  // Negative speed is motion, not a stop (the absolute speed is compared).
  EXPECT_FALSE(stopped_tail::ends_stopped(make_frames({0.0, -2.0})));
}

// ---------------------------------------------------------------------------
// create_ego_sequence hold behaviour, which ends_stopped gates
// ---------------------------------------------------------------------------

namespace
{

using autoware::diffusion_planner::OUTPUT_T;
using autoware::diffusion_planner::POSE_DIM;

// The GT future of frame `index`, as process_sequence requests it.
std::optional<std::vector<float>> future_at(
  const std::vector<FrameData> & data_list, const int64_t index, const bool use_interpolation,
  const bool allow_hold_past_end)
{
  const rclcpp::Time current_time(
    data_list[static_cast<size_t>(index)].kinematic_state.header.stamp);
  const rclcpp::Time reference_time =
    current_time + rclcpp::Duration::from_seconds(
                     OUTPUT_T * autoware::diffusion_planner::constants::PREDICTION_TIME_STEP_S);
  return create_ego_sequence(
    data_list, index + 1, OUTPUT_T, Eigen::Matrix4d::Identity(), reference_time, use_interpolation,
    allow_hold_past_end);
}

}  // namespace

TEST(EgoFutureHoldTest, GivesUpPastTheEndWhenHoldIsNotAllowed)
{
  // 10 stopped frames is far less than the OUTPUT_T the future needs.
  const std::vector<FrameData> data_list = make_frames(std::vector<double>(10, 0.0));
  EXPECT_FALSE(future_at(data_list, 0, true, false).has_value());
  EXPECT_FALSE(future_at(data_list, 0, false, false).has_value());
}

TEST(EgoFutureHoldTest, HoldsTheFinalPoseWhenAllowed)
{
  const std::vector<FrameData> data_list = make_frames(std::vector<double>(10, 0.0));
  const double last_x = data_list.back().kinematic_state.pose.pose.position.x;

  for (const bool use_interpolation : {true, false}) {
    const auto future = future_at(data_list, 0, use_interpolation, true);
    ASSERT_TRUE(future.has_value()) << "use_interpolation=" << use_interpolation;
    ASSERT_EQ(future->size(), static_cast<size_t>(OUTPUT_T * POSE_DIM));
    // Every timestep past the recording sits on the final pose (frame 0 is the origin here).
    for (int64_t t = 0; t < OUTPUT_T; ++t) {
      EXPECT_NEAR((*future)[t * POSE_DIM + 0], last_x, 1e-3)
        << "t=" << t << " use_interpolation=" << use_interpolation;
      EXPECT_NEAR((*future)[t * POSE_DIM + 1], 0.0, 1e-3);
    }
  }
}

TEST(EgoFutureHoldTest, LastFrameOfTheSequenceStillGetsAFuture)
{
  // start_idx is one past the last frame here: the whole future is held.
  const std::vector<FrameData> data_list = make_frames(std::vector<double>(10, 0.0));
  const int64_t last = static_cast<int64_t>(data_list.size()) - 1;
  EXPECT_TRUE(future_at(data_list, last, true, true).has_value());
  EXPECT_TRUE(future_at(data_list, last, false, true).has_value());
}

TEST(EgoFutureHoldTest, HoldIsNotUsedWhileTheRecordingStillCoversTheHorizon)
{
  // Moving ego with more than OUTPUT_T frames left: the future comes entirely from the data, so
  // allowing the hold changes nothing.
  const std::vector<FrameData> data_list = make_frames(std::vector<double>(OUTPUT_T + 5, 2.0));
  const auto without_hold = future_at(data_list, 0, true, false);
  const auto with_hold = future_at(data_list, 0, true, true);
  ASSERT_TRUE(without_hold.has_value());
  ASSERT_TRUE(with_hold.has_value());
  EXPECT_EQ(*without_hold, *with_hold);
  // The last future timestep advanced by 2 m/s * 8 s from the current pose.
  EXPECT_NEAR((*with_hold)[(OUTPUT_T - 1) * POSE_DIM + 0], 16.0, 1e-2);
}
