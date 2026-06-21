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

#include "io/frame_writer.hpp"

#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// build_frame_json
// ---------------------------------------------------------------------------

TEST(BuildFrameJsonTest, AcceptedFrameFields)
{
  nav_msgs::msg::Odometry odom;
  odom.pose.pose.position.x = 1.1;
  odom.pose.pose.position.y = 2.2;
  odom.pose.pose.position.z = 3.3;
  odom.pose.pose.orientation.x = 0.0;
  odom.pose.pose.orientation.y = 0.0;
  odom.pose.pose.orientation.z = 0.0;
  odom.pose.pose.orientation.w = 1.0;

  const SkippingInfo info = SkippingInfo::accepted();
  const int64_t ts = 123456789LL;

  const nlohmann::json j = build_frame_json(odom, ts, info);

  EXPECT_FALSE(j["is_skipped"].get<bool>());
  EXPECT_EQ(j["timestamp"].get<int64_t>(), ts);
  EXPECT_DOUBLE_EQ(j["x"].get<double>(), 1.1);
  EXPECT_DOUBLE_EQ(j["y"].get<double>(), 2.2);
  EXPECT_DOUBLE_EQ(j["z"].get<double>(), 3.3);
  EXPECT_EQ(j["skipping_info"]["label"].get<int>(), static_cast<int>(SkippingLabel::NotSkipped));
}

TEST(BuildFrameJsonTest, SkippedFrameIsSkippedTrue)
{
  nav_msgs::msg::Odometry odom;
  const SkippingInfo info = SkippingInfo::stale_data(600'000'000LL);

  const nlohmann::json j = build_frame_json(odom, 0LL, info);

  EXPECT_TRUE(j["is_skipped"].get<bool>());
  EXPECT_EQ(
    j["skipping_info"]["label"].get<int>(), static_cast<int>(SkippingLabel::IncompleteData));
}

// ---------------------------------------------------------------------------
// build_route_json
// ---------------------------------------------------------------------------

TEST(BuildRouteJsonTest, BasicFieldsPresent)
{
  const SkippingInfo info = SkippingInfo::accepted();
  timestamp_stats::TimestampStatsMap stats_map({});  // empty map

  const nlohmann::json j = build_route_json(42, 150.5, 1000000LL, 2000000LL, info, stats_map);

  EXPECT_FALSE(j["is_skipped"].get<bool>());
  EXPECT_EQ(j["num_frames"].get<int64_t>(), 42);
  EXPECT_DOUBLE_EQ(j["traveled_distance_m"].get<double>(), 150.5);
  EXPECT_EQ(j["start_timestamp"].get<int64_t>(), 1000000LL);
  EXPECT_EQ(j["end_timestamp"].get<int64_t>(), 2000000LL);
  EXPECT_EQ(j["skipping_info"]["label"].get<int>(), static_cast<int>(SkippingLabel::NotSkipped));
}

TEST(BuildRouteJsonTest, SkippedRouteFieldsPresent)
{
  const SkippingInfo info = SkippingInfo::insufficient_frames(100, 1700);
  timestamp_stats::TimestampStatsMap stats_map({});

  const nlohmann::json j = build_route_json(100, 10.0, 0LL, 1000LL, info, stats_map);

  EXPECT_TRUE(j["is_skipped"].get<bool>());
  EXPECT_EQ(
    j["skipping_info"]["label"].get<int>(), static_cast<int>(SkippingLabel::InsufficientFrames));
}
