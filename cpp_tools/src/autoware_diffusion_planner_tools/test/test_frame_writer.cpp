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

#include <autoware/diffusion_planner/dimensions.hpp>

#include <gtest/gtest.h>

#include <vector>

// ---------------------------------------------------------------------------
// Helpers to build correctly-sized zero vectors
// ---------------------------------------------------------------------------

namespace
{

using namespace autoware::diffusion_planner;

inline int64_t ego_past_size()
{
  return EGO_HISTORY_SHAPE[1] * EGO_HISTORY_SHAPE[2];
}
inline int64_t ego_current_size()
{
  return EGO_CURRENT_STATE_SHAPE[1];
}
inline int64_t ego_future_size()
{
  return OUTPUT_T * EGO_HISTORY_SHAPE[2];
}
inline int64_t neighbor_past_size()
{
  return MAX_NUM_NEIGHBORS * INPUT_T_WITH_CURRENT * NEIGHBOR_SHAPE[3];
}
inline int64_t neighbor_future_size()
{
  return MAX_NUM_NEIGHBORS * OUTPUT_T * 4;  // NEIGHBOR_FUTURE_DIM = 4
}
inline int64_t static_objects_size()
{
  return STATIC_OBJECTS_SHAPE[1] * STATIC_OBJECTS_SHAPE[2];
}
inline int64_t lanes_size()
{
  return NUM_SEGMENTS_IN_LANE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM;
}
inline int64_t route_lanes_size()
{
  return NUM_SEGMENTS_IN_ROUTE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM;
}
inline int64_t polygons_size()
{
  return NUM_POLYGONS * POINTS_PER_POLYGON * (2 + POLYGON_TYPE_NUM);
}
inline int64_t line_strings_size()
{
  return NUM_LINE_STRINGS * POINTS_PER_LINE_STRING * (2 + LINE_STRING_TYPE_NUM);
}

}  // namespace

// ---------------------------------------------------------------------------
// build_training_data
// ---------------------------------------------------------------------------

TEST(BuildTrainingDataTest, EgoPastIsCopiedCorrectly)
{
  const int64_t n = ego_past_size();
  std::vector<float> ego_past(n, 1.5f);
  std::vector<float> ego_current(ego_current_size(), 0.0f);
  std::vector<float> ego_future(ego_future_size(), 0.0f);
  std::vector<float> neighbor_past(neighbor_past_size(), 0.0f);
  std::vector<float> neighbor_future(neighbor_future_size(), 0.0f);
  std::vector<float> static_objects(static_objects_size(), 0.0f);
  std::vector<float> lanes(lanes_size(), 0.0f);
  std::vector<float> lanes_speed_limit(NUM_SEGMENTS_IN_LANE, 0.0f);
  std::vector<bool> lanes_has_speed_limit(NUM_SEGMENTS_IN_LANE, false);
  std::vector<float> route_lanes(route_lanes_size(), 0.0f);
  std::vector<float> route_lanes_speed_limit(NUM_SEGMENTS_IN_ROUTE, 0.0f);
  std::vector<bool> route_lanes_has_speed_limit(NUM_SEGMENTS_IN_ROUTE, false);
  std::vector<float> polygons(polygons_size(), 0.0f);
  std::vector<float> line_strings(line_strings_size(), 0.0f);
  std::vector<float> goal_pose(4, 0.0f);
  std::vector<int32_t> turn_indicators(INPUT_T_WITH_CURRENT, 0);
  std::vector<float> ego_shape = {2.75f, 4.34f, 1.70f};

  const TrainingDataBinary data = build_training_data(
    ego_past, ego_current, ego_future, neighbor_past, neighbor_future, static_objects, lanes,
    lanes_speed_limit, lanes_has_speed_limit, route_lanes, route_lanes_speed_limit,
    route_lanes_has_speed_limit, polygons, line_strings, goal_pose, turn_indicators, ego_shape);

  // All ego_past elements must be 1.5
  for (int64_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(data.ego_agent_past[i], 1.5f) << "mismatch at index " << i;
  }
}

TEST(BuildTrainingDataTest, EgoShapeIsCopiedCorrectly)
{
  std::vector<float> ego_past(ego_past_size(), 0.0f);
  std::vector<float> ego_current(ego_current_size(), 0.0f);
  std::vector<float> ego_future(ego_future_size(), 0.0f);
  std::vector<float> neighbor_past(neighbor_past_size(), 0.0f);
  std::vector<float> neighbor_future(neighbor_future_size(), 0.0f);
  std::vector<float> static_objects(static_objects_size(), 0.0f);
  std::vector<float> lanes(lanes_size(), 0.0f);
  std::vector<float> lanes_speed_limit(NUM_SEGMENTS_IN_LANE, 0.0f);
  std::vector<bool> lanes_has_speed_limit(NUM_SEGMENTS_IN_LANE, false);
  std::vector<float> route_lanes(route_lanes_size(), 0.0f);
  std::vector<float> route_lanes_speed_limit(NUM_SEGMENTS_IN_ROUTE, 0.0f);
  std::vector<bool> route_lanes_has_speed_limit(NUM_SEGMENTS_IN_ROUTE, false);
  std::vector<float> polygons(polygons_size(), 0.0f);
  std::vector<float> line_strings(line_strings_size(), 0.0f);
  std::vector<float> goal_pose = {3.0f, 4.0f, 1.0f, 0.0f};
  std::vector<int32_t> turn_indicators(INPUT_T_WITH_CURRENT, 0);
  const std::vector<float> ego_shape = {2.75f, 4.34f, 1.70f};

  const TrainingDataBinary data = build_training_data(
    ego_past, ego_current, ego_future, neighbor_past, neighbor_future, static_objects, lanes,
    lanes_speed_limit, lanes_has_speed_limit, route_lanes, route_lanes_speed_limit,
    route_lanes_has_speed_limit, polygons, line_strings, goal_pose, turn_indicators, ego_shape);

  EXPECT_FLOAT_EQ(data.ego_shape[0], 2.75f);
  EXPECT_FLOAT_EQ(data.ego_shape[1], 4.34f);
  EXPECT_FLOAT_EQ(data.ego_shape[2], 1.70f);
}

TEST(BuildTrainingDataTest, GoalPoseIsCopiedCorrectly)
{
  std::vector<float> ego_past(ego_past_size(), 0.0f);
  std::vector<float> ego_current(ego_current_size(), 0.0f);
  std::vector<float> ego_future(ego_future_size(), 0.0f);
  std::vector<float> neighbor_past(neighbor_past_size(), 0.0f);
  std::vector<float> neighbor_future(neighbor_future_size(), 0.0f);
  std::vector<float> static_objects(static_objects_size(), 0.0f);
  std::vector<float> lanes(lanes_size(), 0.0f);
  std::vector<float> lanes_speed_limit(NUM_SEGMENTS_IN_LANE, 0.0f);
  std::vector<bool> lanes_has_speed_limit(NUM_SEGMENTS_IN_LANE, false);
  std::vector<float> route_lanes(route_lanes_size(), 0.0f);
  std::vector<float> route_lanes_speed_limit(NUM_SEGMENTS_IN_ROUTE, 0.0f);
  std::vector<bool> route_lanes_has_speed_limit(NUM_SEGMENTS_IN_ROUTE, false);
  std::vector<float> polygons(polygons_size(), 0.0f);
  std::vector<float> line_strings(line_strings_size(), 0.0f);
  const std::vector<float> goal_pose = {10.0f, 20.0f, 0.5f, 0.866f};
  std::vector<int32_t> turn_indicators(INPUT_T_WITH_CURRENT, 0);
  std::vector<float> ego_shape = {2.75f, 4.34f, 1.70f};

  const TrainingDataBinary data = build_training_data(
    ego_past, ego_current, ego_future, neighbor_past, neighbor_future, static_objects, lanes,
    lanes_speed_limit, lanes_has_speed_limit, route_lanes, route_lanes_speed_limit,
    route_lanes_has_speed_limit, polygons, line_strings, goal_pose, turn_indicators, ego_shape);

  EXPECT_FLOAT_EQ(data.goal_pose[0], 10.0f);
  EXPECT_FLOAT_EQ(data.goal_pose[1], 20.0f);
  EXPECT_FLOAT_EQ(data.goal_pose[2], 0.5f);
  EXPECT_FLOAT_EQ(data.goal_pose[3], 0.866f);
}

TEST(BuildTrainingDataTest, HasSpeedLimitBoolConversion)
{
  std::vector<float> ego_past(ego_past_size(), 0.0f);
  std::vector<float> ego_current(ego_current_size(), 0.0f);
  std::vector<float> ego_future(ego_future_size(), 0.0f);
  std::vector<float> neighbor_past(neighbor_past_size(), 0.0f);
  std::vector<float> neighbor_future(neighbor_future_size(), 0.0f);
  std::vector<float> static_objects(static_objects_size(), 0.0f);
  std::vector<float> lanes(lanes_size(), 0.0f);
  std::vector<float> lanes_speed_limit(NUM_SEGMENTS_IN_LANE, 0.0f);
  std::vector<bool> lanes_has_speed_limit(NUM_SEGMENTS_IN_LANE, false);
  lanes_has_speed_limit[0] = true;
  std::vector<float> route_lanes(route_lanes_size(), 0.0f);
  std::vector<float> route_lanes_speed_limit(NUM_SEGMENTS_IN_ROUTE, 0.0f);
  std::vector<bool> route_lanes_has_speed_limit(NUM_SEGMENTS_IN_ROUTE, false);
  route_lanes_has_speed_limit[1] = true;
  std::vector<float> polygons(polygons_size(), 0.0f);
  std::vector<float> line_strings(line_strings_size(), 0.0f);
  std::vector<float> goal_pose(4, 0.0f);
  std::vector<int32_t> turn_indicators(INPUT_T_WITH_CURRENT, 0);
  std::vector<float> ego_shape = {2.75f, 4.34f, 1.70f};

  const TrainingDataBinary data = build_training_data(
    ego_past, ego_current, ego_future, neighbor_past, neighbor_future, static_objects, lanes,
    lanes_speed_limit, lanes_has_speed_limit, route_lanes, route_lanes_speed_limit,
    route_lanes_has_speed_limit, polygons, line_strings, goal_pose, turn_indicators, ego_shape);

  EXPECT_EQ(data.lanes_has_speed_limit[0], 1);
  EXPECT_EQ(data.lanes_has_speed_limit[1], 0);
  EXPECT_EQ(data.route_lanes_has_speed_limit[1], 1);
  EXPECT_EQ(data.route_lanes_has_speed_limit[0], 0);
}

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
