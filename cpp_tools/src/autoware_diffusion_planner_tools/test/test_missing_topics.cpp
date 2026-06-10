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

#include "rosbag/parsed_bag_data.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

// Build a ParsedBagData with all required topics populated with one entry each.
ParsedBagData make_full_data()
{
  ParsedBagData data({});

  nav_msgs::msg::Odometry odom;
  data.kinematic_states.push_back({0LL, odom});

  geometry_msgs::msg::AccelWithCovarianceStamped accel;
  data.accelerations.push_back({0LL, accel});

  autoware_perception_msgs::msg::TrackedObjects tracked;
  data.tracked_objects_msgs.push_back({0LL, tracked});

  autoware_planning_msgs::msg::LaneletRoute route;
  data.route_msgs.push_back({0LL, route});

  autoware_vehicle_msgs::msg::TurnIndicatorsReport turn;
  data.turn_indicators.push_back({0LL, turn});

  autoware_perception_msgs::msg::TrafficLightGroupArray traffic;
  data.traffic_signals.push_back({0LL, traffic});

  return data;
}

}  // namespace

// ---------------------------------------------------------------------------
// check_missing_topics tests
// ---------------------------------------------------------------------------

TEST(CheckMissingTopicsTest, AllTopicsPresentReturnsNullopt)
{
  const ParsedBagData data = make_full_data();
  const auto result = check_missing_topics(data);
  EXPECT_FALSE(result.has_value());
}

TEST(CheckMissingTopicsTest, EmptyDataAllTopicsMissing)
{
  ParsedBagData data({});
  const auto result = check_missing_topics(data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->label, SkippingLabel::MissingRequiredTopic);
  // All 6 topics should be listed
  EXPECT_EQ(result->missing_topic_types.size(), 6u);
}

TEST(CheckMissingTopicsTest, MissingKinematicState)
{
  ParsedBagData data = make_full_data();
  data.kinematic_states.clear();

  const auto result = check_missing_topics(data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->label, SkippingLabel::MissingRequiredTopic);
  EXPECT_EQ(result->missing_topic_types.size(), 1u);
  EXPECT_EQ(result->missing_topic_types[0], MissingTopicType::KinematicState);
}

TEST(CheckMissingTopicsTest, MissingAcceleration)
{
  ParsedBagData data = make_full_data();
  data.accelerations.clear();

  const auto result = check_missing_topics(data);
  ASSERT_TRUE(result.has_value());
  const auto & types = result->missing_topic_types;
  EXPECT_EQ(types.size(), 1u);
  EXPECT_EQ(types[0], MissingTopicType::Acceleration);
}

TEST(CheckMissingTopicsTest, MissingTrackedObjects)
{
  ParsedBagData data = make_full_data();
  data.tracked_objects_msgs.clear();

  const auto result = check_missing_topics(data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->missing_topic_types[0], MissingTopicType::TrackedObjects);
}

TEST(CheckMissingTopicsTest, MissingRoute)
{
  ParsedBagData data = make_full_data();
  data.route_msgs.clear();

  const auto result = check_missing_topics(data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->missing_topic_types[0], MissingTopicType::Route);
}

TEST(CheckMissingTopicsTest, MissingTurnIndicators)
{
  ParsedBagData data = make_full_data();
  data.turn_indicators.clear();

  const auto result = check_missing_topics(data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->missing_topic_types[0], MissingTopicType::TurnIndicators);
}

TEST(CheckMissingTopicsTest, MissingTrafficSignals)
{
  ParsedBagData data = make_full_data();
  data.traffic_signals.clear();

  const auto result = check_missing_topics(data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->missing_topic_types[0], MissingTopicType::TrafficSignals);
}

TEST(CheckMissingTopicsTest, MultipleMissingTopicsReported)
{
  ParsedBagData data = make_full_data();
  data.kinematic_states.clear();
  data.route_msgs.clear();

  const auto result = check_missing_topics(data);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->missing_topic_types.size(), 2u);
  const auto & types = result->missing_topic_types;
  EXPECT_TRUE(
    std::find(types.begin(), types.end(), MissingTopicType::KinematicState) != types.end());
  EXPECT_TRUE(
    std::find(types.begin(), types.end(), MissingTopicType::Route) != types.end());
}
