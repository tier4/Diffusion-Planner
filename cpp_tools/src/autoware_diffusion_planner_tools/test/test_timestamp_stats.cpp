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

#include "timestamp_stats.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <string>

using namespace timestamp_stats;

TEST(TimestampStatsTest, BasicStats)
{
  TimestampStats stats("/test/topic");
  std::vector<int64_t> header_ts = {100, 200, 300, 400, 500};
  std::vector<int64_t> rosbag_ts = {110, 210, 310, 410, 510};
  for (size_t i = 0; i < header_ts.size(); ++i) {
    stats.add_header_timestamp(header_ts[i]);
    stats.add_rosbag_timestamp(rosbag_ts[i]);
  }
  stats.calc_stats();

  EXPECT_TRUE(stats.is_monotonic_header());
  EXPECT_TRUE(stats.is_monotonic_rosbag());
  EXPECT_DOUBLE_EQ(stats.diff_mean(), 10.0);
  EXPECT_DOUBLE_EQ(stats.header_diff_mean(), 100.0);
  EXPECT_DOUBLE_EQ(stats.rosbag_diff_mean(), 100.0);
  EXPECT_EQ(stats.diff_min(), 10);
  EXPECT_EQ(stats.diff_max(), 10);
  EXPECT_EQ(stats.header_diff_min(), 100);
  EXPECT_EQ(stats.header_diff_max(), 100);
  EXPECT_EQ(stats.rosbag_diff_min(), 100);
  EXPECT_EQ(stats.rosbag_diff_max(), 100);
}

TEST(TimestampStatsTest, NonMonotonic)
{
  TimestampStats stats("/test/topic");
  std::vector<int64_t> header_ts = {100, 300, 200, 400};
  std::vector<int64_t> rosbag_ts = {110, 310, 210, 410};
  for (size_t i = 0; i < header_ts.size(); ++i) {
    stats.add_header_timestamp(header_ts[i]);
    stats.add_rosbag_timestamp(rosbag_ts[i]);
  }
  stats.calc_stats();
  EXPECT_FALSE(stats.is_monotonic_header());
  EXPECT_FALSE(stats.is_monotonic_rosbag());
}

TEST(TimestampStatsMapTest, AddAndAnalyze)
{
  std::vector<std::string> topics = {"/a", "/b"};
  TimestampStatsMap stats_map(topics);
  stats_map.add_timestamp("/a", 100, 110);
  stats_map.add_timestamp("/a", 200, 210);
  stats_map.add_timestamp("/b", 300, 310);
  stats_map.add_timestamp("/b", 400, 410);
  stats_map.analyze_all();
  EXPECT_TRUE(stats_map.stats_map["/a"].is_monotonic_header());
  EXPECT_TRUE(stats_map.stats_map["/b"].is_monotonic_header());
  EXPECT_DOUBLE_EQ(stats_map.stats_map["/a"].diff_mean(), 10.0);
  EXPECT_DOUBLE_EQ(stats_map.stats_map["/b"].diff_mean(), 10.0);
}

TEST(TimestampStatsMapTest, EmptyTopics)
{
  std::vector<std::string> topics;
  TimestampStatsMap stats_map(topics);
  EXPECT_TRUE(stats_map.stats_map.empty());
  // add timestamp to a new topic not in the initial list
  stats_map.add_timestamp("/new_topic", 100, 200);
  EXPECT_EQ(stats_map.stats_map.size(), 1);
  EXPECT_TRUE(stats_map.stats_map.find("/new_topic") != stats_map.stats_map.end());
}
