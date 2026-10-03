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

#include "conversion/selected_frames.hpp"
#include "utils/cnpy.hpp"

#include <nlohmann/json.hpp>
#include <rclcpp/time.hpp>

#include <gtest/gtest.h>
#include <lanelet2_core/LaneletMap.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>

namespace
{
constexpr int64_t start = 1'000'000'000;
constexpr int64_t step = 100'000'000;

class SelectedFrames : public ::testing::TestWithParam<bool>
{
protected:
  std::filesystem::path directory;
  ParsedBagData bag{{}};
  ConverterOptions options = ConverterOptions::default_converter_options();

  void SetUp() override
  {
    char pattern[] = "/tmp/selected-frames-test-XXXXXX";
    const char * created = mkdtemp(pattern);
    ASSERT_NE(created, nullptr);
    directory = created;
    options.timestamps = (directory / "timestamps.json").string();
    options.use_interpolation = GetParam();
    options.ego_wheel_base = 3.0;
    options.ego_length = 5.0;
    options.ego_width = 2.0;
    for (int i = 0; i < 150; ++i) {
      const int64_t tick = start + i * step;
      nav_msgs::msg::Odometry odom;
      odom.header.stamp = rclcpp::Time(tick - 10'000'000);
      odom.pose.pose.orientation.w = 1.0;
      odom.pose.pose.position.x = i * 0.2;
      odom.twist.twist.linear.x = 2.0;
      bag.kinematic_states.emplace_back(tick, odom);
      bag.accelerations.emplace_back(tick, geometry_msgs::msg::AccelWithCovarianceStamped{});
      bag.tracked_objects_msgs.emplace_back(tick, autoware_perception_msgs::msg::TrackedObjects{});
      bag.turn_indicators.emplace_back(tick, autoware_vehicle_msgs::msg::TurnIndicatorsReport{});
    }
    autoware_planning_msgs::msg::LaneletRoute route;
    route.header.stamp = rclcpp::Time(start);
    route.start_pose.orientation.w = route.goal_pose.orientation.w = 1.0;
    route.goal_pose.position.x = 100.0;
    bag.route_msgs.emplace_back(start, route);
  }

  void TearDown() override
  {
    if (!directory.empty()) std::filesystem::remove_all(directory);
  }

  nlohmann::json run(const std::vector<int> & indices, const int expected_code)
  {
    std::vector<int64_t> ticks;
    for (int i : indices) ticks.push_back(start + i * step);
    std::ofstream(options.timestamps) << nlohmann::json(ticks);
    const ConverterPaths paths{"bag", "unused", (directory / "output").string()};
    auto map = std::make_shared<lanelet::LaneletMap>();
    lanelet::LineString3d left(3, {lanelet::Point3d(1, 0, 2, 0), lanelet::Point3d(2, 100, 2, 0)});
    lanelet::LineString3d right(
      6, {lanelet::Point3d(4, 0, -2, 0), lanelet::Point3d(5, 100, -2, 0)});
    lanelet::Lanelet lane(7, left, right);
    lane.attributes()["subtype"] = "road";
    map->add(lane);
    const autoware::diffusion_planner::preprocess::LaneSegmentContext context(map);
    EXPECT_EQ(export_selected_frames(bag, paths, options, context, {}), expected_code);
    std::ifstream stream(directory / "output/requests.json");
    return nlohmann::json::parse(stream);
  }

  cnpy::npz_t arrays(const nlohmann::json & result)
  {
    return cnpy::npz_load((directory / "output" / result["output"].get<std::string>()).string());
  }
};

TEST_P(SelectedFrames, NoAddedCurrentHistoryOrFutureAgeFilter)
{
  // Native preprocessing handles its existing resampling; no selected-only age gate.
  for (int i : {35, 40, 90}) {
    bag.kinematic_states[i].second.header.stamp = bag.kinematic_states[i - 1].second.header.stamp;
  }
  const auto results = run({40, 41}, 0);
  EXPECT_EQ(results[0]["status"], "ok");
  EXPECT_EQ(results[1]["status"], "ok");
}

TEST_P(SelectedFrames, NativeBoundaryFailureDoesNotSuppressLaterRequests)
{
  const auto results = run({0, 30, 40, 149}, 2);
  EXPECT_EQ(results[0]["status"], "unavailable");
  EXPECT_EQ(results[1]["status"], "ok");
  EXPECT_EQ(results[2]["status"], "ok");
  EXPECT_EQ(results[3]["status"], "unavailable");
}

TEST_P(SelectedFrames, StoppedBagEndCanHoldFutureAtItsLastTick)
{
  bag.kinematic_states.back().second.twist.twist.linear.x = 0.0;
  const auto results = run({149}, 0);
  ASSERT_EQ(results[0]["status"], "ok");
  auto data = arrays(results[0]);
  for (size_t i = 0; i < data.at("ego_agent_future").num_vals; ++i) {
    EXPECT_NEAR(data.at("ego_agent_future").data<float>()[i], 0.0f, 1e-4);
  }
  EXPECT_NEAR(data.at("goal_pose").data<float>()[0], 0.0f, 1e-4);
}

TEST_P(SelectedFrames, RouteGoalsStayLocalWhileTemporalContextCrossesRoutes)
{
  bag.kinematic_states[69].second.twist.twist.linear.x = 0.0;
  bag.kinematic_states.back().second.twist.twist.linear.x = 0.0;
  auto route = bag.route_msgs.front().second;
  route.start_pose.position.x = 14.0;
  route.goal_pose.position.x = 200.0;
  bag.route_msgs.emplace_back(start + 70 * step, route);
  const auto results = run({40, 70, 149}, 0);
  auto early = arrays(results[0]);
  auto boundary = arrays(results[1]);
  EXPECT_NEAR(early.at("goal_pose").data<float>()[0], (69 - 40) * 0.2f, 1e-3);
  EXPECT_NEAR(boundary.at("goal_pose").data<float>()[0], (149 - 70) * 0.2f, 1e-3);
  EXPECT_EQ(results[2]["status"], "ok");
}
INSTANTIATE_TEST_SUITE_P(NativeResamplingModes, SelectedFrames, ::testing::Bool());
}  // namespace
