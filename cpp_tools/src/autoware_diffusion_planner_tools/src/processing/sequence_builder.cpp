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

#include "processing/sequence_builder.hpp"

#include "io/frame_writer.hpp"
#include "utils/timestamp_utils.hpp"

#include <autoware/diffusion_planner/preprocessing/traffic_signals.hpp>
#include <rclcpp/time.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace
{

constexpr int64_t CLOCK_PERIOD_NS = 100'000'000LL;  // 10 Hz
// Traffic-light validity, matching the runtime node's process_traffic_signals TTL.
constexpr double TRAFFIC_TTL_S = 5.0;

// Advance `cursor` (initially -1) to the largest index whose rosbag_time is <= tick.
template <typename T>
void advance_cursor(
  const std::deque<TimedMsg<T>> & msgs, int64_t & cursor, const int64_t tick)
{
  while (cursor + 1 < static_cast<int64_t>(msgs.size()) &&
         msgs[cursor + 1].first <= tick) {
    ++cursor;
  }
}

}  // namespace

// Build per-route sequences of fixed-rate (10 Hz) frames.
//
// This stage is pure assembly: every tick produces exactly one frame and is appended to
// its route's sequence — there is no skipping here. Each frame carries the latest message
// at-or-before the tick for every required topic (zero-order hold, matching how the
// runtime node plans from the most recent message it has received), plus the largest
// message staleness at that tick. All skip decisions (stale data, covariance, stop, red
// light, collision, off-lane) are made later in frame_processor.
//
// The clock starts only once every required topic has produced a message and a route
// exists, so each tick has real (possibly stale) data and a route bin; this removes both
// the "no route yet" and "missing data" early-outs the loop used to have.
std::vector<SequenceData> build_sequences(
  ParsedBagData & data, const int64_t search_nearest_route)
{
  using autoware_perception_msgs::msg::TrackedObjects;
  using autoware_perception_msgs::msg::TrafficLightGroupArray;
  using autoware_vehicle_msgs::msg::TurnIndicatorsReport;
  using geometry_msgs::msg::AccelWithCovarianceStamped;
  using nav_msgs::msg::Odometry;
  using autoware::diffusion_planner::preprocess::process_traffic_signals;
  using autoware::diffusion_planner::preprocess::TrafficSignalStamped;

  if (data.route_msgs.empty()) {
    std::cout << "No route messages; nothing to build." << std::endl;
    return {};
  }

  // Merge consecutive routes that share a start_pose up front. FreeSpacePlanner sometimes
  // only changes goal_pose, so such routes belong to one sequence. route_to_group maps a
  // route index -> its merged-sequence index.
  std::vector<SequenceData> sequences;
  std::vector<int64_t> route_to_group(data.route_msgs.size());
  for (size_t j = 0; j < data.route_msgs.size(); ++j) {
    if (
      j > 0 &&
      data.route_msgs[j].second.start_pose == data.route_msgs[j - 1].second.start_pose) {
      route_to_group[j] = static_cast<int64_t>(sequences.size()) - 1;
    } else {
      route_to_group[j] = static_cast<int64_t>(sequences.size());
      sequences.push_back({{}, data.route_msgs[j].second});
    }
  }

  // Clock range: start when every required topic and a route are available; end at the
  // latest message across topics. Traffic is drop-tolerated and does not gate the clock.
  const auto first_or_max = [](const auto & dq) {
    return dq.empty() ? std::numeric_limits<int64_t>::max() : dq.front().first;
  };
  const auto last_or_min = [](const auto & dq) {
    return dq.empty() ? std::numeric_limits<int64_t>::min() : dq.back().first;
  };
  int64_t earliest_route = std::numeric_limits<int64_t>::max();
  for (const auto & route_entry : data.route_msgs) {
    earliest_route = std::min(earliest_route, route_entry.first);
  }
  const int64_t clock_start = std::max({
    first_or_max(data.kinematic_states), first_or_max(data.accelerations),
    first_or_max(data.tracked_objects_msgs), first_or_max(data.turn_indicators), earliest_route});
  // End once any required topic is exhausted: beyond its last message the loop could only
  // carry stale data forward. Symmetric with clock_start; traffic does not gate the clock.
  const int64_t clock_end = std::min({
    last_or_min(data.kinematic_states), last_or_min(data.accelerations),
    last_or_min(data.tracked_objects_msgs), last_or_min(data.turn_indicators)});
  std::cout << "clock_start=" << clock_start << " clock_end=" << clock_end
            << " period_ns=" << CLOCK_PERIOD_NS << std::endl;

  // Forward-walking cursors.
  int64_t kin_cursor = -1;
  int64_t accel_cursor = -1;
  int64_t tracked_cursor = -1;
  int64_t turn_ind_cursor = -1;
  int64_t traffic_high_cursor = -1;
  int64_t traffic_low_cursor = 0;

  // Persistent traffic-light state, maintained at 10 Hz exactly like the runtime node:
  // each tick folds in the traffic msgs that arrived since the previous tick and expires
  // entries older than TRAFFIC_TTL_S. A snapshot is stored on each frame.
  std::map<lanelet::Id, TrafficSignalStamped> traffic_map;

  for (int64_t tick = clock_start; tick <= clock_end; tick += CLOCK_PERIOD_NS) {
    advance_cursor(data.kinematic_states, kin_cursor, tick);
    advance_cursor(data.accelerations, accel_cursor, tick);
    advance_cursor(data.tracked_objects_msgs, tracked_cursor, tick);
    advance_cursor(data.turn_indicators, turn_ind_cursor, tick);
    advance_cursor(data.traffic_signals, traffic_high_cursor, tick);

    // Required topics: latest message at or before tick (zero-order hold). clock_start
    // guarantees each cursor is valid; the staleness decision is made in frame_processor.
    const Odometry & kinematic = data.kinematic_states[kin_cursor].second;
    const AccelWithCovarianceStamped & accel = data.accelerations[accel_cursor].second;
    const TrackedObjects & tracked = data.tracked_objects_msgs[tracked_cursor].second;
    const TurnIndicatorsReport & turn_ind = data.turn_indicators[turn_ind_cursor].second;

    const int64_t max_msg_age_ns = std::max({
      tick - data.kinematic_states[kin_cursor].first,
      tick - data.accelerations[accel_cursor].first,
      tick - data.tracked_objects_msgs[tracked_cursor].first,
      tick - data.turn_indicators[turn_ind_cursor].first});

    // Traffic: consume every msg that arrived since the previous tick into the persistent
    // map (latest-per-light + TTL), then snapshot the current state for this frame.
    std::vector<TrafficLightGroupArray::ConstSharedPtr> new_traffic;
    for (int64_t k = traffic_low_cursor; k <= traffic_high_cursor; ++k) {
      new_traffic.push_back(
        std::make_shared<TrafficLightGroupArray>(data.traffic_signals[k].second));
    }
    traffic_low_cursor = traffic_high_cursor + 1;
    // Use RCL_ROS_TIME to match the msg header stamps (process_traffic_signals subtracts
    // current_time from each signal's stamp; mixing clock sources throws).
    process_traffic_signals(new_traffic, traffic_map, rclcpp::Time(tick, RCL_ROS_TIME), TRAFFIC_TTL_S);

    // Resolve the route for this tick (latest route with rosbag_time <= tick).
    int64_t max_route_index = 0;
    if (search_nearest_route) {
      int64_t best_route_time = std::numeric_limits<int64_t>::min();
      for (int64_t j = 0; j < static_cast<int64_t>(data.route_msgs.size()); ++j) {
        const int64_t route_time = data.route_msgs[j].first;
        if (route_time <= tick && route_time >= best_route_time) {
          best_route_time = route_time;
          max_route_index = j;
        }
      }
    }

    FrameData frame;
    frame.timestamp = tick;
    frame.tracked_objects = tracked;
    frame.kinematic_state = kinematic;
    frame.acceleration = accel;
    frame.traffic_light_id_map = traffic_map;
    frame.turn_indicator = turn_ind;
    frame.max_msg_age_ns = max_msg_age_ns;
    sequences[route_to_group[max_route_index]].data_list.push_back(frame);
  }

  // Frames are pushed in tick order; keep each sequence's data_list ascending by timestamp.
  for (auto & seq : sequences) {
    std::sort(
      seq.data_list.begin(), seq.data_list.end(),
      [](const FrameData & a, const FrameData & b) { return a.timestamp < b.timestamp; });
  }

  return sequences;
}
