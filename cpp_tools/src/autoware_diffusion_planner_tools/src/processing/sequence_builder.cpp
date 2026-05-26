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

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

constexpr int64_t CLOCK_PERIOD_NS = 100'000'000LL;  // 10 Hz
// Mirrors the original check_and_update_msg threshold of 200 ms.
constexpr int64_t MAX_MSG_AGE_NS = 200'000'000LL;

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

std::vector<SequenceData> build_sequences(
  ParsedBagData & data, const int64_t search_nearest_route, const std::string & save_dir,
  const std::string & rosbag_dir_name)
{
  using autoware_perception_msgs::msg::TrackedObjects;
  using autoware_perception_msgs::msg::TrafficLightGroupArray;
  using autoware_planning_msgs::msg::LaneletRoute;
  using autoware_vehicle_msgs::msg::TurnIndicatorsReport;
  using geometry_msgs::msg::AccelWithCovarianceStamped;
  using nav_msgs::msg::Odometry;

  // One sequence per route message; route is selected per-frame below.
  std::vector<SequenceData> sequences;
  for (const auto & route_entry : data.route_msgs) {
    sequences.push_back({{}, route_entry.second});
  }

  // Clock range = min/max rosbag arrival time across input topics.
  const std::array<int64_t, 5> first_times{
    data.kinematic_states.empty() ? std::numeric_limits<int64_t>::max()
                                  : data.kinematic_states.front().first,
    data.accelerations.empty() ? std::numeric_limits<int64_t>::max()
                               : data.accelerations.front().first,
    data.tracked_objects_msgs.empty() ? std::numeric_limits<int64_t>::max()
                                      : data.tracked_objects_msgs.front().first,
    data.turn_indicators.empty() ? std::numeric_limits<int64_t>::max()
                                 : data.turn_indicators.front().first,
    data.traffic_signals.empty() ? std::numeric_limits<int64_t>::max()
                                 : data.traffic_signals.front().first,
  };
  const std::array<int64_t, 5> last_times{
    data.kinematic_states.empty() ? std::numeric_limits<int64_t>::min()
                                  : data.kinematic_states.back().first,
    data.accelerations.empty() ? std::numeric_limits<int64_t>::min()
                               : data.accelerations.back().first,
    data.tracked_objects_msgs.empty() ? std::numeric_limits<int64_t>::min()
                                      : data.tracked_objects_msgs.back().first,
    data.turn_indicators.empty() ? std::numeric_limits<int64_t>::min()
                                 : data.turn_indicators.back().first,
    data.traffic_signals.empty() ? std::numeric_limits<int64_t>::min()
                                 : data.traffic_signals.back().first,
  };
  const int64_t clock_start = *std::min_element(first_times.begin(), first_times.end());
  const int64_t clock_end = *std::max_element(last_times.begin(), last_times.end());
  std::cout << "clock_start=" << clock_start << " clock_end=" << clock_end
            << " period_ns=" << CLOCK_PERIOD_NS << std::endl;

  // Forward-walking cursors (analogue of the original deque-erase pattern).
  int64_t kin_cursor = -1;
  int64_t accel_cursor = -1;
  int64_t tracked_cursor = -1;
  int64_t turn_ind_cursor = -1;
  int64_t traffic_high_cursor = -1;
  int64_t traffic_low_cursor = 0;
  int64_t frame_idx = 0;

  for (int64_t tick = clock_start; tick <= clock_end; tick += CLOCK_PERIOD_NS, ++frame_idx) {
    advance_cursor(data.kinematic_states, kin_cursor, tick);
    advance_cursor(data.accelerations, accel_cursor, tick);
    advance_cursor(data.tracked_objects_msgs, tracked_cursor, tick);
    advance_cursor(data.turn_indicators, turn_ind_cursor, tick);
    advance_cursor(data.traffic_signals, traffic_high_cursor, tick);

    // Latest within [tick - 200ms, tick] for the four single-msg topics.
    const int64_t freshness_low = tick - MAX_MSG_AGE_NS;

    const Odometry * kinematic_ptr = nullptr;
    if (kin_cursor >= 0 && data.kinematic_states[kin_cursor].first >= freshness_low) {
      kinematic_ptr = &data.kinematic_states[kin_cursor].second;
    }
    const AccelWithCovarianceStamped * accel_ptr = nullptr;
    if (accel_cursor >= 0 && data.accelerations[accel_cursor].first >= freshness_low) {
      accel_ptr = &data.accelerations[accel_cursor].second;
    }
    const TrackedObjects * tracked_ptr = nullptr;
    if (
      tracked_cursor >= 0 &&
      data.tracked_objects_msgs[tracked_cursor].first >= freshness_low) {
      tracked_ptr = &data.tracked_objects_msgs[tracked_cursor].second;
    }
    const TurnIndicatorsReport * turn_ind_ptr = nullptr;
    if (turn_ind_cursor >= 0 && data.turn_indicators[turn_ind_cursor].first >= freshness_low) {
      turn_ind_ptr = &data.turn_indicators[turn_ind_cursor].second;
    }

    std::vector<std::string> incomplete_details;
    bool ok = true;
    if (kinematic_ptr == nullptr) {
      ok = false;
      incomplete_details.emplace_back("KinematicState");
      std::cout << "No matching kinematic_state at tick=" << tick << std::endl;
    }
    if (accel_ptr == nullptr) {
      ok = false;
      incomplete_details.emplace_back("Acceleration");
      std::cout << "No matching acceleration at tick=" << tick << std::endl;
    }
    if (tracked_ptr == nullptr) {
      ok = false;
      incomplete_details.emplace_back("TrackedObjects");
      std::cout << "No matching tracked_objects at tick=" << tick << std::endl;
    }
    if (turn_ind_ptr == nullptr) {
      ok = false;
      incomplete_details.emplace_back("TurnIndicators");
      std::cout << "No matching turn_indicators at tick=" << tick << std::endl;
    }

    // Traffic signals: all msgs in [tick - 200ms, tick] (original window).
    // Drops are tolerated — empty traffic_signal does not fail the frame.
    while (
      traffic_low_cursor <= traffic_high_cursor &&
      data.traffic_signals[traffic_low_cursor].first < freshness_low) {
      ++traffic_low_cursor;
    }
    std::vector<TrafficLightGroupArray> traffic_signal;
    for (int64_t k = traffic_low_cursor; k <= traffic_high_cursor; ++k) {
      traffic_signal.push_back(data.traffic_signals[k].second);
    }
    if (traffic_signal.empty()) {
      std::cout << "No matching traffic_signal at tick=" << tick << " (drop tolerated)"
                << std::endl;
    }

    // Resolve the route for this tick (latest route with rosbag_time <= tick, or first one).
    int64_t max_route_index = -1;
    if (search_nearest_route) {
      int64_t best_route_time = std::numeric_limits<int64_t>::min();
      for (int64_t j = 0; j < static_cast<int64_t>(data.route_msgs.size()); ++j) {
        const int64_t route_time = data.route_msgs[j].first;
        if (route_time <= tick && route_time >= best_route_time) {
          best_route_time = route_time;
          max_route_index = j;
        }
      }
      if (max_route_index == -1) {
        std::cout << "Cannot find route msg at tick=" << tick << std::endl;
        continue;
      }
    } else {
      if (data.route_msgs.empty()) {
        std::cout << "No route msgs available at tick=" << tick << std::endl;
        continue;
      }
      max_route_index = 0;
    }

    // Kinematic covariance validation (matches original).
    Odometry kinematic;
    if (kinematic_ptr != nullptr) {
      kinematic = *kinematic_ptr;
      const std::array<double, 36> & covariance = kinematic.pose.covariance;
      const double covariance_xx = covariance[0];
      const double covariance_yy = covariance[7];
      if (covariance_xx > 1e-1 || covariance_yy > 1e-1) {
        std::cout << "Invalid kinematic_state covariance_xx=" << covariance_xx
                  << ", covariance_yy=" << covariance_yy << std::endl;
        ok = false;
        incomplete_details.emplace_back("InvalidKinematicCovariance");
      }
    }

    SequenceData & sequence = sequences[max_route_index];

    if (!ok) {
      if (sequence.data_list.empty()) {
        // Beginning of recording: skip this frame and record why it was skipped.
        std::vector<IncompleteDataType> incomplete_types;
        for (const auto & s : incomplete_details) {
          if (s == "KinematicState" || s == "InvalidKinematicCovariance") {
            incomplete_types.push_back(IncompleteDataType::KinematicState);
          } else if (s == "Acceleration") {
            incomplete_types.push_back(IncompleteDataType::Acceleration);
          } else if (s == "TrackedObjects") {
            incomplete_types.push_back(IncompleteDataType::TrackedObjects);
          } else if (s == "TrafficSignals") {
            incomplete_types.push_back(IncompleteDataType::TrafficSignals);
          } else if (s == "TurnIndicators") {
            incomplete_types.push_back(IncompleteDataType::TurnIndicators);
          }
        }
        const SkippingInfo skipping_info = SkippingInfo::incomplete_data(incomplete_types);
        Odometry fallback_kinematic;
        if (kinematic_ptr != nullptr) {
          fallback_kinematic = *kinematic_ptr;
        }
        fallback_kinematic.header.stamp.sec = static_cast<int32_t>(tick / 1'000'000'000LL);
        fallback_kinematic.header.stamp.nanosec =
          static_cast<uint32_t>(tick % 1'000'000'000LL);
        save_frame_json(
          save_dir, rosbag_dir_name, create_token(max_route_index, frame_idx), fallback_kinematic,
          tick, skipping_info);
        std::cout << "Skip this frame frame_idx=" << frame_idx << " tick=" << tick << std::endl;
        continue;
      } else {
        // Mid-recording: a required topic stopped — finish this sequence.
        std::cout << "Finish at frame_idx=" << frame_idx << " tick=" << tick << std::endl;
        break;
      }
    }

    const FrameData frame_data{
      tick, *tracked_ptr, kinematic, *accel_ptr, traffic_signal, *turn_ind_ptr};
    sequence.data_list.push_back(frame_data);
  }

  // Because FreeSpacePlanner sometimes changes goal_pose at the end, combine such things.
  for (int64_t i = static_cast<int64_t>(sequences.size()) - 2; i >= 0; --i) {
    const LaneletRoute & route_msg_l = sequences[i].route;
    const LaneletRoute & route_msg_r = sequences[i + 1].route;

    if (route_msg_l.start_pose != route_msg_r.start_pose) {
      std::cout << "Route start pose mismatch: " << i << " != " << i + 1 << std::endl;
      continue;
    }

    std::cout << "Concatenate sequence " << i << " and " << i + 1 << std::endl;
    std::cout << "Before sequence[" << i << "].data_list.size()=" << sequences[i].data_list.size()
              << " frames" << std::endl;

    sequences[i].data_list.insert(
      sequences[i].data_list.end(), sequences[i + 1].data_list.begin(),
      sequences[i + 1].data_list.end());

    std::cout << "After sequence[" << i << "].data_list.size()=" << sequences[i].data_list.size()
              << " frames" << std::endl;

    sequences.erase(sequences.begin() + i + 1);
  }

  // Sort each sequence's data_list by timestamp to ensure ascending order.
  for (auto & seq : sequences) {
    std::sort(
      seq.data_list.begin(), seq.data_list.end(),
      [](const FrameData & a, const FrameData & b) { return a.timestamp < b.timestamp; });
  }

  return sequences;
}
