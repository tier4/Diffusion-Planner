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

#include "sequence_builder.hpp"

#include "io.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace autoware_perception_msgs::msg;
using namespace autoware_planning_msgs::msg;
using namespace autoware_vehicle_msgs::msg;
using namespace geometry_msgs::msg;
using namespace nav_msgs::msg;

std::vector<SequenceData> build_sequences(
  const std::deque<TrackedObjects> & tracked_objects_msgs,
  std::deque<Odometry> & kinematic_states,
  std::deque<AccelWithCovarianceStamped> & accelerations,
  std::deque<TrafficLightGroupArray> & traffic_signals,
  std::deque<TurnIndicatorsReport> & turn_indicators,
  const std::vector<LaneletRoute> & route_msgs, const bool search_nearest_route,
  const std::string & save_dir, const std::string & rosbag_dir_name)
{
  std::vector<SequenceData> sequences;
  for (const LaneletRoute & route : route_msgs) {
    sequences.push_back({{}, route});
  }

  const int64_t n = static_cast<int64_t>(tracked_objects_msgs.size());
  std::cout << "n=" << n << std::endl;

  for (int64_t i = 0; i < n; ++i) {
    const TrackedObjects & tracking = tracked_objects_msgs[i];
    const int64_t timestamp = parse_timestamp(tracking.header.stamp);

    Odometry kinematic;
    AccelWithCovarianceStamped accel;
    std::vector<TrafficLightGroupArray> traffic_signal;
    TurnIndicatorsReport turn_ind;
    std::vector<std::string> incomplete_details;

    bool ok = true;

    const auto kinematic_vec = check_and_update_msg(kinematic_states, tracking.header.stamp);
    if (!kinematic_vec.empty()) {
      kinematic = kinematic_vec.back();
    } else {
      ok = false;
      incomplete_details.emplace_back("KinematicState");
      std::cout << "No matching kinematic_state for tracked_objects at " << i << std::endl;
    }

    const auto accel_vec = check_and_update_msg(accelerations, tracking.header.stamp);
    if (!accel_vec.empty()) {
      accel = accel_vec.back();
    } else {
      ok = false;
      incomplete_details.emplace_back("Acceleration");
      std::cout << "No matching acceleration for tracked_objects at " << i << std::endl;
    }

    const auto traffic_signal_vec = check_and_update_msg(traffic_signals, tracking.header.stamp);
    if (!traffic_signal_vec.empty()) {
      traffic_signal = traffic_signal_vec;
    } else {
      ok = false;
      incomplete_details.emplace_back("TrafficSignals");
      std::cout << "No matching traffic_signal for tracked_objects at " << i << std::endl;
    }

    const auto turn_ind_vec = check_and_update_msg(turn_indicators, tracking.header.stamp);
    if (!turn_ind_vec.empty()) {
      turn_ind = turn_ind_vec.back();
    } else {
      ok = false;
      incomplete_details.emplace_back("TurnIndicators");
      std::cout << "No matching turn_indicators for tracked_objects at " << i << std::endl;
    }

    // Find the appropriate route for this frame
    int64_t max_route_index = -1;
    if (search_nearest_route) {
      int64_t max_route_timestamp = 0;
      for (int64_t j = 0; j < static_cast<int64_t>(route_msgs.size()); ++j) {
        const LaneletRoute & route_msg = route_msgs[j];
        const int64_t route_stamp = parse_timestamp(route_msg.header.stamp);
        if (max_route_timestamp <= route_stamp && route_stamp <= timestamp) {
          max_route_timestamp = route_stamp;
          max_route_index = j;
        }
      }
      if (max_route_index == -1) {
        std::cout << "Cannot find route msg at " << i << std::endl;
        continue;
      }
    } else {
      max_route_index = 0;
    }

    // Validate kinematic_state covariance
    if (ok) {
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
        // At the beginning of recording, some msgs may be missing — skip this frame
        std::vector<IncompleteDataType> incomplete_types;
        for (const auto & s : incomplete_details) {
          if (s == "KinematicState" || s == "InvalidKinematicCovariance")
            incomplete_types.push_back(IncompleteDataType::KinematicState);
          else if (s == "Acceleration")
            incomplete_types.push_back(IncompleteDataType::Acceleration);
          else if (s == "TrackedObjects")
            incomplete_types.push_back(IncompleteDataType::TrackedObjects);
          else if (s == "TrafficSignals")
            incomplete_types.push_back(IncompleteDataType::TrafficSignals);
          else if (s == "TurnIndicators")
            incomplete_types.push_back(IncompleteDataType::TurnIndicators);
        }
        const SkippingInfo skipping_info = SkippingInfo::incomplete_data(incomplete_types);
        Odometry fallback_kinematic = kinematic;
        fallback_kinematic.header.stamp = tracking.header.stamp;
        save_frame_json(
          save_dir, rosbag_dir_name,
          create_token(max_route_index >= 0 ? max_route_index : 0, i), fallback_kinematic,
          timestamp, skipping_info);
        std::cout << "Skip this frame i=" << i << "/n=" << n << std::endl;
        continue;
      } else {
        std::cout << "Finish at this frame i=" << i << "/n=" << n << std::endl;
        break;
      }
    }

    const FrameData frame_data{timestamp, tracking, kinematic, accel, traffic_signal, turn_ind};
    sequence.data_list.push_back(frame_data);
  }

  return sequences;
}

void merge_sequences(std::vector<SequenceData> & sequences)
{
  // FreeSpacePlanner sometimes changes goal_pose at the end; combine such sequences.
  for (int64_t i = static_cast<int64_t>(sequences.size()) - 2; i >= 0; --i) {
    const auto & route_l = sequences[i].route;
    const auto & route_r = sequences[i + 1].route;

    if (route_l.start_pose != route_r.start_pose) {
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
}

void sort_sequences(std::vector<SequenceData> & sequences)
{
  for (auto & seq : sequences) {
    std::sort(
      seq.data_list.begin(), seq.data_list.end(),
      [](const FrameData & a, const FrameData & b) { return a.timestamp < b.timestamp; });
  }
}
