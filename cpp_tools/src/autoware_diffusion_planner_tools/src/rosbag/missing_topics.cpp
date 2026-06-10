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

#include <iostream>
#include <string>
#include <vector>

std::optional<SkippingInfo> check_missing_topics(const ParsedBagData & data)
{
  std::vector<std::string> missing_topics;
  std::vector<MissingTopicType> missing_topic_types;
  if (data.kinematic_states.empty()) {
    missing_topics.emplace_back("/localization/kinematic_state");
    missing_topic_types.push_back(MissingTopicType::KinematicState);
  }
  if (data.accelerations.empty()) {
    missing_topics.emplace_back("/localization/acceleration");
    missing_topic_types.push_back(MissingTopicType::Acceleration);
  }
  if (data.tracked_objects_msgs.empty()) {
    missing_topics.emplace_back("/perception/object_recognition/tracking/objects");
    missing_topic_types.push_back(MissingTopicType::TrackedObjects);
  }
  if (data.route_msgs.empty()) {
    missing_topics.emplace_back("/planning/mission_planning/route");
    missing_topic_types.push_back(MissingTopicType::Route);
  }
  if (data.turn_indicators.empty()) {
    missing_topics.emplace_back("/vehicle/status/turn_indicators_status");
    missing_topic_types.push_back(MissingTopicType::TurnIndicators);
  }
  if (data.traffic_signals.empty()) {
    missing_topics.emplace_back("/perception/traffic_light_recognition/traffic_signals");
    missing_topic_types.push_back(MissingTopicType::TrafficSignals);
  }

  if (missing_topics.empty()) {
    return std::nullopt;
  }

  std::cout << "Skipping rosbag due to missing required topics:" << std::endl;
  for (const auto & topic : missing_topics) {
    std::cout << "  - " << topic << std::endl;
  }
  std::cout << "No training samples will be generated from this rosbag." << std::endl;

  return SkippingInfo::missing_topics(missing_topic_types);
}
