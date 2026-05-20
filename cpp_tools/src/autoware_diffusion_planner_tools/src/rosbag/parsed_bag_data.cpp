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

#include "rosbag_parser.hpp"
#include "utils/timestamp_utils.hpp"

#include <iostream>

ParsedBagData load_rosbag(const std::string & rosbag_path, int64_t limit)
{
  using autoware_perception_msgs::msg::TrackedObjects;
  using autoware_perception_msgs::msg::TrafficLightGroupArray;
  using autoware_planning_msgs::msg::LaneletRoute;
  using autoware_vehicle_msgs::msg::TurnIndicatorsReport;
  using geometry_msgs::msg::AccelWithCovarianceStamped;
  using nav_msgs::msg::Odometry;

  const std::vector<std::string> target_topics = {
    "/localization/kinematic_state",
    "/localization/acceleration",
    "/perception/object_recognition/tracking/objects",
    "/planning/mission_planning/route",
    "/vehicle/status/turn_indicators_status",
    "/perception/traffic_light_recognition/traffic_signals"};

  ParsedBagData data(target_topics);

  rosbag_parser::RosbagParser rosbag_parser(rosbag_path);
  rosbag_parser.create_reader(rosbag_path);

  int64_t parse_count = 0;
  while (rosbag_parser.has_next() && (limit < 0 || parse_count < limit)) {
    const rosbag2_storage::SerializedBagMessageSharedPtr msg = rosbag_parser.read_next();

    if (msg->topic_name == "/localization/kinematic_state") {
      const Odometry odometry = rosbag_parser.deserialize_message<Odometry>(msg);
      data.kinematic_states.push_back(odometry);
      data.timestamp_stats_map.add_timestamp(
        "/localization/kinematic_state", parse_timestamp(odometry.header.stamp),
        static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/localization/acceleration") {
      const AccelWithCovarianceStamped accel =
        rosbag_parser.deserialize_message<AccelWithCovarianceStamped>(msg);
      data.accelerations.push_back(accel);
      data.timestamp_stats_map.add_timestamp(
        "/localization/acceleration", parse_timestamp(accel.header.stamp),
        static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/perception/object_recognition/tracking/objects") {
      const TrackedObjects objects = rosbag_parser.deserialize_message<TrackedObjects>(msg);
      data.tracked_objects_msgs.push_back(objects);
      data.timestamp_stats_map.add_timestamp(
        "/perception/object_recognition/tracking/objects", parse_timestamp(objects.header.stamp),
        static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/planning/mission_planning/route") {
      const LaneletRoute route = rosbag_parser.deserialize_message<LaneletRoute>(msg);
      data.route_msgs.push_back(route);
      data.timestamp_stats_map.add_timestamp(
        "/planning/mission_planning/route", parse_timestamp(route.header.stamp),
        static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/vehicle/status/turn_indicators_status") {
      const TurnIndicatorsReport turn_ind =
        rosbag_parser.deserialize_message<TurnIndicatorsReport>(msg);
      data.turn_indicators.push_back(turn_ind);
      data.timestamp_stats_map.add_timestamp(
        "/vehicle/status/turn_indicators_status", parse_timestamp(turn_ind.stamp),
        static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/perception/traffic_light_recognition/traffic_signals") {
      const TrafficLightGroupArray traffic_signal =
        rosbag_parser.deserialize_message<TrafficLightGroupArray>(msg);
      data.traffic_signals.push_back(traffic_signal);
      data.timestamp_stats_map.add_timestamp(
        "/perception/traffic_light_recognition/traffic_signals",
        parse_timestamp(traffic_signal.stamp), static_cast<int64_t>(msg->time_stamp));
    }

    parse_count++;
  }

  data.timestamp_stats_map.analyze_all();

  std::cout << "Parsed " << data.kinematic_states.size() << " kinematic states" << std::endl;
  std::cout << "Parsed " << data.accelerations.size() << " acceleration messages" << std::endl;
  std::cout << "Parsed " << data.tracked_objects_msgs.size() << " tracked objects" << std::endl;
  std::cout << "Parsed " << data.route_msgs.size() << " route messages" << std::endl;
  std::cout << "Parsed " << data.turn_indicators.size() << " turn indicator messages" << std::endl;
  std::cout << "Parsed " << data.traffic_signals.size() << " traffic signal messages" << std::endl;

  return data;
}

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
