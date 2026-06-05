// Copyright 2025 TIER IV, Inc.
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

#include "data_types.hpp"
#include "frame_processor.hpp"
#include "io.hpp"
#include "rosbag_parser.hpp"
#include "sequence_builder.hpp"
#include "timestamp_stats.hpp"

#include <autoware/diffusion_planner/preprocessing/lane_segments.hpp>
#include <autoware_lanelet2_extension/projection/mgrs_projector.hpp>
#include <autoware_lanelet2_extension/projection/transverse_mercator_projector.hpp>
#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_io/Io.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace autoware::diffusion_planner;
using namespace autoware_perception_msgs::msg;
using namespace autoware_planning_msgs::msg;
using namespace autoware_vehicle_msgs::msg;
using namespace geometry_msgs::msg;
using namespace nav_msgs::msg;

std::unique_ptr<lanelet::Projector> create_projector_from_yaml(
  const std::string & vector_map_path)
{
  const std::filesystem::path map_path_fs(vector_map_path);
  const std::filesystem::path projector_info_yaml =
    map_path_fs.parent_path() / "map_projector_info.yaml";
  if (!std::filesystem::exists(projector_info_yaml)) {
    std::cerr << "WARNING: map_projector_info.yaml not found at " << projector_info_yaml
              << ". Falling back to MGRSProjector (previous default)." << std::endl;
    return std::make_unique<lanelet::projection::MGRSProjector>();
  }

  const YAML::Node data = YAML::LoadFile(projector_info_yaml.string());
  const std::string projector_type = data["projector_type"].as<std::string>();

  if (projector_type == "MGRS") {
    auto mgrs_projector = std::make_unique<lanelet::projection::MGRSProjector>();
    mgrs_projector->setMGRSCode(data["mgrs_grid"].as<std::string>());
    return mgrs_projector;
  }
  if (projector_type == "TransverseMercator") {
    const double lat = data["map_origin"]["latitude"].as<double>();
    const double lon = data["map_origin"]["longitude"].as<double>();
    const double scale_factor = data["scale_factor"].as<double>();
    const lanelet::GPSPoint position{lat, lon, 0.0};
    const lanelet::Origin origin{position};
    return std::make_unique<lanelet::projection::TransverseMercatorProjector>(origin, scale_factor);
  }
  throw std::runtime_error(
    "Unsupported projector_type in map_projector_info.yaml: " + projector_type +
    " (supported: MGRS, TransverseMercator)");
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  if (argc < 4) {
    std::cerr << "Usage: data_converter <rosbag_path> <vector_map_path> <save_dir> [--step=1] "
                 "[--limit=-1] [--min_frames=1700] [--min_distance=50.0] [--convert_yellow=0] "
                 "[--convert_red=0] [--interpolation=1] "
                 "[--ego_wheel_base=2.75] [--ego_length=4.34] [--ego_width=1.70]"
              << std::endl;
    return 1;
  }

  const std::string rosbag_path = argv[1];
  const std::string vector_map_path = argv[2];
  const std::string save_dir = argv[3];
  const std::string rosbag_dir_name = std::filesystem::path(rosbag_path).filename();

  int64_t step = 1;
  int64_t limit = -1;
  int64_t min_frames = 1700;
  int64_t search_nearest_route = 1;
  int64_t convert_yellow = 0;
  int64_t convert_red = 0;
  int64_t interpolation = 1;
  double min_distance = 50.0;
  float ego_wheel_base = -1.0;
  float ego_length = -1.0;
  float ego_width = -1.0;

  for (int64_t i = 4; i < argc; ++i) {
    const std::string arg = argv[i];
    std::cout << "arg[" << i << "] = " << arg << std::endl;
    if (arg.find("--step=") == 0) {
      step = std::stoll(arg.substr(7));
    } else if (arg.find("--limit=") == 0) {
      limit = std::stoll(arg.substr(8));
    } else if (arg.find("--min_frames=") == 0) {
      min_frames = std::stoll(arg.substr(13));
    } else if (arg.find("--min_distance=") == 0) {
      min_distance = std::stod(arg.substr(15));
    } else if (arg.find("--search_nearest_route=") == 0) {
      search_nearest_route = std::stoll(arg.substr(23));
    } else if (arg.find("--convert_yellow=") == 0) {
      convert_yellow = std::stoll(arg.substr(17));
    } else if (arg.find("--convert_red=") == 0) {
      convert_red = std::stoll(arg.substr(14));
    } else if (arg.find("--interpolation=") == 0) {
      interpolation = std::stoll(arg.substr(16));
    } else if (arg.find("--ego_wheel_base=") == 0) {
      ego_wheel_base = std::stof(arg.substr(17));
    } else if (arg.find("--ego_length=") == 0) {
      ego_length = std::stof(arg.substr(13));
    } else if (arg.find("--ego_width=") == 0) {
      ego_width = std::stof(arg.substr(12));
    }
  }

  std::cout << "Ego wheel base: " << ego_wheel_base << ", Ego length: " << ego_length
            << ", Ego width: " << ego_width << std::endl;
  if (ego_wheel_base < 0.0 || ego_length < 0.0 || ego_width < 0.0) {
    std::cerr << "Ego vehicle dimensions must be specified with positive values." << std::endl;
    return 1;
  }
  const std::vector<float> ego_shape = {ego_wheel_base, ego_length, ego_width};

  std::cout << "Processing rosbag: " << rosbag_path << std::endl;
  std::cout << "Vector map: " << vector_map_path << std::endl;
  std::cout << "Save directory: " << save_dir << std::endl;
  const bool use_interpolation = static_cast<bool>(interpolation);
  std::cout << "Step: " << step << ", Limit: " << limit << ", Min frames: " << min_frames
            << ", Min distance: " << min_distance
            << ", Search nearest route: " << search_nearest_route
            << ", Convert yellow: " << convert_yellow << ", Convert red: " << convert_red
            << ", Interpolation: " << use_interpolation << std::endl;

  // Load Lanelet2 map
  lanelet::ErrorMessages errors{};
  const std::unique_ptr<lanelet::Projector> projector =
    create_projector_from_yaml(vector_map_path);
  const std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr =
    lanelet::load(vector_map_path, *projector, &errors);
  std::cout << "Loaded lanelet2 map with " << lanelet_map_ptr->laneletLayer.size() << " lanelets"
            << std::endl;

  const preprocess::LaneSegmentContext lane_segment_context(lanelet_map_ptr);

  // Parse rosbag messages
  rosbag_parser::RosbagParser rosbag_parser(rosbag_path);
  rosbag_parser.create_reader(rosbag_path);

  std::deque<Odometry> kinematic_states;
  std::deque<AccelWithCovarianceStamped> accelerations;
  std::deque<TrackedObjects> tracked_objects_msgs;
  std::deque<TurnIndicatorsReport> turn_indicators;
  std::vector<LaneletRoute> route_msgs;
  std::deque<TrafficLightGroupArray> traffic_signals;

  const std::vector<std::string> target_topics = {
    "/localization/kinematic_state",
    "/localization/acceleration",
    "/perception/object_recognition/tracking/objects",
    "/planning/mission_planning/route",
    "/vehicle/status/turn_indicators_status",
    "/perception/traffic_light_recognition/traffic_signals"};

  timestamp_stats::TimestampStatsMap timestamp_stats_map(target_topics);

  int64_t parse_count = 0;
  while (rosbag_parser.has_next() && (limit < 0 || parse_count < limit)) {
    const rosbag2_storage::SerializedBagMessageSharedPtr msg = rosbag_parser.read_next();

    if (msg->topic_name == "/localization/kinematic_state") {
      const Odometry odometry = rosbag_parser.deserialize_message<Odometry>(msg);
      kinematic_states.push_back(odometry);
      timestamp_stats_map.add_timestamp(
        "/localization/kinematic_state", parse_timestamp(odometry.header.stamp),
        static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/localization/acceleration") {
      const AccelWithCovarianceStamped accel =
        rosbag_parser.deserialize_message<AccelWithCovarianceStamped>(msg);
      accelerations.push_back(accel);
      timestamp_stats_map.add_timestamp(
        "/localization/acceleration", parse_timestamp(accel.header.stamp),
        static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/perception/object_recognition/tracking/objects") {
      const TrackedObjects objects = rosbag_parser.deserialize_message<TrackedObjects>(msg);
      tracked_objects_msgs.push_back(objects);
      timestamp_stats_map.add_timestamp(
        "/perception/object_recognition/tracking/objects",
        parse_timestamp(objects.header.stamp), static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/planning/mission_planning/route") {
      const LaneletRoute route = rosbag_parser.deserialize_message<LaneletRoute>(msg);
      route_msgs.push_back(route);
      timestamp_stats_map.add_timestamp(
        "/planning/mission_planning/route", parse_timestamp(route.header.stamp),
        static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/vehicle/status/turn_indicators_status") {
      const TurnIndicatorsReport turn_ind =
        rosbag_parser.deserialize_message<TurnIndicatorsReport>(msg);
      turn_indicators.push_back(turn_ind);
      timestamp_stats_map.add_timestamp(
        "/vehicle/status/turn_indicators_status", parse_timestamp(turn_ind.stamp),
        static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/perception/traffic_light_recognition/traffic_signals") {
      const TrafficLightGroupArray traffic_signal =
        rosbag_parser.deserialize_message<TrafficLightGroupArray>(msg);
      traffic_signals.push_back(traffic_signal);
      timestamp_stats_map.add_timestamp(
        "/perception/traffic_light_recognition/traffic_signals",
        parse_timestamp(traffic_signal.stamp), static_cast<int64_t>(msg->time_stamp));
    }

    parse_count++;
  }

  timestamp_stats_map.analyze_all();

  std::cout << "Parsed " << kinematic_states.size() << " kinematic states" << std::endl;
  std::cout << "Parsed " << accelerations.size() << " acceleration messages" << std::endl;
  std::cout << "Parsed " << tracked_objects_msgs.size() << " tracked objects" << std::endl;
  std::cout << "Parsed " << route_msgs.size() << " route messages" << std::endl;
  std::cout << "Parsed " << turn_indicators.size() << " turn indicator messages" << std::endl;
  std::cout << "Parsed " << traffic_signals.size() << " traffic signal messages" << std::endl;

  // Check for missing required topics
  std::vector<std::string> missing_topics;
  std::vector<MissingTopicType> missing_topic_types;
  if (kinematic_states.empty()) {
    missing_topics.emplace_back("/localization/kinematic_state");
    missing_topic_types.push_back(MissingTopicType::KinematicState);
  }
  if (accelerations.empty()) {
    missing_topics.emplace_back("/localization/acceleration");
    missing_topic_types.push_back(MissingTopicType::Acceleration);
  }
  if (tracked_objects_msgs.empty()) {
    missing_topics.emplace_back("/perception/object_recognition/tracking/objects");
    missing_topic_types.push_back(MissingTopicType::TrackedObjects);
  }
  if (route_msgs.empty()) {
    missing_topics.emplace_back("/planning/mission_planning/route");
    missing_topic_types.push_back(MissingTopicType::Route);
  }
  if (turn_indicators.empty()) {
    missing_topics.emplace_back("/vehicle/status/turn_indicators_status");
    missing_topic_types.push_back(MissingTopicType::TurnIndicators);
  }
  if (traffic_signals.empty()) {
    missing_topics.emplace_back("/perception/traffic_light_recognition/traffic_signals");
    missing_topic_types.push_back(MissingTopicType::TrafficSignals);
  }

  if (!missing_topics.empty()) {
    std::cout << "Skipping rosbag " << rosbag_path
              << " due to missing required topics:" << std::endl;
    for (const auto & topic : missing_topics) {
      std::cout << "  - " << topic << std::endl;
    }
    std::cout << "No training samples will be generated from this rosbag." << std::endl;
    save_route_json(
      save_dir, rosbag_dir_name, "missing_topics", 0, 0.0, 0, 0,
      SkippingInfo::missing_topics(missing_topic_types), timestamp_stats_map);
    rclcpp::shutdown();
    return 0;
  }

  // Build, merge, and sort sequences
  std::vector<SequenceData> sequences = build_sequences(
    tracked_objects_msgs, kinematic_states, accelerations, traffic_signals, turn_indicators,
    route_msgs, static_cast<bool>(search_nearest_route), save_dir, rosbag_dir_name);

  merge_sequences(sequences);
  sort_sequences(sequences);

  const int64_t sequence_num = static_cast<int64_t>(sequences.size());
  std::cout << "Total " << sequence_num << " sequences" << std::endl;

  // Process each sequence
  for (int64_t seq_id = 0; seq_id < sequence_num; ++seq_id) {
    SequenceData & seq = sequences[seq_id];
    const int64_t n = static_cast<int64_t>(seq.data_list.size());

    std::cout << "Processing sequence " << seq_id + 1 << "/" << sequence_num << " with " << n
              << " frames" << std::endl;

    std::ostringstream seq_id_stream;
    seq_id_stream << "sequence_" << std::setfill('0') << std::setw(8) << seq_id;
    const std::string sequence_id_str = seq_id_stream.str();
    const int64_t start_ts = seq.data_list.empty() ? 0 : seq.data_list.front().timestamp;
    const int64_t end_ts = seq.data_list.empty() ? 0 : seq.data_list.back().timestamp;

    double traveled_distance = 0.0;
    for (int64_t i = 1; i < n; ++i) {
      const auto & pos1 = seq.data_list[i - 1].kinematic_state.pose.pose.position;
      const auto & pos2 = seq.data_list[i].kinematic_state.pose.pose.position;
      const double dx = pos2.x - pos1.x;
      const double dy = pos2.y - pos1.y;
      traveled_distance += std::sqrt(dx * dx + dy * dy);
    }

    if (n < min_frames) {
      std::cout << "Skipping sequence with only " << n << " frames (min: " << min_frames << ")"
                << std::endl;
      save_route_json(
        save_dir, rosbag_dir_name, sequence_id_str, n, traveled_distance, start_ts, end_ts,
        SkippingInfo::insufficient_frames(n, min_frames), timestamp_stats_map);
      continue;
    }

    std::cout << "Traveled distance: " << traveled_distance << " meters" << std::endl;
    if (traveled_distance < min_distance) {
      std::cout << "Skipping sequence with traveled distance " << traveled_distance
                << " meters (min: " << min_distance << " meters)" << std::endl;
      save_route_json(
        save_dir, rosbag_dir_name, sequence_id_str, n, traveled_distance, start_ts, end_ts,
        SkippingInfo::insufficient_distance(traveled_distance, min_distance), timestamp_stats_map);
      continue;
    }

    save_route_json(
      save_dir, rosbag_dir_name, sequence_id_str, n, traveled_distance, start_ts, end_ts,
      SkippingInfo::accepted(), timestamp_stats_map);

    process_sequence(
      seq, seq_id, lane_segment_context, step, use_interpolation, convert_red, convert_yellow,
      ego_wheel_base, ego_shape, save_dir, rosbag_dir_name);
  }

  std::cout << "Data conversion completed!" << std::endl;

  rclcpp::shutdown();
  return 0;
}
