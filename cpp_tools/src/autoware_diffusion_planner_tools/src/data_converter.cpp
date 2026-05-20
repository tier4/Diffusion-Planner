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

#include "io/frame_writer.hpp"
#include "io/projector_factory.hpp"
#include "processing/ego_sequence.hpp"
#include "processing/neighbor_processor.hpp"
#include "rosbag_parser.hpp"
#include "timestamp_stats.hpp"
#include "types/frame_data.hpp"
#include "types/skipping_info.hpp"
#include "utils/timestamp_utils.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <autoware/diffusion_planner/constants.hpp>
#include <autoware/diffusion_planner/conversion/agent.hpp>
#include <autoware/diffusion_planner/conversion/lanelet.hpp>
#include <autoware/diffusion_planner/dimensions.hpp>
#include <autoware/diffusion_planner/preprocessing/lane_segments.hpp>
#include <autoware/diffusion_planner/preprocessing/preprocessing_utils.hpp>
#include <autoware/diffusion_planner/preprocessing/traffic_signals.hpp>
#include <autoware/diffusion_planner/utils/utils.hpp>
#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <rclcpp/rclcpp.hpp>

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

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace autoware::diffusion_planner;
using namespace autoware_perception_msgs::msg;
using namespace autoware_planning_msgs::msg;
using namespace autoware_vehicle_msgs::msg;
using namespace geometry_msgs::msg;
using namespace nav_msgs::msg;

int main(int argc, char ** argv)
{
  // Initialize for route handler functionality
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

  // Parse optional arguments
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

  // Load Lanelet2 map using projector chosen by map_projector_info.yaml.
  lanelet::ErrorMessages errors{};
  const std::unique_ptr<lanelet::Projector> projector =
    create_projector_from_yaml(vector_map_path);
  const std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr =
    lanelet::load(vector_map_path, *projector, &errors);

  std::cout << "Loaded lanelet2 map with " << lanelet_map_ptr->laneletLayer.size() << " lanelets"
            << std::endl;

  const preprocess::LaneSegmentContext lane_segment_context(lanelet_map_ptr);

  rosbag_parser::RosbagParser rosbag_parser(rosbag_path);
  rosbag_parser.create_reader(rosbag_path);

  // Parse messages from specific topics
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
      timestamp_stats_map.add_timestamp("/localization/kinematic_state", parse_timestamp(odometry.header.stamp), static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/localization/acceleration") {
      const AccelWithCovarianceStamped accel =
        rosbag_parser.deserialize_message<AccelWithCovarianceStamped>(msg);
      accelerations.push_back(accel);
      timestamp_stats_map.add_timestamp("/localization/acceleration", parse_timestamp(accel.header.stamp), static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/perception/object_recognition/tracking/objects") {
      const TrackedObjects objects = rosbag_parser.deserialize_message<TrackedObjects>(msg);
      tracked_objects_msgs.push_back(objects);
      timestamp_stats_map.add_timestamp("/perception/object_recognition/tracking/objects", parse_timestamp(objects.header.stamp), static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/planning/mission_planning/route") {
      const LaneletRoute route = rosbag_parser.deserialize_message<LaneletRoute>(msg);
      route_msgs.push_back(route);
      timestamp_stats_map.add_timestamp("/planning/mission_planning/route", parse_timestamp(route.header.stamp), static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/vehicle/status/turn_indicators_status") {
      const TurnIndicatorsReport turn_ind =
        rosbag_parser.deserialize_message<TurnIndicatorsReport>(msg);
      turn_indicators.push_back(turn_ind);
      timestamp_stats_map.add_timestamp("/vehicle/status/turn_indicators_status", parse_timestamp(turn_ind.stamp), static_cast<int64_t>(msg->time_stamp));
    } else if (msg->topic_name == "/perception/traffic_light_recognition/traffic_signals") {
      const TrafficLightGroupArray traffic_signal =
        rosbag_parser.deserialize_message<TrafficLightGroupArray>(msg);
      traffic_signals.push_back(traffic_signal);
      timestamp_stats_map.add_timestamp("/perception/traffic_light_recognition/traffic_signals", parse_timestamp(traffic_signal.stamp), static_cast<int64_t>(msg->time_stamp));
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

  // Create sequences based on tracked objects (base topic at 10Hz)
  std::vector<SequenceData> sequences;
  for (const LaneletRoute & route : route_msgs) {
    sequences.push_back({{}, route});
  }

  // Process each tracked objects message with synchronization like Python version
  const int64_t n = static_cast<int64_t>(tracked_objects_msgs.size());
  std::cout << "n=" << n << std::endl;

  for (int64_t i = 0; i < n; ++i) {
    const TrackedObjects & tracking = tracked_objects_msgs[i];
    const int64_t timestamp = parse_timestamp(tracking.header.stamp);

    // Find matching messages with synchronization check like Python version
    Odometry kinematic;
    AccelWithCovarianceStamped accel;
    std::vector<TrafficLightGroupArray> traffic_signal;
    TurnIndicatorsReport turn_ind;
    std::vector<std::string> incomplete_details;

    bool ok = true;

    // Check all messages
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
      // Tolerate drops: traffic_signal publisher is known to drop messages even in
      // healthy recordings (measured 23 gaps >150ms, up to 393ms, in this bag).
      // Keep the frame with an empty signal vector instead of failing it.
      std::cout << "No matching traffic_signal for tracked_objects at " << i << " (drop tolerated)"
                << std::endl;
    }

    const auto turn_ind_vec = check_and_update_msg(turn_indicators, tracking.header.stamp);
    if (!turn_ind_vec.empty()) {
      turn_ind = turn_ind_vec.back();
    } else {
      ok = false;
      incomplete_details.emplace_back("TurnIndicators");
      std::cout << "No matching turn_indicators for tracked_objects at " << i << std::endl;
    }

    // Check route
    int64_t max_route_index = -1;
    if (search_nearest_route) {
      // Find the latest route msg
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
      // Use the first route msg
      max_route_index = 0;
    }

    // Check kinematic_state covariance validation
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

    // Handle frame based on validation result
    if (!ok) {
      if (sequence.data_list.empty()) {
        // At the beginning of recording, some msgs may be missing - Skip this frame
        // Convert incomplete_details (vector<string>) to vector<IncompleteDataType>
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
          save_dir, rosbag_dir_name, create_token(max_route_index >= 0 ? max_route_index : 0, i),
          fallback_kinematic, timestamp, skipping_info);
        std::cout << "Skip this frame i=" << i << "/n=" << n << std::endl;
        continue;
      } else {
        // If the msg is missing in the middle of recording, we can use the msgs to this point
        std::cout << "Finish at this frame i=" << i << "/n=" << n << std::endl;
        break;
      }
    }

    // Shift kinematic pose to center
    // kinematic.pose.pose = utils::shift_x(kinematic.pose.pose, (ego_wheel_base / 2.0));

    const FrameData frame_data{timestamp, tracking, kinematic, accel, traffic_signal, turn_ind};

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

  // Sort each sequence's data_list by timestamp to ensure ascending order
  for (auto & seq : sequences) {
    std::sort(
      seq.data_list.begin(), seq.data_list.end(),
      [](const FrameData & a, const FrameData & b) { return a.timestamp < b.timestamp; });
  }

  const int64_t sequence_num = static_cast<int64_t>(sequences.size());
  std::cout << "Total " << sequence_num << " sequences" << std::endl;

  // Process sequences
  for (int64_t seq_id = 0; seq_id < static_cast<int64_t>(sequences.size()); ++seq_id) {
    SequenceData & seq = sequences[seq_id];
    const int64_t n = static_cast<int64_t>(seq.data_list.size());

    std::cout << "Processing sequence " << seq_id + 1 << "/" << sequences.size() << " with " << n
              << " frames" << std::endl;

    std::ostringstream seq_id_stream;
    seq_id_stream << "sequence_" << std::setfill('0') << std::setw(8) << seq_id;
    const std::string sequence_id_str = seq_id_stream.str();
    const int64_t start_ts = seq.data_list.empty() ? 0 : seq.data_list.front().timestamp;
    const int64_t end_ts = seq.data_list.empty() ? 0 : seq.data_list.back().timestamp;

    // Calculate the traveled distance
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

    // Replace the goal pose with the last frame's pose
    seq.route.goal_pose = seq.data_list.back().kinematic_state.pose.pose;

    // Process frames with stopping count tracking
    int64_t stopping_count = 0;
    for (int64_t i = INPUT_T_WITH_CURRENT; i < n; i += step) {
      // Create token in canonical format: seq_id(8digits) + "_" + i(8digits)
      const std::string token = create_token(seq_id, i);

      // Get transformation matrix
      const Eigen::Matrix4d bl2map =
        utils::pose_to_matrix4d(seq.data_list[i].kinematic_state.pose.pose);
      const Eigen::Matrix4d map2bl = utils::inverse(bl2map);

      // Create ego sequences
      const rclcpp::Time past_reference_time(seq.data_list[i].kinematic_state.header.stamp);
      const auto ego_past_opt = create_ego_sequence(
        seq.data_list, i - INPUT_T_WITH_CURRENT + 1, INPUT_T_WITH_CURRENT, map2bl,
        past_reference_time, use_interpolation);
      if (!ego_past_opt) {
        std::cout << "Failed to create ego past at frame " << i << std::endl;
        break;
      }
      const std::vector<float> & ego_past = ego_past_opt.value();

      const rclcpp::Time future_reference_time =
        past_reference_time +
        rclcpp::Duration::from_seconds(OUTPUT_T * constants::PREDICTION_TIME_STEP_S);
      const auto ego_future_opt = create_ego_sequence(
        seq.data_list, i + 1, OUTPUT_T, map2bl, future_reference_time, use_interpolation);
      if (!ego_future_opt) {
        std::cout << "Reached end of sequence at frame " << i << "/" << n << std::endl;
        break;
      }
      const std::vector<float> & ego_future = ego_future_opt.value();

      // Create ego current state
      const std::vector<float> ego_current = preprocess::create_ego_current_state(
        seq.data_list[i].kinematic_state, seq.data_list[i].acceleration, ego_wheel_base);

      // Process neighbor agents (both past and future with consistent agent ordering)
      const auto [neighbor_past, neighbor_future] =
        process_neighbor_agents_and_future(seq.data_list, i, map2bl);

      // Process lanes and routes
      const Point & ego_pos = seq.data_list[i].kinematic_state.pose.pose.position;
      const double center_x = ego_pos.x;
      const double center_y = ego_pos.y;
      const double center_z = ego_pos.z;

      // Process traffic signals for this frame using the traffic signals from FrameData
      std::map<lanelet::Id, preprocess::TrafficSignalStamped> traffic_light_id_map;
      const auto current_stamp = seq.data_list[i].tracked_objects.header.stamp;
      const rclcpp::Time current_time(current_stamp);

      std::vector<autoware_perception_msgs::msg::TrafficLightGroupArray::ConstSharedPtr> msg_vec;
      for (const auto & traffic_signal_msg : seq.data_list[i].traffic_signals) {
        msg_vec.push_back(
          std::make_shared<autoware_perception_msgs::msg::TrafficLightGroupArray>(
            traffic_signal_msg));
      }
      preprocess::process_traffic_signals(msg_vec, traffic_light_id_map, current_time, 5.0);

      // Get lanes data with speed limits
      const std::vector<int64_t> lane_segment_indices =
        lane_segment_context.select_lane_segment_indices(
          map2bl, center_x, center_y, NUM_SEGMENTS_IN_LANE);
      const auto [lanes, lanes_speed_limit] = lane_segment_context.create_tensor_data_from_indices(
        map2bl, traffic_light_id_map, lane_segment_indices, NUM_SEGMENTS_IN_LANE);

      // Create has_speed_limit flags based on speed_limit values
      std::vector<bool> lanes_has_speed_limit(lanes_speed_limit.size());
      for (size_t idx = 0; idx < lanes_speed_limit.size(); ++idx) {
        lanes_has_speed_limit[idx] =
          (lanes_speed_limit[idx] > std::numeric_limits<float>::epsilon());
      }

      // Get route lanes data with speed limits
      const std::vector<int64_t> segment_indices =
        lane_segment_context.select_route_segment_indices(
          seq.route, center_x, center_y, center_z, NUM_SEGMENTS_IN_ROUTE);
      const auto [route_lanes, route_lanes_speed_limit] =
        lane_segment_context.create_tensor_data_from_indices(
          map2bl, traffic_light_id_map, segment_indices, NUM_SEGMENTS_IN_ROUTE);

      // Create route_lanes_has_speed_limit based on speed_limit values
      std::vector<bool> route_lanes_has_speed_limit(route_lanes_speed_limit.size());
      for (size_t idx = 0; idx < route_lanes_speed_limit.size(); ++idx) {
        route_lanes_has_speed_limit[idx] =
          (route_lanes_speed_limit[idx] > std::numeric_limits<float>::epsilon());
      }

      const std::vector<float> polygons =
        lane_segment_context.create_polygon_tensor(map2bl, center_x, center_y);
      const std::vector<float> line_strings =
        lane_segment_context.create_line_string_tensor(map2bl, center_x, center_y);

      // Get goal pose
      const geometry_msgs::msg::Pose & goal_pose = seq.route.goal_pose;
      const Eigen::Matrix4d goal_pose_in_map = utils::pose_to_matrix4d(goal_pose);
      const Eigen::Matrix4d goal_pose_in_bl = map2bl * goal_pose_in_map;
      const float goal_x = goal_pose_in_bl(0, 3);
      const float goal_y = goal_pose_in_bl(1, 3);
      const float yaw = std::atan2(goal_pose_in_bl(1, 0), goal_pose_in_bl(0, 0));
      const std::vector<float> goal_pose_vec = {goal_x, goal_y, std::cos(yaw), std::sin(yaw)};

      // Such data should be skipped.
      // (1)Ego vehicle is stopped
      // (2)The lanelet segment in front is a red light
      // (3)The GT trajectory is moving forward.

      // (1)Ego vehicle is stopped
      const bool is_stop = seq.data_list[i].kinematic_state.twist.twist.linear.x < 0.1;
      if (is_stop) {
        stopping_count++;
      } else {
        stopping_count = 0;
      }

      // if ego vehicle is stopped and close to goal, finish
      const float ego_future_last_x = ego_future[(OUTPUT_T - 1) * 4 + 0];
      const float ego_future_last_y = ego_future[(OUTPUT_T - 1) * 4 + 1];
      const float distance_to_goal_pose = std::sqrt(
        (ego_future_last_x - goal_x) * (ego_future_last_x - goal_x) +
        (ego_future_last_y - goal_y) * (ego_future_last_y - goal_y));

      if (stopping_count > INPUT_T && distance_to_goal_pose < 5.0) {
        std::cout << "finish at " << i << " because stopping_count=" << stopping_count
                  << " and distance_to_goal_pose=" << distance_to_goal_pose << std::endl;
        break;
      }

      // Check for red light (next segment)
      // route_tensor[:, 1, 0, -3] corresponds to second segment, first point, red light flag
      const int64_t segment_idx = 1;  // next segment
      const int64_t point_idx = 0;    // first point
      const int64_t red_light_index = segment_idx * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM +
                                      point_idx * SEGMENT_POINT_DIM + TRAFFIC_LIGHT_RED;
      const bool is_red_light = route_lanes[red_light_index] > 0.5 && !convert_red;
      const int64_t yellow_light_index = segment_idx * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM +
                                         point_idx * SEGMENT_POINT_DIM + TRAFFIC_LIGHT_YELLOW;
      const bool is_yellow_light = route_lanes[yellow_light_index] > 0.5 && !convert_yellow;
      const bool is_red_or_yellow = is_red_light || is_yellow_light;

      float sum_mileage = 0.0;
      for (int64_t j = 0; j < OUTPUT_T - 1; ++j) {
        const float dx = ego_future[(j + 1) * 4 + 0] - ego_future[j * 4 + 0];
        const float dy = ego_future[(j + 1) * 4 + 1] - ego_future[j * 4 + 1];
        sum_mileage += std::sqrt(dx * dx + dy * dy);
      }
      const bool is_future_forward = sum_mileage > 1.0;

      // Create placeholder data for static objects
      const std::vector<float> static_objects(
        STATIC_OBJECTS_SHAPE[1] * STATIC_OBJECTS_SHAPE[2], 0.0f);

      // const int64_t turn_indicator = seq.data_list[i].turn_indicator.report;
      std::vector<int32_t> turn_indicators(INPUT_T_WITH_CURRENT);
      for (int64_t t = 0; t < INPUT_T_WITH_CURRENT; ++t) {
        turn_indicators[t] = seq.data_list[std::max(int64_t(0), i - INPUT_T_WITH_CURRENT + 1 + t)]
                               .turn_indicator.report;
      }

      if (is_stop && is_red_or_yellow && is_future_forward) {
        std::cout << "Skip this frame " << i
                  << " because it is stop at red or yellow light and future trajectory is forward"
                  << std::endl;
        save_frame_json(
          save_dir, rosbag_dir_name, token, seq.data_list[i].kinematic_state,
          seq.data_list[i].timestamp, SkippingInfo::red_or_yellow_light());
        continue;
      }
      if (stopping_count > (INPUT_T + 5) && is_red_or_yellow) {
        std::cout << "Skip this frame " << i << " because stopping_count=" << stopping_count
                  << " and red or yellow light" << std::endl;
        save_frame_json(
          save_dir, rosbag_dir_name, token, seq.data_list[i].kinematic_state,
          seq.data_list[i].timestamp, SkippingInfo::vehicle_stopped());
        continue;
      }

      save_frame_data(
        save_dir, rosbag_dir_name, token, ego_past, ego_current, ego_future, neighbor_past,
        neighbor_future, static_objects, lanes, lanes_speed_limit, lanes_has_speed_limit,
        route_lanes, route_lanes_speed_limit, route_lanes_has_speed_limit, polygons, line_strings,
        goal_pose_vec, turn_indicators, ego_shape);
      save_frame_json(
        save_dir, rosbag_dir_name, token, seq.data_list[i].kinematic_state,
        seq.data_list[i].timestamp, SkippingInfo::accepted());

      if (i % 100 == 0) {
        std::cout << "Processed frame " << i << "/" << n << std::endl;
      }
    }
  }

  std::cout << "Data conversion completed!" << std::endl;

  rclcpp::shutdown();
}
