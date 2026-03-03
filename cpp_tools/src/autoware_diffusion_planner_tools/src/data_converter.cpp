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

#include "rosbag_parser.hpp"

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
#include <autoware_lanelet2_extension/projection/mgrs_projector.hpp>
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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace autoware::diffusion_planner;
using namespace autoware_perception_msgs::msg;
using namespace autoware_planning_msgs::msg;
using namespace autoware_vehicle_msgs::msg;
using namespace geometry_msgs::msg;
using namespace nav_msgs::msg;

// Using constants from dimensions.hpp
constexpr int64_t NEIGHBOR_PAST_DIM = NEIGHBOR_SHAPE[3];
constexpr int64_t NEIGHBOR_FUTURE_DIM = 4;  // x, y, cos(yaw), sin(yaw)

struct FrameData
{
  int64_t timestamp;
  TrackedObjects tracked_objects;
  Odometry kinematic_state;
  AccelWithCovarianceStamped acceleration;
  std::vector<TrafficLightGroupArray> traffic_signals;
  TurnIndicatorsReport turn_indicator;
};

struct SequenceData
{
  std::vector<FrameData> data_list;
  LaneletRoute route;
};

// Training data structure for binary file (all fixed size)
struct TrainingDataBinary
{
  // Header information
  uint32_t version;  // Data format version

  // Fixed size data arrays
  float ego_agent_past[EGO_HISTORY_SHAPE[1] * EGO_HISTORY_SHAPE[2]];
  float ego_current_state[EGO_CURRENT_STATE_SHAPE[1]];
  float ego_agent_future[OUTPUT_T * EGO_HISTORY_SHAPE[2]];
  float neighbor_agents_past[MAX_NUM_NEIGHBORS * INPUT_T_WITH_CURRENT * NEIGHBOR_PAST_DIM];
  float neighbor_agents_future[MAX_NUM_NEIGHBORS * OUTPUT_T * NEIGHBOR_FUTURE_DIM];
  float static_objects[STATIC_OBJECTS_SHAPE[1] * STATIC_OBJECTS_SHAPE[2]];
  float lanes[NUM_SEGMENTS_IN_LANE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM];
  float lanes_speed_limit[NUM_SEGMENTS_IN_LANE];
  int32_t lanes_has_speed_limit[NUM_SEGMENTS_IN_LANE];
  float route_lanes[NUM_SEGMENTS_IN_ROUTE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM];
  float route_lanes_speed_limit[NUM_SEGMENTS_IN_ROUTE];
  int32_t route_lanes_has_speed_limit[NUM_SEGMENTS_IN_ROUTE];
  float polygons[NUM_POLYGONS * POINTS_PER_POLYGON * (2 + POLYGON_TYPE_NUM)];
  float line_strings[NUM_LINE_STRINGS * POINTS_PER_LINE_STRING * (2 + LINE_STRING_TYPE_NUM)];
  float goal_pose[NEIGHBOR_FUTURE_DIM];
  int32_t turn_indicators[INPUT_T_WITH_CURRENT];
  float ego_shape[EGO_SHAPE_SHAPE[1]];

  // Constructor with zero initialization
  TrainingDataBinary() : version(2)
  {
    std::fill(std::begin(ego_agent_past), std::end(ego_agent_past), 0.0f);
    std::fill(std::begin(ego_current_state), std::end(ego_current_state), 0.0f);
    std::fill(std::begin(ego_agent_future), std::end(ego_agent_future), 0.0f);
    std::fill(std::begin(neighbor_agents_past), std::end(neighbor_agents_past), 0.0f);
    std::fill(std::begin(neighbor_agents_future), std::end(neighbor_agents_future), 0.0f);
    std::fill(std::begin(static_objects), std::end(static_objects), 0.0f);
    std::fill(std::begin(lanes), std::end(lanes), 0.0f);
    std::fill(std::begin(lanes_speed_limit), std::end(lanes_speed_limit), 0.0f);
    std::fill(std::begin(lanes_has_speed_limit), std::end(lanes_has_speed_limit), 0);
    std::fill(std::begin(route_lanes), std::end(route_lanes), 0.0f);
    std::fill(std::begin(route_lanes_speed_limit), std::end(route_lanes_speed_limit), 0.0f);
    std::fill(std::begin(route_lanes_has_speed_limit), std::end(route_lanes_has_speed_limit), 0);
    std::fill(std::begin(polygons), std::end(polygons), 0.0f);
    std::fill(std::begin(line_strings), std::end(line_strings), 0.0f);
    std::fill(std::begin(goal_pose), std::end(goal_pose), 0.0f);
    std::fill(std::begin(turn_indicators), std::end(turn_indicators), 0);
    std::fill(std::begin(ego_shape), std::end(ego_shape), 0.0f);
  }
};

int64_t parse_timestamp(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}

template <typename T>
std::vector<T> check_and_update_msg(
  std::deque<T> & msgs, const builtin_interfaces::msg::Time & target_stamp)
{
  const int64_t target_time = parse_timestamp(target_stamp);
  std::vector<T> result;
  int64_t best_index = -1;

  for (int64_t i = 0; i < static_cast<int64_t>(msgs.size()); ++i) {
    const auto & msg = msgs[i];
    builtin_interfaces::msg::Time msg_stamp;
    if constexpr (
      std::is_same_v<T, Odometry> || std::is_same_v<T, TrackedObjects> ||
      std::is_same_v<T, AccelWithCovarianceStamped>) {
      msg_stamp = msg.header.stamp;
    } else if constexpr (
      std::is_same_v<T, TurnIndicatorsReport> || std::is_same_v<T, TrafficLightGroupArray>) {
      msg_stamp = msg.stamp;
    }

    const int64_t msg_time = parse_timestamp(msg_stamp);
    const int64_t time_diff = target_time - msg_time;

    // Only consider past messages, break if future message is encountered
    if (time_diff < 0) {
      break;
    }

    if (time_diff <= static_cast<int64_t>(2e8)) {  // 200 msec within loop
      result.push_back(msg);                       // collect all within threshold
      best_index = i;
    }
  }

  // Remove processed messages up to the selected index
  if (best_index >= 0) {
    msgs.erase(msgs.begin(), msgs.begin() + best_index);
  }
  return result;
}

std::vector<float> create_ego_sequence(
  const std::vector<FrameData> & data_list, const int64_t start_idx, const int64_t time_steps,
  const Eigen::Matrix4d & map2bl_matrix)
{
  // Extract pose messages from FrameData
  std::deque<nav_msgs::msg::Odometry> odom_deque;
  for (int64_t j = 0; j < time_steps; ++j) {
    const int64_t index = std::min(start_idx + j, static_cast<int64_t>(data_list.size()) - 1);
    odom_deque.push_back(data_list[index].kinematic_state);
  }
  return preprocess::create_ego_agent_past(odom_deque, time_steps, map2bl_matrix);
}

std::pair<std::vector<float>, std::vector<float>> process_neighbor_agents_and_future(
  const std::vector<FrameData> & data_list, const int64_t current_idx,
  const Eigen::Matrix4d & map2bl_matrix)
{
  // Build agent histories using AgentData::update_histories
  const int64_t start_idx =
    std::max(static_cast<int64_t>(0), current_idx - INPUT_T_WITH_CURRENT + 1);
  const bool ignore_unknown_agents = true;
  autoware::diffusion_planner::AgentData agent_data_past;
  for (int64_t t = 0; t < INPUT_T_WITH_CURRENT; ++t) {
    const int64_t frame_idx = start_idx + t;
    if (frame_idx >= static_cast<int64_t>(data_list.size())) {
      break;
    }
    agent_data_past.update_histories(data_list[frame_idx].tracked_objects, ignore_unknown_agents);
  }
  const auto transformed_histories =
    agent_data_past.transformed_and_trimmed_histories(map2bl_matrix, MAX_NUM_NEIGHBORS);
  const std::vector<float> neighbor_past =
    flatten_histories_to_vector(transformed_histories, MAX_NUM_NEIGHBORS, INPUT_T_WITH_CURRENT);

  // Build id -> AgentHistory map for future filling
  const std::vector<AgentHistory> agent_histories = transformed_histories;
  std::unordered_map<std::string, AgentHistory> id_to_history;
  for (size_t i = 0; i < agent_histories.size(); ++i) {
    const auto object_id = agent_histories[i].get_latest_state().object_id;
    id_to_history.emplace(object_id, AgentHistory(OUTPUT_T));
    id_to_history.at(object_id).update(
      agent_histories[i].get_latest_state().original_info,
      agent_histories[i].get_latest_state().timestamp);
  }

  // Future data: use AgentHistory for each agent
  std::vector<float> neighbor_future(MAX_NUM_NEIGHBORS * OUTPUT_T * NEIGHBOR_FUTURE_DIM, 0.0f);
  for (int64_t agent_idx = 0; agent_idx < static_cast<int64_t>(agent_histories.size());
       ++agent_idx) {
    const std::string & agent_id_str = agent_histories[agent_idx].get_latest_state().object_id;
    AgentHistory & future_history = id_to_history.at(agent_id_str);
    for (int64_t t = 1; t <= OUTPUT_T; ++t) {
      const int64_t future_frame_idx = current_idx + t;
      if (future_frame_idx >= static_cast<int64_t>(data_list.size())) {
        break;
      }
      // Find object with same id in future frame
      const auto & future_objects = data_list[future_frame_idx].tracked_objects.objects;
      bool found = false;
      for (const auto & obj : future_objects) {
        const std::string obj_id = autoware_utils_uuid::to_hex_string(obj.object_id);
        if (obj_id == agent_id_str) {
          future_history.update(obj, data_list[future_frame_idx].kinematic_state.header.stamp);
          found = true;
          break;
        }
      }
      if (!found) {
        break;
      }
    }
    future_history.apply_transform(map2bl_matrix);

    // Fill future array for this agent
    const std::vector<float> arr = future_history.as_array();
    for (int64_t t = 0; t < OUTPUT_T; ++t) {
      const int64_t base_idx = agent_idx * OUTPUT_T * NEIGHBOR_FUTURE_DIM + t * NEIGHBOR_FUTURE_DIM;
      for (int64_t d = 0; d < NEIGHBOR_FUTURE_DIM; ++d) {
        if (t * AGENT_STATE_DIM + d >= arr.size()) {
          break;
        }
        neighbor_future[base_idx + d] = arr[t * AGENT_STATE_DIM + d];
      }
    }
  }

  return std::make_pair(neighbor_past, neighbor_future);
}

void save_binary_data(
  const std::string & output_path, const std::string & rosbag_dir_name, const std::string & token,
  const std::vector<float> & ego_past, const std::vector<float> & ego_current,
  const std::vector<float> & ego_future, const std::vector<float> & neighbor_past,
  const std::vector<float> & neighbor_future, const std::vector<float> & static_objects,
  const std::vector<float> & lanes, const std::vector<float> & lanes_speed_limit,
  const std::vector<bool> & lanes_has_speed_limit, const std::vector<float> & route_lanes,
  const std::vector<float> & route_lanes_speed_limit,
  const std::vector<bool> & route_lanes_has_speed_limit, const std::vector<float> & polygons,
  const std::vector<float> & line_strings, const std::vector<float> & goal_pose,
  const std::vector<int32_t> & turn_indicators, const std::vector<float> & ego_shape,
  const Odometry & kinematic_state, const int64_t timestamp)
{
  namespace fs = std::filesystem;

  fs::create_directories(output_path);

  // Copy data to struct
  TrainingDataBinary data;

  // Copy vector data to fixed arrays
  std::copy(ego_past.begin(), ego_past.end(), data.ego_agent_past);
  std::copy(ego_current.begin(), ego_current.end(), data.ego_current_state);
  std::copy(ego_future.begin(), ego_future.end(), data.ego_agent_future);
  std::copy(neighbor_past.begin(), neighbor_past.end(), data.neighbor_agents_past);
  std::copy(neighbor_future.begin(), neighbor_future.end(), data.neighbor_agents_future);
  std::copy(static_objects.begin(), static_objects.end(), data.static_objects);
  std::copy(lanes.begin(), lanes.end(), data.lanes);
  std::copy(lanes_speed_limit.begin(), lanes_speed_limit.end(), data.lanes_speed_limit);
  std::copy(route_lanes.begin(), route_lanes.end(), data.route_lanes);
  std::copy(
    route_lanes_speed_limit.begin(), route_lanes_speed_limit.end(), data.route_lanes_speed_limit);
  std::copy(polygons.begin(), polygons.end(), data.polygons);
  std::copy(line_strings.begin(), line_strings.end(), data.line_strings);
  std::copy(goal_pose.begin(), goal_pose.end(), data.goal_pose);

  // Convert bool to int32_t and copy
  for (size_t i = 0; i < lanes_has_speed_limit.size(); ++i) {
    data.lanes_has_speed_limit[i] = static_cast<int32_t>(lanes_has_speed_limit[i]);
  }
  for (size_t i = 0; i < route_lanes_has_speed_limit.size(); ++i) {
    data.route_lanes_has_speed_limit[i] = static_cast<int32_t>(route_lanes_has_speed_limit[i]);
  }

  std::copy(turn_indicators.begin(), turn_indicators.end(), data.turn_indicators);
  std::copy(ego_shape.begin(), ego_shape.end(), data.ego_shape);

  // Save to binary file with rosbag directory name prefix (same format as Python version)
  const std::string binary_filename = output_path + "/" + rosbag_dir_name + "_" + token + ".bin";
  std::ofstream file(binary_filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << binary_filename << std::endl;
    return;
  }

  file.write(reinterpret_cast<const char *>(&data), sizeof(TrainingDataBinary));

  if (file.fail()) {
    std::cerr << "Failed to write data to file: " << binary_filename << std::endl;
    return;
  }

  file.close();

  // Save JSON file with pose information (same as Python version)
  const std::string json_filename = output_path + "/" + rosbag_dir_name + "_" + token + ".json";
  std::ofstream json_file(json_filename);
  if (json_file.is_open()) {
    json_file << std::fixed << std::setprecision(15);
    json_file << "{\n";
    json_file << "  \"timestamp\": " << timestamp << ",\n";
    json_file << "  \"x\": " << kinematic_state.pose.pose.position.x << ",\n";
    json_file << "  \"y\": " << kinematic_state.pose.pose.position.y << ",\n";
    json_file << "  \"z\": " << kinematic_state.pose.pose.position.z << ",\n";
    json_file << "  \"qx\": " << kinematic_state.pose.pose.orientation.x << ",\n";
    json_file << "  \"qy\": " << kinematic_state.pose.pose.orientation.y << ",\n";
    json_file << "  \"qz\": " << kinematic_state.pose.pose.orientation.z << ",\n";
    json_file << "  \"qw\": " << kinematic_state.pose.pose.orientation.w << "\n";
    json_file << "}\n";
    json_file.close();
  }
}

int main(int argc, char ** argv)
{
  // Initialize for route handler functionality
  rclcpp::init(argc, argv);

  if (argc < 4) {
    std::cerr << "Usage: data_converter <rosbag_path> <vector_map_path> <save_dir> [--step=1] "
                 "[--limit=-1] [--min_frames=1700] [--min_distance=50.0] [--convert_yellow=0] "
                 "[--convert_red=0] "
                 "[--ego_wheel_base=2.75] [--ego_length=4.34] [--ego_width=1.70]"
              << std::endl;
    return 1;
  }

  const std::string rosbag_path = argv[1];
  const std::string vector_map_path = argv[2];
  const std::string save_dir = argv[3];

  int64_t step = 1;
  int64_t limit = -1;
  int64_t min_frames = 1700;
  int64_t search_nearest_route = 1;
  int64_t convert_yellow = 0;
  int64_t convert_red = 0;
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
  std::cout << "Step: " << step << ", Limit: " << limit << ", Min frames: " << min_frames
            << ", Min distance: " << min_distance
            << ", Search nearest route: " << search_nearest_route
            << ", Convert yellow: " << convert_yellow << ", Convert red: " << convert_red
            << std::endl;

  // Load Lanelet2 map and create context like in diffusion_planner_node
  lanelet::ErrorMessages errors{};
  lanelet::projection::MGRSProjector projector{};
  const std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr =
    lanelet::load(vector_map_path, projector, &errors);

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

  int64_t parse_count = 0;
  while (rosbag_parser.has_next() && (limit < 0 || parse_count < limit)) {
    const rosbag2_storage::SerializedBagMessageSharedPtr msg = rosbag_parser.read_next();

    if (msg->topic_name == "/localization/kinematic_state") {
      const Odometry odometry = rosbag_parser.deserialize_message<Odometry>(msg);
      kinematic_states.push_back(odometry);
    } else if (msg->topic_name == "/localization/acceleration") {
      const AccelWithCovarianceStamped accel =
        rosbag_parser.deserialize_message<AccelWithCovarianceStamped>(msg);
      accelerations.push_back(accel);
    } else if (msg->topic_name == "/perception/object_recognition/tracking/objects") {
      const TrackedObjects objects = rosbag_parser.deserialize_message<TrackedObjects>(msg);
      tracked_objects_msgs.push_back(objects);
    } else if (msg->topic_name == "/planning/mission_planning/route") {
      const LaneletRoute route = rosbag_parser.deserialize_message<LaneletRoute>(msg);
      route_msgs.push_back(route);
    } else if (msg->topic_name == "/vehicle/status/turn_indicators_status") {
      const TurnIndicatorsReport turn_ind =
        rosbag_parser.deserialize_message<TurnIndicatorsReport>(msg);
      turn_indicators.push_back(turn_ind);
    } else if (msg->topic_name == "/perception/traffic_light_recognition/traffic_signals") {
      const TrafficLightGroupArray traffic_signal =
        rosbag_parser.deserialize_message<TrafficLightGroupArray>(msg);
      traffic_signals.push_back(traffic_signal);
    }

    parse_count++;
  }

  std::cout << "Parsed " << kinematic_states.size() << " kinematic states" << std::endl;
  std::cout << "Parsed " << accelerations.size() << " acceleration messages" << std::endl;
  std::cout << "Parsed " << tracked_objects_msgs.size() << " tracked objects" << std::endl;
  std::cout << "Parsed " << route_msgs.size() << " route messages" << std::endl;
  std::cout << "Parsed " << turn_indicators.size() << " turn indicator messages" << std::endl;
  std::cout << "Parsed " << traffic_signals.size() << " traffic signal messages" << std::endl;

  std::vector<std::string> missing_topics;
  if (kinematic_states.empty()) {
    missing_topics.emplace_back("/localization/kinematic_state");
  }
  if (accelerations.empty()) {
    missing_topics.emplace_back("/localization/acceleration");
  }
  if (tracked_objects_msgs.empty()) {
    missing_topics.emplace_back("/perception/object_recognition/tracking/objects");
  }
  if (route_msgs.empty()) {
    missing_topics.emplace_back("/planning/mission_planning/route");
  }
  if (turn_indicators.empty()) {
    missing_topics.emplace_back("/vehicle/status/turn_indicators_status");
  }
  if (traffic_signals.empty()) {
    missing_topics.emplace_back("/perception/traffic_light_recognition/traffic_signals");
  }

  if (!missing_topics.empty()) {
    std::cout << "Skipping rosbag " << rosbag_path
              << " due to missing required topics:" << std::endl;
    for (const auto & topic : missing_topics) {
      std::cout << "  - " << topic << std::endl;
    }
    std::cout << "No training samples will be generated from this rosbag." << std::endl;
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

    bool ok = true;

    // Check all messages
    const auto kinematic_vec = check_and_update_msg(kinematic_states, tracking.header.stamp);
    if (!kinematic_vec.empty()) {
      kinematic = kinematic_vec.back();
    } else {
      ok = false;
      std::cout << "No matching kinematic_state for tracked_objects at " << i << std::endl;
    }

    const auto accel_vec = check_and_update_msg(accelerations, tracking.header.stamp);
    if (!accel_vec.empty()) {
      accel = accel_vec.back();
    } else {
      ok = false;
      std::cout << "No matching acceleration for tracked_objects at " << i << std::endl;
    }

    const auto traffic_signal_vec = check_and_update_msg(traffic_signals, tracking.header.stamp);
    if (!traffic_signal_vec.empty()) {
      traffic_signal = traffic_signal_vec;
    } else {
      ok = false;
      std::cout << "No matching traffic_signal for tracked_objects at " << i << std::endl;
    }

    const auto turn_ind_vec = check_and_update_msg(turn_indicators, tracking.header.stamp);
    if (!turn_ind_vec.empty()) {
      turn_ind = turn_ind_vec.back();
    } else {
      ok = false;
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
      }
    }

    SequenceData & sequence = sequences[max_route_index];

    // Handle frame based on validation result
    if (!ok) {
      if (sequence.data_list.empty()) {
        // At the beginning of recording, some msgs may be missing - Skip this frame
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

  const std::string rosbag_dir_name = std::filesystem::path(rosbag_path).filename();
  const int64_t sequence_num = static_cast<int64_t>(sequences.size());
  std::cout << "Total " << sequence_num << " sequences" << std::endl;

  // Process sequences
  for (int64_t seq_id = 0; seq_id < static_cast<int64_t>(sequences.size()); ++seq_id) {
    SequenceData & seq = sequences[seq_id];
    const int64_t n = static_cast<int64_t>(seq.data_list.size());

    std::cout << "Processing sequence " << seq_id + 1 << "/" << sequences.size() << " with " << n
              << " frames" << std::endl;

    if (n < min_frames) {
      std::cout << "Skipping sequence with only " << n << " frames (min: " << min_frames << ")"
                << std::endl;
      continue;
    }

    // Calculate the traveled distance and skip if it's too short
    double traveled_distance = 0.0;
    for (int64_t i = 1; i < n; ++i) {
      const auto & pos1 = seq.data_list[i - 1].kinematic_state.pose.pose.position;
      const auto & pos2 = seq.data_list[i].kinematic_state.pose.pose.position;
      const double dx = pos2.x - pos1.x;
      const double dy = pos2.y - pos1.y;
      traveled_distance += std::sqrt(dx * dx + dy * dy);
    }
    std::cout << "Traveled distance: " << traveled_distance << " meters" << std::endl;
    if (traveled_distance < min_distance) {
      std::cout << "Skipping sequence with traveled distance " << traveled_distance
                << " meters (min: " << min_distance << " meters)" << std::endl;
      continue;
    }

    // Replace the goal pose with the last frame's pose
    seq.route.goal_pose = seq.data_list.back().kinematic_state.pose.pose;

    // Process frames with stopping count tracking
    int64_t stopping_count = 0;
    for (int64_t i = INPUT_T_WITH_CURRENT; i < n; i += step) {
      // Create token in same format as Python version: seq_id(8digits) + i(8digits)
      std::ostringstream token_stream;
      token_stream << std::setfill('0') << std::setw(8) << seq_id << std::setw(8) << i;
      const std::string token = token_stream.str();

      // Get transformation matrix
      const Eigen::Matrix4d bl2map =
        utils::pose_to_matrix4d(seq.data_list[i].kinematic_state.pose.pose);
      const Eigen::Matrix4d map2bl = utils::inverse(bl2map);

      // Create ego sequences
      const std::vector<float> ego_past = create_ego_sequence(
        seq.data_list, i - INPUT_T_WITH_CURRENT + 1, INPUT_T_WITH_CURRENT, map2bl);
      const std::vector<float> ego_future =
        create_ego_sequence(seq.data_list, i + 1, OUTPUT_T, map2bl);

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

      if (is_stop && is_red_or_yellow && is_future_forward) {
        std::cout << "Skip this frame " << i
                  << " because it is stop at red or yellow light and future trajectory is forward"
                  << std::endl;
        continue;
      }
      if (stopping_count > (INPUT_T + 5) && is_red_or_yellow) {
        std::cout << "Skip this frame " << i << " because stopping_count=" << stopping_count
                  << " and red or yellow light" << std::endl;
        continue;
      }

      // Create placeholder data for static objects
      const std::vector<float> static_objects(
        STATIC_OBJECTS_SHAPE[1] * STATIC_OBJECTS_SHAPE[2], 0.0f);

      // const int64_t turn_indicator = seq.data_list[i].turn_indicator.report;
      std::vector<int32_t> turn_indicators(INPUT_T_WITH_CURRENT);
      for (int64_t t = 0; t < INPUT_T_WITH_CURRENT; ++t) {
        turn_indicators[t] = seq.data_list[std::max(int64_t(0), i - INPUT_T_WITH_CURRENT + 1 + t)]
                               .turn_indicator.report;
      }

      // Save data
      save_binary_data(
        save_dir, rosbag_dir_name, token, ego_past, ego_current, ego_future, neighbor_past,
        neighbor_future, static_objects, lanes, lanes_speed_limit, lanes_has_speed_limit,
        route_lanes, route_lanes_speed_limit, route_lanes_has_speed_limit, polygons, line_strings,
        goal_pose_vec, turn_indicators, ego_shape, seq.data_list[i].kinematic_state,
        seq.data_list[i].timestamp);

      if (i % 100 == 0) {
        std::cout << "Processed frame " << i << "/" << n << std::endl;
      }
    }
  }

  std::cout << "Data conversion completed!" << std::endl;

  rclcpp::shutdown();
}
