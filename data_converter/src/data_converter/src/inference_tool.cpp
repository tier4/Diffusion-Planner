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

#include "rosbag_parser.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <autoware/diffusion_planner/diffusion_planner_core.hpp>
#include <autoware/diffusion_planner/preprocessing/preprocessing_utils.hpp>
#include <autoware/diffusion_planner/utils/utils.hpp>
#include <autoware/vehicle_info_utils/vehicle_info.hpp>
#include <autoware_lanelet2_extension/projection/mgrs_projector.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <lanelet2_io/Io.h>
#include <rcl_yaml_param_parser/parser.h>

#include <deque>
#include <iostream>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace autoware::diffusion_planner;
using namespace autoware_perception_msgs::msg;
using namespace autoware_planning_msgs::msg;
using namespace autoware_vehicle_msgs::msg;
using namespace geometry_msgs::msg;
using namespace nav_msgs::msg;

// --- Substitution resolution for yaml parameter strings ---
std::string resolve_substitutions(const std::string & str)
{
  std::string result = str;

  // Resolve $(env VAR)
  std::regex env_re(R"(\$\(env\s+(\w+)\))");
  std::smatch match;
  while (std::regex_search(result, match, env_re)) {
    const char * val = std::getenv(match[1].str().c_str());
    result = match.prefix().str() + (val ? val : "") + match.suffix().str();
  }

  // Resolve $(find-pkg-share PKG)
  std::regex pkg_re(R"(\$\(find-pkg-share\s+([\w-]+)\))");
  while (std::regex_search(result, match, pkg_re)) {
    const std::string pkg_dir = ament_index_cpp::get_package_share_directory(match[1].str());
    result = match.prefix().str() + pkg_dir + match.suffix().str();
  }

  return result;
}

// --- Parameter loading from yaml ---
using ParamMap = std::unordered_map<std::string, rclcpp::Parameter>;

ParamMap load_param_map(const std::string & yaml_path)
{
  rcl_params_t * params_st = rcl_yaml_node_struct_init(rcl_get_default_allocator());
  if (!rcl_parse_yaml_file(yaml_path.c_str(), params_st)) {
    std::cerr << "Failed to parse yaml file: " << yaml_path << std::endl;
    std::exit(1);
  }

  const rclcpp::ParameterMap param_map = rclcpp::parameter_map_from(params_st, "");
  rcl_yaml_node_struct_fini(params_st);

  ParamMap flat_map;
  for (const auto & [ns, params] : param_map) {
    for (const auto & p : params) {
      flat_map[p.get_name()] = p;
    }
  }
  return flat_map;
}

template <typename T>
T get_param(const ParamMap & params, const std::string & name, const T & default_val)
{
  const auto it = params.find(name);
  if (it == params.end()) return default_val;
  return it->second.get_value<T>();
}

DiffusionPlannerParams read_planner_params(const ParamMap & params)
{
  DiffusionPlannerParams p;
  p.model_path = resolve_substitutions(get_param<std::string>(params, "onnx_model_path", ""));
  p.args_path = resolve_substitutions(get_param<std::string>(params, "args_path", ""));
  p.plugins_path = resolve_substitutions(get_param<std::string>(params, "plugins_path", ""));
  p.build_only = false;
  p.planning_frequency_hz = get_param<double>(params, "planning_frequency_hz", 10.0);
  p.ignore_neighbors = get_param<bool>(params, "ignore_neighbors", false);
  p.ignore_unknown_neighbors = get_param<bool>(params, "ignore_unknown_neighbors", false);
  p.predict_neighbor_trajectory = get_param<bool>(params, "predict_neighbor_trajectory", false);
  p.traffic_light_group_msg_timeout_seconds =
    get_param<double>(params, "traffic_light_group_msg_timeout_seconds", 0.2);
  p.batch_size = static_cast<int>(get_param<int64_t>(params, "batch_size", 1));
  p.temperature_list = get_param<std::vector<double>>(params, "temperature", {0.0});
  p.velocity_smoothing_window = get_param<int64_t>(params, "velocity_smoothing_window", 8);
  p.stopping_threshold = get_param<double>(params, "stopping_threshold", 0.3);
  p.turn_indicator_keep_offset =
    static_cast<float>(get_param<double>(params, "turn_indicator_keep_offset", -1.25));
  p.turn_indicator_hold_duration = get_param<double>(params, "turn_indicator_hold_duration", 0.0);
  p.shift_x = get_param<bool>(params, "shift_x", false);
  return p;
}

// --- Topic names ---
constexpr const char * TOPIC_KINEMATIC_STATE = "/localization/kinematic_state";
constexpr const char * TOPIC_ACCELERATION = "/localization/acceleration";
constexpr const char * TOPIC_TRACKED_OBJECTS = "/perception/object_recognition/tracking/objects";
constexpr const char * TOPIC_TRAFFIC_SIGNALS =
  "/perception/traffic_light_recognition/traffic_signals";
constexpr const char * TOPIC_TURN_INDICATORS = "/vehicle/status/turn_indicators_status";
constexpr const char * TOPIC_ROUTE = "/planning/mission_planning/route";

constexpr const char * TOPIC_OUT_TRAJECTORY = "/diffusion_planner/output/trajectory";
constexpr const char * TOPIC_OUT_TRAJECTORIES = "/diffusion_planner/output/trajectories";
constexpr const char * TOPIC_OUT_PREDICTED_OBJECTS = "/diffusion_planner/output/predicted_objects";
constexpr const char * TOPIC_OUT_TURN_INDICATORS = "/diffusion_planner/output/turn_indicators";

constexpr int64_t SYNC_WINDOW_NS = 200'000'000;  // 200ms

int64_t to_nanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1'000'000'000LL + stamp.nanosec;
}

// --- Message synchronization (same approach as data_converter) ---
template <typename T, typename StampExtractor>
std::optional<T> find_nearest(
  std::deque<T> & msgs, int64_t target_ns, StampExtractor get_stamp)
{
  int64_t best_idx = -1;
  int64_t best_diff = SYNC_WINDOW_NS + 1;

  for (int64_t i = 0; i < static_cast<int64_t>(msgs.size()); ++i) {
    const int64_t msg_ns = to_nanoseconds(get_stamp(msgs[i]));
    const int64_t diff = target_ns - msg_ns;
    if (diff < 0) break;  // future message
    if (diff <= SYNC_WINDOW_NS && diff < best_diff) {
      best_diff = diff;
      best_idx = i;
    }
  }

  if (best_idx < 0) return std::nullopt;

  T result = msgs[best_idx];
  msgs.erase(msgs.begin(), msgs.begin() + best_idx);
  return result;
}

template <typename T, typename StampExtractor>
std::vector<std::shared_ptr<const T>> collect_within_window(
  std::deque<T> & msgs, int64_t target_ns, StampExtractor get_stamp)
{
  std::vector<std::shared_ptr<const T>> result;
  int64_t last_consumed = -1;

  for (int64_t i = 0; i < static_cast<int64_t>(msgs.size()); ++i) {
    const int64_t msg_ns = to_nanoseconds(get_stamp(msgs[i]));
    const int64_t diff = target_ns - msg_ns;
    if (diff < 0) break;
    if (diff <= SYNC_WINDOW_NS) {
      result.push_back(std::make_shared<const T>(msgs[i]));
    }
    last_consumed = i;
  }

  if (last_consumed >= 0) {
    msgs.erase(msgs.begin(), msgs.begin() + last_consumed);
  }
  return result;
}

int main(int argc, char ** argv)
{
  std::cout << "Diffusion Planner Inference Tool" << std::endl;

  if (argc < 4) {
    std::cerr
      << "Usage: diffusion_planner_inference_tool <rosbag_path> <vector_map_path> <output_path>\n"
      << "  [--vehicle_model_path=<path>] [--step=1] [--limit=-1]\n";
    return 1;
  }

  const std::string rosbag_path = argv[1];
  const std::string vector_map_path = argv[2];
  const std::string output_path = argv[3];

  std::string vehicle_model_path =
    ament_index_cpp::get_package_share_directory("autoware_vehicle_info_utils") +
    "/config/vehicle_info.param.yaml";
  int64_t step = 1;
  int64_t limit = -1;

  for (int i = 4; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.find("--vehicle_model_path=") == 0) {
      vehicle_model_path = arg.substr(21);
    } else if (arg.find("--step=") == 0) {
      step = std::stoll(arg.substr(7));
    } else if (arg.find("--limit=") == 0) {
      limit = std::stoll(arg.substr(8));
    }
  }

  // --- 1. Load parameters ---
  std::cout << "Loading parameters..." << std::endl;
  const std::string planner_yaml_path =
    ament_index_cpp::get_package_share_directory("autoware_diffusion_planner") +
    "/config/diffusion_planner.param.yaml";
  auto param_map = load_param_map(planner_yaml_path);
  for (const auto & [k, v] : load_param_map(vehicle_model_path)) {
    param_map[k] = v;
  }

  const auto params = read_planner_params(param_map);
  const auto vehicle_info = autoware::vehicle_info_utils::createVehicleInfo(
    get_param<double>(param_map, "wheel_radius", 0.0),
    get_param<double>(param_map, "wheel_width", 0.0),
    get_param<double>(param_map, "wheel_base", 0.0),
    get_param<double>(param_map, "wheel_tread", 0.0),
    get_param<double>(param_map, "front_overhang", 0.0),
    get_param<double>(param_map, "rear_overhang", 0.0),
    get_param<double>(param_map, "left_overhang", 0.0),
    get_param<double>(param_map, "right_overhang", 0.0),
    get_param<double>(param_map, "vehicle_height", 0.0),
    get_param<double>(param_map, "max_steer_angle", 0.0));

  std::cout << "  model_path: " << params.model_path << std::endl;
  std::cout << "  args_path: " << params.args_path << std::endl;
  std::cout << "  batch_size: " << params.batch_size << std::endl;

  // --- 2. Load lanelet map ---
  std::cout << "Loading lanelet map: " << vector_map_path << std::endl;
  lanelet::ErrorMessages errors;
  lanelet::projection::MGRSProjector projector;
  const std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr =
    lanelet::load(vector_map_path, projector, &errors);
  if (!errors.empty()) {
    for (const auto & e : errors) {
      std::cerr << "  Map warning: " << e << std::endl;
    }
  }

  // --- 3. Create core and load model ---
  std::cout << "Loading model..." << std::endl;
  DiffusionPlannerCore core(params, vehicle_info);
  core.load_model();
  core.set_map(lanelet_map_ptr);
  std::cout << "Model loaded successfully." << std::endl;

  const auto generator_uuid = autoware_utils_uuid::generate_uuid();

  // --- 4. Read input rosbag ---
  std::cout << "Reading rosbag: " << rosbag_path << std::endl;
  rosbag_parser::RosbagParser parser(rosbag_path);
  parser.create_reader(rosbag_path);

  std::deque<Odometry> odometry_msgs;
  std::deque<AccelWithCovarianceStamped> acceleration_msgs;
  std::deque<TrackedObjects> tracked_objects_msgs;
  std::deque<TrafficLightGroupArray> traffic_signal_msgs;
  std::deque<TurnIndicatorsReport> turn_indicator_msgs;
  std::deque<LaneletRoute> route_msgs;

  // Store raw messages for pass-through
  std::vector<rosbag2_storage::SerializedBagMessageSharedPtr> raw_messages;

  int64_t read_count = 0;
  while (parser.has_next() && (limit < 0 || read_count < limit)) {
    const auto msg = parser.read_next();
    raw_messages.push_back(msg);

    const auto & topic = msg->topic_name;
    if (topic == TOPIC_KINEMATIC_STATE) {
      odometry_msgs.push_back(parser.deserialize_message<Odometry>(msg));
    } else if (topic == TOPIC_ACCELERATION) {
      acceleration_msgs.push_back(parser.deserialize_message<AccelWithCovarianceStamped>(msg));
    } else if (topic == TOPIC_TRACKED_OBJECTS) {
      tracked_objects_msgs.push_back(parser.deserialize_message<TrackedObjects>(msg));
    } else if (topic == TOPIC_TRAFFIC_SIGNALS) {
      traffic_signal_msgs.push_back(parser.deserialize_message<TrafficLightGroupArray>(msg));
    } else if (topic == TOPIC_TURN_INDICATORS) {
      turn_indicator_msgs.push_back(parser.deserialize_message<TurnIndicatorsReport>(msg));
    } else if (topic == TOPIC_ROUTE) {
      route_msgs.push_back(parser.deserialize_message<LaneletRoute>(msg));
    }
    ++read_count;
  }

  std::cout << "  Odometry: " << odometry_msgs.size()
            << ", Acceleration: " << acceleration_msgs.size()
            << ", TrackedObjects: " << tracked_objects_msgs.size()
            << ", TrafficSignals: " << traffic_signal_msgs.size()
            << ", TurnIndicators: " << turn_indicator_msgs.size()
            << ", Route: " << route_msgs.size() << std::endl;

  if (odometry_msgs.empty()) {
    std::cerr << "No odometry messages found. Exiting." << std::endl;
    return 1;
  }

  // --- 5. Create output rosbag ---
  std::cout << "Creating output rosbag: " << output_path << std::endl;
  rosbag_parser::RosbagParser writer_parser(rosbag_path);
  writer_parser.create_writer(output_path);

  // Create pass-through topics
  for (const auto & topic_meta : writer_parser.get_all_topic_data()) {
    writer_parser.create_topic(topic_meta);
  }

  // Create output topics
  writer_parser.create_topic(TOPIC_OUT_TRAJECTORY, "autoware_planning_msgs/msg/Trajectory");
  writer_parser.create_topic(
    TOPIC_OUT_TRAJECTORIES, "autoware_internal_planning_msgs/msg/CandidateTrajectories");
  writer_parser.create_topic(
    TOPIC_OUT_PREDICTED_OBJECTS, "autoware_perception_msgs/msg/PredictedObjects");
  writer_parser.create_topic(
    TOPIC_OUT_TURN_INDICATORS, "autoware_vehicle_msgs/msg/TurnIndicatorsCommand");

  // Pass-through all raw input messages
  for (const auto & msg : raw_messages) {
    writer_parser.write_topic(msg);
  }
  raw_messages.clear();  // free memory

  // --- 6. Process frames ---
  std::cout << "Processing frames (step=" << step << ")..." << std::endl;

  auto accel_stamp = [](const AccelWithCovarianceStamped & m) { return m.header.stamp; };
  auto objects_stamp = [](const TrackedObjects & m) { return m.header.stamp; };
  auto traffic_stamp = [](const TrafficLightGroupArray & m) { return m.stamp; };
  auto turn_stamp = [](const TurnIndicatorsReport & m) { return m.stamp; };

  LaneletRoute::SharedPtr current_route;
  if (!route_msgs.empty()) {
    current_route = std::make_shared<LaneletRoute>(route_msgs.front());
  }

  int64_t total_frames = 0;
  int64_t processed_frames = 0;
  int64_t skipped_frames = 0;
  int64_t failed_frames = 0;

  for (size_t odom_idx = 0; odom_idx < odometry_msgs.size();
       odom_idx += static_cast<size_t>(step)) {
    ++total_frames;

    const auto & odom = odometry_msgs[odom_idx];
    const int64_t target_ns = to_nanoseconds(odom.header.stamp);

    // Update sticky route
    while (!route_msgs.empty()) {
      const int64_t route_ns = to_nanoseconds(route_msgs.front().header.stamp);
      if (route_ns > target_ns) break;
      current_route = std::make_shared<LaneletRoute>(route_msgs.front());
      route_msgs.pop_front();
    }

    // Synchronize
    const auto accel = find_nearest(acceleration_msgs, target_ns, accel_stamp);
    const auto objects = find_nearest(tracked_objects_msgs, target_ns, objects_stamp);
    const auto turn_ind = find_nearest(turn_indicator_msgs, target_ns, turn_stamp);
    const auto traffic_signals =
      collect_within_window(traffic_signal_msgs, target_ns, traffic_stamp);

    if (!accel || !objects || !turn_ind || !current_route) {
      ++skipped_frames;
      continue;
    }

    // Create shared_ptrs for core API
    const auto odom_ptr = std::make_shared<const Odometry>(odom);
    const auto accel_ptr = std::make_shared<const AccelWithCovarianceStamped>(*accel);
    const auto objects_ptr = std::make_shared<const TrackedObjects>(*objects);
    const auto turn_ind_ptr = std::make_shared<const TurnIndicatorsReport>(*turn_ind);
    const auto route_ptr = std::const_pointer_cast<const LaneletRoute>(current_route);

    const rclcpp::Time current_time(odom.header.stamp);

    // Core pipeline
    const auto frame_context = core.create_frame_context(
      odom_ptr, accel_ptr, objects_ptr, traffic_signals, turn_ind_ptr, route_ptr, current_time);

    if (!frame_context) {
      ++skipped_frames;
      continue;
    }

    auto input_data_map = core.create_input_data(*frame_context);
    preprocess::normalize_input_data(input_data_map, core.get_normalization_map());

    if (!utils::check_input_map(input_data_map)) {
      ++skipped_frames;
      continue;
    }

    const auto inference_result = core.run_inference(input_data_map);
    if (!inference_result.outputs) {
      std::cerr << "  Frame " << total_frames << ": inference failed - "
                << inference_result.error_msg << std::endl;
      ++failed_frames;
      continue;
    }

    const auto & [predictions, turn_indicator_logit] = inference_result.outputs.value();
    const rclcpp::Time frame_time(frame_context->frame_time);

    PlannerOutput planner_output;
    try {
      planner_output = core.create_planner_output(
        predictions, turn_indicator_logit, *frame_context, frame_time, generator_uuid);
    } catch (const std::exception & e) {
      std::cerr << "  Frame " << total_frames << ": postprocessing failed - " << e.what()
                << std::endl;
      ++failed_frames;
      continue;
    }

    // Write output to rosbag
    writer_parser.write_topic(planner_output.trajectory, frame_time, TOPIC_OUT_TRAJECTORY);
    writer_parser.write_topic(
      planner_output.candidate_trajectories, frame_time, TOPIC_OUT_TRAJECTORIES);
    writer_parser.write_topic(
      planner_output.predicted_objects, frame_time, TOPIC_OUT_PREDICTED_OBJECTS);
    writer_parser.write_topic(
      planner_output.turn_indicator_command, frame_time, TOPIC_OUT_TURN_INDICATORS);

    ++processed_frames;

    if (processed_frames % 100 == 0) {
      std::cout << "  Processed " << processed_frames << " frames..." << std::endl;
    }
  }

  // --- 7. Summary ---
  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "  Total frames:     " << total_frames << std::endl;
  std::cout << "  Processed:        " << processed_frames << std::endl;
  std::cout << "  Skipped:          " << skipped_frames << std::endl;
  std::cout << "  Failed:           " << failed_frames << std::endl;
  std::cout << "  Output written to: " << output_path << std::endl;

  return 0;
}
