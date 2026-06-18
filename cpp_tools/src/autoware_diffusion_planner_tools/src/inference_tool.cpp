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
#include <autoware/diffusion_planner/dimensions.hpp>
#include <autoware/diffusion_planner/preprocessing/preprocessing_utils.hpp>
#include <autoware/diffusion_planner/utils/marker_utils.hpp>
#include <autoware/diffusion_planner/utils/utils.hpp>
#include <autoware/vehicle_info_utils/vehicle_info.hpp>
#include <autoware_lanelet2_extension/projection/mgrs_projector.hpp>
#include <autoware_utils_geometry/boost_polygon_utils.hpp>
#include <autoware_utils_geometry/geometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/algorithms/intersects.hpp>

#include <lanelet2_io/Io.h>
#include <rcl_yaml_param_parser/parser.h>

#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
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
  if (it == params.end()) {
    return default_val;
  }
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
constexpr const char * TOPIC_OUT_GT_TRAJECTORY = "/diffusion_planner/output/gt_trajectory";
constexpr const char * TOPIC_OUT_DEBUG_ROUTE_MARKER = "/diffusion_planner/debug/route_marker";
constexpr const char * TOPIC_OUT_DEBUG_LANE_MARKER = "/diffusion_planner/debug/lane_marker";

constexpr int64_t SYNC_WINDOW_NS = 200'000'000;  // 200ms

int64_t to_nanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1'000'000'000LL + stamp.nanosec;
}

// --- Message synchronization (same approach as data_converter) ---
template <typename T, typename StampExtractor>
std::optional<T> find_nearest(std::deque<T> & msgs, int64_t target_ns, StampExtractor get_stamp)
{
  int64_t best_idx = -1;
  int64_t best_diff = SYNC_WINDOW_NS + 1;

  for (int64_t i = 0; i < static_cast<int64_t>(msgs.size()); ++i) {
    const int64_t msg_ns = to_nanoseconds(get_stamp(msgs[i]));
    const int64_t diff = target_ns - msg_ns;
    if (diff < 0) {
      break;  // future message
    }
    if (diff <= SYNC_WINDOW_NS && diff < best_diff) {
      best_diff = diff;
      best_idx = i;
    }
  }

  if (best_idx < 0) {
    return std::nullopt;
  }

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
    if (diff < 0) {
      break;
    }
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

// --- Odometry interpolation at a target time ---
// Uses linear interpolation for position/velocity and slerp for orientation.
// search_hint is updated to speed up subsequent calls with increasing target times.
Odometry interpolate_odometry(
  const std::deque<Odometry> & odom_msgs, int64_t target_ns, size_t & search_hint)
{
  auto stamp_sec = [](const Odometry & m) -> double {
    return static_cast<double>(m.header.stamp.sec) +
           static_cast<double>(m.header.stamp.nanosec) * 1e-9;
  };

  const double target_sec = static_cast<double>(target_ns) / 1e9;
  const double first_sec = stamp_sec(odom_msgs.front());
  const double last_sec = stamp_sec(odom_msgs.back());

  if (target_sec <= first_sec) {
    return odom_msgs.front();
  }
  if (target_sec >= last_sec) {
    return odom_msgs.back();
  }

  // Find bracketing odom messages, continuing from previous position
  for (; search_hint + 1 < odom_msgs.size(); ++search_hint) {
    const double t_next = stamp_sec(odom_msgs[search_hint + 1]);
    if (target_sec <= t_next) {
      break;
    }
  }

  const auto & odom0 = odom_msgs[search_hint];
  const auto & odom1 = odom_msgs[search_hint + 1];
  const double t0 = stamp_sec(odom0);
  const double t1 = stamp_sec(odom1);
  const double ratio = (t1 > t0) ? (target_sec - t0) / (t1 - t0) : 0.0;
  const double r = std::clamp(ratio, 0.0, 1.0);

  Odometry result;
  result.header = odom0.header;

  // Interpolate pose (position linear + orientation slerp)
  result.pose.pose =
    autoware_utils_geometry::calc_interpolated_pose(odom0.pose.pose, odom1.pose.pose, r, false);

  // Interpolate twist linearly
  result.twist.twist.linear.x =
    odom0.twist.twist.linear.x * (1.0 - r) + odom1.twist.twist.linear.x * r;
  result.twist.twist.linear.y =
    odom0.twist.twist.linear.y * (1.0 - r) + odom1.twist.twist.linear.y * r;
  result.twist.twist.linear.z =
    odom0.twist.twist.linear.z * (1.0 - r) + odom1.twist.twist.linear.z * r;
  result.twist.twist.angular.x =
    odom0.twist.twist.angular.x * (1.0 - r) + odom1.twist.twist.angular.x * r;
  result.twist.twist.angular.y =
    odom0.twist.twist.angular.y * (1.0 - r) + odom1.twist.twist.angular.y * r;
  result.twist.twist.angular.z =
    odom0.twist.twist.angular.z * (1.0 - r) + odom1.twist.twist.angular.z * r;

  return result;
}

int main(int argc, char ** argv)
{
  std::cout << "Diffusion Planner Inference Tool" << std::endl;

  if (argc < 4) {
    std::cerr
      << "Usage: diffusion_planner_inference_tool <rosbag_path> <vector_map_path> <output_path>\n"
      << "  [--vehicle_model_path=<path>] [--planner_config_path=<path>]\n"
      << "  [--step=1] [--limit=-1]\n"
      << "  [--metrics_output_path=<path>]  (output per-frame evaluation metrics CSV)\n"
      << "  [--<yaml_param_name>=<value>]  (override any parameter from the planner config YAML)\n";
    return 1;
  }

  const std::string rosbag_path = argv[1];
  const std::string vector_map_path = argv[2];
  const std::string output_path = argv[3];

  std::string vehicle_model_path =
    ament_index_cpp::get_package_share_directory("autoware_vehicle_info_utils") +
    "/config/vehicle_info.param.yaml";
  std::string planner_config_path =
    ament_index_cpp::get_package_share_directory("autoware_diffusion_planner") +
    "/config/diffusion_planner.param.yaml";
  int64_t step = 1;
  int64_t limit = -1;
  std::string metrics_output_path;

  // CLI overrides for planner parameters
  std::unordered_map<std::string, std::string> cli_overrides;

  for (int i = 4; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.find("--vehicle_model_path=") == 0) {
      vehicle_model_path = arg.substr(21);
    } else if (arg.find("--planner_config_path=") == 0) {
      planner_config_path = arg.substr(22);
    } else if (arg.find("--step=") == 0) {
      step = std::stoll(arg.substr(7));
    } else if (arg.find("--limit=") == 0) {
      limit = std::stoll(arg.substr(8));
    } else if (arg.find("--metrics_output_path=") == 0) {
      metrics_output_path = arg.substr(22);
    } else if (arg.substr(0, 2) == "--") {
      const auto eq_pos = arg.find('=');
      if (eq_pos != std::string::npos) {
        cli_overrides[arg.substr(2, eq_pos - 2)] = arg.substr(eq_pos + 1);
      }
    }
  }

  // --- 1. Load parameters ---
  std::cout << "Loading parameters..." << std::endl;
  std::cout << "  planner_config: " << planner_config_path << std::endl;
  std::cout << "  vehicle_model:  " << vehicle_model_path << std::endl;
  auto param_map = load_param_map(planner_config_path);
  for (const auto & [k, v] : load_param_map(vehicle_model_path)) {
    param_map[k] = v;
  }

  // Apply CLI overrides to param_map (type is inferred from existing YAML parameter)
  for (const auto & [key, value] : cli_overrides) {
    const auto it = param_map.find(key);
    if (it != param_map.end()) {
      switch (it->second.get_type()) {
        case rclcpp::ParameterType::PARAMETER_BOOL:
          param_map[key] = rclcpp::Parameter(key, value == "true" || value == "1");
          break;
        case rclcpp::ParameterType::PARAMETER_INTEGER:
          param_map[key] = rclcpp::Parameter(key, static_cast<int64_t>(std::stoll(value)));
          break;
        case rclcpp::ParameterType::PARAMETER_DOUBLE:
          param_map[key] = rclcpp::Parameter(key, std::stod(value));
          break;
        case rclcpp::ParameterType::PARAMETER_STRING:
          param_map[key] = rclcpp::Parameter(key, value);
          break;
        case rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY: {
          std::vector<double> vals;
          std::istringstream ss(value);
          std::string token;
          while (std::getline(ss, token, ',')) {
            vals.push_back(std::stod(token));
          }
          param_map[key] = rclcpp::Parameter(key, vals);
          break;
        }
        case rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY: {
          std::vector<int64_t> vals;
          std::istringstream ss(value);
          std::string token;
          while (std::getline(ss, token, ',')) {
            vals.push_back(std::stoll(token));
          }
          param_map[key] = rclcpp::Parameter(key, vals);
          break;
        }
        case rclcpp::ParameterType::PARAMETER_STRING_ARRAY: {
          std::vector<std::string> vals;
          std::istringstream ss(value);
          std::string token;
          while (std::getline(ss, token, ',')) {
            vals.push_back(token);
          }
          param_map[key] = rclcpp::Parameter(key, vals);
          break;
        }
        default:
          std::cerr << "  Warning: unsupported type for CLI override: " << key << std::endl;
          break;
      }
      std::cout << "  CLI override: " << key << " = " << value << std::endl;
    } else {
      std::cerr << "  Warning: unknown parameter: " << key << std::endl;
    }
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
  std::cout << "  plugins_path: " << params.plugins_path << std::endl;
  std::cout << "  planning_frequency_hz: " << params.planning_frequency_hz << std::endl;
  std::cout << "  ignore_neighbors: " << std::boolalpha << params.ignore_neighbors << std::endl;
  std::cout << "  ignore_unknown_neighbors: " << params.ignore_unknown_neighbors << std::endl;
  std::cout << "  traffic_light_group_msg_timeout_seconds: "
            << params.traffic_light_group_msg_timeout_seconds << std::endl;
  std::cout << "  batch_size: " << params.batch_size << std::endl;
  std::cout << "  temperature: [";
  for (size_t i = 0; i < params.temperature_list.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << params.temperature_list[i];
  }
  std::cout << "]" << std::endl;
  std::cout << "  velocity_smoothing_window: " << params.velocity_smoothing_window << std::endl;
  std::cout << "  stopping_threshold: " << params.stopping_threshold << std::endl;
  std::cout << "  turn_indicator_keep_offset: " << params.turn_indicator_keep_offset << std::endl;
  std::cout << "  turn_indicator_hold_duration: " << params.turn_indicator_hold_duration
            << std::endl;
  std::cout << "  shift_x: " << params.shift_x << std::endl;
  std::cout << "  vehicle_info:" << std::endl;
  std::cout << "    wheel_radius: " << vehicle_info.wheel_radius_m << std::endl;
  std::cout << "    wheel_base: " << vehicle_info.wheel_base_m << std::endl;
  std::cout << "    wheel_tread: " << vehicle_info.wheel_tread_m << std::endl;
  std::cout << "    front_overhang: " << vehicle_info.front_overhang_m << std::endl;
  std::cout << "    rear_overhang: " << vehicle_info.rear_overhang_m << std::endl;
  std::cout << "    left_overhang: " << vehicle_info.left_overhang_m << std::endl;
  std::cout << "    right_overhang: " << vehicle_info.right_overhang_m << std::endl;
  std::cout << "    vehicle_height: " << vehicle_info.vehicle_height_m << std::endl;
  std::cout << "    vehicle_length: " << vehicle_info.vehicle_length_m << std::endl;
  std::cout << "    vehicle_width: " << vehicle_info.vehicle_width_m << std::endl;
  std::cout << "    max_steer_angle: " << vehicle_info.max_steer_angle_rad << std::endl;

  // Ego footprint dimensions for metrics
  const double ego_base_to_front = vehicle_info.front_overhang_m + vehicle_info.wheel_base_m;
  const double ego_base_to_rear = vehicle_info.rear_overhang_m;
  const double ego_width = vehicle_info.vehicle_width_m;

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

  unique_identifier_msgs::msg::UUID generator_uuid{};
  generator_uuid.uuid.fill(0);

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
  writer_parser.create_topic(TOPIC_OUT_GT_TRAJECTORY, "autoware_planning_msgs/msg/Trajectory");
  writer_parser.create_topic(TOPIC_OUT_DEBUG_ROUTE_MARKER, "visualization_msgs/msg/MarkerArray");
  writer_parser.create_topic(TOPIC_OUT_DEBUG_LANE_MARKER, "visualization_msgs/msg/MarkerArray");

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

  const int64_t first_odom_ns = to_nanoseconds(odometry_msgs.front().header.stamp);
  const int64_t last_odom_ns = to_nanoseconds(odometry_msgs.back().header.stamp);
  const int64_t timer_interval_ns =
    static_cast<int64_t>(1.0e9 / params.planning_frequency_hz) * step;
  size_t odom_search_idx = 0;

  const int64_t expected_frames = (last_odom_ns - first_odom_ns) / timer_interval_ns + 1;
  std::cout << "  timer_interval: " << timer_interval_ns / 1'000'000 << "ms"
            << ", expected_frames: " << expected_frames << std::endl;

  // --- Metrics CSV output ---
  std::ofstream metrics_file;
  if (!metrics_output_path.empty()) {
    metrics_file.open(metrics_output_path);
    if (!metrics_file.is_open()) {
      std::cerr << "Failed to open metrics output: " << metrics_output_path << std::endl;
      return 1;
    }
    metrics_file << "timestamp_ns,ade,fde,min_road_border_dist,road_border_contact,"
                    "min_neighbor_dist,neighbor_collision\n";
  }

  for (int64_t timer_ns = first_odom_ns; timer_ns <= last_odom_ns; timer_ns += timer_interval_ns) {
    ++total_frames;

    // Find most recent odometry at or before timer tick
    for (size_t i = odom_search_idx + 1; i < odometry_msgs.size(); ++i) {
      if (to_nanoseconds(odometry_msgs[i].header.stamp) > timer_ns) {
        break;
      }
      odom_search_idx = i;
    }

    const auto & odom = odometry_msgs[odom_search_idx];

    // Update sticky route
    while (!route_msgs.empty()) {
      const int64_t route_ns = to_nanoseconds(route_msgs.front().header.stamp);
      if (route_ns > timer_ns) {
        break;
      }
      current_route = std::make_shared<LaneletRoute>(route_msgs.front());
      route_msgs.pop_front();
    }

    // Synchronize (use timer_ns as reference for all messages)
    const auto accel = find_nearest(acceleration_msgs, timer_ns, accel_stamp);
    const auto objects = find_nearest(tracked_objects_msgs, timer_ns, objects_stamp);
    const auto turn_ind = find_nearest(turn_indicator_msgs, timer_ns, turn_stamp);
    const auto traffic_signals =
      collect_within_window(traffic_signal_msgs, timer_ns, traffic_stamp);

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
    const rclcpp::Time frame_time(frame_context->frame_time);

    // Write debug markers before normalization
    {
      const auto lifetime = rclcpp::Duration::from_seconds(0.2);
      const auto route_markers = utils::create_lane_marker(
        frame_context->ego_to_map_transform, input_data_map.at("route_lanes"),
        std::vector<int64_t>(ROUTE_LANES_SHAPE.begin(), ROUTE_LANES_SHAPE.end()), frame_time,
        lifetime, {0.8f, 0.8f, 0.8f, 0.8f}, "map", true);
      writer_parser.write_topic(route_markers, frame_time, TOPIC_OUT_DEBUG_ROUTE_MARKER);

      const auto lane_markers = utils::create_lane_marker(
        frame_context->ego_to_map_transform, input_data_map.at("lanes"),
        std::vector<int64_t>(LANES_SHAPE.begin(), LANES_SHAPE.end()), frame_time, lifetime,
        {0.1f, 0.1f, 0.7f, 0.8f}, "map", true);
      writer_parser.write_topic(lane_markers, frame_time, TOPIC_OUT_DEBUG_LANE_MARKER);
    }

    // Extract road_border linestrings from input data (before normalization, in ego frame)
    // line_strings shape: [1, NUM_LINE_STRINGS(60), POINTS_PER_LINE_STRING(20),
    // 2+LINE_STRING_TYPE_NUM(4)] last dim: (x, y, stop_line_type, road_border_type)
    std::vector<autoware_utils_geometry::LineString2d> frame_road_borders;
    if (metrics_file.is_open()) {
      constexpr int64_t feat_dim = 2 + LINE_STRING_TYPE_NUM;                 // 4
      constexpr int64_t road_border_idx = 2 + LINE_STRING_TYPE_ROAD_BORDER;  // 3
      const auto & ls_data = input_data_map.at("line_strings");
      const Eigen::Matrix4d & ego_to_map = frame_context->ego_to_map_transform;

      for (int64_t ls_i = 0; ls_i < NUM_LINE_STRINGS; ++ls_i) {
        const int64_t ls_offset = ls_i * POINTS_PER_LINE_STRING * feat_dim;
        const float rb_flag = ls_data[ls_offset + road_border_idx];
        if (rb_flag < 0.5f) {
          continue;
        }

        autoware_utils_geometry::LineString2d ls2d;
        bool has_nonzero = false;
        for (int64_t pt_i = 0; pt_i < POINTS_PER_LINE_STRING; ++pt_i) {
          const int64_t pt_offset = ls_offset + pt_i * feat_dim;
          const double ex = static_cast<double>(ls_data[pt_offset + 0]);
          const double ey = static_cast<double>(ls_data[pt_offset + 1]);
          // Skip zero-padded points
          if (std::abs(ex) < 1e-6 && std::abs(ey) < 1e-6 && has_nonzero) {
            break;
          }
          if (std::abs(ex) > 1e-6 || std::abs(ey) > 1e-6) {
            has_nonzero = true;
          }
          // Transform ego frame -> map frame
          const Eigen::Vector4d ego_pt(ex, ey, 0.0, 1.0);
          const Eigen::Vector4d map_pt = ego_to_map * ego_pt;
          ls2d.emplace_back(map_pt(0), map_pt(1));
        }
        if (ls2d.size() >= 2) {
          frame_road_borders.push_back(std::move(ls2d));
        }
      }
    }

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

    // Build ground truth trajectory from future odometry with interpolation
    autoware_planning_msgs::msg::Trajectory gt_trajectory;
    {
      const int64_t GT_DT_NS =
        static_cast<int64_t>(constants::PREDICTION_TIME_STEP_S * 1e9);  // 0.1s

      gt_trajectory.header = odom.header;

      size_t gt_search_hint = odom_search_idx;
      for (int64_t k = 0; k < OUTPUT_T; ++k) {
        const int64_t offset_ns = (k + 1) * GT_DT_NS;
        const int64_t target_ns = timer_ns + offset_ns;
        if (target_ns > last_odom_ns) {
          break;
        }

        const auto interp_odom = interpolate_odometry(odometry_msgs, target_ns, gt_search_hint);

        autoware_planning_msgs::msg::TrajectoryPoint tp;
        tp.time_from_start.sec = static_cast<int32_t>(offset_ns / 1'000'000'000LL);
        tp.time_from_start.nanosec = static_cast<uint32_t>(offset_ns % 1'000'000'000LL);
        tp.pose = interp_odom.pose.pose;
        tp.longitudinal_velocity_mps = static_cast<float>(interp_odom.twist.twist.linear.x);
        tp.lateral_velocity_mps = static_cast<float>(interp_odom.twist.twist.linear.y);
        tp.heading_rate_rps = static_cast<float>(interp_odom.twist.twist.angular.z);

        gt_trajectory.points.push_back(tp);
      }

      writer_parser.write_topic(gt_trajectory, frame_time, TOPIC_OUT_GT_TRAJECTORY);
    }

    // --- Per-frame metrics ---
    if (metrics_file.is_open()) {
      // 1. ADE and FDE
      double ade = std::numeric_limits<double>::quiet_NaN();
      double fde = std::numeric_limits<double>::quiet_NaN();
      {
        const auto & pred_pts = planner_output.trajectory.points;
        const auto & gt_pts = gt_trajectory.points;
        const size_t n = std::min(pred_pts.size(), gt_pts.size());
        if (n > 0) {
          double sum_disp = 0.0;
          for (size_t i = 0; i < n; ++i) {
            const double dx = pred_pts[i].pose.position.x - gt_pts[i].pose.position.x;
            const double dy = pred_pts[i].pose.position.y - gt_pts[i].pose.position.y;
            sum_disp += std::sqrt(dx * dx + dy * dy);
          }
          ade = sum_disp / static_cast<double>(n);
          const double fdx = pred_pts[n - 1].pose.position.x - gt_pts[n - 1].pose.position.x;
          const double fdy = pred_pts[n - 1].pose.position.y - gt_pts[n - 1].pose.position.y;
          fde = std::sqrt(fdx * fdx + fdy * fdy);
        }
      }

      // 2. Road border contact (using per-frame input line_strings, not entire map)
      double min_road_border_dist = std::numeric_limits<double>::quiet_NaN();
      bool road_border_contact = false;
      if (!frame_road_borders.empty()) {
        min_road_border_dist = std::numeric_limits<double>::max();
        const auto & traj_pts = planner_output.trajectory.points;
        for (size_t i = 0; i < traj_pts.size(); ++i) {
          const auto ego_poly = autoware_utils_geometry::to_footprint(
            traj_pts[i].pose, ego_base_to_front, ego_base_to_rear, ego_width);
          for (const auto & border_ls : frame_road_borders) {
            const double dist = boost::geometry::distance(ego_poly, border_ls);
            if (dist < min_road_border_dist) {
              min_road_border_dist = dist;
            }
          }
        }
        if (min_road_border_dist < 1e-6) {
          road_border_contact = true;
        }
      }

      // 3. Ego-neighbor collision
      double min_neighbor_dist = std::numeric_limits<double>::quiet_NaN();
      bool neighbor_collision = false;
      {
        const auto & traj_pts = planner_output.trajectory.points;
        const auto & pred_objects = planner_output.predicted_objects;
        bool has_neighbor = false;

        for (const auto & obj : pred_objects.objects) {
          if (obj.kinematics.predicted_paths.empty()) {
            continue;
          }
          const auto & pred_path = obj.kinematics.predicted_paths[0];
          const size_t n = std::min(traj_pts.size(), pred_path.path.size());
          if (n == 0) {
            continue;
          }
          if (!has_neighbor) {
            min_neighbor_dist = std::numeric_limits<double>::max();
            has_neighbor = true;
          }
          for (size_t t = 0; t < n; ++t) {
            const auto ego_poly = autoware_utils_geometry::to_footprint(
              traj_pts[t].pose, ego_base_to_front, ego_base_to_rear, ego_width);
            const auto obj_poly =
              autoware_utils_geometry::to_polygon2d(pred_path.path[t], obj.shape);
            if (boost::geometry::intersects(ego_poly, obj_poly)) {
              min_neighbor_dist = 0.0;
              neighbor_collision = true;
            } else {
              const double dist = boost::geometry::distance(ego_poly, obj_poly);
              if (dist < min_neighbor_dist) {
                min_neighbor_dist = dist;
              }
            }
          }
        }
      }

      // Write CSV row
      metrics_file << timer_ns << "," << ade << "," << fde << "," << min_road_border_dist << ","
                   << (road_border_contact ? 1 : 0) << "," << min_neighbor_dist << ","
                   << (neighbor_collision ? 1 : 0) << "\n";
    }

    ++processed_frames;

    if (processed_frames % 100 == 0) {
      std::cout << "  Processed " << processed_frames << " / " << expected_frames << " frames..."
                << std::endl;
    }
  }

  // Close metrics file
  if (metrics_file.is_open()) {
    metrics_file.close();
    std::cout << "  Metrics written to: " << metrics_output_path << std::endl;
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
