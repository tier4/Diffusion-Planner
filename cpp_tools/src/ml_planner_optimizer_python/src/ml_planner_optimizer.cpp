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

// The post-processing ml_planner_node applies to the model output -- road border avoidance, then
// the acados trajectory optimizer -- driven from Python one planning cycle at a time. The two
// stages are autoware_ml_planner's own classes; this file only converts arrays to the messages
// they take, in the order MLPlannerCore::create_planner_output calls them.

#include "autoware/ml_planner/optimization/trajectory_optimizer.hpp"
#include "autoware/ml_planner/postprocessing/road_border_avoidance.hpp"

#include <autoware/vehicle_info_utils/vehicle_info.hpp>
#include <autoware_utils/geometry/geometry.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{
using autoware::ml_planner::optimization::TrajectoryOptimizationParams;
using autoware::ml_planner::optimization::TrajectoryOptimizer;
using autoware::ml_planner::postprocess::RoadBorderAvoidance;
using autoware::ml_planner::postprocess::RoadBorderAvoidanceParams;
using autoware_planning_msgs::msg::Trajectory;
using autoware_planning_msgs::msg::TrajectoryPoint;

constexpr size_t horizon = autoware::ml_planner::optimization::opt_horizon;
constexpr double dt = autoware::ml_planner::optimization::opt_dt_s;
constexpr int output_columns = 7;  // x, y, yaw, v, front wheel angle, acceleration, heading rate

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

YAML::Node ros_parameters(const std::string & path)
{
  const YAML::Node root = YAML::LoadFile(path);
  for (const auto & entry : root) {
    if (entry.second["ros__parameters"]) {
      return entry.second["ros__parameters"];
    }
  }
  throw std::runtime_error(path + ": no ros__parameters");
}

// Reads a dotted parameter name the way the node declares it; absent keys keep the default.
template <typename T>
void read(const YAML::Node & params, const std::string & name, T & value)
{
  YAML::Node node;
  node.reset(params);
  std::istringstream parts(name);
  std::string part;
  while (std::getline(parts, part, '.')) {
    // Const lookup: the non-const operator[] inserts the key it does not find.
    const YAML::Node & current = node;
    const YAML::Node child = current.IsMap() ? current[part] : YAML::Node();
    if (!child.IsDefined() || child.IsNull()) {
      return;
    }
    // reset() rebinds; operator= would overwrite the node the handle points at.
    node.reset(child);
  }
  value = node.as<T>();
}

TrajectoryOptimizationParams optimization_params(const YAML::Node & p)
{
  TrajectoryOptimizationParams o;
  const std::string n = "trajectory_optimization.";
  read(p, n + "enable", o.enable);
  read(p, n + "weight_longitudinal", o.weight_longitudinal);
  read(p, n + "weight_lateral", o.weight_lateral);
  read(p, n + "weight_yaw", o.weight_yaw);
  read(p, n + "weight_velocity", o.weight_velocity);
  read(p, n + "weight_steering_angle", o.weight_steering_angle);
  read(p, n + "weight_acceleration", o.weight_acceleration);
  read(p, n + "weight_steering_rate", o.weight_steering_rate);
  read(p, n + "terminal_weight_scale", o.terminal_weight_scale);
  read(p, n + "goal.weight_longitudinal", o.goal.weight_longitudinal);
  read(p, n + "goal.weight_lateral", o.goal.weight_lateral);
  read(p, n + "goal.weight_yaw", o.goal.weight_yaw);
  read(p, n + "goal.weight_velocity", o.goal.weight_velocity);
  read(p, n + "goal.snap_distance_m", o.goal.snap_distance_m);
  read(p, n + "min_velocity_mps", o.min_velocity_mps);
  read(p, n + "max_velocity_mps", o.max_velocity_mps);
  read(p, n + "min_acceleration_mps2", o.min_acceleration_mps2);
  read(p, n + "max_acceleration_mps2", o.max_acceleration_mps2);
  read(p, n + "max_steering_rate_rps", o.max_steering_rate_rps);
  read(p, n + "max_lateral_acceleration_mps2", o.max_lateral_acceleration_mps2);
  read(p, n + "max_sqp_iterations", o.max_sqp_iterations);
  const std::string t = n + "temporal_consistency.";
  read(p, t + "enable", o.temporal_consistency.enable);
  read(p, t + "weight_longitudinal", o.temporal_consistency.weight_longitudinal);
  read(p, t + "weight_lateral", o.temporal_consistency.weight_lateral);
  read(p, t + "weight_yaw", o.temporal_consistency.weight_yaw);
  read(p, t + "weight_velocity", o.temporal_consistency.weight_velocity);
  read(p, t + "decay_time_constant_s", o.temporal_consistency.decay_time_constant_s);
  read(p, t + "far_weight_ratio", o.temporal_consistency.far_weight_ratio);
  return o;
}

RoadBorderAvoidanceParams border_params(const YAML::Node & p)
{
  RoadBorderAvoidanceParams b;
  const std::string n = "road_border_avoidance.";
  read(p, n + "enable", b.enable);
  read(p, n + "start_time_s", b.start_time_s);
  read(p, n + "footprint_margin_m", b.footprint_margin_m);
  read(p, n + "search_radius_m", b.search_radius_m);
  read(p, n + "shift_step_m", b.shift_step_m);
  read(p, n + "max_lateral_shift_m", b.max_lateral_shift_m);
  read(p, n + "propagate_shift", b.propagate_shift);
  return b;
}

// vehicle_info.param.yaml, with any of its keys replaced by the simulated ego's own dimensions.
autoware::vehicle_info_utils::VehicleInfo vehicle_info(
  const std::string & path, const std::map<std::string, double> & overrides)
{
  const YAML::Node p = ros_parameters(path);
  std::map<std::string, double> v;
  for (const auto & entry : p) {
    v[entry.first.as<std::string>()] = entry.second.as<double>();
  }
  for (const auto & [key, value] : overrides) {
    if (v.count(key) == 0) {
      throw std::invalid_argument("unknown vehicle_info key: " + key);
    }
    v[key] = value;
  }
  return autoware::vehicle_info_utils::createVehicleInfo(
    v.at("wheel_radius"), v.at("wheel_width"), v.at("wheel_base"), v.at("wheel_tread"),
    v.at("front_overhang"), v.at("rear_overhang"), v.at("left_overhang"), v.at("right_overhang"),
    v.at("vehicle_height"), v.at("max_steer_angle"));
}

builtin_interfaces::msg::Time to_stamp(const double seconds)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(std::floor(seconds));
  stamp.nanosec = static_cast<uint32_t>(std::llround((seconds - stamp.sec) * 1e9) % 1000000000LL);
  return stamp;
}

geometry_msgs::msg::Pose to_pose(const double x, const double y, const double yaw)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.orientation = autoware_utils::create_quaternion_from_yaw(yaw);
  return pose;
}

double yaw_of(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

// The raw trajectory as create_ego_trajectory builds it: point k at t = (k + 1) dt, with the
// speed of the step from the previous point (the ego position for the first).
Trajectory raw_trajectory(const Array & raw, const double ego_x, const double ego_y, double stamp)
{
  if (raw.ndim() != 2 || raw.shape(0) != static_cast<py::ssize_t>(horizon) || raw.shape(1) != 3) {
    throw std::invalid_argument("raw must be (80, 3): x, y, yaw in the map frame");
  }
  const auto r = raw.unchecked<2>();
  Trajectory trajectory;
  trajectory.header.stamp = to_stamp(stamp);
  trajectory.header.frame_id = "map";
  double px = ego_x;
  double py = ego_y;
  for (size_t i = 0; i < horizon; ++i) {
    const auto k = static_cast<py::ssize_t>(i);
    TrajectoryPoint p;
    const double t = dt * static_cast<double>(i + 1);
    p.time_from_start.sec = static_cast<int32_t>(t);
    p.time_from_start.nanosec = static_cast<uint32_t>((t - p.time_from_start.sec) * 1e9);
    p.pose = to_pose(r(k, 0), r(k, 1), r(k, 2));
    p.longitudinal_velocity_mps = static_cast<float>(std::hypot(r(k, 0) - px, r(k, 1) - py) / dt);
    px = r(k, 0);
    py = r(k, 1);
    trajectory.points.push_back(p);
  }
  return trajectory;
}

Array to_array(const Trajectory & trajectory)
{
  Array out({static_cast<py::ssize_t>(trajectory.points.size()), py::ssize_t{output_columns}});
  auto o = out.mutable_unchecked<2>();
  for (size_t i = 0; i < trajectory.points.size(); ++i) {
    const auto k = static_cast<py::ssize_t>(i);
    const auto & p = trajectory.points[i];
    o(k, 0) = p.pose.position.x;
    o(k, 1) = p.pose.position.y;
    o(k, 2) = yaw_of(p.pose.orientation);
    o(k, 3) = p.longitudinal_velocity_mps;
    o(k, 4) = p.front_wheel_angle_rad;
    o(k, 5) = p.acceleration_mps2;
    o(k, 6) = p.heading_rate_rps;
  }
  return out;
}

class Optimizer
{
public:
  Optimizer(
    const std::string & param_yaml, const std::string & vehicle_yaml,
    const std::map<std::string, double> & vehicle_overrides)
  {
    const YAML::Node params = ros_parameters(param_yaml);
    const auto vehicle = vehicle_info(vehicle_yaml, vehicle_overrides);
    const auto border = border_params(params);
    if (border.enable) {
      border_ = std::make_unique<RoadBorderAvoidance>(border, vehicle);
    }
    const auto optimization = optimization_params(params);
    if (optimization.enable) {
      optimizer_ = std::make_unique<TrajectoryOptimizer>(optimization, vehicle, 1);
    }
  }

  void set_road_borders(const std::vector<Array> & borders)
  {
    if (!border_) {
      return;
    }
    std::vector<autoware_utils_geometry::LineString2d> lines;
    lines.reserve(borders.size());
    for (const auto & b : borders) {
      if (b.ndim() != 2 || b.shape(1) < 2) {
        throw std::invalid_argument("each road border must be (K, 2) in the map frame");
      }
      const auto r = b.unchecked<2>();
      autoware_utils_geometry::LineString2d line;
      for (py::ssize_t k = 0; k < r.shape(0); ++k) {
        line.emplace_back(r(k, 0), r(k, 1));
      }
      if (line.size() >= 2) {
        lines.push_back(std::move(line));
      }
    }
    border_->set_road_borders(std::move(lines));
  }

  // One planning cycle. `trajectory` is None when the optimizer failed: the node drops such a
  // candidate rather than publishing the raw output.
  py::dict step(
    const Array & raw, const std::vector<double> & ego, const double steering_angle_rad,
    const double stamp_s, const std::optional<std::vector<double>> & goal)
  {
    if (ego.size() != 4) {
      throw std::invalid_argument("ego must be (x, y, yaw, v)");
    }
    if (goal && goal->size() != 3) {
      throw std::invalid_argument("goal must be (x, y, yaw)");
    }
    Trajectory trajectory = raw_trajectory(raw, ego[0], ego[1], stamp_s);
    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = trajectory.header.stamp;
    odometry.header.frame_id = "map";
    odometry.pose.pose = to_pose(ego[0], ego[1], ego[2]);
    odometry.twist.twist.linear.x = ego[3];

    py::dict out;
    size_t shifted = 0;
    size_t unresolved = 0;
    double max_shift = 0.0;
    if (border_) {
      auto adjusted = border_->adjust(trajectory, odometry.pose.pose);
      shifted = adjusted.num_shifted_points;
      unresolved = adjusted.num_unresolved_points;
      for (size_t i = 0; i < trajectory.points.size(); ++i) {
        const auto & a = trajectory.points[i].pose.position;
        const auto & b = adjusted.trajectory.points[i].pose.position;
        max_shift = std::max(max_shift, std::hypot(b.x - a.x, b.y - a.y));
      }
      trajectory = std::move(adjusted.trajectory);
    }
    out["border_shifted_points"] = shifted;
    out["border_unresolved_points"] = unresolved;
    // How far the reference was moved off the model's own path, at its worst point.
    out["border_max_shift_m"] = max_shift;

    bool failed = false;
    if (optimizer_) {
      std::optional<geometry_msgs::msg::Pose> goal_pose;
      if (goal) {
        goal_pose = to_pose((*goal)[0], (*goal)[1], (*goal)[2]);
      }
      auto result = optimizer_->optimize(trajectory, odometry, steering_angle_rad, 0, goal_pose);
      out["optimized"] = result.optimized;
      out["solver_status"] = result.solver_status;
      out["solve_time_ms"] = result.solve_time_ms;
      failed = !result.optimized;
      trajectory = std::move(result.trajectory);
    }
    out["trajectory"] = failed ? py::object(py::none()) : py::object(to_array(trajectory));
    return out;
  }

  bool optimizes() const { return optimizer_ != nullptr; }
  bool avoids_road_borders() const { return border_ != nullptr; }

private:
  std::unique_ptr<RoadBorderAvoidance> border_;
  std::unique_ptr<TrajectoryOptimizer> optimizer_;
};
}  // namespace

PYBIND11_MODULE(ml_planner_optimizer, m)
{
  m.doc() = "autoware_ml_planner's road border avoidance and trajectory optimizer";
  m.attr("HORIZON") = horizon;
  m.attr("DT") = dt;
  py::class_<Optimizer>(m, "Optimizer")
    .def(
      py::init<const std::string &, const std::string &, const std::map<std::string, double> &>(),
      py::arg("param_yaml"), py::arg("vehicle_yaml"),
      py::arg("vehicle_overrides") = std::map<std::string, double>{})
    .def("set_road_borders", &Optimizer::set_road_borders, py::arg("borders"))
    .def(
      "step", &Optimizer::step, py::arg("raw"), py::arg("ego"), py::arg("steering_angle_rad"),
      py::arg("stamp_s"), py::arg("goal") = py::none())
    .def_property_readonly("optimizes", &Optimizer::optimizes)
    .def_property_readonly("avoids_road_borders", &Optimizer::avoids_road_borders);
}
