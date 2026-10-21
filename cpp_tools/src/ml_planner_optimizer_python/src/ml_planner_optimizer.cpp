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

// ml_planner_node's road border avoidance and acados trajectory optimizer, driven from Python one
// planning cycle at a time. Both are autoware_ml_planner's own classes, called in the order
// MLPlannerCore::create_planner_output calls them; stop point fixing is not included.

#include "autoware/ml_planner/optimization/trajectory_optimizer.hpp"
#include "autoware/ml_planner/postprocessing/road_border_avoidance.hpp"

#include <autoware/vehicle_info_utils/vehicle_info.hpp>
#include <autoware_utils_geometry/geometry.hpp>
#include <rclcpp/time.hpp>

#include <nav_msgs/msg/odometry.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
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

constexpr size_t horizon = autoware::ml_planner::optimization::opt_horizon;
constexpr double dt = autoware::ml_planner::optimization::opt_dt_s;
// x, y, yaw, v, front wheel angle, acceleration, heading rate
constexpr py::ssize_t output_columns = 7;

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

YAML::Node ros_parameters(const std::string & path)
{
  for (const auto & entry : YAML::LoadFile(path)) {
    if (entry.second["ros__parameters"]) {
      return entry.second["ros__parameters"];
    }
  }
  throw std::runtime_error(path + ": no ros__parameters");
}

// Reads a dotted parameter name the way the node declares it. Absent keys keep the struct default,
// which equals the node's declare_parameter default.
template <typename T>
void read(const YAML::Node & node, const std::string & name, T & value)
{
  const auto dot = name.find('.');
  const YAML::Node child = node.IsMap() ? node[name.substr(0, dot)] : YAML::Node();
  if (!child || child.IsNull()) {
    return;
  }
  if (dot == std::string::npos) {
    value = child.as<T>();
  } else {
    read(child, name.substr(dot + 1), value);
  }
}

// The fields carry the names of their parameters below the struct's namespace.
#define READ_PARAM(field) read(root, ns + #field, params.field)

TrajectoryOptimizationParams optimization_params(const YAML::Node & root)
{
  TrajectoryOptimizationParams params;
  const std::string ns = "trajectory_optimization.";
  READ_PARAM(enable);
  READ_PARAM(weight_longitudinal);
  READ_PARAM(weight_lateral);
  READ_PARAM(weight_yaw);
  READ_PARAM(weight_velocity);
  READ_PARAM(weight_steering_angle);
  READ_PARAM(weight_acceleration);
  READ_PARAM(weight_steering_rate);
  READ_PARAM(terminal_weight_scale);
  READ_PARAM(goal.weight_longitudinal);
  READ_PARAM(goal.weight_lateral);
  READ_PARAM(goal.weight_yaw);
  READ_PARAM(goal.weight_velocity);
  READ_PARAM(goal.snap_distance_m);
  READ_PARAM(goal.unlatch_horizon_s);
  READ_PARAM(goal.unlatch_min_speed_mps);
  READ_PARAM(min_velocity_mps);
  READ_PARAM(max_velocity_mps);
  READ_PARAM(min_acceleration_mps2);
  READ_PARAM(max_acceleration_mps2);
  READ_PARAM(max_steering_rate_rps);
  READ_PARAM(max_lateral_acceleration_mps2);
  READ_PARAM(max_sqp_iterations);
  READ_PARAM(temporal_consistency.enable);
  READ_PARAM(temporal_consistency.weight_longitudinal);
  READ_PARAM(temporal_consistency.weight_lateral);
  READ_PARAM(temporal_consistency.weight_yaw);
  READ_PARAM(temporal_consistency.weight_velocity);
  READ_PARAM(temporal_consistency.decay_time_constant_s);
  READ_PARAM(temporal_consistency.far_weight_ratio);
  return params;
}

RoadBorderAvoidanceParams border_params(const YAML::Node & root)
{
  RoadBorderAvoidanceParams params;
  const std::string ns = "road_border_avoidance.";
  READ_PARAM(enable);
  READ_PARAM(start_time_s);
  READ_PARAM(footprint_margin_m);
  READ_PARAM(search_radius_m);
  READ_PARAM(shift_step_m);
  READ_PARAM(max_lateral_shift_m);
  READ_PARAM(propagate_shift);
  return params;
}

#undef READ_PARAM

// vehicle_info.param.yaml with some of its keys replaced, e.g. by the simulated ego's dimensions.
autoware::vehicle_info_utils::VehicleInfo vehicle_info(
  const std::string & path, const std::map<std::string, double> & overrides)
{
  std::map<std::string, double> v;
  for (const auto & entry : ros_parameters(path)) {
    v[entry.first.as<std::string>()] = entry.second.as<double>();
  }
  for (const auto & [key, value] : overrides) {
    v.at(key) = value;
  }
  return autoware::vehicle_info_utils::createVehicleInfo(
    v.at("wheel_radius"), v.at("wheel_width"), v.at("wheel_base"), v.at("wheel_tread"),
    v.at("front_overhang"), v.at("rear_overhang"), v.at("left_overhang"), v.at("right_overhang"),
    v.at("vehicle_height"), v.at("max_steer_angle"));
}

geometry_msgs::msg::Pose to_pose(const double x, const double y, const double yaw)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.orientation = autoware_utils_geometry::create_quaternion_from_yaw(yaw);
  return pose;
}

// The raw trajectory as create_ego_trajectory builds it, in the plane: point k at t = (k + 1) dt,
// with the speed of the step onto it (from the ego for the first) and the forward difference of
// those speeds as its acceleration.
Trajectory raw_trajectory(const Array & raw, const double ego_x, const double ego_y)
{
  const auto r = raw.unchecked<2>();
  if (r.shape(0) != static_cast<py::ssize_t>(horizon) || r.shape(1) != 3) {
    throw std::invalid_argument("raw must be (80, 3): x, y, yaw in the map frame");
  }
  Trajectory trajectory;
  trajectory.header.frame_id = "map";
  auto & points = trajectory.points;
  double px = ego_x;
  double py = ego_y;
  for (py::ssize_t k = 0; k < r.shape(0); ++k) {
    auto & p = points.emplace_back();
    // Truncated as create_ego_trajectory does; road border start_time_s compares against it.
    const double t = dt * static_cast<double>(k + 1);
    p.time_from_start.sec = static_cast<int32_t>(t);
    p.time_from_start.nanosec = static_cast<uint32_t>((t - p.time_from_start.sec) * 1e9);
    p.pose = to_pose(r(k, 0), r(k, 1), r(k, 2));
    p.longitudinal_velocity_mps = static_cast<float>(std::hypot(r(k, 0) - px, r(k, 1) - py) / dt);
    px = r(k, 0);
    py = r(k, 1);
  }
  for (size_t i = 0; i + 1 < points.size(); ++i) {
    const double v0 = points[i].longitudinal_velocity_mps;
    const double v1 = points[i + 1].longitudinal_velocity_mps;
    points[i].acceleration_mps2 = static_cast<float>((v1 - v0) / dt);
  }
  return trajectory;
}

Array to_array(const Trajectory & trajectory)
{
  Array out({static_cast<py::ssize_t>(trajectory.points.size()), output_columns});
  auto o = out.mutable_unchecked<2>();
  for (py::ssize_t k = 0; k < o.shape(0); ++k) {
    const auto & p = trajectory.points[k];
    o(k, 0) = p.pose.position.x;
    o(k, 1) = p.pose.position.y;
    o(k, 2) = autoware_utils_geometry::get_rpy(p.pose.orientation).z;
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
    if (const auto border = border_params(params); border.enable) {
      border_ = std::make_unique<RoadBorderAvoidance>(border, vehicle);
    }
    if (const auto optimization = optimization_params(params); optimization.enable) {
      optimizer_ = std::make_unique<TrajectoryOptimizer>(optimization, vehicle, 1);
    }
  }

  void set_road_borders(const std::vector<Array> & borders)
  {
    if (!border_) {
      return;
    }
    std::vector<autoware_utils_geometry::LineString2d> lines;
    for (const auto & b : borders) {
      const auto r = b.unchecked<2>();
      if (r.shape(1) < 2) {
        throw std::invalid_argument("each road border must be (K, 2) in the map frame");
      }
      autoware_utils_geometry::LineString2d line;
      line.reserve(r.shape(0));
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
    const Array & raw, const std::array<double, 4> & ego, const double steering_angle_rad,
    const double stamp_s, const std::optional<std::array<double, 3>> & goal)
  {
    Trajectory trajectory = raw_trajectory(raw, ego[0], ego[1]);
    trajectory.header.stamp = rclcpp::Time(std::llround(stamp_s * 1e9));
    nav_msgs::msg::Odometry odometry;
    odometry.pose.pose = to_pose(ego[0], ego[1], ego[2]);
    odometry.twist.twist.linear.x = ego[3];

    py::dict out;
    out["border_shifted_points"] = 0;
    out["border_unresolved_points"] = 0;
    // How far the reference was moved off the model's own path, at its worst point.
    out["border_max_shift_m"] = 0.0;
    if (border_) {
      auto adjusted = border_->adjust(trajectory, odometry.pose.pose);
      double max_shift = 0.0;
      for (size_t i = 0; i < trajectory.points.size(); ++i) {
        const auto & a = trajectory.points[i].pose.position;
        const auto & b = adjusted.trajectory.points[i].pose.position;
        max_shift = std::max(max_shift, std::hypot(b.x - a.x, b.y - a.y));
      }
      out["border_shifted_points"] = adjusted.num_shifted_points;
      out["border_unresolved_points"] = adjusted.num_unresolved_points;
      out["border_max_shift_m"] = max_shift;
      trajectory = std::move(adjusted.trajectory);
    }

    if (optimizer_) {
      std::optional<geometry_msgs::msg::Pose> goal_pose;
      if (goal) {
        goal_pose = to_pose((*goal)[0], (*goal)[1], (*goal)[2]);
      }
      auto result = optimizer_->optimize(trajectory, odometry, steering_angle_rad, 0, goal_pose);
      out["optimized"] = result.optimized;
      out["solver_status"] = result.solver_status;
      out["solve_time_ms"] = result.solve_time_ms;
      if (!result.optimized) {
        out["trajectory"] = py::none();
        return out;
      }
      trajectory = std::move(result.trajectory);
    }
    out["trajectory"] = to_array(trajectory);
    return out;
  }

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
      py::arg("stamp_s"), py::arg("goal") = py::none());
}
