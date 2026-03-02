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

#include "autoware/diffusion_planner/conversion/lanelet.hpp"
#include "autoware/diffusion_planner/dimensions.hpp"

#include <autoware_lanelet2_extension/projection/mgrs_projector.hpp>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_io/Io.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{
using autoware::diffusion_planner::LaneletMap;
using autoware::diffusion_planner::LanePoint;
using autoware::diffusion_planner::POINTS_PER_LINE_STRING;
using autoware::diffusion_planner::POINTS_PER_POLYGON;
using autoware::diffusion_planner::POINTS_PER_SEGMENT;
using nlohmann::json;

json to_json_point(const LanePoint & p)
{
  return json::array({p.x(), p.y(), p.z()});
}

json to_json_polyline(const std::vector<LanePoint> & polyline)
{
  json out = json::array();
  for (const auto & p : polyline) {
    out.push_back(to_json_point(p));
  }
  return out;
}

template <typename LineStringT>
json lanelet_line_string_to_json(const LineStringT & ls)
{
  json out = json::array();
  for (const auto & pt : ls) {
    out.push_back(json::array({pt.x(), pt.y(), pt.z()}));
  }
  return out;
}

json to_json_lanelet_map(const LaneletMap & map)
{
  json root;
  root["lane_segments"] = json::array();
  for (const auto & lane : map.lane_segments) {
    root["lane_segments"].push_back({
      {"id", lane.id},
      {"centerline", to_json_polyline(lane.centerline)},
      {"left_boundary", to_json_polyline(lane.left_boundary)},
      {"right_boundary", to_json_polyline(lane.right_boundary)},
    });
  }

  root["polygons"] = json::array();
  for (const auto & poly : map.polygons) {
    root["polygons"].push_back(json{{"points", to_json_polyline(poly.points)}});
  }

  root["line_strings"] = json::array();
  for (const auto & line : map.line_strings) {
    root["line_strings"].push_back(json{{"points", to_json_polyline(line.points)}});
  }
  return root;
}

json to_json_raw_lanelet_map(const lanelet::LaneletMapConstPtr & map_ptr)
{
  json root;
  root["lane_segments"] = json::array();
  for (const auto & ll : map_ptr->laneletLayer) {
    const auto center = lanelet_line_string_to_json(ll.centerline3d());
    const auto left = lanelet_line_string_to_json(ll.leftBound3d());
    const auto right = lanelet_line_string_to_json(ll.rightBound3d());
    if (center.empty() || left.empty() || right.empty()) {
      continue;
    }
    root["lane_segments"].push_back(json{
      {"id", static_cast<int64_t>(ll.id())},
      {"centerline", center},
      {"left_boundary", left},
      {"right_boundary", right}});
  }

  root["polygons"] = json::array();
  for (const auto & poly : map_ptr->polygonLayer) {
    const auto pts = lanelet_line_string_to_json(poly.basicLineString());
    if (!pts.empty()) {
      root["polygons"].push_back(json{{"points", pts}});
    }
  }

  root["line_strings"] = json::array();
  for (const auto & ls : map_ptr->lineStringLayer) {
    const auto pts = lanelet_line_string_to_json(ls);
    if (!pts.empty()) {
      root["line_strings"].push_back(json{{"points", pts}});
    }
  }
  return root;
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  const auto node = std::make_shared<rclcpp::Node>("map_exporter");

  node->declare_parameter<std::string>("map_path", "");
  node->declare_parameter<std::string>("internal_out", "");
  node->declare_parameter<std::string>("reference_out", "");

  const std::string map_path = node->get_parameter("map_path").as_string();
  const std::string internal_out = node->get_parameter("internal_out").as_string();
  const std::string reference_out = node->get_parameter("reference_out").as_string();

  auto fail_and_shutdown = [&](const std::string & msg) {
    RCLCPP_ERROR(node->get_logger(), "%s", msg.c_str());
    rclcpp::shutdown();
    return 1;
  };

  if (map_path.empty()) {
    return fail_and_shutdown("Parameter 'map_path' is required.");
  }
  if (internal_out.empty()) {
    return fail_and_shutdown("Parameter 'internal_out' is required.");
  }
  if (reference_out.empty()) {
    return fail_and_shutdown("Parameter 'reference_out' is required.");
  }

  RCLCPP_INFO(
    node->get_logger(), "map_exporter params: map_path=%s internal_out=%s reference_out=%s",
    map_path.c_str(), internal_out.c_str(), reference_out.c_str());

  lanelet::ErrorMessages errors;
  lanelet::projection::MGRSProjector projector;
  auto lanelet_map_unique_ptr = lanelet::load(map_path, projector, &errors);
  if (!errors.empty()) {
    for (const auto & e : errors) {
      RCLCPP_WARN(node->get_logger(), "%s", e.c_str());
    }
  }
  if (!lanelet_map_unique_ptr) {
    return fail_and_shutdown("Failed to load map from: " + map_path);
  }
  std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr(lanelet_map_unique_ptr.release());
  const lanelet::LaneletMapConstPtr lanelet_map_const_ptr = lanelet_map_ptr;

  const LaneletMap internal_map =
    autoware::diffusion_planner::convert_to_internal_lanelet_map(lanelet_map_const_ptr);

  json internal_json;
  internal_json["meta"] = {
    {"source_map_path", map_path},
    {"map_type", "internal"},
    {"point_constants",
     {{"points_per_segment", POINTS_PER_SEGMENT},
      {"points_per_polygon", POINTS_PER_POLYGON},
      {"points_per_line_string", POINTS_PER_LINE_STRING}}}};
  internal_json.update(to_json_lanelet_map(internal_map));

  json reference_json;
  reference_json["meta"] = {
    {"source_map_path", map_path},
    {"map_type", "reference_raw"},
    {"description",
     "All lanelets, polygons, and line strings from raw Lanelet2 map (no subtype/type filtering)"},
  };
  reference_json.update(to_json_raw_lanelet_map(lanelet_map_const_ptr));

  {
    std::ofstream ofs(internal_out);
    ofs << internal_json.dump(2) << "\n";
  }
  {
    std::ofstream ofs(reference_out);
    ofs << reference_json.dump(2) << "\n";
  }

  RCLCPP_INFO(node->get_logger(), "Wrote internal map JSON: %s", internal_out.c_str());
  RCLCPP_INFO(node->get_logger(), "Wrote reference JSON: %s", reference_out.c_str());
  rclcpp::shutdown();
  return 0;
}
