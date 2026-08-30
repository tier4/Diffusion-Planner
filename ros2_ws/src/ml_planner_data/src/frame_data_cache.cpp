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

#include "frame_data_cache.hpp"

#include "label_builder.hpp"

#include <autoware/geography_utils/lanelet2_projector.hpp>
#include <autoware/map_projection_loader/load_info_from_lanelet2_map.hpp>
#include <autoware/map_projection_loader/map_projection_loader.hpp>

#include <autoware_map_msgs/msg/map_projector_info.hpp>

#include <lanelet2_io/Io.h>
#include <lanelet2_io/Projection.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace autoware::ml_planner::data {
namespace {

class LocalProjector : public lanelet::Projector {
public:
  LocalProjector() : Projector(lanelet::Origin(lanelet::GPSPoint{})) {}

  lanelet::BasicPoint3d forward(const lanelet::GPSPoint &gps) const override {
    return {0.0, 0.0, gps.ele};
  }

  lanelet::GPSPoint reverse(const lanelet::BasicPoint3d &point) const override {
    return {0.0, 0.0, point.z()};
  }
};

lanelet::LaneletMapPtr load_map(const std::string &map_path) {
  const std::filesystem::path projector_info_path =
      std::filesystem::path(map_path).parent_path() / "map_projector_info.yaml";
  const auto projector_info =
      std::filesystem::exists(projector_info_path)
          ? autoware::map_projection_loader::load_info_from_yaml(
                projector_info_path.string())
          : autoware::map_projection_loader::load_info_from_lanelet2_map(
                map_path);

  lanelet::ErrorMessages errors;
  lanelet::LaneletMapPtr map;
  if (projector_info.projector_type ==
      autoware_map_msgs::msg::MapProjectorInfo::LOCAL) {
    LocalProjector projector;
    map = lanelet::load(map_path, projector, &errors);
    for (lanelet::Point3d point : map->pointLayer) {
      if (point.hasAttribute("local_x")) {
        point.x() = point.attribute("local_x").asDouble().value();
      }
      if (point.hasAttribute("local_y")) {
        point.y() = point.attribute("local_y").asDouble().value();
      }
    }
  } else {
    const std::unique_ptr<lanelet::Projector> projector =
        autoware::geography_utils::get_lanelet2_projector(projector_info);
    map = lanelet::load(map_path, *projector, &errors);
  }
  if (!errors.empty()) {
    std::string message = "failed to load lanelet map " + map_path;
    for (const std::string &error : errors) {
      message += "\n" + error;
    }
    throw std::runtime_error(message);
  }
  return map;
}

} // namespace

FrameDataCache::FrameDataCache(const size_t reader_capacity,
                               const size_t map_capacity,
                               const TopicConfig &topics,
                               const double line_string_max_step_m)
    : readers_(reader_capacity), map_contexts_(map_capacity), topics_(topics),
      line_string_max_step_m_(line_string_max_step_m) {}

BagFrameReader &FrameDataCache::reader_for(const std::string &bag_path) {
  return *readers_.get_or_create(bag_path, [&]() {
    return std::make_shared<BagFrameReader>(bag_path, topics_);
  });
}

FrameDataResult FrameDataCache::create_frame_data(
    const std::string &bag_path, const std::string &map_path,
    const int64_t frame_time_ns, const VehicleSpec &vehicle_spec,
    const double traffic_light_timeout_s, const int64_t num_future_steps,
    const double neighbor_observation_timeout_s) {
  auto &map_context = map_contexts_.get_or_create(map_path, [&]() {
    const lanelet::LaneletMapPtr map = load_map(map_path);
    return std::shared_ptr<const preprocess::LaneSegmentContext>(
        preprocess::build_map_context(map, line_string_max_step_m_));
  });

  BagFrameReader &reader = reader_for(bag_path);
  const rclcpp::Time frame_time(frame_time_ns, RCL_ROS_TIME);

  preprocess::InputBuilderParams input_params;
  input_params.traffic_light_group_msg_timeout_seconds =
      traffic_light_timeout_s;
  preprocess::InputBuilderResult input_result = reader.create_input_data(
      frame_time, *map_context, vehicle_spec, input_params);
  if (!input_result) {
    return tl::unexpected(input_result.error());
  }
  auto input_output = std::move(input_result.value());

  LabelBuilderParams label_params;
  label_params.num_future_steps = num_future_steps;
  label_params.neighbor_observation_timeout_s = neighbor_observation_timeout_s;
  label_params.traffic_light_timeout_s = traffic_light_timeout_s;
  preprocess::TensorMapResult label_result = reader.create_label_data(
      frame_time, *map_context, label_params, input_output.selected_agents);
  if (!label_result) {
    return label_result;
  }
  for (auto &[key, value] : label_result.value()) {
    input_output.tensors[key] = std::move(value);
  }

  return std::move(input_output.tensors);
}

} // namespace autoware::ml_planner::data
