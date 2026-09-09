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

#include "topic_config.hpp"

#include <yaml-cpp/yaml.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace autoware::ml_planner::data {

TopicConfig load_topic_config() {
  const std::string path =
      ament_index_cpp::get_package_share_directory("ml_planner_data") +
      "/config/ml_planner_data.param.yaml";
  const YAML::Node topics = YAML::LoadFile(path)["topics"];

  TopicConfig config;
  config.kinematic_state = topics["kinematic_state"].as<std::string>();
  config.tracked_objects = topics["tracked_objects"].as<std::string>();
  config.turn_indicators = topics["turn_indicators"].as<std::string>();
  config.traffic_signals = topics["traffic_signals"].as<std::string>();
  config.route = topics["route"].as<std::string>();
  return config;
}

} // namespace autoware::ml_planner::data
