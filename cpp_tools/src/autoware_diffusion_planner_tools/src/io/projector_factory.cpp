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

#include "io/projector_factory.hpp"

#include <autoware_lanelet2_extension/projection/mgrs_projector.hpp>
#include <autoware_lanelet2_extension/projection/transverse_mercator_projector.hpp>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>

std::unique_ptr<lanelet::Projector> create_projector_from_yaml(const std::string & vector_map_path)
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
