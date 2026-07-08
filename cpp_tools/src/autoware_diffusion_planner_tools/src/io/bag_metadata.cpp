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

#include "io/bag_metadata.hpp"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

namespace
{
std::string get_string_or_empty(const nlohmann::json & j, const std::string & key)
{
  if (j.contains(key) && j[key].is_string()) {
    return j[key].get<std::string>();
  }
  return "";
}

void parse_date_and_bag_time(
  const std::string & log_file_name, std::string & date, std::string & bag_time)
{
  // Pattern: {vehicle_uuid}_{YYYY-MM-DD}-{HH-MM-SS}_{suffix}
  // The date and bag_time are separated by a hyphen between DD and HH,
  // so the full datetime portion is: YYYY-MM-DD-HH-MM-SS
  // We look for the date-time pattern after the first underscore.
  static const std::regex re(R"((\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})_)");
  std::smatch match;
  if (std::regex_search(log_file_name, match, re)) {
    date = match[1].str();
    bag_time = match[2].str();
  }
}
}  // namespace

BagMetadata load_bag_metadata(const std::string & rosbag_path)
{
  BagMetadata meta;

  try {
    const std::filesystem::path info_path =
      std::filesystem::path(rosbag_path) / "log_file_info.json";

    if (!std::filesystem::is_regular_file(info_path)) {
      return meta;
    }

    std::ifstream file(info_path);
    const nlohmann::json j = nlohmann::json::parse(file);

    meta.log_file_id = get_string_or_empty(j, "id");
    meta.vehicle_id = get_string_or_empty(j, "vehicle_id");
    meta.project_id = get_string_or_empty(j, "project_id");
    meta.map_version_id = get_string_or_empty(j, "area_map_version_id");

    const std::string log_file_name = get_string_or_empty(j, "log_file_name");
    if (!log_file_name.empty()) {
      parse_date_and_bag_time(log_file_name, meta.date, meta.bag_time);
    }
  } catch (const std::exception & e) {
    std::cerr << "Warning: failed to load bag metadata from " << rosbag_path << ": " << e.what()
              << std::endl;
    return meta;
  }

  return meta;
}

void write_bag_metadata(nlohmann::json & j, const BagMetadata & bag_metadata)
{
  j["log_file_id"] = bag_metadata.log_file_id;
  j["vehicle_id"] = bag_metadata.vehicle_id;
  j["project_id"] = bag_metadata.project_id;
  j["map_version_id"] = bag_metadata.map_version_id;
  j["date"] = bag_metadata.date;
  j["bag_time"] = bag_metadata.bag_time;
}
