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

#include "cli/converter_options.hpp"

#include <filesystem>
#include <iostream>
#include <string>

std::optional<ConverterOptions> parse_arguments(int argc, char ** argv)
{
  if (argc < 4) {
    std::cerr << "Usage: data_converter <rosbag_path> <vector_map_path> <save_dir> [--step=1] "
                 "[--limit=-1] [--min_frames=1700] [--min_distance=50.0] [--convert_yellow=0] "
                 "[--convert_red=0] [--interpolation=1] "
                 "[--ego_wheel_base=2.75] [--ego_length=4.34] [--ego_width=1.70]"
              << std::endl;
    return std::nullopt;
  }

  ConverterOptions options;
  options.rosbag_path = argv[1];
  options.vector_map_path = argv[2];
  options.save_dir = argv[3];
  options.rosbag_dir_name = std::filesystem::path(options.rosbag_path).filename();

  options.step = 1;
  options.limit = -1;
  options.min_frames = 1700;
  options.search_nearest_route = 1;
  options.convert_yellow = 0;
  options.convert_red = 0;
  options.interpolation = 1;
  options.min_distance = 50.0;
  options.ego_wheel_base = -1.0;
  options.ego_length = -1.0;
  options.ego_width = -1.0;

  // Collision-free filter defaults match filter_collision_free_npz.py.
  options.static_object_margin = 0.0f;
  options.neighbor_margin = 0.0f;
  options.road_border_margin = 0.0f;
  options.collision_time_stride = 5;

  // In-lanelet filter defaults match filter_in_lanelet_npz.py.
  options.offlane_max_score = 6.0f;
  options.offlane_time_stride = 1;

  // Inspection-only: production keeps this off so skipped frames write no npz.
  options.write_skipped_npz = false;

  for (int64_t i = 4; i < argc; ++i) {
    const std::string arg = argv[i];
    std::cout << "arg[" << i << "] = " << arg << std::endl;
    if (arg.find("--step=") == 0) {
      options.step = std::stoll(arg.substr(7));
    } else if (arg.find("--limit=") == 0) {
      options.limit = std::stoll(arg.substr(8));
    } else if (arg.find("--min_frames=") == 0) {
      options.min_frames = std::stoll(arg.substr(13));
    } else if (arg.find("--min_distance=") == 0) {
      options.min_distance = std::stod(arg.substr(15));
    } else if (arg.find("--search_nearest_route=") == 0) {
      options.search_nearest_route = std::stoll(arg.substr(23));
    } else if (arg.find("--convert_yellow=") == 0) {
      options.convert_yellow = std::stoll(arg.substr(17));
    } else if (arg.find("--convert_red=") == 0) {
      options.convert_red = std::stoll(arg.substr(14));
    } else if (arg.find("--interpolation=") == 0) {
      options.interpolation = std::stoll(arg.substr(16));
    } else if (arg.find("--ego_wheel_base=") == 0) {
      options.ego_wheel_base = std::stof(arg.substr(17));
    } else if (arg.find("--ego_length=") == 0) {
      options.ego_length = std::stof(arg.substr(13));
    } else if (arg.find("--ego_width=") == 0) {
      options.ego_width = std::stof(arg.substr(12));
    } else if (arg.find("--static_object_margin=") == 0) {
      options.static_object_margin = std::stof(arg.substr(23));
    } else if (arg.find("--neighbor_margin=") == 0) {
      options.neighbor_margin = std::stof(arg.substr(18));
    } else if (arg.find("--road_border_margin=") == 0) {
      options.road_border_margin = std::stof(arg.substr(21));
    } else if (arg.find("--collision_time_stride=") == 0) {
      options.collision_time_stride = std::stoll(arg.substr(24));
    } else if (arg.find("--offlane_max_score=") == 0) {
      options.offlane_max_score = std::stof(arg.substr(20));
    } else if (arg.find("--offlane_time_stride=") == 0) {
      options.offlane_time_stride = std::stoll(arg.substr(22));
    } else if (arg.find("--write_skipped_npz=") == 0) {
      options.write_skipped_npz = static_cast<bool>(std::stoll(arg.substr(20)));
    }
  }

  std::cout << "Ego wheel base: " << options.ego_wheel_base << ", Ego length: " << options.ego_length
            << ", Ego width: " << options.ego_width << std::endl;
  if (options.ego_wheel_base < 0.0 || options.ego_length < 0.0 || options.ego_width < 0.0) {
    std::cerr << "Ego vehicle dimensions must be specified with positive values." << std::endl;
    return std::nullopt;
  }
  options.ego_shape = {options.ego_wheel_base, options.ego_length, options.ego_width};

  std::cout << "Processing rosbag: " << options.rosbag_path << std::endl;
  std::cout << "Vector map: " << options.vector_map_path << std::endl;
  std::cout << "Save directory: " << options.save_dir << std::endl;
  options.use_interpolation = static_cast<bool>(options.interpolation);
  std::cout << "Step: " << options.step << ", Limit: " << options.limit
            << ", Min frames: " << options.min_frames << ", Min distance: " << options.min_distance
            << ", Search nearest route: " << options.search_nearest_route
            << ", Convert yellow: " << options.convert_yellow
            << ", Convert red: " << options.convert_red
            << ", Interpolation: " << options.use_interpolation << std::endl;
  std::cout << "Collision filter static_object_margin: " << options.static_object_margin
            << ", neighbor_margin: " << options.neighbor_margin
            << ", road_border_margin: " << options.road_border_margin
            << ", collision_time_stride: " << options.collision_time_stride << std::endl;
  std::cout << "Off-lane filter max_score: " << options.offlane_max_score
            << ", offlane_time_stride: " << options.offlane_time_stride << std::endl;
  std::cout << "Write skipped npz: " << options.write_skipped_npz << std::endl;

  return options;
}
