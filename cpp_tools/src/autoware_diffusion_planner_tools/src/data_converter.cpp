// Copyright 2025 TIER IV, Inc.
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

#include "conversion/data_converter.hpp"

#include "cli/converter_options.hpp"

#include <CLI/CLI.hpp>

#include <fmt/core.h>

#include <iostream>

namespace
{

void print_options(const ConverterPaths & paths, const ConverterOptions & converter)
{
  fmt::print(
    "Ego wheel base: {}, Ego length: {}, Ego width: {}\n"
    "Processing rosbag: {}\n"
    "Vector map: {}\n"
    "Save directory: {}\n"
    "Step: {}, Limit: {}, Min frames: {}, Min distance: {}, Search nearest route: {}, "
    "Convert yellow: {}, Convert red: {}, Interpolation: {}\n"
    "Collision filter static_object_margin: {}, neighbor_margin: {}, road_border_margin: {}, "
    "collision_time_stride: {}\n"
    "Off-lane filter max_score: {}, offlane_time_stride: {}\n"
    "Write skipped npz: {}\n",
    converter.ego_wheel_base, converter.ego_length, converter.ego_width, paths.rosbag_path,
    paths.vector_map_path, paths.save_dir, converter.step, converter.limit, converter.min_frames,
    converter.min_distance, converter.search_nearest_route, converter.convert_yellow,
    converter.convert_red, converter.use_interpolation, converter.static_object_margin,
    converter.neighbor_margin, converter.road_border_margin, converter.collision_time_stride,
    converter.offlane_max_score, converter.offlane_time_stride, converter.write_skipped_npz);
}

bool parse_arguments(int argc, char ** argv, ConverterPaths & paths, ConverterOptions & converter)
{
  converter = ConverterOptions::default_converter_options();

  CLI::App app{"Convert one rosbag to Diffusion Planner npz files"};
  app.add_option("rosbag_path", paths.rosbag_path, "Input rosbag directory.")->required();
  app
    .add_option(
      "vector_map_path", paths.vector_map_path,
      "Path to the Lanelet2 vector map file, typically lanelet2_map.osm.")
    ->required();
  app
    .add_option(
      "save_dir", paths.save_dir, "Directory where converted npz and json files are written.")
    ->required();
  converter.add_converter_options(app);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError & e) {
    app.exit(e);
    return false;
  }

  if (const auto err = validate_options(converter)) {
    std::cerr << *err << std::endl;
    return false;
  }
  print_options(paths, converter);
  return true;
}

}  // namespace

int main(int argc, char ** argv)
{
  ConverterPaths paths;
  ConverterOptions converter;
  if (!parse_arguments(argc, argv, paths, converter)) {
    return 1;
  }

  return run_data_converter(paths, converter);
}
