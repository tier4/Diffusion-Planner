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

#ifndef CLI__CONVERTER_OPTIONS_HPP_
#define CLI__CONVERTER_OPTIONS_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ConverterOptions
{
  std::string rosbag_path;
  std::string vector_map_path;
  std::string save_dir;
  std::string rosbag_dir_name;

  int64_t step;
  int64_t limit;
  int64_t min_frames;
  int64_t search_nearest_route;
  int64_t convert_yellow;
  int64_t convert_red;
  int64_t interpolation;
  double min_distance;
  float ego_wheel_base;
  float ego_length;
  float ego_width;

  std::vector<float> ego_shape;
  bool use_interpolation;

  // Collision-free filter (ported from filter_collision_free_npz.py), always applied.
  // A frame whose GT ego trajectory collides with a static object, neighbor, or
  // road border is skipped during conversion (no npz written, like other skips).
  float static_object_margin;
  float neighbor_margin;
  float road_border_margin;
  int64_t collision_time_stride;

  // In-lanelet filter (ported from filter_in_lanelet_npz.py), always applied.
  // A frame whose GT ego trajectory is on average >= offlane_max_score metres from
  // any lane centerline is skipped during conversion.
  float offlane_max_score;
  int64_t offlane_time_stride;

  // When true, also write the npz for frame-level skipped frames (collision,
  // off-lane, red/yellow light, vehicle stopped) so they can be visualised with
  // their skip reason. Intended for inspection/testing only; off in production.
  bool write_skipped_npz;
};

std::optional<ConverterOptions> parse_arguments(int argc, char ** argv);

#endif  // CLI__CONVERTER_OPTIONS_HPP_
