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

#ifndef IO__NPZ_FRAME_WRITER_HPP_
#define IO__NPZ_FRAME_WRITER_HPP_

#include <cstdint>
#include <string>
#include <vector>

void save_frame_data_npz(
  const std::string & output_path, const std::string & rosbag_dir_name, const std::string & token,
  const std::vector<float> & ego_past, const std::vector<float> & ego_current,
  const std::vector<float> & ego_future, const std::vector<float> & neighbor_past,
  const std::vector<float> & neighbor_future, const std::vector<float> & static_objects,
  const std::vector<float> & lanes, const std::vector<float> & lanes_speed_limit,
  const std::vector<uint8_t> & lanes_has_speed_limit, const std::vector<float> & route_lanes,
  const std::vector<float> & route_lanes_speed_limit,
  const std::vector<uint8_t> & route_lanes_has_speed_limit, const std::vector<float> & polygons,
  const std::vector<float> & line_strings, const std::vector<float> & goal_pose,
  const std::vector<int32_t> & turn_indicators, const std::vector<float> & ego_shape);

#endif  // IO__NPZ_FRAME_WRITER_HPP_
