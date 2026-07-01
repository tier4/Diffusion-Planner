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

// ---------------------------------------------------------------------------
// Pack-sequence mode: accumulate every frame of a sequence, then write a single
// npz whose arrays are stacked along a leading frame axis (shape {num_frames, ...}).
// ---------------------------------------------------------------------------

// Holds the concatenated per-frame buffers for one sequence. Each buffer is the row-major
// concatenation of every frame's data (in frame order), so reshaping to {num_frames, ...}
// yields the stacked array. The cos/sin->heading conversion matches save_frame_data_npz.
struct SequenceNpzData
{
  int64_t num_frames;
  // Per-frame flat sizes captured from the appended frames; constant across a sequence.
  size_t ego_current_dim;
  size_t turn_indicators_dim;
  size_t ego_shape_dim;
  // In-sequence frame index for each stacked slice, shape {num_frames}.
  std::vector<int64_t> frame_indices;
  std::vector<float> ego_agent_past;
  std::vector<float> ego_current_state;
  std::vector<float> ego_agent_future;
  std::vector<float> neighbor_agents_past;
  std::vector<float> neighbor_agents_future;
  std::vector<float> static_objects;
  std::vector<float> lanes;
  std::vector<float> lanes_speed_limit;
  std::vector<uint8_t> lanes_has_speed_limit;
  std::vector<float> route_lanes;
  std::vector<float> route_lanes_speed_limit;
  std::vector<uint8_t> route_lanes_has_speed_limit;
  std::vector<float> polygons;
  std::vector<float> line_strings;
  std::vector<float> goal_pose;
  std::vector<int32_t> turn_indicators;
  std::vector<float> ego_shape;

  SequenceNpzData();
};

// Append one frame's tensors to the accumulator (same args as save_frame_data_npz, minus the
// output path/token; frame_index identifies the in-sequence frame for the frame_indices array).
void add_frame_to_sequence_npz(
  SequenceNpzData & acc, const int64_t frame_index, const std::vector<float> & ego_past,
  const std::vector<float> & ego_current, const std::vector<float> & ego_future,
  const std::vector<float> & neighbor_past, const std::vector<float> & neighbor_future,
  const std::vector<float> & static_objects, const std::vector<float> & lanes,
  const std::vector<float> & lanes_speed_limit, const std::vector<uint8_t> & lanes_has_speed_limit,
  const std::vector<float> & route_lanes, const std::vector<float> & route_lanes_speed_limit,
  const std::vector<uint8_t> & route_lanes_has_speed_limit, const std::vector<float> & polygons,
  const std::vector<float> & line_strings, const std::vector<float> & goal_pose,
  const std::vector<int32_t> & turn_indicators, const std::vector<float> & ego_shape);

// Write the accumulated sequence as a single <rosbag>_<sequence_id>.npz with frame-stacked arrays.
void save_sequence_data_npz(
  const std::string & output_path, const std::string & rosbag_dir_name,
  const std::string & sequence_id, const SequenceNpzData & acc);

#endif  // IO__NPZ_FRAME_WRITER_HPP_
