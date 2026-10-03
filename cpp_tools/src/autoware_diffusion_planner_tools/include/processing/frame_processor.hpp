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

#ifndef PROCESSING__FRAME_PROCESSOR_HPP_
#define PROCESSING__FRAME_PROCESSOR_HPP_

#include "cli/converter_options.hpp"
#include "io/bag_metadata.hpp"
#include "processing/neighbor_processor.hpp"
#include "timestamp_stats.hpp"
#include "types/frame_data.hpp"

#include <autoware/diffusion_planner/preprocessing/lane_segments.hpp>

#include <cstdint>
#include <optional>

// Native tensor construction shared by training sequences and selected-frame export.
// Selection, filtering and file writing are intentionally left to the callers.
struct FrameTensors
{
  std::vector<float> ego_past, ego_current, ego_future;
  NeighborResult neighbor_result;
  std::vector<float> lanes, lanes_speed_limit;
  std::vector<uint8_t> lanes_has_speed_limit;
  std::vector<float> route_lanes, route_lanes_speed_limit;
  std::vector<uint8_t> route_lanes_has_speed_limit;
  std::vector<float> polygons, line_strings, goal_pose_vec, static_objects;
  std::vector<int32_t> turn_indicators;
  std::vector<float> ego_shape;
};

std::optional<FrameTensors> create_frame_tensors(
  const std::vector<FrameData> & data_list, int64_t index,
  const autoware_planning_msgs::msg::LaneletRoute & route, const ConverterOptions & options,
  const autoware::diffusion_planner::preprocess::LaneSegmentContext & lane_segment_context,
  bool allow_hold_past_end);

void process_sequence(
  SequenceData & seq, const int64_t seq_id, const ConverterPaths & paths,
  const ConverterOptions & options,
  const autoware::diffusion_planner::preprocess::LaneSegmentContext & lane_segment_context,
  const timestamp_stats::TimestampStatsMap & timestamp_stats_map, const BagMetadata & bag_metadata);

#endif  // PROCESSING__FRAME_PROCESSOR_HPP_
