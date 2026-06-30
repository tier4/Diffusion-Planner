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
#include "timestamp_stats.hpp"
#include "types/frame_data.hpp"

#include <autoware/diffusion_planner/preprocessing/lane_segments.hpp>

#include <cstdint>

void process_sequence(
  SequenceData & seq, const int64_t seq_id, const ConverterPaths & paths,
  const ConverterOptions & options,
  const autoware::diffusion_planner::preprocess::LaneSegmentContext & lane_segment_context,
  const timestamp_stats::TimestampStatsMap & timestamp_stats_map);

#endif  // PROCESSING__FRAME_PROCESSOR_HPP_
