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

#ifndef CONVERSION__SELECTED_FRAMES_HPP_
#define CONVERSION__SELECTED_FRAMES_HPP_

#include "processing/frame_processor.hpp"
#include "rosbag/parsed_bag_data.hpp"

// Writes requests.json with an outcome for each requested tick. Returns 2 for partial
// availability, 0 for complete success; malformed requests and I/O errors throw.
int export_selected_frames(
  ParsedBagData & data, const ConverterPaths & paths, const ConverterOptions & options,
  const autoware::diffusion_planner::preprocess::LaneSegmentContext & lane_context,
  const BagMetadata & bag_metadata);

#endif
