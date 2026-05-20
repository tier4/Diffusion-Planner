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

#ifndef PROCESSING__NEIGHBOR_PROCESSOR_HPP_
#define PROCESSING__NEIGHBOR_PROCESSOR_HPP_

#include "types/frame_data.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <utility>
#include <vector>

std::pair<std::vector<float>, std::vector<float>> process_neighbor_agents_and_future(
  const std::vector<FrameData> & data_list, const int64_t current_idx,
  const Eigen::Matrix4d & map2bl_matrix);

#endif  // PROCESSING__NEIGHBOR_PROCESSOR_HPP_
