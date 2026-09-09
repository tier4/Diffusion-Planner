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

#ifndef ML_PLANNER_DATA__SRC__SKIP_INDEX_HPP_
#define ML_PLANNER_DATA__SRC__SKIP_INDEX_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace autoware::ml_planner::data {

struct DatasetBuilderParam;
struct TopicConfig;

struct InvalidFrameRange {
  double from;
  double until;
};

struct FrameRange {
  double first_valid_t;
  double last_valid_t;
  std::vector<InvalidFrameRange> invalid_ranges;
  std::vector<std::string> warnings;
  size_t usable_frames;
};

FrameRange calculate_frame_range(const TopicConfig &topics,
                                 const DatasetBuilderParam &param,
                                 const std::vector<double> &ego_stamps,
                                 const std::vector<double> &turn_stamps,
                                 const std::vector<double> &objects_stamps,
                                 const std::vector<double> &traffic_stamps,
                                 const std::vector<double> &route_stamps,
                                 size_t num_frames);

bool is_frame_invalid(const std::vector<InvalidFrameRange> &invalid_ranges,
                      double frame_time, size_t &range_cursor);

std::optional<std::string> check_min_travel_distance(
    const std::vector<std::pair<double, double>> &ego_positions,
    double min_travel_distance);

} // namespace autoware::ml_planner::data

#endif // ML_PLANNER_DATA__SRC__SKIP_INDEX_HPP_
