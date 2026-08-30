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

#include "skip_index.hpp"

#include "bag_dataset_builder.hpp"
#include "topic_config.hpp"

#include "autoware/ml_planner/constants.hpp"
#include "autoware/ml_planner/dimensions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace autoware::ml_planner::data {

namespace {

constexpr double history_window_s = HISTORY_WINDOW_S;
constexpr double future_horizon_s =
    static_cast<double>(OUTPUT_T) * constants::PREDICTION_TIME_STEP_S;

struct TopicTimeline {
  const char *topic;
  const std::vector<double> *stamps;
  double threshold;
};

bool check_enabled(const double threshold) {
  return std::isfinite(threshold) && threshold > 0.0;
}

std::string make_gap_warning(const TopicTimeline &timeline,
                             const double gap_from, const double gap_until,
                             const double ego_first_sec) {
  std::ostringstream message;
  message << std::fixed << std::setprecision(2) << timeline.topic
          << " stopped publishing for " << gap_until - gap_from
          << " s at t=" << gap_from - ego_first_sec << "-"
          << gap_until - ego_first_sec
          << " s; frames whose history/future window intersects the gap were "
             "skipped";
  return message.str();
}

void add_topic_ranges(const TopicTimeline &timeline, const double ego_first_sec,
                      std::vector<InvalidFrameRange> &ranges,
                      std::vector<std::string> &warnings) {
  if (!check_enabled(timeline.threshold)) {
    return;
  }
  if (timeline.stamps->empty()) {
    warnings.emplace_back(std::string{timeline.topic} + " never published");
    return;
  }

  const auto &stamps = *timeline.stamps;
  for (size_t i = 0; i + 1 < stamps.size(); ++i) {
    if (stamps[i + 1] - stamps[i] <= timeline.threshold) {
      continue;
    }
    ranges.push_back(
        {stamps[i] - future_horizon_s, stamps[i + 1] + history_window_s});
    warnings.push_back(
        make_gap_warning(timeline, stamps[i], stamps[i + 1], ego_first_sec));
  }
}

std::vector<InvalidFrameRange>
merge_ranges(std::vector<InvalidFrameRange> ranges) {
  std::sort(ranges.begin(), ranges.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.from < rhs.from;
  });
  std::vector<InvalidFrameRange> merged;
  for (const auto &range : ranges) {
    if (merged.empty() || range.from >= merged.back().until) {
      merged.push_back(range);
    } else {
      merged.back().until = std::max(merged.back().until, range.until);
    }
  }
  return merged;
}

size_t count_grid_frames(const double range_from, const double range_until,
                         const double grid_from, const double time_step_s,
                         const size_t num_frames) {
  if (range_from > range_until) {
    return 0;
  }
  const auto first_i =
      static_cast<int64_t>(std::ceil((range_from - grid_from) / time_step_s));
  const auto last_i =
      static_cast<int64_t>(std::floor((range_until - grid_from) / time_step_s));
  const int64_t begin = std::max<int64_t>(first_i, 0);
  const int64_t end =
      std::min<int64_t>(last_i, static_cast<int64_t>(num_frames) - 1);
  return end >= begin ? static_cast<size_t>(end - begin + 1) : 0;
}

} // namespace

FrameRange calculate_frame_range(const TopicConfig &topics,
                                 const DatasetBuilderParam &param,
                                 const std::vector<double> &ego_stamps,
                                 const std::vector<double> &turn_stamps,
                                 const std::vector<double> &objects_stamps,
                                 const std::vector<double> &traffic_stamps,
                                 const std::vector<double> &route_stamps,
                                 const size_t num_frames) {
  const double infinity = std::numeric_limits<double>::infinity();
  const double ego_first_sec = ego_stamps.front();
  const double ego_last_sec = ego_stamps.back();
  const std::array timelines = {
      TopicTimeline{topics.kinematic_state.c_str(), &ego_stamps,
                    param.topic_drop_thresholds.kinematic_state},
      TopicTimeline{topics.tracked_objects.c_str(), &objects_stamps,
                    param.topic_drop_thresholds.tracked_objects},
      TopicTimeline{topics.turn_indicators.c_str(), &turn_stamps,
                    param.topic_drop_thresholds.turn_indicators},
      TopicTimeline{topics.traffic_signals.c_str(), &traffic_stamps,
                    param.topic_drop_thresholds.traffic_signals},
  };

  const double turn_first_sec =
      turn_stamps.empty() ? ego_last_sec : turn_stamps.front();
  const double objects_first_sec =
      objects_stamps.empty() ? ego_last_sec : objects_stamps.front();
  const double structural_first_t =
      std::max({ego_first_sec, turn_first_sec, objects_first_sec}) +
      history_window_s;
  const double route_first_t =
      route_stamps.empty() ? infinity : route_stamps.front() - 1e-6;
  const double usable_from = std::max(structural_first_t, route_first_t);
  const double usable_until = ego_last_sec - future_horizon_s;

  double first_valid_t = usable_from;
  double last_valid_t = usable_until;
  std::vector<InvalidFrameRange> invalid_ranges;
  std::vector<std::string> warnings;
  for (const auto &timeline : timelines) {
    if (!check_enabled(timeline.threshold)) {
      continue;
    }
    if (timeline.stamps->empty()) {
      first_valid_t = infinity;
      last_valid_t = -infinity;
    } else {
      first_valid_t =
          std::max(first_valid_t, timeline.stamps->front() + history_window_s);
      last_valid_t =
          std::min(last_valid_t, timeline.stamps->back() + timeline.threshold -
                                     future_horizon_s);
    }
    add_topic_ranges(timeline, ego_first_sec, invalid_ranges, warnings);
  }

  return {
      first_valid_t,
      last_valid_t,
      merge_ranges(std::move(invalid_ranges)),
      std::move(warnings),
      count_grid_frames(usable_from, usable_until, ego_first_sec,
                        param.frame_interval_s, num_frames),
  };
}

bool is_frame_invalid(const std::vector<InvalidFrameRange> &invalid_ranges,
                      const double frame_time, size_t &range_cursor) {
  while (range_cursor < invalid_ranges.size() &&
         frame_time >= invalid_ranges[range_cursor].until) {
    ++range_cursor;
  }
  return range_cursor < invalid_ranges.size() &&
         frame_time > invalid_ranges[range_cursor].from &&
         frame_time < invalid_ranges[range_cursor].until;
}

std::optional<std::string> check_min_travel_distance(
    const std::vector<std::pair<double, double>> &ego_positions,
    const double min_travel_distance) {
  if (min_travel_distance <= 0.0) {
    return std::nullopt;
  }

  double travel_distance = 0.0;
  for (size_t i = 1; i < ego_positions.size(); ++i) {
    travel_distance +=
        std::hypot(ego_positions[i].first - ego_positions[i - 1].first,
                   ego_positions[i].second - ego_positions[i - 1].second);
  }
  if (travel_distance > min_travel_distance) {
    return std::nullopt;
  }

  std::ostringstream message;
  message << std::fixed << std::setprecision(2) << "bag traveled "
          << travel_distance
          << " m, at or below min_travel_distance=" << min_travel_distance
          << " m; skipped the entire bag";
  return message.str();
}

} // namespace autoware::ml_planner::data
