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

#include "conversion/selected_frames.hpp"

#include "io/frame_writer.hpp"
#include "io/npz_frame_writer.hpp"
#include "processing/sequence_builder.hpp"
#include "processing/stopped_tail.hpp"

#include <nlohmann/json.hpp>
#include <rclcpp/time.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>

int export_selected_frames(
  ParsedBagData & data, const ConverterPaths & paths, const ConverterOptions & options,
  const autoware::diffusion_planner::preprocess::LaneSegmentContext & lane_context,
  const BagMetadata & bag_metadata)
{
  namespace fs = std::filesystem;
  std::ifstream input(options.timestamps);
  const auto requested = nlohmann::json::parse(input);
  if (!requested.is_array() || requested.empty()) {
    throw std::invalid_argument("--timestamps must contain a nonempty JSON array");
  }
  std::vector<int64_t> ticks;
  for (const auto & value : requested) {
    if (
      !value.is_number_integer() || value.get<double>() <= 0 ||
      (value.is_number_unsigned() &&
       value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
      throw std::invalid_argument("timestamps must be positive int64 nanoseconds");
    }
    ticks.push_back(value.get<int64_t>());
  }
  if (
    !std::is_sorted(ticks.begin(), ticks.end()) ||
    std::adjacent_find(ticks.begin(), ticks.end()) != ticks.end()) {
    throw std::invalid_argument("timestamps must be strictly increasing");
  }
  const fs::path directory = fs::path(paths.save_dir) / "selected";
  const fs::path report_path = fs::path(paths.save_dir) / "requests.json";
  if (fs::exists(directory) || fs::exists(report_path)) {
    throw std::invalid_argument("selected export requires a fresh output directory");
  }

  // Reuse the native clock/message assembly, but do not let route boundaries truncate
  // real temporal context. Moving frames avoids retaining two copies of the bag.
  auto sequences = build_sequences(data, options.search_nearest_route);
  std::vector<FrameData> frames;
  std::map<int64_t, autoware_planning_msgs::msg::LaneletRoute> routes;
  size_t count = 0;
  for (const auto & seq : sequences) count += seq.data_list.size();
  frames.reserve(count);
  for (auto & seq : sequences) {
    if (seq.data_list.empty()) continue;
    // Preserve the native route assignment and that route's stopped-end goal.
    // An earlier route must never inherit the full bag's final goal.
    stopped_tail::prepare_sequence_goal(seq);
    routes.emplace(seq.data_list.front().timestamp, std::move(seq.route));
    for (auto & frame : seq.data_list) frames.push_back(std::move(frame));
  }
  sequences.clear();
  std::sort(frames.begin(), frames.end(), [](const auto & a, const auto & b) {
    return a.timestamp < b.timestamp;
  });
  // Temporal context is the full bag, so only its actual end can be held.
  const bool ends_stopped = stopped_tail::ends_stopped(frames);
  nlohmann::json results = nlohmann::json::array();
  bool failed = false;
  for (const int64_t tick : ticks) {
    nlohmann::json result = {{"requested_time_ns", tick}};
    const auto pos = std::lower_bound(
      frames.begin(), frames.end(), tick,
      [](const auto & frame, int64_t t) { return frame.timestamp < t; });
    std::optional<std::string> issue;
    if (pos == frames.end() || pos->timestamp != tick) {
      issue = "request is outside the native bag clock or off its 10 Hz grid";
    }
    if (!issue) {
      const auto & route = std::prev(routes.upper_bound(tick))->second;
      const auto tensors = create_frame_tensors(
        frames, pos - frames.begin(), route, options, lane_context, ends_stopped);
      if (!tensors) {
        issue = "native ego tensors unavailable";
      } else {
        const auto & t = *tensors;
        const std::string token = std::to_string(tick);
        save_frame_data_npz(
          directory.string(), paths.get_rosbag_dir_name(), token, t.ego_past, t.ego_current,
          t.ego_future, t.neighbor_result.neighbor_past, t.neighbor_result.neighbor_future,
          t.static_objects, t.lanes, t.lanes_speed_limit, t.lanes_has_speed_limit, t.route_lanes,
          t.route_lanes_speed_limit, t.route_lanes_has_speed_limit, t.polygons, t.line_strings,
          t.goal_pose_vec, t.turn_indicators, t.ego_shape);
        save_frame_json(
          directory.string(), paths.get_rosbag_dir_name(), token, pos->kinematic_state, tick,
          SkippingInfo::accepted(), t.neighbor_result.neighbor_ids, bag_metadata);
        result["status"] = "ok";
        result["output"] = "selected/" + paths.get_rosbag_dir_name() + "_" + token + ".npz";
        result["actual_odometry_time_ns"] =
          rclcpp::Time(pos->kinematic_state.header.stamp).nanoseconds();
      }
    }
    if (issue) {
      failed = true;
      result["status"] = "unavailable";
      result["reason"] = *issue;
    }
    results.push_back(std::move(result));
  }
  fs::create_directories(paths.save_dir);
  std::ofstream report(report_path);
  report.exceptions(std::ios::failbit | std::ios::badbit);
  report << results.dump(2) << '\n';
  return failed ? 2 : 0;
}
