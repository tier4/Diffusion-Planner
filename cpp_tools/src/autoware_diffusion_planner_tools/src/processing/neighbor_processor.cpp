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

#include "processing/neighbor_processor.hpp"

#include <autoware/diffusion_planner/conversion/agent.hpp>
#include <autoware/diffusion_planner/dimensions.hpp>
#include <autoware/object_recognition_utils/object_recognition_utils.hpp>
#include <autoware_utils_uuid/uuid_helper.hpp>
#include <rclcpp/time.hpp>

#include <autoware_perception_msgs/msg/object_classification.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

constexpr int64_t NEIGHBOR_FUTURE_DIM = 4;  // x, y, cos(yaw), sin(yaw)

namespace
{

bool is_unknown_object(const autoware_perception_msgs::msg::TrackedObject & object)
{
  const auto label = autoware::object_recognition_utils::getHighestProbLabel(object.classification);
  return label == autoware_perception_msgs::msg::ObjectClassification::UNKNOWN;
}

std::vector<autoware::diffusion_planner::AgentHistory> build_past_histories(
  const std::vector<FrameData> & data_list, const int64_t current_idx,
  const Eigen::Matrix4d & map2bl_matrix)
{
  using autoware::diffusion_planner::AgentHistory;
  using autoware::diffusion_planner::INPUT_T_WITH_CURRENT;
  using autoware::diffusion_planner::MAX_NUM_NEIGHBORS;

  const int64_t start_idx =
    std::max(static_cast<int64_t>(0), current_idx - INPUT_T_WITH_CURRENT + 1);
  std::unordered_map<std::string, AgentHistory> histories_map;

  for (int64_t t = 0; t < INPUT_T_WITH_CURRENT; ++t) {
    const int64_t frame_idx = start_idx + t;
    if (frame_idx >= static_cast<int64_t>(data_list.size())) {
      break;
    }

    const auto & tracked_objects = data_list[frame_idx].tracked_objects;
    const rclcpp::Time objects_timestamp(tracked_objects.header.stamp);
    std::vector<std::string> found_ids;
    found_ids.reserve(tracked_objects.objects.size());

    for (const auto & object : tracked_objects.objects) {
      if (is_unknown_object(object)) {
        continue;
      }
      const std::string object_id = autoware_utils_uuid::to_hex_string(object.object_id);
      auto it = histories_map.find(object_id);
      if (it == histories_map.end()) {
        it = histories_map.emplace(object_id, AgentHistory(INPUT_T_WITH_CURRENT)).first;
      }
      it->second.update(object, objects_timestamp);
      found_ids.push_back(object_id);
    }

    for (auto it = histories_map.begin(); it != histories_map.end();) {
      if (std::find(found_ids.begin(), found_ids.end(), it->first) == found_ids.end()) {
        it = histories_map.erase(it);
      } else {
        ++it;
      }
    }
  }

  std::vector<AgentHistory> histories;
  histories.reserve(histories_map.size());
  for (const auto & [_, history] : histories_map) {
    histories.push_back(history);
  }
  for (auto & history : histories) {
    history.apply_transform(map2bl_matrix);
  }
  std::sort(histories.begin(), histories.end(), [](const AgentHistory & a, const AgentHistory & b) {
    const double a_dist =
      std::hypot(a.get_latest_state().pose(0, 3), a.get_latest_state().pose(1, 3));
    const double b_dist =
      std::hypot(b.get_latest_state().pose(0, 3), b.get_latest_state().pose(1, 3));
    return a_dist < b_dist;
  });
  if (histories.size() > MAX_NUM_NEIGHBORS) {
    histories.erase(
      histories.begin() + static_cast<std::ptrdiff_t>(MAX_NUM_NEIGHBORS), histories.end());
  }

  return histories;
}

std::vector<float> flatten_histories_with_leading_padding(
  const std::vector<autoware::diffusion_planner::AgentHistory> & histories,
  const size_t max_num_agent, const size_t time_length)
{
  using autoware::diffusion_planner::AGENT_STATE_DIM;

  std::vector<float> data;
  data.reserve(max_num_agent * time_length * AGENT_STATE_DIM);

  for (const auto & history : histories) {
    const auto history_array = history.as_array();
    const size_t observed_steps = history_array.size() / AGENT_STATE_DIM;
    const size_t padding_steps = time_length > observed_steps ? time_length - observed_steps : 0;
    data.insert(data.end(), padding_steps * AGENT_STATE_DIM, 0.0f);
    data.insert(data.end(), history_array.begin(), history_array.end());
  }

  data.resize(max_num_agent * time_length * AGENT_STATE_DIM, 0.0f);
  return data;
}

}  // namespace

NeighborResult process_neighbor_agents_and_future(
  const std::vector<FrameData> & data_list, const int64_t current_idx,
  const Eigen::Matrix4d & map2bl_matrix)
{
  using autoware::diffusion_planner::AGENT_STATE_DIM;
  using autoware::diffusion_planner::AgentHistory;
  using autoware::diffusion_planner::INPUT_T_WITH_CURRENT;
  using autoware::diffusion_planner::MAX_NUM_NEIGHBORS;
  using autoware::diffusion_planner::OUTPUT_T;

  const auto transformed_histories = build_past_histories(data_list, current_idx, map2bl_matrix);
  const std::vector<float> neighbor_past = flatten_histories_with_leading_padding(
    transformed_histories, MAX_NUM_NEIGHBORS, INPUT_T_WITH_CURRENT);

  // Build id -> AgentHistory map for future filling
  const std::vector<AgentHistory> agent_histories = transformed_histories;
  // Ordered track UUIDs, aligned 1:1 with the neighbor_past slots (same sorted +
  // trimmed order as flatten_histories_to_vector above). Persisted to the sidecar.
  std::vector<std::string> neighbor_ids;
  neighbor_ids.reserve(agent_histories.size());
  for (const auto & h : agent_histories) {
    neighbor_ids.push_back(h.get_latest_state().object_id);
  }
  std::unordered_map<std::string, AgentHistory> id_to_history;
  for (size_t i = 0; i < agent_histories.size(); ++i) {
    const auto object_id = agent_histories[i].get_latest_state().object_id;
    // Capacity must hold the current state plus OUTPUT_T future states. With only
    // OUTPUT_T slots, pushing the current state followed by OUTPUT_T future frames
    // evicts the current state, leaving the last future timestep unfilled (zeros).
    // The `(t + 1)` offset below skips the current state at index 0.
    id_to_history.emplace(object_id, AgentHistory(OUTPUT_T + 1));
    id_to_history.at(object_id).update(
      agent_histories[i].get_latest_state().original_info,
      agent_histories[i].get_latest_state().timestamp);
  }

  // Future data: use AgentHistory for each agent
  std::vector<float> neighbor_future(MAX_NUM_NEIGHBORS * OUTPUT_T * NEIGHBOR_FUTURE_DIM, 0.0f);
  for (int64_t agent_idx = 0; agent_idx < static_cast<int64_t>(agent_histories.size());
       ++agent_idx) {
    const std::string & agent_id_str = agent_histories[agent_idx].get_latest_state().object_id;
    AgentHistory & future_history = id_to_history.at(agent_id_str);
    for (int64_t t = 1; t <= OUTPUT_T; ++t) {
      const int64_t future_frame_idx = current_idx + t;
      if (future_frame_idx >= static_cast<int64_t>(data_list.size())) {
        break;
      }
      // Find object with same id in future frame
      const auto & future_objects = data_list[future_frame_idx].tracked_objects.objects;
      bool found = false;
      for (const auto & obj : future_objects) {
        const std::string obj_id = autoware_utils_uuid::to_hex_string(obj.object_id);
        if (obj_id == agent_id_str) {
          future_history.update(obj, data_list[future_frame_idx].kinematic_state.header.stamp);
          found = true;
          break;
        }
      }
      if (!found) {
        break;
      }
    }
    future_history.apply_transform(map2bl_matrix);

    // Fill future array for this agent
    const std::vector<float> arr = future_history.as_array();
    for (int64_t t = 0; t < OUTPUT_T; ++t) {
      const int64_t base_idx = agent_idx * OUTPUT_T * NEIGHBOR_FUTURE_DIM + t * NEIGHBOR_FUTURE_DIM;
      for (int64_t d = 0; d < NEIGHBOR_FUTURE_DIM; ++d) {
        const size_t arr_idx = (t + 1) * AGENT_STATE_DIM + d;
        if (arr_idx >= arr.size()) {
          break;
        }
        neighbor_future[base_idx + d] = arr[arr_idx];
      }
    }
  }

  return NeighborResult{neighbor_past, neighbor_future, neighbor_ids};
}
