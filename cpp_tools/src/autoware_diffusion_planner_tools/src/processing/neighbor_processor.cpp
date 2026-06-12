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

#include "types/training_data_binary.hpp"

#include <autoware/diffusion_planner/conversion/agent.hpp>
#include <autoware/diffusion_planner/dimensions.hpp>
#include <autoware_utils_uuid/uuid_helper.hpp>

#include <algorithm>
#include <string>
#include <unordered_map>

std::pair<std::vector<float>, std::vector<float>> process_neighbor_agents_and_future(
  const std::vector<FrameData> & data_list, const int64_t current_idx,
  const Eigen::Matrix4d & map2bl_matrix)
{
  using autoware::diffusion_planner::AgentHistory;
  using autoware::diffusion_planner::AGENT_STATE_DIM;
  using autoware::diffusion_planner::flatten_histories_to_vector;
  using autoware::diffusion_planner::INPUT_T_WITH_CURRENT;
  using autoware::diffusion_planner::MAX_NUM_NEIGHBORS;
  using autoware::diffusion_planner::OUTPUT_T;

  // Build agent histories using AgentData::update_histories
  const int64_t start_idx =
    std::max(static_cast<int64_t>(0), current_idx - INPUT_T_WITH_CURRENT + 1);
  const bool ignore_unknown_agents = true;
  autoware::diffusion_planner::AgentData agent_data_past;
  for (int64_t t = 0; t < INPUT_T_WITH_CURRENT; ++t) {
    const int64_t frame_idx = start_idx + t;
    if (frame_idx >= static_cast<int64_t>(data_list.size())) {
      break;
    }
    agent_data_past.update_histories(data_list[frame_idx].tracked_objects, ignore_unknown_agents);
  }
  const auto transformed_histories =
    agent_data_past.transformed_and_trimmed_histories(map2bl_matrix, MAX_NUM_NEIGHBORS);
  const std::vector<float> neighbor_past =
    flatten_histories_to_vector(transformed_histories, MAX_NUM_NEIGHBORS, INPUT_T_WITH_CURRENT);

  // Build id -> AgentHistory map for future filling
  const std::vector<AgentHistory> agent_histories = transformed_histories;
  std::unordered_map<std::string, AgentHistory> id_to_history;
  for (size_t i = 0; i < agent_histories.size(); ++i) {
    const auto object_id = agent_histories[i].get_latest_state().object_id;
    id_to_history.emplace(object_id, AgentHistory(OUTPUT_T));
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
        if (t * AGENT_STATE_DIM + d >= arr.size()) {
          break;
        }
        neighbor_future[base_idx + d] = arr[t * AGENT_STATE_DIM + d];
      }
    }
  }

  return std::make_pair(neighbor_past, neighbor_future);
}
