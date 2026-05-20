// Copyright 2025 TIER IV, Inc.
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

#ifndef TYPES__TRAINING_DATA_BINARY_HPP_
#define TYPES__TRAINING_DATA_BINARY_HPP_

#include <autoware/diffusion_planner/dimensions.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>

// Using constants from dimensions.hpp
constexpr int64_t NEIGHBOR_PAST_DIM = autoware::diffusion_planner::NEIGHBOR_SHAPE[3];
constexpr int64_t NEIGHBOR_FUTURE_DIM = 4;  // x, y, cos(yaw), sin(yaw)

// Training data structure for binary file (all fixed size)
struct TrainingDataBinary
{
  // Header information
  uint32_t version;  // Data format version

  // Fixed size data arrays
  float ego_agent_past[autoware::diffusion_planner::EGO_HISTORY_SHAPE[1] *
                       autoware::diffusion_planner::EGO_HISTORY_SHAPE[2]];
  float ego_current_state[autoware::diffusion_planner::EGO_CURRENT_STATE_SHAPE[1]];
  float ego_agent_future[autoware::diffusion_planner::OUTPUT_T *
                         autoware::diffusion_planner::EGO_HISTORY_SHAPE[2]];
  float neighbor_agents_past[autoware::diffusion_planner::MAX_NUM_NEIGHBORS *
                             autoware::diffusion_planner::INPUT_T_WITH_CURRENT * NEIGHBOR_PAST_DIM];
  float neighbor_agents_future[autoware::diffusion_planner::MAX_NUM_NEIGHBORS *
                               autoware::diffusion_planner::OUTPUT_T * NEIGHBOR_FUTURE_DIM];
  float static_objects[autoware::diffusion_planner::STATIC_OBJECTS_SHAPE[1] *
                       autoware::diffusion_planner::STATIC_OBJECTS_SHAPE[2]];
  float lanes[autoware::diffusion_planner::NUM_SEGMENTS_IN_LANE *
              autoware::diffusion_planner::POINTS_PER_SEGMENT *
              autoware::diffusion_planner::SEGMENT_POINT_DIM];
  float lanes_speed_limit[autoware::diffusion_planner::NUM_SEGMENTS_IN_LANE];
  int32_t lanes_has_speed_limit[autoware::diffusion_planner::NUM_SEGMENTS_IN_LANE];
  float route_lanes[autoware::diffusion_planner::NUM_SEGMENTS_IN_ROUTE *
                    autoware::diffusion_planner::POINTS_PER_SEGMENT *
                    autoware::diffusion_planner::SEGMENT_POINT_DIM];
  float route_lanes_speed_limit[autoware::diffusion_planner::NUM_SEGMENTS_IN_ROUTE];
  int32_t route_lanes_has_speed_limit[autoware::diffusion_planner::NUM_SEGMENTS_IN_ROUTE];
  float polygons[autoware::diffusion_planner::NUM_POLYGONS *
                 autoware::diffusion_planner::POINTS_PER_POLYGON *
                 (2 + autoware::diffusion_planner::POLYGON_TYPE_NUM)];
  float line_strings[autoware::diffusion_planner::NUM_LINE_STRINGS *
                     autoware::diffusion_planner::POINTS_PER_LINE_STRING *
                     (2 + autoware::diffusion_planner::LINE_STRING_TYPE_NUM)];
  float goal_pose[NEIGHBOR_FUTURE_DIM];
  int32_t turn_indicators[autoware::diffusion_planner::INPUT_T_WITH_CURRENT];
  float ego_shape[autoware::diffusion_planner::EGO_SHAPE_SHAPE[1]];

  // Constructor with zero initialization
  TrainingDataBinary() : version(2)
  {
    std::fill(std::begin(ego_agent_past), std::end(ego_agent_past), 0.0f);
    std::fill(std::begin(ego_current_state), std::end(ego_current_state), 0.0f);
    std::fill(std::begin(ego_agent_future), std::end(ego_agent_future), 0.0f);
    std::fill(std::begin(neighbor_agents_past), std::end(neighbor_agents_past), 0.0f);
    std::fill(std::begin(neighbor_agents_future), std::end(neighbor_agents_future), 0.0f);
    std::fill(std::begin(static_objects), std::end(static_objects), 0.0f);
    std::fill(std::begin(lanes), std::end(lanes), 0.0f);
    std::fill(std::begin(lanes_speed_limit), std::end(lanes_speed_limit), 0.0f);
    std::fill(std::begin(lanes_has_speed_limit), std::end(lanes_has_speed_limit), 0);
    std::fill(std::begin(route_lanes), std::end(route_lanes), 0.0f);
    std::fill(std::begin(route_lanes_speed_limit), std::end(route_lanes_speed_limit), 0.0f);
    std::fill(std::begin(route_lanes_has_speed_limit), std::end(route_lanes_has_speed_limit), 0);
    std::fill(std::begin(polygons), std::end(polygons), 0.0f);
    std::fill(std::begin(line_strings), std::end(line_strings), 0.0f);
    std::fill(std::begin(goal_pose), std::end(goal_pose), 0.0f);
    std::fill(std::begin(turn_indicators), std::end(turn_indicators), 0);
    std::fill(std::begin(ego_shape), std::end(ego_shape), 0.0f);
  }
};

#endif  // TYPES__TRAINING_DATA_BINARY_HPP_
