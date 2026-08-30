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

#ifndef ML_PLANNER_DATA__SRC__FRAME_DATA_CACHE_HPP_
#define ML_PLANNER_DATA__SRC__FRAME_DATA_CACHE_HPP_

#include "bag_frame_reader.hpp"
#include "frame_data.hpp"
#include "lru_cache.hpp"
#include "topic_config.hpp"

#include "autoware/ml_planner/preprocessing/input_builder.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace autoware::ml_planner::data {

/**
 * @brief LRU-cached bag readers and map contexts.
 *
 * Intended usage: one instance per DataLoader worker process. Access is
 * expected to be (near-)chronological within a bag for good performance.
 */
class FrameDataCache {
public:
  FrameDataCache(size_t reader_capacity, size_t map_capacity,
                 const TopicConfig &topics, double line_string_max_step_m);

  /**
   * @brief Build both the model inputs and the training labels for one frame,
   * merged into a single map. Returns an error if the frame is not usable.
   */
  FrameDataResult
  create_frame_data(const std::string &bag_path, const std::string &map_path,
                    int64_t frame_time_ns, const VehicleSpec &vehicle_spec,
                    double traffic_light_timeout_s, int64_t num_future_steps,
                    double neighbor_observation_timeout_s);

private:
  BagFrameReader &reader_for(const std::string &bag_path);

  LruCache<std::shared_ptr<BagFrameReader>> readers_;
  LruCache<std::shared_ptr<const preprocess::LaneSegmentContext>> map_contexts_;
  TopicConfig topics_;
  double line_string_max_step_m_;
};

} // namespace autoware::ml_planner::data

#endif // ML_PLANNER_DATA__SRC__FRAME_DATA_CACHE_HPP_
