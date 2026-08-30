// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef ML_PLANNER_DATA__SRC__BAG_DATASET_BUILDER_HPP_
#define ML_PLANNER_DATA__SRC__BAG_DATASET_BUILDER_HPP_

#include "frame_data.hpp"

#include "autoware/ml_planner/dimensions.hpp"
#include "autoware/ml_planner/preprocessing/input_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace autoware::ml_planner::data {

struct TopicConfig;

struct TopicDropThresholds {
  double kinematic_state{0.5};
  double tracked_objects{0.5};
  double turn_indicators{0.5};
  double traffic_signals{0.5};
};

struct DatasetBuilderParam {
  double frame_interval_s{0.1};
  double min_travel_distance{0.0};
  TopicDropThresholds topic_drop_thresholds{};
  double traffic_light_timeout_s{0.2};
  double neighbor_observation_timeout_s{0.3};
  int64_t num_future_steps{autoware::ml_planner::OUTPUT_T};
};

struct BagFrameMetadata {
  int64_t frame_time_ns;
  float ego_speed_mps;
  float ego_yaw_rate_rps;
  uint8_t turn_indicator;
  int32_t num_objects;
};

struct BagDataResult {
  std::vector<FrameData> frames;
  std::vector<BagFrameMetadata> metadata;
  std::vector<std::string> warnings;
  size_t all_frames{0};
  size_t usable_frames{0};
  size_t failed_frames{0};
  bool skipped{false};
};

/**
 * @brief Build every usable training frame in one bag.
 *
 * Frame selection, dropout checks, shared inference preprocessing, and label
 * generation are performed in this call. No intermediate frame index is
 * exposed or persisted.
 */
BagDataResult create_bag_frame_data(const std::string &bag_path,
                                    const std::string &map_path,
                                    const VehicleSpec &vehicle_spec,
                                    const DatasetBuilderParam &param,
                                    const TopicConfig &topics);

} // namespace autoware::ml_planner::data

#endif // ML_PLANNER_DATA__SRC__BAG_DATASET_BUILDER_HPP_
