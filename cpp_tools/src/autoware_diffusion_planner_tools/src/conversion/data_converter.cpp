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

#include "conversion/data_converter.hpp"

#include "io/frame_writer.hpp"
#include "io/projector_factory.hpp"
#include "processing/frame_processor.hpp"
#include "processing/sequence_builder.hpp"
#include "rosbag/parsed_bag_data.hpp"

#include <autoware/diffusion_planner/preprocessing/lane_segments.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_io/Io.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

int run_data_converter(const ConverterPaths & paths, const ConverterOptions & converter)
{
  lanelet::ErrorMessages errors{};
  const std::unique_ptr<lanelet::Projector> projector =
    create_projector_from_yaml(paths.vector_map_path);
  const std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr =
    lanelet::load(paths.vector_map_path, *projector, &errors);

  std::cout << "Loaded lanelet2 map with " << lanelet_map_ptr->laneletLayer.size() << " lanelets"
            << std::endl;

  const autoware::diffusion_planner::preprocess::LaneSegmentContext lane_segment_context(
    lanelet_map_ptr);
  const std::string rosbag_dir_name = paths.get_rosbag_dir_name();

  ParsedBagData bag_data = load_rosbag(paths.rosbag_path, converter.limit);

  const auto missing_topics_skip = check_missing_topics(bag_data);
  if (missing_topics_skip) {
    std::cout << "Skipping rosbag due to missing required topics:" << std::endl;
    for (const auto & t : missing_topics_skip->missing_topic_types) {
      std::cout << "  - " << to_topic_name(t) << std::endl;
    }
    std::cout << "No training samples will be generated from this rosbag." << std::endl;
    save_route_json(
      paths.save_dir, rosbag_dir_name, "missing_topics", 0, 0.0, 0, 0, missing_topics_skip.value(),
      bag_data.timestamp_stats_map);
    return 0;
  }

  std::vector<SequenceData> sequences = build_sequences(bag_data, converter.search_nearest_route);

  std::cout << "Total " << sequences.size() << " sequences" << std::endl;

  for (int64_t seq_id = 0; seq_id < static_cast<int64_t>(sequences.size()); ++seq_id) {
    process_sequence(
      sequences[seq_id], seq_id, paths, converter, lane_segment_context,
      bag_data.timestamp_stats_map);
  }

  std::cout << "Data conversion completed!" << std::endl;
  return 0;
}
