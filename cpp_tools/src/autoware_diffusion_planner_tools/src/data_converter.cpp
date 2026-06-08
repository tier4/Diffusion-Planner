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

#include "cli/converter_options.hpp"
#include "io/frame_writer.hpp"
#include "io/projector_factory.hpp"
#include "processing/frame_processor.hpp"
#include "processing/sequence_builder.hpp"
#include "rosbag/parsed_bag_data.hpp"

#include <autoware/diffusion_planner/preprocessing/lane_segments.hpp>
#include <rclcpp/rclcpp.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_io/Io.h>

#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  const auto options_opt = parse_arguments(argc, argv);
  if (!options_opt) {
    return 1;
  }
  const ConverterOptions & options = options_opt.value();

  // Load Lanelet2 map using projector chosen by map_projector_info.yaml.
  lanelet::ErrorMessages errors{};
  const std::unique_ptr<lanelet::Projector> projector =
    create_projector_from_yaml(options.vector_map_path);
  const std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr =
    lanelet::load(options.vector_map_path, *projector, &errors);

  std::cout << "Loaded lanelet2 map with " << lanelet_map_ptr->laneletLayer.size() << " lanelets"
            << std::endl;

  const autoware::diffusion_planner::preprocess::LaneSegmentContext lane_segment_context(
    lanelet_map_ptr);

  ParsedBagData bag_data = load_rosbag(options.rosbag_path, options.limit);

  const auto missing_topics_skip = check_missing_topics(bag_data);
  if (missing_topics_skip) {
    save_route_json(
      options.save_dir, options.rosbag_dir_name, "missing_topics", 0, 0.0, 0, 0,
      missing_topics_skip.value(), bag_data.timestamp_stats_map);
    rclcpp::shutdown();
    return 0;
  }

  std::vector<SequenceData> sequences =
    build_sequences(bag_data, options.search_nearest_route);

  std::cout << "Total " << sequences.size() << " sequences" << std::endl;

  for (int64_t seq_id = 0; seq_id < static_cast<int64_t>(sequences.size()); ++seq_id) {
    process_sequence(
      sequences[seq_id], seq_id, options, lane_segment_context, bag_data.timestamp_stats_map);
  }

  std::cout << "Data conversion completed!" << std::endl;

  rclcpp::shutdown();
}
